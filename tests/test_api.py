"""Contract tests for every API route, on a generated tiny artifact set."""

import json
import os
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from plastic.api.app import create_app
from plastic.config import ModelConfig
from plastic.data.text import encode_documents_to_bin
from plastic.harness.canary import CanarySuite
from plastic.store import ArtifactStore
from plastic.tokenizer.bpe import Tokenizer
from plastic.train.loop import TrainConfig, train

DOCS = ["alpha beta gamma delta epsilon " * 80, "one two three four five six " * 80]
PROMPT = "alpha beta gamma delta epsilon alpha beta gamma delta epsilon one two three four"
TRANSACTION_KEYS = {"index", "t_unix", "pos_start", "pos_end", "decision", "requested", "signals", "accepted",
                    "read_only", "read_only_reason", "seconds"}
ACCEPTED_KEYS = {"delta_norm", "budget_charge", "budget_used", "budget_remaining"}
SUMMARY_KEYS = {"pos", "pending", "budget_used", "read_only", "n_transactions", "cusum", "state_norms", "drift_from_anchor"}


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("api"))
    data_dir = os.path.join(root, "data", "tiny")
    os.makedirs(data_dir)
    tok = Tokenizer.train(DOCS, vocab_size=300)
    tok.save(os.path.join(data_dir, "tokenizer.json"))
    n_train = encode_documents_to_bin(tok, DOCS, os.path.join(data_dir, "train.bin"))
    n_val = encode_documents_to_bin(tok, DOCS, os.path.join(data_dir, "validation.bin"))
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "corpus": "tiny",
                "vocab_size": tok.vocab_size,
                "splits": {"train": n_train, "validation": n_val, "test": 0},
                "created_at_unix": int(time.time()),
                "tokenizer_lines": len(DOCS),
            },
            f,
        )

    text_id = train(
        TrainConfig(
            domain="text",
            model=ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=8, vocab_size=tok.vocab_size),
            artifacts_root=root, model_id="lm_tiny", data_dir=data_dir, steps=3, batch_size=2, seq_len=32,
            warmup_steps=1, eval_every=0, save_every=0, eval_batches=1, log_every=1, device="cpu", mqar_frac=0.0,
        ),
        log=lambda s: None,
    )
    physics_id = train(
        TrainConfig(
            domain="physics",
            model=ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=1, chunk=8),
            artifacts_root=root, model_id="phys_tiny", steps=3, batch_size=2, seq_len=32, episodes_per_seq=2,
            warmup_steps=1, eval_every=0, save_every=0, eval_batches=1, log_every=1, device="cpu", use_muon=False,
        ),
        log=lambda s: None,
    )

    store = ArtifactStore(root)
    CanarySuite.default_text(
        os.path.join(data_dir, "validation.bin"), vocab_size=tok.vocab_size, n_probe=2, probe_len=16
    ).save(store.canary_path(text_id))
    CanarySuite.default_physics(n_probe=2, steps=8).save(store.canary_path(physics_id))

    app = create_app(root, device="cpu")
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, store=store, root=root, data=data_dir, text=text_id, physics=physics_id)


# ---------------------------------------------------------------------- health and data
def test_health(api):
    body = api.client.get("/api/health").json()
    assert body["capabilities"] == {"create_session": True, "fork": True, "reset": True, "delete": True, "resume": True, "calibrate": True, "sleep": True}
    assert body["public"] is False
    assert body["ok"] is True
    assert body["artifacts_root"] == api.store.root and body["device"] == "cpu"
    assert body["n_models"] >= 2 and body["n_sessions"] >= 0


# ---------------------------------------------------------------------- models
@pytest.mark.parametrize('backend', ['plastic', 'qwen'])
def test_model_identity_survives_list_and_detail_without_inventing_config(tmp_path, backend):
    store = ArtifactStore(str(tmp_path))
    record = {'domain': 'text', 'status': 'completed', 'params': 12}
    if backend == 'qwen':
        record['backend'] = backend
    store.register_model('model', record)
    with TestClient(create_app(str(tmp_path), device='cpu')) as client:
        assert client.get('/api/models').json()[0]['backend'] == backend
        detail = client.get('/api/models/model').json()
        assert detail['record']['backend'] == backend
        assert detail['config'] == {}


def test_models_listing(api):
    body = api.client.get("/api/models").json()
    by_id = {m["model_id"]: m for m in body}
    assert {api.text, api.physics} <= set(by_id)
    text = by_id[api.text]
    assert text["domain"] == "text" and text["status"] == "completed"
    assert text["params"] > 0 and text["steps"] == 3 and text["has_canary"] is True
    assert text["calibrated"] is False
    assert text["eval"]["heldout_loss"] > 0 and "memory_value" in text["eval"]
    assert by_id[api.physics]["domain"] == "physics"


def test_models_listing_survives_non_finite_eval_and_sleep(api):
    # A model whose consolidation ran against an empty canary set carries NaN in its sleep
    # manifest (score_suite returns NaN for an empty probe set); an unevaluated model's eval
    # can be NaN/Inf too. GET /api/models is the first call every dashboard tab makes, so a
    # single non-finite float there must render as null, not 500 the whole response.
    api.store.register_model(
        "nan_probe",
        {
            "domain": "text",
            "status": "completed",
            "params": 1,
            "eval": {"heldout_loss": float("nan"), "memory_value": float("inf")},
            "sleep": {
                "accepted": True,
                "canary_before": {"coherence": float("nan"), "poison": 0.1},
                "canary_after": {"coherence": float("nan"), "poison": float("-inf")},
            },
        },
    )
    resp = api.client.get("/api/models")
    assert resp.status_code == 200
    rec = {m["model_id"]: m for m in resp.json()}["nan_probe"]
    assert rec["eval"]["heldout_loss"] is None and rec["eval"]["memory_value"] is None
    assert rec["sleep"]["canary_before"]["coherence"] is None and rec["sleep"]["canary_before"]["poison"] == 0.1
    assert rec["sleep"]["canary_after"]["coherence"] is None and rec["sleep"]["canary_after"]["poison"] is None


def test_model_detail(api):
    body = api.client.get(f"/api/models/{api.text}").json()
    assert body["record"]["model_id"] == api.text
    assert body["config"]["d_model"] == 32 and body["config"]["chunk"] == 8
    assert body["eval"]["heldout_loss"] > 0 and "beta_hist" in body["eval"]  # the detail eval keeps the histogram
    assert body["calibration"] is None
    assert body["canary"] == {"n_coherence": 2, "n_poison": 3}
    assert body["log"] and any("loss" in rec for rec in body["log"])



def test_calibrate_dispatches_pretrained_records_to_the_real_chat_calibration(api, tmp_path, monkeypatch):
    from plastic.harness import calibrate as calibrate_mod
    from plastic.harness.calibration_prompts import DEFAULT_CALIBRATION_PROMPTS

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    api.store.register_model("chat_pre", {"backend": "ttt", "domain": "text", "checkpoint_dir": str(ckpt), "params": 1})
    calls = []

    def fake_qwen(store, model_id, prompts, **kw):
        calls.append((model_id, list(prompts), kw))
        return calibrate_mod.Calibration(thresholds={"chunk_loss": 2.0}, achievable_fpr={"chunk_loss": 0.1},
                                         reference={"chunk_loss": [1.0, 2.0]}, n_chunks=2, target_fpr=kw["target_fpr"])

    monkeypatch.setattr(calibrate_mod, "calibrate_qwen", fake_qwen)
    monkeypatch.setattr(calibrate_mod, "calibrate_model", lambda *a, **k: pytest.fail("toy path used for a pretrained record"))
    r = api.client.post("/api/models/chat_pre/calibrate", json={})
    assert r.status_code == 200, r.text
    assert r.json()["thresholds"] == {"chunk_loss": 2.0} and r.json()["n_chunks"] == 2
    assert calls[-1][1] == list(DEFAULT_CALIBRATION_PROMPTS) and calls[-1][2]["max_new_tokens"] == 32
    r = api.client.post("/api/models/chat_pre/calibrate", json={"prompts": ["hi there"], "cusum_prompts": ["again"], "max_new_tokens": 4, "fpr": 0.2})
    assert r.status_code == 200 and calls[-1][1] == ["hi there"]
    assert calls[-1][2]["cusum_prompts"] == ["again"] and calls[-1][2]["max_new_tokens"] == 4 and calls[-1][2]["target_fpr"] == 0.2
    # request-shape family: empty prompt list and an out-of-range generation cap are rejected before any work
    assert api.client.post("/api/models/chat_pre/calibrate", json={"prompts": []}).status_code == 422
    assert api.client.post("/api/models/chat_pre/calibrate", json={"max_new_tokens": 0}).status_code == 422
    # a pretrained record whose checkpoint directory is gone is "no checkpoint yet"
    api.store.register_model("chat_gone", {"backend": "ttt", "domain": "text", "checkpoint_dir": str(tmp_path / "missing")})
    assert api.client.post("/api/models/chat_gone/calibrate", json={}).status_code == 400


def test_sleep_routes_start_a_process_and_report_its_outcome(api, tmp_path, monkeypatch):
    """The route validates the record and body, starts one process per run, and reads the outcome back from
    the report the process writes. The process here is a stand-in that writes an accepted/rejected report."""
    import sys

    from plastic.harness.config import HarnessConfig

    from plastic.api import sleep_jobs

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    api.store.register_model("chat_sleep", {"backend": "ttt", "domain": "text", "checkpoint_dir": str(ckpt), "params": 1})
    api.store.create_session("teach", model_id="chat_sleep", domain="text", harness_cfg=HarnessConfig())
    seen = {}

    def fake_argv(model_id, run_dir, artifacts_root, device, options, sessions, probes_path):
        seen.update(model_id=model_id, options=options, sessions=sessions, probes_path=probes_path, device=device)
        outcome = "rejected" if options.get("method") == "anchor" else "accepted"
        script = (
            "import json,sys,os; d=sys.argv[1]; open(os.path.join(d,'log.txt'),'a').write('[sleep] stand-in\\n');"
            f"json.dump({{'run_id': os.path.basename(d), 'parent_model_id': {model_id!r}, 'status': {outcome!r}, 'model_id': 'sleep_child' if {outcome!r}=='accepted' else None,"
            " 'gate': {'passed': " + ("False" if outcome == "rejected" else "True") + ", 'checks': []}, 'losses': [1.0]}, open(os.path.join(d,'sleep_report.json'),'w'))"
        )
        return [sys.executable, "-c", script, run_dir]

    monkeypatch.setattr(sleep_jobs, "build_sleep_argv", fake_argv)
    # a toy record has no fast weights to consolidate
    assert api.client.post(f"/api/models/{api.text}/sleep", json={}).status_code == 400
    assert api.client.post("/api/models/nope/sleep", json={}).status_code == 404
    # request-shape family
    for bad in ({"method": "nap"}, {"steps": 0}, {"replay_ratio": 2}, {"sessions": []}, {"probes": [{"question": "", "answer": "x"}]}):
        assert api.client.post("/api/models/chat_sleep/sleep", json=bad).status_code == 422, bad
    assert api.client.post("/api/models/chat_sleep/sleep", json={"sessions": ["missing"]}).status_code == 400
    r = api.client.post("/api/models/chat_sleep/sleep", json={"method": "distill", "steps": 3, "sessions": ["teach"],
                                                            "probes": [{"question": "q", "answer": "a", "paraphrase": "q2"}]})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["model_id"] == "chat_sleep" and job["status"] in ("running", "accepted") and job["n_probes"] == 1
    assert seen["options"]["method"] == "distill" and seen["options"]["steps"] == 3 and seen["sessions"] == ["teach"]
    assert seen["probes_path"] and seen["probes_path"].endswith("probes.json") and seen["device"] == "cpu"
    run_id = job["run_id"]
    for _ in range(100):
        s = api.client.get(f"/api/sleep/{run_id}").json()
        if s["status"] != "running":
            break
        time.sleep(0.05)
    assert s["status"] == "accepted" and s["report"]["model_id"] == "sleep_child" and s["log_tail"] == ["[sleep] stand-in"]
    r2 = api.client.post("/api/models/chat_sleep/sleep", json={"method": "anchor"}).json()
    for _ in range(100):
        s2 = api.client.get(f"/api/sleep/{r2['run_id']}").json()
        if s2["status"] != "running":
            break
        time.sleep(0.05)
    assert s2["status"] == "rejected" and s2["report"]["gate"]["passed"] is False
    listing = api.client.get("/api/sleep").json()
    assert {j["run_id"] for j in listing} >= {run_id, r2["run_id"]} and all("losses" not in (j["report"] or {}) for j in listing)
    assert api.client.get("/api/sleep/nope").status_code == 404


def test_model_unknown_is_404(api):
    assert api.client.get("/api/models/nope").status_code == 404
    assert api.client.post("/api/models/nope/calibrate", json={}).status_code == 404


def test_calibrate(api):
    body = api.client.post(
        f"/api/models/{api.text}/calibrate",
        json={"data_dir": api.data, "chunks": 16, "fisher_chunks": 2, "fpr": 0.1},
    ).json()
    assert body["n_chunks"] >= 16 and body["target_fpr"] == 0.1
    assert body["thresholds"] and all(v is None or isinstance(v, float) for v in body["thresholds"].values())
    assert body["reference_sizes"] and "reference" not in body
    assert body["created_at_unix"] > 0
    # the rate the sample can support, which is not the requested target_fpr
    assert set(body["achievable_fpr"]) <= set(body["thresholds"])
    assert body["achievable_fpr"] and all(v > 0 for v in body["achievable_fpr"].values())

    detail = api.client.get(f"/api/models/{api.text}").json()
    assert detail["record"]["calibrated"] is True
    assert detail["calibration"]["thresholds"] == body["thresholds"]
    assert detail["calibration"]["achievable_fpr"] == body["achievable_fpr"]


def test_an_unbounded_threshold_is_null(api):
    """A threshold can be infinite (an unbounded bound); JSON carries it as null."""
    path = os.path.join(api.store.model_dir(api.text), "calibration.json")
    original = open(path, "rb").read()
    payload = json.loads(original)
    signal = sorted(payload["thresholds"])[0]
    payload["thresholds"][signal] = float("inf")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        body = api.client.get(f"/api/models/{api.text}")
        assert body.status_code == 200
        assert body.json()["calibration"]["thresholds"][signal] is None
    finally:
        with open(path, "wb") as f:
            f.write(original)


# ---------------------------------------------------------------------- sessions
def test_create_sessions(api):
    text = api.client.post("/api/sessions", json={"model_id": api.text, "session_id": "t1",
                                                  "harness": {"enable_projection": False}})
    assert text.status_code == 200
    body = text.json()
    assert body["session_id"] == "t1" and body["domain"] == "text" and body["model_id"] == api.text
    assert body["pos"] == 0 and body["n_transactions"] == 0 and body["read_only"] is False

    physics = api.client.post("/api/sessions", json={"model_id": api.physics, "session_id": "p1"}).json()
    assert physics["session_id"] == "p1" and physics["domain"] == "physics"

    auto = api.client.post("/api/sessions", json={"model_id": api.text}).json()
    assert auto["session_id"] and auto["session_id"] not in ("t1", "p1")
    api.client.delete(f"/api/sessions/{auto['session_id']}")


def test_create_session_errors(api):
    assert api.client.post("/api/sessions", json={"model_id": "nope"}).status_code == 404
    assert api.client.post("/api/sessions", json={"model_id": api.text, "session_id": "t1"}).status_code == 409
    bad = api.client.post("/api/sessions", json={"model_id": api.text, "session_id": "bad", "harness": {"nope": 1}})
    assert bad.status_code == 400 and "nope" in bad.json()["detail"]


def test_sessions_listing(api):
    body = api.client.get("/api/sessions").json()
    by_id = {s["session_id"]: s for s in body}
    assert {"t1", "p1"} <= set(by_id)
    assert by_id["t1"]["model_id"] == api.text and by_id["p1"]["domain"] == "physics"


def test_chat(api):
    body = api.client.post("/api/sessions/t1/chat", json={"prompt": PROMPT, "max_new_tokens": 4, "seed": 0}).json()
    assert body["prompt"] == PROMPT and isinstance(body["completion"], str)
    assert body["n_tokens_in"] > 0 and body["n_tokens_out"] <= 4
    assert body["transactions"] and TRANSACTION_KEYS <= set(body["transactions"][0])
    assert body["transactions"][0]["decision"]["kind"] in ("commit", "rollback", "scale", "project", "readonly")
    signals = body["transactions"][0]["signals"]
    assert signals["n_tokens"] > 0 and "log_delta_norm" in signals and "z" in signals
    # signals are the proposed update; accepted is what the decision actually committed
    accepted = body["transactions"][0]["accepted"]
    assert ACCEPTED_KEYS <= set(accepted) and accepted["delta_norm"] >= 0.0
    assert "canary_delta_coherence" in accepted
    assert SUMMARY_KEYS <= set(body["summary"])


def test_session_detail(api):
    body = api.client.get("/api/sessions/t1").json()
    assert body["meta"]["session_id"] == "t1"
    assert body["meta"]["harness"]["enable_projection"] is False and body["meta"]["model_signature"]
    assert body["lineage"] == ["t1"]
    assert body["summary"]["n_transactions"] == body["meta"]["n_transactions"]
    assert body["summary"]["state_norms"]["s_norm"] and body["summary"]["cusum"]
    assert body["transactions"] and TRANSACTION_KEYS <= set(body["transactions"][0])
    assert body["trace"] and body["trace"][0]["kind"] == "chat"


def test_transactions_paging(api):
    full = api.client.get("/api/sessions/t1/transactions", params={"limit": 100}).json()
    assert full["total"] == len(full["items"]) >= 1
    page = api.client.get("/api/sessions/t1/transactions", params={"limit": 1, "offset": 0}).json()
    assert page["total"] == full["total"] and len(page["items"]) == 1
    assert page["items"][0]["index"] == full["items"][0]["index"]
    empty = api.client.get("/api/sessions/t1/transactions", params={"limit": 5, "offset": full["total"] + 10}).json()
    assert empty["items"] == [] and empty["total"] == full["total"]


def test_state(api):
    body = api.client.get("/api/sessions/t1/state").json()
    assert body["kind"] == "plastic"  # the toy per-layer S/h shape, discriminated for the client
    assert body["pos"] > 0 and len(body["layers"]) == 2
    layer = body["layers"][0]
    assert len(layer["s_norm_per_head"]) == 2 and layer["h_norm"] >= 0.0
    assert len(layer["singular_values"]) == 2 and len(layer["singular_values"][0]) <= 8
    assert layer["drift_from_anchor"] >= 0.0


def test_state_payload_native_backend_has_no_plastic_layers():
    # ASTRA-081 #5: a pretrained (Qwen) session's committed state has no per-layer S/h shape, so the
    # /state route must return an honest recurrent payload instead of a 500 (committed.layers) or
    # fabricated tensors. Driven with a fake session so it needs no model/isolated deps.
    import torch

    from plastic.api.service import state_payload

    class _QwenLikeState:  # no `.layers`, exactly like QwenState
        pass

    class _FakeBackend:
        def state_norms(self, _s):
            return {"recurrent_norm": [1.5, 2.0], "recurrent_norm_total": 2.5}

        def state_delta(self, _a, _b):
            return [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]  # per-unit norms 5.0, 0.0

    class _FakeRunner:
        committed = _QwenLikeState()
        anchor = _QwenLikeState()
        pos = 12
        backend = _FakeBackend()

    class _FakeSession:
        runner = _FakeRunner()
        backend_kind = "qwen"

    body = state_payload(_FakeSession())
    assert body["kind"] == "recurrent" and body["backend"] == "qwen" and body["pos"] == 12
    assert body["recurrent_norm_total"] == 2.5
    assert [u["recurrent_norm"] for u in body["units"]] == [1.5, 2.0]
    assert body["units"][0]["drift_from_anchor"] == 5.0 and body["units"][1]["drift_from_anchor"] == 0.0
    assert "layers" not in body  # never the plastic shape, never fabricated S/h tensors


def test_fork(api):
    child = api.client.post("/api/sessions/t1/fork", json={"child_session_id": "t1_fork"}).json()
    assert child["session_id"] == "t1_fork" and child["parent_session_id"] == "t1"
    assert child["root_session_id"] == "t1" and child["pos"] == api.store.load_session_meta("t1")["pos"]

    grand = api.client.post("/api/sessions/t1_fork/fork", json={}).json()
    detail = api.client.get(f"/api/sessions/{grand['session_id']}").json()
    assert detail["lineage"] == ["t1", "t1_fork", grand["session_id"]]

    assert api.client.post("/api/sessions/t1/fork", json={"child_session_id": "t1_fork"}).status_code == 409
    assert api.client.post("/api/sessions/nope/fork", json={}).status_code == 404


def test_reset_and_resume(api):
    before = api.client.get("/api/sessions/t1_fork").json()["meta"]
    assert before["pos"] > 0
    reset = api.client.post("/api/sessions/t1_fork/reset").json()
    assert reset["pos"] == 0 and reset["read_only"] is False
    resumed = api.client.post("/api/sessions/t1_fork/resume").json()
    assert resumed["read_only"] is False and resumed["session_id"] == "t1_fork"
    assert api.client.post("/api/sessions/nope/reset").status_code == 404
    assert api.client.post("/api/sessions/nope/resume").status_code == 404


def test_delete(api):
    created = api.client.post("/api/sessions", json={"model_id": api.text, "session_id": "doomed"}).json()
    assert created["session_id"] == "doomed"
    assert api.client.delete("/api/sessions/doomed").json() == {"deleted": True}
    assert api.client.get("/api/sessions/doomed").status_code == 404
    assert api.client.delete("/api/sessions/doomed").status_code == 404
    assert "doomed" not in {s["session_id"] for s in api.client.get("/api/sessions").json()}


def test_session_unknown_is_404(api):
    assert api.client.get("/api/sessions/nope").status_code == 404
    assert api.client.get("/api/sessions/nope/state").status_code == 404
    assert api.client.get("/api/sessions/nope/transactions").status_code == 404
    assert api.client.post("/api/sessions/nope/chat", json={"prompt": "x"}).status_code == 404


# ---------------------------------------------------------------------- training jobs


# ---------------------------------------------------------------------- red team


# ---------------------------------------------------------------------- sleep


def test_model_summary_exposes_sleep_field():
    from plastic.api.service import model_summary
    from plastic.store import ArtifactStore

    store = ArtifactStore("/nonexistent-root-for-summary-only")
    rec = {"model_id": "m", "domain": "text", "type": "sleep_consolidation", "parent_model_id": "base",
           "sleep": {"accepted": True, "delta_coherence": 0.01, "delta_poison": -0.2}}
    out = model_summary(store, rec)
    assert out["sleep"] == rec["sleep"]
    assert out["type"] == "sleep_consolidation" and out["parent_model_id"] == "base"
    # a model without a sleep record reports None, not a fabricated value
    assert model_summary(store, {"model_id": "m2", "domain": "text"})["sleep"] is None


def test_state_payload_native_missing_or_failed_drift_is_null():
    # ASTRA-094: a failed / missing / nonfinite measurement is null (unavailable), never a measured
    # zero; a genuine zero is preserved. Driven with fakes, no model.
    import torch

    from plastic.api.service import state_payload

    class _QwenLikeState:
        pass

    def _sess(norms, delta):
        class _Backend:
            def state_norms(self, _s):
                return norms

            def state_delta(self, _a, _b):
                if isinstance(delta, Exception):
                    raise delta
                return delta

        class _Runner:
            committed = _QwenLikeState()
            anchor = _QwenLikeState()
            pos = 5
            backend = _Backend()

        class _Session:
            runner = _Runner()
            backend_kind = "qwen"

        return _Session()

    # state_delta raises -> every drift null, but the recurrent norms are still reported
    body = state_payload(_sess({"recurrent_norm": [1.0, 2.0], "recurrent_norm_total": 3.0}, RuntimeError("boom")))
    assert [u["drift_from_anchor"] for u in body["units"]] == [None, None]
    assert [u["recurrent_norm"] for u in body["units"]] == [1.0, 2.0]

    # a short delta list -> the unmatched unit is null; a genuine zero drift is preserved as 0.0
    body = state_payload(_sess({"recurrent_norm": [1.0, 2.0, 3.0], "recurrent_norm_total": 4.0},
                               [torch.tensor([0.0]), torch.tensor([4.0])]))
    assert [u["drift_from_anchor"] for u in body["units"]] == [0.0, 4.0, None]

    # an absent total is null, not zero
    body = state_payload(_sess({"recurrent_norm": [1.0]}, [torch.tensor([2.0])]))
    assert body["recurrent_norm_total"] is None

    # nonfinite norm / total / drift are all null
    body = state_payload(_sess({"recurrent_norm": [float("inf")], "recurrent_norm_total": float("nan")},
                               [torch.tensor([float("inf")])]))
    assert body["units"][0]["recurrent_norm"] is None
    assert body["units"][0]["drift_from_anchor"] is None
    assert body["recurrent_norm_total"] is None


def test_session_detail_exposes_loaded_calibration_not_replaced_artifact(api):
    # ASTRA-096 #2: a session's active calibration is the one it verified at open, not the model's
    # CURRENT saved artifact. A separate calibration process replacing the artifact same-model must
    # not change what the open session (and its running policy) uses or what the UI labels active.
    from plastic.harness.calibrate import Calibration

    mid = api.text
    api.client.post(f"/api/models/{mid}/calibrate",
                    json={"data_dir": api.data, "chunks": 16, "fisher_chunks": 2, "fpr": 0.1})
    model_dir = api.store.model_dir(mid)

    sid = api.client.post("/api/sessions", json={"model_id": mid, "session_id": "cal_identity"}).json()["session_id"]
    detail = api.client.get(f"/api/sessions/{sid}").json()
    assert detail["summary"]["calibration"] == "installed"
    assert detail["calibration"] is not None
    loaded_thresholds = detail["calibration"]["thresholds"]
    assert loaded_thresholds  # the session exposes its loaded calibration's thresholds

    # a SEPARATE calibration process replaces the saved artifact with a valid same-signature one whose
    # thresholds differ (bumped by 100); the model signature is unchanged so it is still "installable"
    replacement = Calibration.load(model_dir)
    replacement.thresholds = {k: (None if v is None else float(v) + 100.0) for k, v in replacement.thresholds.items()}
    replacement.save(model_dir)

    model_after = api.client.get(f"/api/models/{mid}").json()
    assert model_after["calibration"]["thresholds"] != loaded_thresholds  # the model detail shows the replacement

    # the OPEN session still exposes the calibration it actually loaded, not the replacement
    detail_after = api.client.get(f"/api/sessions/{sid}").json()
    assert detail_after["summary"]["calibration"] == "installed"
    assert detail_after["calibration"]["thresholds"] == loaded_thresholds
    assert detail_after["calibration"]["thresholds"] != model_after["calibration"]["thresholds"]


def test_session_summary_reports_effective_generation_write_policy(api):
    # ASTRA-101: writes_generation is the EFFECTIVE policy (backend source capability OR
    # learn_from_generation). A plastic session freezes generation by default; the flag makes it
    # write-eligible. (Qwen's generation writes regardless -- that path is model-gated.)
    sid = api.client.post("/api/sessions", json={"model_id": api.text}).json()["session_id"]
    body = api.client.get(f"/api/sessions/{sid}").json()
    assert body["summary"]["backend"] == "plastic"
    assert body["summary"]["writes_generation"] is False  # default: generation frozen read-only

    sid2 = api.client.post(
        "/api/sessions", json={"model_id": api.text, "harness": {"learn_from_generation": True}}
    ).json()["session_id"]
    body2 = api.client.get(f"/api/sessions/{sid2}").json()
    assert body2["summary"]["writes_generation"] is True  # the override makes generation write-eligible


def test_sleep_argv_forwards_the_replay_revision_and_rejects_a_malformed_one(api):
    """ASTRA-181: the API's replay_revision reaches the sleep process; a non-hex value is a request-shape error."""
    from plastic.api.schemas import SleepRequest
    from plastic.api.sleep_jobs import build_sleep_argv

    argv = build_sleep_argv("m", "/run", "/root", "cpu", {"method": "replay", "replay_revision": "5feaf2fd3ffca7"}, None, None)
    assert argv[argv.index("--replay-revision") + 1] == "5feaf2fd3ffca7"
    assert "--replay-revision" not in build_sleep_argv("m", "/run", "/root", "cpu", {"method": "replay"}, None, None)
    assert SleepRequest().model_dump(exclude_none=True).get("replay_revision") is None
    assert api.client.post("/api/models/nope/sleep", json={"replay_revision": "not-a-sha!"}).status_code == 422
