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
TRANSACTION_KEYS = {"index", "t_unix", "pos_start", "pos_end", "decision", "requested", "signals", "read_only", "seconds"}
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
    assert body["ok"] is True
    assert body["artifacts_root"] == api.store.root and body["device"] == "cpu"
    assert body["n_models"] == 2 and body["n_sessions"] >= 0


def test_data_listing(api):
    body = api.client.get("/api/data").json()
    entry = next(d for d in body if d["name"] == "tiny")
    assert entry["dir"] == api.data and entry["corpus"] == "tiny"
    assert entry["vocab_size"] > 0 and entry["splits"]["train"] > 0


# ---------------------------------------------------------------------- models
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


def test_model_detail_and_log(api):
    body = api.client.get(f"/api/models/{api.text}").json()
    assert body["record"]["model_id"] == api.text
    assert body["config"]["d_model"] == 32 and body["config"]["chunk"] == 8
    assert body["eval"]["heldout_loss"] > 0 and "beta_hist" in body["eval"]  # the detail eval keeps the histogram
    assert body["calibration"] is None
    assert body["canary"] == {"n_coherence": 2, "n_poison": 3}
    assert body["log"] and any("loss" in rec for rec in body["log"])

    log = api.client.get(f"/api/models/{api.text}/log", params={"limit": 2}).json()
    assert len(log) <= 2 and log == body["log"][-len(log) :]


def test_model_unknown_is_404(api):
    assert api.client.get("/api/models/nope").status_code == 404
    assert api.client.get("/api/models/nope/log").status_code == 404
    assert api.client.post("/api/models/nope/calibrate", json={}).status_code == 404


def test_calibrate(api):
    body = api.client.post(
        f"/api/models/{api.text}/calibrate",
        json={"data_dir": api.data, "chunks": 16, "fisher_chunks": 2, "fpr": 0.1},
    ).json()
    assert body["n_chunks"] >= 16 and body["target_fpr"] == 0.1
    assert body["thresholds"] and all(isinstance(v, float) for v in body["thresholds"].values())
    assert body["reference_sizes"] and "reference" not in body
    assert body["created_at_unix"] > 0

    detail = api.client.get(f"/api/models/{api.text}").json()
    assert detail["record"]["calibrated"] is True
    assert detail["calibration"]["thresholds"] == body["thresholds"]


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
    assert SUMMARY_KEYS <= set(body["summary"])


def test_chat_on_a_physics_session_is_400(api):
    r = api.client.post("/api/sessions/p1/chat", json={"prompt": "hello"})
    assert r.status_code == 400 and "text session" in r.json()["detail"]


def test_physics_episode(api):
    body = api.client.post("/api/sessions/p1/physics", json={"steps": 16, "mu": 0.12, "seed": 0}).json()
    assert body["mu"] == 0.12 and body["steps"] == 16
    assert len(body["per_step"]) == 16 and set(body["per_step"][0]) == {"t", "base_mse", "frozen_mse", "adaptive_mse"}
    assert set(body["means"]) == {"base_mse", "frozen_mse", "adaptive_mse"}
    assert len(body["transactions"]) == 2  # 16 steps of chunk 8
    assert SUMMARY_KEYS <= set(body["summary"])


def test_physics_on_a_text_session_is_400(api):
    r = api.client.post("/api/sessions/t1/physics", json={"steps": 8})
    assert r.status_code == 400 and "physics session" in r.json()["detail"]


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
    assert body["pos"] > 0 and len(body["layers"]) == 2
    layer = body["layers"][0]
    assert len(layer["s_norm_per_head"]) == 2 and layer["h_norm"] >= 0.0
    assert len(layer["singular_values"]) == 2 and len(layer["singular_values"][0]) <= 8
    assert layer["drift_from_anchor"] >= 0.0


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
def test_train_job(api):
    started = api.client.post(
        "/api/train",
        json={
            "domain": "text", "data_dir": api.data, "steps": 2, "batch_size": 2, "seq_len": 96,
            "d_model": 32, "layers": 1, "heads": 2, "chunk": 8, "eval_every": 0, "save_every": 0, "device": "cpu",
        },
    ).json()
    model_id = started["model_id"]
    assert model_id and started["pid"] > 0

    jobs = api.client.get("/api/train/jobs").json()
    assert model_id in {j["model_id"] for j in jobs}
    assert {"model_id", "pid", "status", "exit_code", "started_at_unix"} == set(jobs[0])

    deadline = time.time() + 300
    status = api.client.get(f"/api/train/{model_id}").json()
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(1.0)
        status = api.client.get(f"/api/train/{model_id}").json()
    assert status["status"] == "finished", status
    assert status["exit_code"] == 0, status.get("error")
    assert status["phase"] == "completed"
    assert status["latest"]["step"] >= 1 and status["eval"]["heldout_loss"] > 0

    cancelled = api.client.post(f"/api/train/{model_id}/cancel").json()
    assert cancelled == {"model_id": model_id, "status": "finished"}
    assert model_id in {m["model_id"] for m in api.client.get("/api/models").json()}


def test_train_errors(api):
    assert api.client.get("/api/train/nope").status_code == 404
    assert api.client.post("/api/train/nope/cancel").status_code == 404
    assert api.client.post("/api/train", json={"domain": "text", "steps": 1, "batch_size": 1, "seq_len": 32}).status_code == 400
    duplicate = api.client.post(
        "/api/train",
        json={"domain": "text", "data_dir": api.data, "model_id": api.text, "steps": 1, "batch_size": 1, "seq_len": 32},
    )
    assert duplicate.status_code == 409


# ---------------------------------------------------------------------- red team
def test_redteam(api):
    summary = api.client.post(
        "/api/redteam",
        json={"model_id": api.text, "data_dir": api.data, "prefixes": 1, "prefix_len": 16,
              "suffix_len": 8, "steps": 2, "families": ["pgd", "random"]},
    ).json()
    run_id = summary["run_id"]
    assert summary["model_id"] == api.text and summary["created_at_unix"] > 0
    assert set(summary["families"]) == {"pgd", "random"}
    family = summary["families"]["pgd"]
    assert family["n"] == 1 and 0.0 <= family["gated_fraction"] <= 1.0
    assert {"damage_mean", "damage_max", "provisional_damage_max", "constraint_violated_fraction",
            "over_threshold_fraction"} <= set(family)

    listing = api.client.get("/api/redteam").json()
    assert run_id in {r["run_id"] for r in listing}

    run = api.client.get(f"/api/redteam/{run_id}").json()
    assert run["summary"]["run_id"] == run_id
    assert len(run["results"]) == 2
    result = run["results"][0]
    assert result["family"] in ("pgd", "random") and len(result["payload_ids"]) == 8
    assert result["decisions"] and result["nll_payload"] > 0

    assert api.client.get("/api/redteam/nope").status_code == 404
    assert api.client.post("/api/redteam", json={"model_id": "nope"}).status_code == 404
    assert api.client.post("/api/redteam", json={"model_id": api.physics, "data_dir": api.data}).status_code == 400
    bad = api.client.post("/api/redteam", json={"model_id": api.text, "data_dir": api.data, "families": ["nope"]})
    assert bad.status_code == 400


# ---------------------------------------------------------------------- sleep
def test_sleep(api):
    api.client.post("/api/sessions", json={"model_id": api.text, "session_id": "sleepy",
                                           "harness": {"enable_projection": False}})
    for seed in range(3):
        assert api.client.post(
            "/api/sessions/sleepy/chat", json={"prompt": PROMPT, "max_new_tokens": 6, "seed": seed}
        ).status_code == 200

    manifest = api.client.post(
        "/api/sleep",
        json={"model_id": api.text, "sessions": ["sleepy"], "core_data_dir": api.data, "steps": 2,
              "seq_len": 16, "batch_size": 4, "tolerance": {"coherence": 10.0, "poison": 10.0}},
    ).json()
    assert manifest["accepted"] is True
    assert manifest["model_id"].startswith("sleep_")
    assert manifest["tolerance"]["coherence"] == 10.0

    child = api.client.get(f"/api/models/{manifest['model_id']}").json()
    assert child["record"]["parent_model_id"] == api.text
    assert child["record"]["type"] == "sleep_consolidation"

    rejected = api.client.post(
        "/api/sleep",
        json={"model_id": api.text, "sessions": ["sleepy"], "core_data_dir": api.data, "steps": 2,
              "seq_len": 16, "batch_size": 4, "tolerance": {"coherence": -1.0, "poison": 10.0}},
    ).json()
    assert rejected["accepted"] is False and "model_id" not in rejected

    assert api.client.post("/api/sleep", json={"model_id": "nope"}).status_code == 404
    assert api.client.post("/api/sleep", json={"model_id": api.physics}).status_code == 400
