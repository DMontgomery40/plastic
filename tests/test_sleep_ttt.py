"""Sleep for the TTT backend: provenance harvesting, packing, anchoring math, recall scoring, config
validation and CLI dispatch are tested without a model. The end-to-end run (anchor and a two-step replay
on the 760M base) is skipped unless the local checkpoint exists, like tests/test_ttt_backend.py."""

import json
import os

import pytest
import torch

from plastic.harness.config import HarnessConfig
from plastic.sleep.recall import RecallProbe, RecallReport, load_probes, normalize, run_probes, score_reply
from plastic.sleep.ttt import (
    SleepConfig,
    anchor_update,
    fast_weight_parameters,
    harvest_sessions,
    harvest_summary,
    pack_examples,
)
from plastic.store import ArtifactStore

CKPT = os.environ.get("TTT_CHECKPOINT", "artifacts/models/ttt-mlp/TTT-MLP-760M-Base-Pile-8k")
HAVE_CKPT = os.path.exists(os.path.join(CKPT, "config.json"))


def _tx(index, start, end, kind="commit", *, read_only=False):
    return {"index": index, "pos_start": start, "pos_end": end, "decision": {"kind": kind, "reasons": [], "scale": 1.0},
            "requested": {"kind": kind, "reasons": [], "scale": 1.0}, "signals": {"pos_start": start, "pos_end": end, "n_tokens": end - start},
            "accepted": {"delta_norm": 0.0 if kind == "rollback" else 1.0}, "read_only": read_only, "read_only_reason": None, "t_unix": 0}


def _session(store, sid, model_id, turns, *, log_only=False, legacy=False):
    """turns: list of (prompt, completion, [chunk kinds]) laid out consecutively in 8-token chunks; a kind of
    "reset" restarts the position (the runner's transaction counter keeps counting). ``legacy`` writes traces
    without the transaction-index range, as sessions recorded before that field existed."""
    store.register_model(model_id, {"backend": "ttt", "domain": "text"}) if model_id not in {r["model_id"] for r in store.list_models()} else None
    store.create_session(sid, model_id=model_id, domain="text", harness_cfg=HarnessConfig(log_only=log_only))
    pos = 0
    idx = 0
    for prompt, completion, kinds in turns:
        if kinds == ["reset"]:
            pos = 0
            continue
        first = idx
        for kind in kinds:
            ro = kind == "readonly"
            store.append_transaction(sid, _tx(idx, pos, pos + 8, kind, read_only=ro))
            pos += 8
            idx += 1
        rec = {"t_unix": 0, "kind": "chat", "prompt": prompt, "completion": completion, "pos_end": pos, "n_transactions": len(kinds)}
        if not legacy:
            rec.update(tx_start=first if kinds else None, tx_end=idx if kinds else None)
        store.append_trace(sid, rec)


def test_harvest_keeps_only_fully_accepted_turns_and_counts_exclusions(tmp_path):
    store = ArtifactStore(str(tmp_path))
    _session(store, "a", "m", [
        ("teach one", "kept one", ["commit", "commit"]),
        ("teach two", "dropped: rolled back", ["commit", "rollback"]),
        ("teach three", "kept three", ["scale", "project"]),
        ("teach four", "dropped: read only", ["readonly"]),
        ("teach five", "", ["commit"]),
    ])
    _session(store, "b", "m", [("obs", "kept in observation", ["commit"])], log_only=True)
    _session(store, "other", "n", [("x", "another model", ["commit"])])
    hs = harvest_sessions(store, "m")
    by = {h.session_id: h for h in hs}
    assert set(by) == {"a", "b"}
    assert [t.reason for t in by["a"].turns] == ["accepted", "rolled_back", "accepted", "read_only", "empty_completion"]
    assert [t.completion for t in by["a"].accepted_turns] == ["kept one", "kept three"]
    assert by["b"].log_only is True and by["b"].accepted_turns[0].completion == "kept in observation"
    s = harvest_summary(hs)
    assert s["turns_by_reason"] == {"accepted": 3, "rolled_back": 1, "read_only": 1, "empty_completion": 1}
    assert s["accepted_tokens"] == 16 + 16 + 8 and s["excluded_tokens"] == 16 + 8 + 8
    assert all(sess["has_committed_state"] is False for sess in s["sessions"])
    with pytest.raises(ValueError, match="not found"):
        harvest_sessions(store, "m", ["a", "zzz"])
    assert [h.session_id for h in harvest_sessions(store, "m", ["b"])] == ["b"]


def test_reset_between_turns_keeps_each_epoch_apart(tmp_path):
    """Positions restart after a reset while transaction indices keep counting: the second epoch's committed
    turn must be accepted on its own, not merged with the first epoch's rolled-back one (ASTRA-156 #1)."""
    store = ArtifactStore(str(tmp_path))
    _session(store, "s", "m", [
        ("first epoch", "rolled back", ["commit", "rollback"]),
        ("", "", ["reset"]),
        ("after reset", "kept", ["commit", "commit"]),
    ])
    turns = harvest_sessions(store, "m")[0].turns
    assert [(t.prompt, t.reason, t.n_chunks, t.n_tokens) for t in turns] == [("first epoch", "rolled_back", 2, 16), ("after reset", "accepted", 2, 16)]


def test_legacy_traces_group_by_position_only_when_positions_never_restart(tmp_path):
    store = ArtifactStore(str(tmp_path))
    _session(store, "mono", "m", [("a", "kept a", ["commit"]), ("b", "kept b", ["commit", "scale"])], legacy=True)
    _session(store, "reset", "m", [("a", "x", ["commit"]), ("", "", ["reset"]), ("b", "y", ["commit"])], legacy=True)
    by = {h.session_id: h for h in harvest_sessions(store, "m")}
    assert [(t.reason, t.n_chunks) for t in by["mono"].turns] == [("accepted", 1), ("accepted", 2)]
    assert [t.reason for t in by["reset"].turns] == ["ambiguous_provenance", "ambiguous_provenance"]
    assert harvest_summary([by["reset"]])["turns_by_reason"] == {"ambiguous_provenance": 2}


def test_trace_whose_transactions_are_missing_is_not_accepted(tmp_path):
    store = ArtifactStore(str(tmp_path))
    store.register_model("m", {"backend": "ttt", "domain": "text"})
    store.create_session("s", model_id="m", domain="text", harness_cfg=HarnessConfig())
    store.append_transaction("s", _tx(0, 0, 8))
    store.append_trace("s", {"t_unix": 0, "kind": "chat", "prompt": "p", "completion": "c", "pos_end": 16, "n_transactions": 2, "tx_start": 0, "tx_end": 2})
    assert harvest_sessions(store, "m")[0].turns[0].reason == "missing_chunks"


def test_turn_with_no_chunks_is_excluded_not_guessed(tmp_path):
    store = ArtifactStore(str(tmp_path))
    store.register_model("m", {"backend": "ttt", "domain": "text"})
    store.create_session("s", model_id="m", domain="text", harness_cfg=HarnessConfig())
    store.append_trace("s", {"t_unix": 0, "kind": "chat", "prompt": "p", "completion": "c", "pos_end": 0, "n_transactions": 0, "tx_start": 0, "tx_end": 0})
    h = harvest_sessions(store, "m")[0]
    assert h.turns[0].reason == "no_chunks" and not h.accepted_turns


def test_pack_examples_respects_window_masks_padding_and_carries_weights():
    ex = [([1, 2, 3], [-100, 2, 3]), ([4, 5], [4, 5]), ([6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11])]
    packed = pack_examples(ex, 6, pad_id=0)
    # a (ids, labels) example gets weight 1 on every target and 0 elsewhere; padding is label -100, weight 0
    assert packed[0] == ([1, 2, 3, 4, 5, 0], [-100, 2, 3, 4, 5, -100], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    assert packed[1] == ([6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11], [1.0] * 6)
    # explicit weights travel with their tokens
    weighted = pack_examples([([1, 2, 3], [-100, 2, 3], [0.0, 0.2, 1.0])], 4, 0)
    assert weighted == [([1, 2, 3, 0], [-100, 2, 3, -100], [0.0, 0.2, 1.0, 0.0])]
    # an over-long example is truncated, never dropped
    assert pack_examples([(list(range(10)), list(range(10)))], 4, 0) == [([0, 1, 2, 3], [0, 1, 2, 3], [1.0] * 4)]


def test_session_weights_scale_prompt_tokens_and_zero_unlabeled():
    from plastic.sleep.ttt import session_weights

    sft = [-100, -100, -100, 41, 2]          # BOS and user tokens masked in SFT; assistant reply and EOS kept
    labels_all = [-100, 30, 31, 41, 2]        # session_loss="all"
    assert session_weights(sft, labels_all, 0.2) == [0.0, 0.2, 0.2, 1.0, 1.0]
    assert session_weights(sft, sft, 0.2) == [0.0, 0.0, 0.0, 1.0, 1.0]  # assistant-only labels: prompt weight irrelevant


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.ones(2, 3, 4))
        self.b1 = torch.nn.Parameter(torch.zeros(2, 1, 4))
        self.W2 = torch.nn.Parameter(torch.ones(2, 4, 3))
        self.b2 = torch.nn.Parameter(torch.zeros(2, 1, 3))
        self.other = torch.nn.Parameter(torch.ones(1))


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_modeling_block = _Block()


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
        self.head = torch.nn.Linear(3, 3)


def test_anchor_moves_w0_toward_the_mean_session_state_and_leaves_other_params():
    m = _Toy()
    names = [n for n, _ in fast_weight_parameters(m)]
    assert names == [f"layers.{l}.seq_modeling_block.{n}" for l in (0, 1) for n in ("W1", "b1", "W2", "b2")]
    params = [p for _, p in fast_weight_parameters(m)]
    s1 = [p.detach().unsqueeze(0) + 1.0 for p in params]
    s2 = [p.detach().unsqueeze(0) + 3.0 for p in params]
    head_before = m.head.weight.detach().clone()
    rel = anchor_update(m, [s1, s2], lam=0.5)
    # mean offset 2.0, lambda 0.5 -> every fast weight moved by exactly +1.0
    assert torch.allclose(m.layers[0].seq_modeling_block.W1, torch.full((2, 3, 4), 2.0))
    assert torch.allclose(m.layers[1].seq_modeling_block.b2, torch.full((2, 1, 3), 1.0))
    assert torch.equal(m.head.weight, head_before) and float(m.layers[0].seq_modeling_block.other) == 1.0
    assert set(rel) == set(names) and all(v > 0 for v in rel.values())
    with pytest.raises(ValueError, match="leaves"):
        anchor_update(m, [s1[:3]], lam=0.5)
    with pytest.raises(ValueError, match="at least one"):
        anchor_update(m, [], lam=0.5)


def test_recall_scoring_and_probe_loading(tmp_path):
    assert normalize("  The CAT's name is Marlowe! ") == "the cat s name is marlowe"
    assert score_reply("Marlowe", "Your cat is called Marlowe, I believe.") == {"contains": True, "exact": False}
    assert score_reply("Marlowe", "marlowe") == {"contains": True, "exact": True}
    assert score_reply("Marlowe", "I do not know.") == {"contains": False, "exact": False}
    p = tmp_path / "probes.json"
    p.write_text(json.dumps([{"question": "What is my cat's name?", "answer": "Marlowe", "paraphrase": "Remind me what I call my cat."},
                             {"question": "Where do I live?", "answer": "Denver"}]), encoding="utf-8")
    probes = load_probes(str(p))
    assert len(probes) == 2 and probes[0].paraphrase and probes[1].paraphrase is None
    replies = {"What is my cat's name?": "Marlowe.", "Remind me what I call my cat.": "A cat.", "Where do I live?": "You live in Denver."}
    rep = run_probes(probes, lambda q: replies[q], lambda q, a: -0.5 if a == "Marlowe" else -2.0)
    d = rep.to_dict()
    assert d["mean_answer_logprob"] == (-0.5 + -2.0) / 2 and rep.results[1].answer_logprob == -0.5  # paraphrase row carries it too
    assert run_probes(probes, lambda q: replies[q]).to_dict()["mean_answer_logprob"] is None
    assert d["distinct_ratio"] == 1.0 and d["max_cluster_share"] == 1 / 3
    collapsed = run_probes(probes, lambda q: "A pleasure to meet you, Marlowe. I am glad you asked.").to_dict()
    assert collapsed["distinct_ratio"] == 1 / 3 and collapsed["max_cluster_share"] == 1.0 and collapsed["recalled"] == 1  # one reply for all; only the cat probe "hits"
    assert (d["n_probes"], d["recalled"], d["recalled_exact"], d["n_paraphrase"], d["recalled_paraphrase"]) == (2, 2, 1, 1, 0)
    assert isinstance(rep, RecallReport) and rep.count(variant="paraphrase") == 0
    with pytest.raises(ValueError):
        RecallProbe.from_dict({"question": "", "answer": "x"})


def test_gate_treats_nonfinite_measurements_as_failures():
    """The gate logic is a closure inside sleep_ttt; this pins the same rule through the module's helper."""
    from plastic.sleep.ttt import gate_from_measurements

    before = {"heldout_nll": {"mean": 2.0, "median": 1.5, "tokens": 10}, "canary": {"coherence": 3.0, "poison": 5.0}, "recall": None}
    ok = gate_from_measurements(before, {"heldout_nll": {"mean": 2.01, "median": 1.5, "tokens": 10}, "canary": {"coherence": 3.05, "poison": 5.0}, "recall": None},
                                tolerance_nll=0.05, tolerance_canary={"coherence": 0.1, "poison": 0.1})
    assert ok["measured"] and ok["passed"] is True and [c["passed"] for c in ok["checks"]] == [True, True, True]
    nan = gate_from_measurements(before, {"heldout_nll": {"mean": float("nan"), "median": 1.5, "tokens": 10}, "canary": {"coherence": float("inf"), "poison": 5.0}, "recall": None},
                                 tolerance_nll=0.05, tolerance_canary={"coherence": 0.1, "poison": 0.1})
    assert nan["passed"] is False and [c["passed"] for c in nan["checks"]] == [False, False, True] and nan["checks"][0]["value"] is None
    none = gate_from_measurements({"heldout_nll": None, "canary": None, "recall": None}, {"heldout_nll": None, "canary": None, "recall": None},
                                  tolerance_nll=0.05, tolerance_canary={})
    assert none["measured"] is False and none["passed"] is None and "exploratory" in none["note"]
    # behavioral collapse: NLL improves but one sentence answers many questions -> the gate fails
    collapsed = gate_from_measurements({"heldout_nll": {"mean": 1.68, "median": 0.9, "tokens": 800}, "canary": None, "recall": {"max_cluster_share": 0.03}},
                                       {"heldout_nll": {"mean": 1.63, "median": 0.4, "tokens": 800}, "canary": None, "recall": {"max_cluster_share": 0.47}},
                                       tolerance_nll=0.05, tolerance_canary={})
    assert collapsed["passed"] is False and [c["name"] for c in collapsed["checks"] if not c["passed"]] == ["reply_cluster_share"]
    # a model that was already repetitive is not blamed for staying so; fewer repeats always pass
    same = gate_from_measurements({"heldout_nll": None, "canary": None, "recall": {"max_cluster_share": 0.6}},
                                  {"heldout_nll": None, "canary": None, "recall": {"max_cluster_share": 0.55}}, tolerance_nll=0.05, tolerance_canary={})
    assert same["passed"] is True
    # legacy reports without the field measure nothing on this axis
    legacy = gate_from_measurements({"heldout_nll": None, "canary": None, "recall": {"recalled": 0}}, {"heldout_nll": None, "canary": None, "recall": {"recalled": 1}},
                                    tolerance_nll=0.05, tolerance_canary={})
    assert legacy["measured"] is False


def test_gate_rejects_the_saved_step100_collapsed_runs_and_passes_their_baselines():
    """ASTRA-167: replay the ACTUAL fresh-session reply lists saved by the step-100 all×40 runs (both arms,
    before and after, verbatim and paraphrase) through RecallReport and the gate. Both runs lowered held-out
    NLL and must be rejected; the before-lists are the baseline and pass."""
    import json

    from plastic.sleep.recall import RecallReport, RecallResult
    from plastic.sleep.ttt import gate_from_measurements

    fix = json.load(open("tests/fixtures/sleep_step100_all40_replies.json"))
    for arm in ("replay", "ungated"):
        reports = {k: RecallReport([RecallResult(**x) for x in fix[arm][k]]).to_dict() for k in ("before", "after")}
        assert reports["before"]["max_cluster_share"] < 0.05 and reports["before"]["distinct_ratio"] == 1.0
        assert reports["after"]["max_cluster_share"] > 0.4, arm  # one sentence for 13-14 of 30 probes
        gate = gate_from_measurements({"heldout_nll": fix[arm]["heldout_nll"]["before"], "canary": None, "recall": reports["before"]},
                                      {"heldout_nll": fix[arm]["heldout_nll"]["after"], "canary": None, "recall": reports["after"]},
                                      tolerance_nll=0.05, tolerance_canary={})
        names = {c["name"]: c["passed"] for c in gate["checks"]}
        assert names["heldout_nll_mean_rise"] is True and names["reply_cluster_share"] is False and gate["passed"] is False, (arm, names)
        # the baseline against itself passes on every axis
        base = gate_from_measurements({"heldout_nll": fix[arm]["heldout_nll"]["before"], "canary": None, "recall": reports["before"]},
                                      {"heldout_nll": fix[arm]["heldout_nll"]["before"], "canary": None, "recall": reports["before"]},
                                      tolerance_nll=0.05, tolerance_canary={})
        assert base["passed"] is True


def test_batch_mix_preserves_batch_size_and_keeps_a_session_row():
    """ASTRA-170: round(batch*ratio) with max(1, batch-n) changed the batch size and the realized ratio."""
    from plastic.sleep.ttt import batch_mix

    assert batch_mix(2, 0.5, True) == (1, 1)
    assert batch_mix(2, 0.8, True) == (1, 1)     # not realizable at batch 2: the caller records 0.5
    assert batch_mix(5, 0.8, True) == (1, 4)
    assert batch_mix(4, 1.0, True) == (1, 3)     # a session row is always kept
    assert batch_mix(3, 0.0, True) == (3, 0)
    assert batch_mix(3, 0.5, False) == (3, 0)    # no replay data: every row is a session row
    assert batch_mix(1, 0.9, True) == (1, 0)
    for b, r in ((2, 0.5), (2, 0.8), (5, 0.8), (4, 1.0), (7, 0.3)):
        assert sum(batch_mix(b, r, True)) == b
    with pytest.raises(ValueError):
        batch_mix(0, 0.5, True)


def test_session_labels_supervise_user_tokens_by_default_and_keep_sft_masks_otherwise():
    from plastic.sleep.ttt import session_labels

    ids = [1, 30, 31, 32, 40, 41, 2]            # BOS, user tokens, assistant tag+reply, EOS
    sft = [-100, -100, -100, -100, 41, 41, 2]   # what the SFT encoder produces: prompt masked
    assert session_labels(ids, sft, "all") == [-100, 30, 31, 32, 40, 41, 2]
    assert session_labels(ids, sft, "assistant") == sft
    assert session_labels([], [], "all") == []


def test_sleep_config_validation():
    SleepConfig().validate()
    SleepConfig(provenance="all").validate()
    SleepConfig(session_loss="assistant").validate()
    with pytest.raises(ValueError):
        SleepConfig(session_loss="user").validate()
    for bad in ({"method": "nap"}, {"target": "lora"}, {"steps": 0}, {"replay_ratio": 1.5}, {"anchor_lambda": -0.1}, {"lr": 0.0}, {"seq_len": 8}, {"provenance": "some"}, {"prompt_loss_weight": 1.5}):
        with pytest.raises(ValueError):
            SleepConfig(**bad).validate()


def test_cli_sleep_dispatches_ttt_records(tmp_path, monkeypatch, capsys):
    from plastic import cli
    from plastic.sleep import ttt as ttt_mod

    root = str(tmp_path / "artifacts")
    ArtifactStore(root).register_model("chat_m", {"backend": "ttt", "domain": "text"})
    calls = []
    monkeypatch.setattr(ttt_mod, "sleep_ttt", lambda store, mid, cfg, **kw: calls.append((mid, cfg, kw)) or {"status": "accepted", "model_id": "sleep_1"})
    probes = tmp_path / "p.json"
    probes.write_text('[{"question": "q", "answer": "a"}]', encoding="utf-8")
    rc = cli.main(["sleep", "chat_m", "--artifacts-root", root, "--method", "distill", "--target", "all", "--steps", "3",
                   "--sessions", "s1", "s2", "--recall", str(probes), "--device", "cpu", "--provenance", "all"])
    assert rc == 0
    mid, cfg, kw = calls[-1]
    assert mid == "chat_m" and cfg.method == "distill" and cfg.target == "all" and cfg.steps == 3 and cfg.device == "cpu"
    assert cfg.provenance == "all"
    assert kw["session_ids"] == ["s1", "s2"] and [p.question for p in kw["probes"]] == ["q"]
    assert json.loads(capsys.readouterr().out)["model_id"] == "sleep_1"


@pytest.mark.skipif(not HAVE_CKPT, reason="TTT checkpoint not available")
def test_sleep_end_to_end_on_the_base_model(tmp_path):
    """anchor from a real committed state, then a two-step replay on w0: both must produce a loadable child
    with lineage and a report that keeps before/after measurements apart. The gate is exercised with a tiny
    held-out set; the run stays under a minute on MPS."""
    from plastic.backends.ttt_lm.backend import TTTBackend, _checkpoint_digest
    from plastic.session.runner import Session
    from plastic.sleep.ttt import sleep_ttt

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    store = ArtifactStore(str(tmp_path / "artifacts"))
    store.register_model("base", {"backend": "ttt", "domain": "text", "status": "completed", "params": 759_000_000, "chunk": 16,
                                  "checkpoint_dir": os.path.abspath(CKPT), "checkpoint_digest": _checkpoint_digest(CKPT), "chat_tuned": False})
    s = Session.create(store, model_id="base", session_id="teach", device=dev,
                       harness_cfg=HarnessConfig(log_only=True, learn_from_generation=True, enable_projection=False, enable_budget=False))
    s.chat("My cat is called Marlowe. Please remember that.", max_new_tokens=8, seed=0)
    probes = [RecallProbe("What is my cat called?", "Marlowe")]
    cfg = SleepConfig(method="anchor", anchor_lambda=0.2, device=dev, heldout_rows=2, replay_rows=0, replay_ratio=0.0, seq_len=128,
                      recall_max_new_tokens=8, tolerance_nll=10.0)
    rep = sleep_ttt(store, "base", cfg, probes=probes, log=lambda m: None)
    assert rep["status"] == "accepted", rep
    assert rep["harvest"]["turns_by_reason"] == {"accepted": 1}
    assert rep["before"]["recall"]["n_probes"] == 1 and rep["after"]["recall"]["n_probes"] == 1
    child = store.load_model_record(rep["model_id"])
    assert child["parent_model_id"] == "base" and child["type"] == "sleep" and child["backend"] == "ttt"
    assert child["checkpoint_digest"] != store.load_model_record("base")["checkpoint_digest"]
    assert os.path.exists(os.path.join(store.model_dir(rep["model_id"]), "sleep_report.json"))
    be = TTTBackend.load(child["checkpoint_dir"], device=dev)
    assert torch.isfinite(be.logits_full([1, 2, 3])).all()
    # the parent's checkpoint is untouched
    assert _checkpoint_digest(CKPT) == store.load_model_record("base")["checkpoint_digest"]

    cfg2 = SleepConfig(method="replay", target="w0", steps=2, batch_size=1, seq_len=64, device=dev, heldout_rows=0, replay_rows=0,
                       replay_ratio=0.0, recall_max_new_tokens=4, scan_checkpoint_groups=4)
    rep2 = sleep_ttt(store, "base", cfg2, log=lambda m: None)
    # no held-out rows and no canary suite: the child exists but the run is marked unmeasured, never verified
    assert rep2["status"] == "accepted_unmeasured" and len(rep2["losses"]) == 2 and all(l == l for l in rep2["losses"])
    assert rep2["gate"]["measured"] is False and rep2["gate"]["passed"] is None and "exploratory" in rep2["gate"]["note"]
    assert rep["gate"]["measured"] is True and rep["gate"]["passed"] is True


def test_dream_selection_drops_degenerate_and_duplicate_dreams_and_keeps_high_gain():
    from plastic.sleep.dream import Dream, DreamReport, is_degenerate, select_dreams

    assert is_degenerate("yes")
    assert is_degenerate("the the the the the the cat")
    assert is_degenerate("meet you meet you meet you meet you")
    assert not is_degenerate("My cat is called Marlowe and I live in Denver.")

    def d(text, t_lp, s_lp):
        return Dream("p", text, [1, 5, 6], [-100, 5, 6], t_lp, s_lp, "s")

    cands = [d("My cat is called Marlowe.", -1.0, -3.0), d("My cat is called Marlowe!", -1.1, -3.1), d("I hope you find what you are looking for.", -1.0, -1.05),
             d("hi", -0.1, -5.0), d("I live in Denver, Colorado, near the mountains.", -1.5, -2.5)]
    rep = DreamReport(generated=len(cands))
    kept = select_dreams(cands, min_gain=0.5, max_keep=10, report=rep)
    assert [k.text for k in kept] == ["My cat is called Marlowe.", "I live in Denver, Colorado, near the mountains."]  # sorted by gain
    assert (rep.degenerate, rep.duplicate, rep.low_gain) == (1, 1, 1)
    assert select_dreams(cands, min_gain=0.5, max_keep=1, report=DreamReport()) == kept[:1]
    assert rep.to_dict()["kept"][0]["gain"] == 2.0


def test_sleep_config_accepts_dream_and_bounds_its_knobs():
    SleepConfig(method="dream").validate()
    with pytest.raises(ValueError):
        SleepConfig(method="dream", dream_per_prompt=0).validate()
    with pytest.raises(ValueError):
        SleepConfig(method="dream", dream_max_new_tokens=2).validate()


def test_frozen_teacher_backend_is_isolated_from_student_updates_when_all_parameters_train():
    """ASTRA-175 #1: with target all, the teacher must read from weights the student optimizer cannot move."""
    import types

    from plastic.sleep.ttt import frozen_teacher_backend

    class FakeBackend:
        def __init__(self, model):
            self.model, self.tokenizer, self.config, self.device, self.checkpoint_digest = model, None, types.SimpleNamespace(vocab_size=4, mini_batch_size=16), torch.device("cpu"), "d"

    m = _Toy()
    same = frozen_teacher_backend(FakeBackend(m), "w0")
    assert same.model is m  # W0 changes are overridden by the loaded session state: the live model is the teacher

    captured = {}

    def fake_factory(model, tok, cfg, *, device, checkpoint_digest):
        captured["model"] = model
        return "teacher-backend"

    assert frozen_teacher_backend(FakeBackend(m), "all", factory=fake_factory) == "teacher-backend"
    teacher_model = captured["model"]
    assert teacher_model is not m
    before = teacher_model.head.weight.detach().clone()
    with torch.no_grad():
        m.head.weight.add_(1.0)  # a student step
    assert torch.equal(teacher_model.head.weight, before) and not any(p.requires_grad for p in teacher_model.parameters())


def test_reply_slices_align_the_same_reply_tokens_in_both_renderings():
    from plastic.sleep.dream import reply_slices

    ts, ss = reply_slices(teacher_prefix=7, student_prefix=9, reply_len=5)
    assert (ts.start, ts.stop, ss.start, ss.stop) == (6, 11, 8, 13)
    teacher_ids = list(range(100, 107)) + [1, 2, 3, 4, 5]
    student_ids = list(range(200, 209)) + [1, 2, 3, 4, 5]
    # logits at position t predict token t+1: the sliced positions predict exactly the reply in both sequences
    assert [teacher_ids[i + 1] for i in range(ts.start, ts.stop)] == [1, 2, 3, 4, 5] == [student_ids[i + 1] for i in range(ss.start, ss.stop)]


def test_teacher_states_keep_session_identity_when_a_state_is_skipped(tmp_path):
    """ASTRA-175 #3: an incompatible first state must not relabel the next session's state."""
    from plastic.sleep.ttt import _teacher_states

    store = ArtifactStore(str(tmp_path))
    _session(store, "old", "m", [("a", "x", ["commit"])])
    _session(store, "valid", "m", [("b", "y", ["commit"])])
    _session(store, "late", "m", [("c", "z", ["commit"])])
    store.save_runner_state("old", {"committed": {"tag": "old"}}, summary={})
    store.save_runner_state("valid", {"committed": {"tag": "valid"}}, summary={})
    store.save_runner_state("late", {"committed": {"tag": "late"}}, summary={})

    class FakeBackend:
        def load_state_dict(self, d):
            if d["tag"] in ("old", "late"):
                raise ValueError(f"incompatible {d['tag']}")
            return f"state:{d['tag']}"

    report = {}
    pairs = _teacher_states(store, FakeBackend(), harvest_sessions(store, "m"), report, lambda s: None)
    assert pairs == [("valid", "state:valid")]
    assert sorted(x["session_id"] for x in report["skipped_states"]) == ["late", "old"]


def test_dream_report_records_every_rejection_with_its_reason():
    from plastic.sleep.dream import Dream, DreamReport, select_dreams

    def d(text, gain):
        return Dream("p", text, [1, 2], [-100, 2], gain, 0.0, "s")

    rep = DreamReport()
    select_dreams([d("a fine dream about cats and cities", 2.0), d("a fine dream about cats and cities!", 1.9), d("hi", 3.0),
                   d("generic text with nothing new here", 0.0), d("another good dream about the cello lesson", 1.0)],
                  min_gain=0.5, max_keep=1, report=rep)
    reasons = sorted(r["reason"].split(" ")[0] for r in rep.to_dict()["rejected"])
    assert reasons == ["degenerate", "duplicate", "gain", "over"] and len(rep.to_dict()["kept"]) == 1
