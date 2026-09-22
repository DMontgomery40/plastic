"""Synthetic-record / fake-runner tests for the operating-point screen driver (ASTRA-085).

These exercise the driver's conformance logic without a model run: chain remainder distribution,
context-group split leakage, and the full pass criterion (retained-write, completion, invalid
records). Model performance is a separate matter.
"""

from scripts.experiments.qwen_operating_point import _check_criterion, _split_into_chains, build_split


def test_split_into_chains_distributes_remainder_without_dropping():
    # ASTRA-085: five prompts into two chains must keep all five (integer division drops the tail)
    chains = _split_into_chains([1, 2, 3, 4, 5], 2)
    assert [len(c) for c in chains] == [3, 2]
    assert sorted(x for c in chains for x in c) == [1, 2, 3, 4, 5]
    assert _split_into_chains([1, 2, 3], 4) == [[1], [2], [3]]  # more chains than prompts -> singletons
    assert _split_into_chains([], 3) == [] and _split_into_chains([1], 0) == []


def _cell(eligible, eir, ror):
    return {"eligible": eligible, "eligible_intervention_rate": eir, "readonly_rate": ror,
            "accepted_change": 0, "readonly": 0, "readonly_reasons": {}}


def _op(*, anomalies=None, eir=0.0, ror=0.0, eligible=10):
    a = anomalies or {"mixed_source": 0, "missing_source": 0, "unknown_kind": 0}
    return {"prompt": _cell(eligible, eir, ror), "generation": _cell(eligible, eir, ror), "anomalies": a}


def _regime(op, sessions, complete):
    return {"operating_point": op, "sessions": sessions, "complete": complete}


def _sess(retained):
    return {"retained_both": retained}


def _report(*, fresh_op, fresh_sessions, fresh_complete, carried_op, carried_sessions, carried_complete):
    return {"fresh": _regime(fresh_op, fresh_sessions, fresh_complete),
            "carried": _regime(carried_op, carried_sessions, carried_complete)}


def test_check_criterion_passes_only_when_everything_holds():
    ok_sessions = [_sess(True)] * 10
    rep = _report(fresh_op=_op(), fresh_sessions=ok_sessions, fresh_complete=True,
                  carried_op=_op(), carried_sessions=ok_sessions, carried_complete=True)
    v = _check_criterion(rep)
    assert v["valid"] and v["pass"]


def test_check_criterion_fails_on_zero_retained_writes():
    # ASTRA-085 false-pass repro: rates fine, but NO session retained a change from both sources
    no_retain = [_sess(False)] * 10
    rep = _report(fresh_op=_op(), fresh_sessions=no_retain, fresh_complete=True,
                  carried_op=_op(), carried_sessions=no_retain, carried_complete=True)
    v = _check_criterion(rep)
    assert v["pass"] is False and v["regimes"]["fresh"]["retained_ok"] is False


def test_check_criterion_fails_when_incomplete_or_invalid():
    ok = [_sess(True)] * 10
    incomplete = _report(fresh_op=_op(), fresh_sessions=ok, fresh_complete=False,
                         carried_op=_op(), carried_sessions=ok, carried_complete=True)
    assert _check_criterion(incomplete)["pass"] is False  # not all requested work ran
    dirty = _report(fresh_op=_op(anomalies={"mixed_source": 1, "missing_source": 0, "unknown_kind": 0}), fresh_sessions=ok, fresh_complete=True,
                    carried_op=_op(), carried_sessions=ok, carried_complete=True)
    vv = _check_criterion(dirty)
    assert vv["valid"] is False and vv["pass"] is False  # a reporting anomaly invalidates the run


def test_check_criterion_fails_over_threshold_or_zero_eligible():
    ok = [_sess(True)] * 10
    hot = _report(fresh_op=_op(eir=0.2), fresh_sessions=ok, fresh_complete=True,
                  carried_op=_op(), carried_sessions=ok, carried_complete=True)
    assert _check_criterion(hot)["pass"] is False  # 20% > 10% eligible-intervention
    empty = _report(fresh_op=_op(eligible=0, eir=None, ror=0.0), fresh_sessions=ok, fresh_complete=True,
                    carried_op=_op(), carried_sessions=ok, carried_complete=True)
    assert _check_criterion(empty)["pass"] is False  # zero eligible denominator cannot pass


def test_build_split_groups_shared_context_into_one_split():
    # ASTRA-085: four rows with distinct instructions but ONE shared context must not scatter that
    # context across fit/cusum/dev/eval
    ctx = "A shared passage of context text used by several distinct questions."
    shared = [{"id": i, "instruction": f"Question {i}?", "context": ctx, "category": "closed_qa",
               "prompt": f"Question {i}?\n\n{ctx}", "text_sha256": f"h{i}"} for i in range(4)]
    singles = [{"id": 100 + i, "instruction": f"Solo {i}?", "context": "", "category": "open_qa",
                "prompt": f"Solo {i}?", "text_sha256": f"s{i}"} for i in range(4)]
    counts = {"fit": 1, "cusum": 1, "dev": 1, "eval": 1}
    split, manifest = build_split(shared + singles, lambda s: [0] * 10, counts=counts, seed=20260922, max_prompt_tokens=256)
    # the four shared-context rows land in exactly ONE split (a whole group), never scattered
    holders = [name for name in ("fit", "cusum", "dev", "eval") if any(r["context"] == ctx for r in split[name])]
    assert len(holders) == 1
    assert sum(1 for r in split[holders[0]] if r["context"] == ctx) == 4 and manifest["n_context_groups"] == 1


class _FakeRunner:
    """Minimal runner for driver logic tests: a decision-kind script, a clearable transaction log."""

    def __init__(self, kinds):
        self._kinds, self._i = list(kinds), 0
        self.transactions = []
        self.read_only, self.read_only_reason, self.committed = False, None, object()

        class _B:
            def is_finite(self, _s):
                return True

        self.backend = _B()

    def reset(self):
        self.transactions = []  # clears the log; the kind script continues (does not reset _i)


def _fake_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
    kind = runner._kinds[runner._i]
    runner._i += 1
    runner.transactions.append({"sources": {"user": 0, "model": 8}, "decision": {"kind": kind},
                                "eligible": kind != "readonly",
                                "accepted": {"delta_norm": 1.0 if kind == "commit" else 0.0}})
    return ("an answer", [1, 2, 3], [4, 5, 6])


def test_followup_isolation_detects_all_readonly_turns():
    # ASTRA-086: without clearing transactions per turn, an early commit masks later all-read-only
    # turns (false all_ok). With the fix, each turn is judged on its own records.
    from scripts.experiments.qwen_operating_point import _run_followups
    fixture = {"sessions": [{"id": f"s{i}", "turns": ["a", "b", "c"]} for i in range(4)]}  # commit,readonly,readonly each
    runner = _FakeRunner(kinds=["commit", "readonly", "readonly"] * 4)
    out = _run_followups(None, None, None, {"max_new_tokens": 4, "temperature": 0.9, "top_k": 50}, 0,
                         "unused", None, deadline=1e18, _runner=runner, _drive=_fake_drive, _fixture=fixture)
    assert out["all_ok"] is False  # the read-only turns are caught, not masked
    for s in out["sessions"]:
        assert [t["all_readonly"] for t in s["turns"]] == [False, True, True]


def _gen_settings():
    return {"max_new_tokens": 4, "temperature": 0.9, "top_k": 50}


def test_eval_uses_matched_seeds_across_fresh_and_carried():
    # ASTRA-083/086: the same per-row seed in fresh and carried (only the state differs)
    from scripts.experiments.qwen_operating_point import _eval_sessions
    seeds = []

    def rec_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        seeds.append(gen.initial_seed())
        runner.transactions.append({"sources": {"user": 8, "model": 0}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 1.0}})
        runner.transactions.append({"sources": {"user": 0, "model": 8}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 1.0}})
        return ("x", [1], [2])

    prompts = [{"id": i, "prompt": f"p{i}"} for i in range(4)]
    _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=20260922, chains=2,
                   deadline=1e18, _runner=_FakeRunner(["commit"] * 100), _drive=rec_drive)
    fresh_seeds, carried_seeds = seeds[:4], seeds[4:8]
    assert fresh_seeds == carried_seeds == [20260922 + i for i in range(4)]


def test_eval_keeps_completed_turns_when_a_chain_is_cut():
    # ASTRA-086: a deadline mid-chain keeps the completed turns (durable evidence) with an incomplete
    # flag, and the regime is marked not complete
    import time as _t
    from scripts.experiments.qwen_operating_point import _eval_sessions

    calls = {"n": 0}

    def cutting_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        calls["n"] += 1
        runner.transactions.append({"sources": {"user": 8, "model": 0}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 1.0}})
        runner.transactions.append({"sources": {"user": 0, "model": 8}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 1.0}})
        return ("x", [1], [2])

    prompts = [{"id": i, "prompt": f"p{i}"} for i in range(4)]
    # deadline in the near past for CARRIED only: run fresh with a far deadline, then check carried cut
    rep = _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=0, chains=1,
                         deadline=_t.time() - 1, _runner=_FakeRunner(["commit"] * 100), _drive=cutting_drive)
    # the far-past deadline stops both regimes immediately: nothing processed, marked incomplete
    assert rep["fresh"]["complete"] is False and rep["carried"]["complete"] is False
    assert rep["fresh"]["processed_ids"] == [] and rep["fresh"]["n_expected_sessions"] == 4


def test_summarize_flags_nonfinite_accepted_delta():
    # ASTRA-086: a nonfinite accepted delta is NOT counted as retention and is surfaced as an anomaly
    from plastic.harness.calibrate import summarize_operating_point
    txns = [
        {"sources": {"user": 8, "model": 0}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": float("inf")}},
        {"sources": {"user": 8, "model": 0}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 0.5}},
    ]
    rep = summarize_operating_point(txns)
    assert rep["prompt"]["accepted_change"] == 1  # only the finite positive one
    assert rep["anomalies"]["nonfinite_accepted"] == 1
