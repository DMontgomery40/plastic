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


def test_eval_marks_incomplete_when_deadline_precedes_all_work():
    # ASTRA-086 boundary: a deadline already in the past before any turn runs processes nothing in
    # either regime and marks both not complete (the timing deadline is not evidence work ran)
    import time as _t
    from scripts.experiments.qwen_operating_point import _eval_sessions

    def cutting_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        runner.transactions.append({"sources": {"user": 8, "model": 0}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 1.0}})
        runner.transactions.append({"sources": {"user": 0, "model": 8}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 1.0}})
        return ("x", [1], [2])

    prompts = [{"id": i, "prompt": f"p{i}"} for i in range(4)]
    rep = _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=0, chains=1,
                         deadline=_t.time() - 1, _runner=_FakeRunner(["commit"] * 100), _drive=cutting_drive)
    # nothing processed, both regimes incomplete, no phantom sessions kept
    assert rep["fresh"]["complete"] is False and rep["carried"]["complete"] is False
    assert rep["fresh"]["processed_ids"] == [] and rep["fresh"]["n_expected_sessions"] == 4
    assert rep["fresh"]["sessions"] == [] and rep["carried"]["sessions"] == []


def test_eval_preserves_completed_turns_then_interruption(monkeypatch):
    # ASTRA-087: the transition the boundary test above does NOT reach -- a deadline that trips AFTER
    # some turns complete must KEEP those turns' records, seeds and transactions as durable evidence,
    # mark the chain incomplete, and mark the regime not complete. Fresh (singleton groups) finishes
    # fully; the carried chain is cut mid-way after two successful turns.
    import scripts.experiments.qwen_operating_point as qop
    from scripts.experiments.qwen_operating_point import _eval_sessions

    clock = {"t": 0.0}
    monkeypatch.setattr(qop.time, "time", lambda: clock["t"])
    seeds = []

    def clock_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        seeds.append(gen.initial_seed())
        runner.transactions.append({"sources": {"user": 8, "model": 0}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 1.0}})
        runner.transactions.append({"sources": {"user": 0, "model": 8}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 1.0}})
        clock["t"] += 1.0  # each completed turn advances the wall clock by one unit
        return ("x", [1], [2])

    prompts = [{"id": i, "prompt": f"p{i}"} for i in range(4)]
    # deadline=5: fresh's four singleton turns (clock 0->4) all finish; the carried four-turn chain
    # starts at clock 4, runs rows 0 and 1 (clock ->6), then the row-2 pre-check trips at 6 > 5.
    rep = _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=700,
                         chains=1, deadline=5.0, _runner=_FakeRunner(["commit"] * 100), _drive=clock_drive)

    # fresh ran to completion; nothing marked incomplete
    assert rep["fresh"]["complete"] is True
    assert rep["fresh"]["processed_ids"] == [0, 1, 2, 3]
    assert all(not s.get("incomplete") for s in rep["fresh"]["sessions"])

    # carried cut mid-chain after two successful turns: those two are retained as durable evidence
    assert rep["carried"]["complete"] is False
    assert rep["carried"]["processed_ids"] == [0, 1]
    assert rep["carried"]["n_expected_sessions"] == 1 and rep["carried"]["n_complete_sessions"] == 0
    (sess,) = rep["carried"]["sessions"]
    assert sess["incomplete"] is True
    assert [t["id"] for t in sess["turns"]] == [0, 1]  # only the completed turns
    assert len(rep["carried"]["raw_transactions"]) == 4  # both completed turns' transactions kept (2 each)
    # matched per-row seeds: fresh row i and carried row i share seed_base + i
    assert seeds[:4] == [700 + i for i in range(4)]
    assert seeds[4:6] == [700, 701]  # carried only reached rows 0 and 1


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


def _rep(fresh, carried):
    return {"fresh": {"complete": fresh}, "carried": {"complete": carried}}


_PROV_OK = {"ok": True, "reasons": [], "distinct": []}


def _fu(**kw):
    """Follow-up output that ran the DECLARED fixture unless overridden."""
    return {"fixture_is_declared": True, "fixture_sha256": "fx", **kw}


def test_screen_verdict_separates_completion_from_pass():
    # ASTRA-092: `complete` means all requested work RAN (fit + both eval regimes + follow-ups, not
    # skipped/deadline-cut); the follow-up all_ok is a quality result folded into PASS, not completion.
    from scripts.experiments.qwen_operating_point import _screen_verdict

    ok_v = {"valid": True, "pass": True}

    def sv(**kw):
        return _screen_verdict(**{"fit_complete": True, "report": _rep(True, True), "verdict": ok_v, "run_provenance": _PROV_OK, **kw})

    # everything ran and passed
    v = sv(followups=_fu(all_ok=True))
    assert v["complete"] is True and v["pass"] is True and v["followups_ok"] is True and v["valid"] is True

    # KEY separation: fully RAN but follow-ups flagged read-only turns -> complete, NOT pass
    v = sv(followups=_fu(all_ok=False))
    assert v["complete"] is True and v["followups_ran"] is True and v["followups_ok"] is False and v["pass"] is False

    # follow-ups skipped (missing fixture): followups_ok null (never False), not complete
    v = sv(followups={"skipped": "fixture missing", "fixture_is_declared": False})
    assert v["followups_ok"] is None and v["followups_ran"] is False and v["complete"] is False and v["pass"] is False

    # follow-ups deadline-cut: ran but incomplete -> not complete
    v = sv(followups=_fu(all_ok=False, incomplete=True))
    assert v["followups_ran"] is False and v["complete"] is False and v["pass"] is False

    # an incomplete eval regime -> not complete regardless of follow-ups
    v = sv(report=_rep(True, False), followups=_fu(all_ok=True))
    assert v["complete"] is False and v["pass"] is False

    # everything ran but the criterion failed -> complete, not pass
    v = sv(followups=_fu(all_ok=True), verdict={"valid": True, "pass": False})
    assert v["complete"] is True and v["pass"] is False

    # fit incomplete -> not complete
    v = sv(fit_complete=False, followups=_fu(all_ok=True))
    assert v["complete"] is False


def test_screen_verdict_never_credits_an_undeclared_fixture_or_mixed_provenance():
    # ASTRA-109: a DIFFERENT fixture than the one declared at run creation is a different follow-up
    # attempt -- its all_ok never reads as the declared fixture passing; and a run whose invocations do
    # not share one clean identified source is kept (complete) but is not a VALID fixed screen
    from scripts.experiments.qwen_operating_point import _screen_verdict
    ok_v = {"valid": True, "pass": True}

    v = _screen_verdict(fit_complete=True, report=_rep(True, True), verdict=ok_v, run_provenance=_PROV_OK,
                        followups=_fu(all_ok=True, fixture_is_declared=False, fixture_sha256="other"))
    assert v["followups_ok"] is None and v["followups_ran"] is False
    assert v["complete"] is False and v["pass"] is False
    assert (v["followups_fixture_sha256"], v["followups_fixture_is_declared"]) == ("other", False)
    # a follow-up output missing the declaration flag is never assumed declared
    v = _screen_verdict(fit_complete=True, report=_rep(True, True), verdict=ok_v, run_provenance=_PROV_OK,
                        followups={"all_ok": True})
    assert v["followups_ok"] is None and v["pass"] is False

    bad = {"ok": False, "reasons": ["2 distinct invocation provenances (mixed source or runtime)"], "distinct": []}
    v = _screen_verdict(fit_complete=True, report=_rep(True, True), verdict=ok_v, run_provenance=bad, followups=_fu(all_ok=True))
    assert v["complete"] is True                    # the work ran ...
    assert v["valid"] is False and v["pass"] is False and v["provenance_ok"] is False  # ... but is not a valid fixed screen
    assert v["provenance_reasons"] == bad["reasons"]


def _fake_args(**over):
    from types import SimpleNamespace
    base = dict(n_fit=8, n_cusum=4, n_dev=4, n_eval=4, max_prompt_tokens=256, max_new_tokens=8, eval_chains=2, smoke=False, exclusions=None,
                device="cpu")
    base.update(over)
    return SimpleNamespace(**base)


def test_settings_identity_changes_with_each_determining_setting():
    from scripts.experiments.qwen_operating_point import _settings_identity
    base = _settings_identity(_fake_args(), "digestX", "rev1", protocol=_proto())
    assert base == _settings_identity(_fake_args(), "digestX", "rev1", protocol=_proto())           # stable
    assert base != _settings_identity(_fake_args(max_new_tokens=16), "digestX", "rev1", protocol=_proto())  # decoding changed
    assert base != _settings_identity(_fake_args(n_fit=16), "digestX", "rev1", protocol=_proto())   # counts changed
    assert base != _settings_identity(_fake_args(), "digestY", "rev1", protocol=_proto())           # checkpoint changed
    assert base != _settings_identity(_fake_args(), "digestX", "rev2", protocol=_proto())           # dataset revision changed
    assert base != _settings_identity(_fake_args(), "digestX", "rev1", "excl_v2", protocol=_proto())  # exclusion CONTENT bound
    assert base != _settings_identity(_fake_args(device="mps"), "digestX", "rev1", protocol=_proto())  # device bound (FABLE-085 #1)


def _split_of(ids_per):
    return {name: [{"id": i, "prompt": f"p{i}", "context": "", "category": "c", "text_sha256": f"h{i}"} for i in ids]
            for name, ids in ids_per.items()}


def _manifest_of(corpus_hash, split):
    return {
        "corpus_hash": corpus_hash,
        "actual_counts": {k: len(v) for k, v in split.items()},
        "excluded_over_cap": 0,
        "selected_records": {name: [{"id": r["id"], "prompt": r["prompt"], "instruction": r["prompt"],
                                     "context": r["context"], "category": r["category"], "sha256": r["text_sha256"]}
                                    for r in recs]
                             for name, recs in split.items()},
    }


def test_split_identity_detects_membership_and_order_changes():
    from scripts.experiments.qwen_operating_point import _split_identity
    base = _split_of({"fit": [0, 1], "cusum": [2], "dev": [3], "eval": [4, 5]})
    assert _split_identity(base) == _split_identity(_split_of({"fit": [0, 1], "cusum": [2], "dev": [3], "eval": [4, 5]}))
    swap = _split_of({"fit": [4, 1], "cusum": [2], "dev": [3], "eval": [0, 5]})  # 0<->4 cross-split swap
    assert _split_identity(base) != _split_identity(swap)
    reorder = _split_of({"fit": [1, 0], "cusum": [2], "dev": [3], "eval": [4, 5]})  # within-fit reorder
    assert _split_identity(base) != _split_identity(reorder)


def test_split_from_manifest_reconstructs_ordered_split_and_refuses_legacy():
    import pytest

    from scripts.experiments.qwen_operating_point import RunConflict, _split_from_manifest
    split = _split_of({"fit": [0, 1], "cusum": [2], "dev": [3], "eval": [4, 5]})
    recon = _split_from_manifest(_manifest_of("h1", split))
    assert [r["id"] for r in recon["eval"]] == [4, 5]
    assert [r["prompt"] for r in recon["fit"]] == ["p0", "p1"]
    # a manifest predating ordered-split resume (no prompt) cannot be safely reconstructed
    with pytest.raises(RunConflict):
        _split_from_manifest({"selected_records": {"fit": [{"id": 0, "sha256": "h0"}], "cusum": [], "dev": [], "eval": []}})


def test_reconcile_run_resume_restores_pinned_split_not_the_rebuild(tmp_path):
    # ASTRA-098 [P1]: a resume must evaluate the ORIGINAL split, never a rebuild whose membership
    # changed while the union corpus_hash stayed the same -- else a completed fit prompt leaks into eval
    from scripts.experiments.qwen_operating_point import _reconcile_run
    split_a = _split_of({"fit": [0], "cusum": [1], "dev": [2], "eval": [3]})
    minted = []

    def mint():
        minted.append(f"qwen_{len(minted) + 1}")
        return minted[-1]

    mode, mid, _m, _s = _reconcile_run(str(tmp_path), _manifest_of("h1", split_a), split_a, "idA", mint)
    assert mode == "fresh" and mid == "qwen_1" and len(minted) == 1
    manifest_bytes = (tmp_path / "split-manifest.json").read_bytes()

    # resume passing a DIFFERENT rebuilt split with the SAME union corpus_hash (cross-split swap)
    split_b = _split_of({"fit": [3], "cusum": [2], "dev": [1], "eval": [0]})
    mode2, mid2, _m2, s2 = _reconcile_run(str(tmp_path), _manifest_of("h1", split_b), split_b, "idA", mint)
    assert mode2 == "resume" and mid2 == "qwen_1" and len(minted) == 1
    assert [r["id"] for r in s2["eval"]] == [3] and [r["id"] for r in s2["fit"]] == [0]  # the ORIGINAL split
    assert (tmp_path / "split-manifest.json").read_bytes() == manifest_bytes  # manifest untouched


def test_reconcile_run_refuses_on_settings_corpus_or_prior_artifacts(tmp_path):
    import pytest

    from scripts.experiments.qwen_operating_point import RunConflict, _reconcile_run
    split_a = _split_of({"fit": [0], "cusum": [1], "dev": [2], "eval": [3]})
    _reconcile_run(str(tmp_path), _manifest_of("h1", split_a), split_a, "idA", lambda: "qwen_1")
    before = (tmp_path / "split-manifest.json").read_bytes()
    with pytest.raises(RunConflict):  # a different configuration
        _reconcile_run(str(tmp_path), _manifest_of("h1", split_a), split_a, "idB", lambda: "qwen_x")
    with pytest.raises(RunConflict):  # same settings but a drifted corpus union
        _reconcile_run(str(tmp_path), _manifest_of("h2", split_a), split_a, "idA", lambda: "qwen_x")
    assert (tmp_path / "split-manifest.json").read_bytes() == before  # never overwritten on a conflict


def test_reconcile_run_preserves_prior_artifacts_without_run_record(tmp_path):
    # ASTRA-098 [P2]: a manifest present with NO run-record (interrupted init, or a pre-run-record dir)
    # must be preserved, not overwritten as a fresh run
    import pytest

    from scripts.experiments.qwen_operating_point import RunConflict, _reconcile_run
    (tmp_path / "split-manifest.json").write_text('{"corpus_hash": "old"}')
    before = (tmp_path / "split-manifest.json").read_bytes()
    split_new = _split_of({"fit": [0], "cusum": [1], "dev": [2], "eval": [3]})
    with pytest.raises(RunConflict):
        _reconcile_run(str(tmp_path), _manifest_of("new", split_new), split_new, "idA", lambda: "qwen_x")
    assert (tmp_path / "split-manifest.json").read_bytes() == before
    assert not (tmp_path / "run-record.json").exists()


def test_apply_exclusions_reserves_ids_duplicates_and_context_groups():
    # ASTRA-086: reserve prior-exposed rows by id or normalized-text-hash prefix (duplicates), and
    # expand to whole nonempty-context groups, WITHOUT over-excluding context-less rows
    from scripts.experiments.qwen_operating_point import _apply_exclusions
    rows = [
        {"id": 1, "context": "Passage  A", "text_sha256": "aaa111", "prompt": "q1"},  # excluded by id
        {"id": 2, "context": "PASSAGE A", "text_sha256": "bbb222", "prompt": "q2"},    # same context as id 1 -> excluded
        {"id": 3, "context": "", "text_sha256": "ccc333", "prompt": "q3"},             # context-less, kept
        {"id": 4, "context": "", "text_sha256": "ded444", "prompt": "q4"},             # excluded by hash prefix
        {"id": 5, "context": "Passage B", "text_sha256": "eee555", "prompt": "q5"},     # unrelated, kept
    ]
    kept, report = _apply_exclusions(rows, ids={1}, hash_prefixes=("ded4",))
    assert {r["id"] for r in kept} == {3, 5}          # id 1, its context sibling 2, and prefix-match 4 reserved
    assert report["excluded_directly"] == 2           # id 1 and the prefix match 4
    assert report["excluded_context_groups"] == 1     # the "passage a" group (normalized, case-insensitive)
    assert report["n_kept"] == 2 and report["n_excluded_total"] == 3

    # an empty spec keeps everything
    kept2, report2 = _apply_exclusions(rows, ids=set(), hash_prefixes=())
    assert len(kept2) == 5 and report2["n_excluded_total"] == 0


def _seed_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
    # two commits per turn (prompt + generation source); the completion encodes the seed so a resume's
    # equivalence can be checked by the per-turn completion
    runner.transactions.append({"sources": {"user": 8, "model": 0}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 1.0}})
    runner.transactions.append({"sources": {"user": 0, "model": 8}, "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 1.0}})
    return (f"s{gen.initial_seed()}", [1], [2])


def _eval_sig(rep):
    return {reg: ([s["ids"] for s in rep[reg]["sessions"]],
                  [[t["completion"] for t in s["turns"]] for s in rep[reg]["sessions"]],
                  rep[reg]["complete"]) for reg in ("fresh", "carried")}


def test_eval_resume_extends_to_uninterrupted_equivalence(monkeypatch):
    # ASTRA-092: an interrupted-then-resumed eval reproduces the uninterrupted result exactly -- same
    # per-regime session ids, per-turn seeds (via completions) and complete flags -- with no leakage
    # (the same pinned groups) and matched seeds preserved across the interruption
    import scripts.experiments.qwen_operating_point as qop
    from scripts.experiments.qwen_operating_point import _eval_sessions

    prompts = [{"id": i, "prompt": f"p{i}"} for i in range(6)]
    ref = _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=700, chains=3,
                         deadline=1e18, _runner=_FakeRunner(["commit"] * 400), _drive=_seed_drive)
    assert ref["fresh"]["complete"] and ref["carried"]["complete"]
    full = _eval_sig(ref)

    clock = {"t": 0.0}
    monkeypatch.setattr(qop.time, "time", lambda: clock["t"])

    def clock_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        out = _seed_drive(runner, tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, gen=gen)
        clock["t"] += 1.0
        return out

    cap = {"fresh": [], "carried": []}

    def prog(regime, rec, txns, next_group):
        cap[regime].append({"record": rec, "txns": txns})

    # deadline mid-way: some fresh sessions complete, then the run is cut
    _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=700, chains=3,
                   deadline=3.5, on_progress=prog, _runner=_FakeRunner(["commit"] * 400), _drive=clock_drive)
    assert 0 < len(cap["fresh"]) < 6  # genuinely interrupted mid fresh regime

    resumed = _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=700, chains=3,
                             deadline=1e18, restore=cap, _runner=_FakeRunner(["commit"] * 400), _drive=clock_drive)
    assert _eval_sig(resumed) == full  # the resumed run reproduces the uninterrupted result exactly


def test_eval_operating_point_aggregates_complete_sessions_only(monkeypatch):
    # ASTRA-092: a cut group is durable evidence in raw_transactions/sessions but must NOT enter the
    # operating_point aggregate (no mixed denominator)
    import scripts.experiments.qwen_operating_point as qop
    from scripts.experiments.qwen_operating_point import _eval_sessions

    clock = {"t": 0.0}
    monkeypatch.setattr(qop.time, "time", lambda: clock["t"])

    def clock_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        out = _seed_drive(runner, tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, gen=gen)
        clock["t"] += 1.0
        return out

    prompts = [{"id": i, "prompt": f"p{i}"} for i in range(4)]
    # fresh's 4 singleton turns finish (clock 0->4); the carried chain of 4 starts at clock 4 and is
    # cut after 2 turns (clock ->6), so the carried regime has one partial (incomplete) group
    rep = _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=0, chains=1,
                         deadline=5.5, _runner=_FakeRunner(["commit"] * 100), _drive=clock_drive)
    carried = rep["carried"]
    assert carried["complete"] is False and carried["n_complete_sessions"] == 0
    assert len(carried["raw_transactions"]) == 4          # the 2 completed turns' transactions are evidence
    # but the operating point aggregated zero complete sessions -> no eligible denominator from the partial
    assert carried["operating_point"]["prompt"]["eligible"] == 0
    assert carried["operating_point"]["generation"]["eligible"] == 0


def test_eval_progress_round_trips_and_rejects_mismatched_identity(tmp_path):
    # ASTRA-100/103: eval progress restores complete sessions and refuses reuse under a different
    # content/config/calibration/policy identity, never a silent fresh start
    from scripts.experiments.qwen_operating_point import (
        RunConflict, _append_eval_progress, _init_eval_progress, _load_eval_progress,
    )
    import pytest

    path = str(tmp_path / "eval-progress.jsonl")
    assert _load_eval_progress(path, "idA") is None  # absent -> fresh start

    _init_eval_progress(path, "idA")
    _append_eval_progress(path, "fresh", _wrec([0], "fresh"), [_wtx()])
    _append_eval_progress(path, "carried", _wrec([1, 2], "carried"), [_wtx(), _wtx("rollback")])
    _append_eval_progress(path, "fresh", _wrec([3], "fresh"), [_wtx()])

    restore = _load_eval_progress(path, "idA")
    assert [s["record"]["ids"] for s in restore["fresh"]] == [[0], [3]]       # order preserved per regime
    assert [s["record"]["ids"] for s in restore["carried"]] == [[1, 2]]
    assert restore["carried"][0]["txns"] == [_wtx(), _wtx("rollback")]

    with pytest.raises(RunConflict):  # a different identity is refused, the log preserved
        _load_eval_progress(path, "idB")
    assert _load_eval_progress(path, "idA") is not None  # still usable under the right identity


def test_eval_progress_skips_a_malformed_trailing_line(tmp_path):
    from scripts.experiments.qwen_operating_point import _append_eval_progress, _init_eval_progress, _load_eval_progress
    path = str(tmp_path / "eval-progress.jsonl")
    _init_eval_progress(path, "idA")
    _append_eval_progress(path, "fresh", _wrec([0], "fresh"), [])
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"regime": "fresh", "record": {"ids": [1]')  # a crash mid-append: truncated JSON
    restore = _load_eval_progress(path, "idA")
    assert [s["record"]["ids"] for s in restore["fresh"]] == [[0]]  # the completed one kept, the partial skipped


def test_eval_identity_binds_calibration_content_and_policy():
    from types import SimpleNamespace
    from scripts.experiments.qwen_operating_point import _eval_identity
    def cal(thresholds, reference, cusum_reference, sig="qwen:x"):
        return SimpleNamespace(thresholds=thresholds, reference=reference, cusum_reference=cusum_reference, model_signature=sig)

    cal_a = cal({"chunk_loss": 1.0}, {"chunk_loss": [0.1, 0.2]}, [1, 2, 3])
    recs1 = [{"id": 0, "text_sha256": "h0", "prompt": "p0"}, {"id": 1, "text_sha256": "h1", "prompt": "p1"}]
    recs_diff_content = [{"id": 0, "text_sha256": "hZ", "prompt": "different"}, {"id": 1, "text_sha256": "h1", "prompt": "p1"}]
    base = _eval_identity(recs1, "settings1", cal_a, {"freeze_on_alarm": True})
    assert base == _eval_identity(recs1, "settings1", cal_a, {"freeze_on_alarm": True})  # stable
    # decision-distinct calibrations must never share a progress identity (ASTRA-105):
    assert base != _eval_identity(recs1, "settings1", cal({"chunk_loss": 9.0}, {"chunk_loss": [0.1, 0.2]}, [1, 2, 3]), {"freeze_on_alarm": True})  # thresholds
    assert base != _eval_identity(recs1, "settings1", cal({"chunk_loss": 1.0}, {"chunk_loss": [9.9, 9.9]}, [1, 2, 3]), {"freeze_on_alarm": True})  # reference window values
    assert base != _eval_identity(recs1, "settings1", cal({"chunk_loss": 1.0}, {"chunk_loss": [0.1, 0.2]}, [7, 8, 9]), {"freeze_on_alarm": True})  # cusum VALUES, same length
    assert base != _eval_identity(recs_diff_content, "settings1", cal_a, {"freeze_on_alarm": True})  # SAME ids, diff CONTENT
    assert base != _eval_identity(recs1, "settings2", cal_a, {"freeze_on_alarm": True})  # settings
    assert base != _eval_identity(recs1, "settings1", cal_a, {"freeze_on_alarm": False})  # eval policy


def test_eval_restore_rejects_records_not_matching_pinned_groups():
    # ASTRA-104: restored progress must be the EXACT completed prefix of the regime's groups; a
    # record whose ids do not match the pinned group at that position is refused, not silently reused
    import pytest

    from scripts.experiments.qwen_operating_point import RunConflict, _eval_sessions
    prompts = [{"id": i, "prompt": f"p{i}"} for i in range(4)]
    bad = {"fresh": [{"record": {"ids": [99]}, "txns": []}], "carried": []}  # group 0's id is 0, not 99
    with pytest.raises(RunConflict):
        _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=0, chains=2,
                       deadline=1e18, restore=bad, _runner=_FakeRunner(["commit"] * 100), _drive=_seed_drive)


def test_eval_persistence_round_trip_no_regeneration_or_mixing(tmp_path, monkeypatch):
    # ASTRA-104: prove that across a simulated interruption/re-invocation, the real eval-progress
    # persistence round-trip resumes to the uninterrupted result, re-appends no completed group, and
    # never writes a partial (incomplete) group into the completed log
    import scripts.experiments.qwen_operating_point as qop
    from scripts.experiments.qwen_operating_point import (
        _append_eval_progress, _init_eval_progress, _load_eval_progress, _eval_sessions,
    )

    prompts = [{"id": i, "prompt": f"p{i}"} for i in range(6)]
    path = str(tmp_path / "eval-progress.jsonl")
    identity = "eid"

    ref = _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=700, chains=3,
                         deadline=1e18, _runner=_FakeRunner(["commit"] * 400), _drive=_seed_drive)
    full = _eval_sig(ref)

    clock = {"t": 0.0}
    monkeypatch.setattr(qop.time, "time", lambda: clock["t"])

    def clock_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        out = _seed_drive(runner, tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, gen=gen)
        clock["t"] += 1.0
        return out

    def append(regime, rec, txns, ng):
        _append_eval_progress(path, regime, rec, txns)

    # invocation 1: init + run cut by a deadline, appending complete sessions to the real on-disk log
    _init_eval_progress(path, identity)
    _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=700, chains=3,
                   deadline=3.5, restore={"fresh": [], "carried": []}, on_progress=append,
                   _runner=_FakeRunner(["commit"] * 400), _drive=clock_drive)
    restore = _load_eval_progress(path, identity)  # a fresh process reloads from disk
    assert 0 < sum(len(v) for v in restore.values()) < 9  # genuinely partial across the two regimes

    # invocation 2: resume from the reloaded cursor to completion
    resumed = _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=700, chains=3,
                             deadline=1e18, restore=restore, on_progress=append,
                             _runner=_FakeRunner(["commit"] * 400), _drive=clock_drive)
    assert _eval_sig(resumed) == full  # no regeneration: the resumed screen equals the uninterrupted one

    final = _load_eval_progress(path, identity)
    ids = [(regime, tuple(s["record"]["ids"])) for regime, v in final.items() for s in v]
    assert len(ids) == len(set(ids))  # each completed session written exactly once (no re-append)
    assert sum(len(v) for v in final.values()) == 9  # fresh 6 singletons + carried 3 chains, all complete
    assert all(s["record"].get("incomplete") is False for v in final.values() for s in v)  # no partial persisted


def test_eval_progress_valid_final_record_without_newline_survives_append(tmp_path):
    # ASTRA-106: a crash before the final newline leaves a VALID record; the next append must restore
    # the delimiter rather than concatenate, and neither record may be lost
    from scripts.experiments.qwen_operating_point import _append_eval_progress, _init_eval_progress, _load_eval_progress
    path = str(tmp_path / "p.jsonl")
    _init_eval_progress(path, "idA")
    _append_eval_progress(path, "fresh", _wrec([0], "fresh"), [])
    _append_eval_progress(path, "fresh", _wrec([1], "fresh"), [])
    with open(path, "rb+") as f:  # simulate a crash before the final newline
        data = f.read()
        assert data.endswith(b"\n")
        f.seek(0); f.truncate(); f.write(data[:-1])
    assert [s["record"]["ids"] for s in _load_eval_progress(path, "idA")["fresh"]] == [[0], [1]]  # both restored
    _append_eval_progress(path, "fresh", _wrec([2], "fresh"), [])  # repairs the delimiter, no concatenation
    assert [s["record"]["ids"] for s in _load_eval_progress(path, "idA")["fresh"]] == [[0], [1], [2]]


def test_eval_progress_truncated_final_fragment_dropped_on_append(tmp_path):
    from scripts.experiments.qwen_operating_point import _append_eval_progress, _init_eval_progress, _load_eval_progress
    path = str(tmp_path / "p.jsonl")
    _init_eval_progress(path, "idA")
    _append_eval_progress(path, "fresh", _wrec([0], "fresh"), [])
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"regime": "fresh", "record": {"ids": [1')  # truncated fragment, no newline
    assert [s["record"]["ids"] for s in _load_eval_progress(path, "idA")["fresh"]] == [[0]]  # quarantined
    _append_eval_progress(path, "fresh", _wrec([1], "fresh"), [])  # rerun: repair drops the fragment
    _append_eval_progress(path, "fresh", _wrec([2], "fresh"), [])
    assert [s["record"]["ids"] for s in _load_eval_progress(path, "idA")["fresh"]] == [[0], [1], [2]]


def test_eval_progress_repeated_recover_append_reload(tmp_path):
    from scripts.experiments.qwen_operating_point import _append_eval_progress, _init_eval_progress, _load_eval_progress
    path = str(tmp_path / "p.jsonl")
    _init_eval_progress(path, "idA")
    for i in range(4):
        _append_eval_progress(path, "fresh", _wrec([i], "fresh"), [])
        with open(path, "rb+") as f:  # crash before the newline each round
            data = f.read()
            if data.endswith(b"\n"):
                f.seek(0); f.truncate(); f.write(data[:-1])
        assert [s["record"]["ids"] for s in _load_eval_progress(path, "idA")["fresh"]] == [[j] for j in range(i + 1)]
    _append_eval_progress(path, "fresh", _wrec([9], "fresh"), [])
    assert [s["record"]["ids"] for s in _load_eval_progress(path, "idA")["fresh"]] == [[0], [1], [2], [3], [9]]


def test_eval_progress_interior_corruption_is_refused(tmp_path):
    import pytest

    from scripts.experiments.qwen_operating_point import RunConflict, _append_eval_progress, _init_eval_progress, _load_eval_progress
    path = str(tmp_path / "p.jsonl")
    _init_eval_progress(path, "idA")
    _append_eval_progress(path, "fresh", _wrec([0], "fresh"), [])
    _append_eval_progress(path, "fresh", _wrec([1], "fresh"), [])
    lines = open(path, encoding="utf-8").read().splitlines()
    lines[1] = "{bad interior json"  # corrupt the FIRST session record (interior: another follows)
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with pytest.raises(RunConflict):
        _load_eval_progress(path, "idA")  # interior corruption is refused, not silently dropped


# ---- FABLE-085 hardening: crash-shape recovery, record schema, header, provenance, follow-up identity ----

import hashlib as _hashlib
import json as _json
import os as _os
import re as _re

import pytest as _pytest

def _wrec(ids, regime="fresh"):
    """A completed-session record in the exact shape ``_eval_sessions`` writes (a writer-shaped fixture;
    the loader validates this shape, so minimal stand-ins would be refused as corruption)."""
    return {"ids": list(ids), "regime": regime, "read_only": False, "read_only_reason": None,
            "eligible_by_source": {"prompt": 1, "generation": 1}, "accepted_by_source": {"prompt": 1, "generation": 1},
            "retained_both": True,
            "anomalies": {"mixed_source": 0, "missing_source": 0, "unknown_kind": 0, "nonfinite_accepted": 0},
            "turns": [{"id": i, "seed": 100 + i, "completion": f"c{i}", "n_in": 3, "n_out": 2, "outcome": "eos",
                       "seconds": 0.1} for i in ids],
            "incomplete": False}


def _wtx(kind="commit"):
    """A transaction in the shape TransactionRunner records (the fields aggregation reads, and more)."""
    return {"index": 0, "decision": {"kind": kind, "reasons": [], "scale": 1.0}, "sources": {"user": 8, "model": 0},
            "eligible": True, "accepted": {"delta_norm": 1.0}, "read_only": False, "read_only_reason": None}


def _line(ids, regime="fresh"):
    return _json.dumps({"regime": regime, "record": _wrec(ids, regime), "txns": [_wtx()]})


_FRAGMENT = _line([1])[:40]            # a kill mid-append: a truncated JSON object never parses
_VALID_1 = _line([1])
_VALID_2 = _line([2])


def _progress_with(tmp_path, tail):
    """A log with one completed session, then ``tail`` written verbatim (a crash shape or corruption)."""
    from scripts.experiments.qwen_operating_point import _append_eval_progress, _init_eval_progress
    path = str(tmp_path / "eval-progress.jsonl")
    _init_eval_progress(path, "idA")
    _append_eval_progress(path, "fresh", _wrec([0], "fresh"), [])
    with open(path, "a", encoding="utf-8") as f:
        f.write(tail)
    return path


def _fresh_ids(path):
    from scripts.experiments.qwen_operating_point import _load_eval_progress
    return [s["record"]["ids"] for s in _load_eval_progress(path, "idA", log=lambda *_: None)["fresh"]]


@_pytest.mark.parametrize("tail, expect", [
    (_FRAGMENT, [[0]]),                 # unterminated truncated record: the crash shape -> quarantined
    ("garbage", [[0]]),                 # unterminated non-JSON final segment -> quarantined
    (_VALID_1, [[0], [1]]),             # complete record that lost only its newline -> restored
    (_VALID_1 + "\n", [[0], [1]]),      # complete terminated record -> restored
    (_FRAGMENT + "\n", "refuse"),       # newline-terminated malformed final line: NOT a crash fragment
    ("garbage\n", "refuse"),
])
def test_eval_progress_final_line_recovery_matrix(tmp_path, tail, expect):
    # FABLE-085 #2a: only an UNTERMINATED malformed final line is quarantined (each record and its
    # newline are one append); a terminated malformed final line is corruption, refused and preserved
    from scripts.experiments.qwen_operating_point import RunConflict, _append_eval_progress, _load_eval_progress
    path = _progress_with(tmp_path, tail)
    if expect == "refuse":
        before = open(path, "rb").read()
        with _pytest.raises(RunConflict, match="newline-terminated final"):
            _load_eval_progress(path, "idA")
        assert open(path, "rb").read() == before
        return
    assert _fresh_ids(path) == expect
    _append_eval_progress(path, "fresh", _wrec([7], "fresh"), [])  # the next append lands on a clean boundary
    assert _fresh_ids(path) == expect + [[7]]


def _mut(fn):
    line = {"regime": "fresh", "record": _wrec([1]), "txns": [_wtx()]}
    fn(line)
    return _json.dumps(line)


def _set(path, value):
    def fn(line):
        *head, last = path
        d = line
        for k in head:
            d = d[k]
        d[last] = value
    return fn


def _del(path):
    def fn(line):
        *head, last = path
        d = line
        for k in head:
            d = d[k]
        del d[last]
    return fn


_SCHEMA_CASES = {
    "list-line": ("[1, 2]", "not a JSON object"),
    "unknown-regime": (_mut(_set(["regime"], "stale")), "unknown regime"),
    "no-record": (_mut(_del(["record"])), "missing session record"),
    "record-not-object": (_mut(_set(["record"], [1])), "missing session record"),
    "no-ids": (_mut(_del(["record", "ids"])), "nonempty integer ids"),
    "bool-ids": (_mut(_set(["record", "ids"], [True])), "nonempty integer ids"),     # True == 1 in Python
    "float-ids": (_mut(_set(["record", "ids"], [1.0])), "nonempty integer ids"),
    "empty-ids": (_mut(_set(["record", "ids"], [])), "nonempty integer ids"),
    "record-regime-differs": (_mut(_set(["record", "regime"], "carried")), "differs from the line regime"),
    "record-regime-missing": (_mut(_del(["record", "regime"])), "differs from the line regime"),
    "persisted-incomplete": (_mut(_set(["record", "incomplete"], True)), "incomplete session was persisted"),
    "no-incomplete-flag": (_mut(_del(["record", "incomplete"])), "no incomplete=false"),
    "no-retained_both": (_mut(_del(["record", "retained_both"])), "boolean retained_both"),
    "int-retained_both": (_mut(_set(["record", "retained_both"], 1)), "boolean retained_both"),
    "no-read_only": (_mut(_del(["record", "read_only"])), "boolean read_only"),
    "bad-read_only_reason": (_mut(_set(["record", "read_only_reason"], 3)), "non-string read_only_reason"),
    "partial-eligible": (_mut(_set(["record", "eligible_by_source"], {"prompt": 1})), "eligible_by_source is not"),
    "bool-accepted": (_mut(_set(["record", "accepted_by_source", "generation"], True)), "accepted_by_source is not"),
    "no-anomalies": (_mut(_del(["record", "anomalies"])), "anomalies map"),
    "bad-provenance": (_mut(_set(["record", "provenance"], "x")), "provenance is not an object"),
    "no-turns": (_mut(_del(["record", "turns"])), "turns do not match"),
    "short-turns": (_mut(_set(["record", "turns"], [])), "turns do not match"),
    "turn-id-differs": (_mut(_set(["record", "turns", 0, "id"], 99)), "does not carry the session's id"),
    "turn-no-seed": (_mut(_del(["record", "turns", 0, "seed"])), "missing a writer field"),
    "turn-bad-outcome": (_mut(_set(["record", "turns", 0, "outcome"], "weird")), "missing a writer field"),
    "no-txns": (_mut(_del(["txns"])), "missing transactions list"),
    "txns-not-list": (_mut(_set(["txns"], {})), "missing transactions list"),
    "txn-not-object": (_mut(_set(["txns"], [1])), "transaction 0 is not an object"),
    "txn-no-decision": (_mut(_del(["txns", 0, "decision"])), "no decision kind"),
    "txn-bool-sources": (_mut(_set(["txns", 0, "sources"], {"user": True, "model": 0})), "source counts"),
    "txn-no-eligible": (_mut(_del(["txns", 0, "eligible"])), "boolean eligibility"),
    "txn-no-accepted": (_mut(_del(["txns", 0, "accepted"])), "accepted delta_norm"),
    "txn-str-delta": (_mut(_set(["txns", 0, "accepted", "delta_norm"], "x")), "accepted delta_norm"),
}


@_pytest.mark.parametrize("case", sorted(_SCHEMA_CASES))
@_pytest.mark.parametrize("position", ["interior", "final_terminated", "final_unterminated"])
def test_eval_progress_schema_invalid_record_is_refused(tmp_path, case, position):
    # FABLE-085 #2b / CODEX-001 #1: a PARSEABLE line that is not a completed-session record the writer
    # could produce -- down to integer (never boolean) ids, every field the criterion/aggregation reads,
    # turns consistent with ids and transaction structure -- is corruption wherever it sits (a truncated
    # JSON object never parses): refused with RunConflict, never ignored nor a later bare KeyError
    from scripts.experiments.qwen_operating_point import RunConflict, _load_eval_progress
    bad, why = _SCHEMA_CASES[case]
    tail = {"interior": bad + "\n" + _VALID_2 + "\n", "final_terminated": bad + "\n", "final_unterminated": bad}[position]
    path = _progress_with(tmp_path, tail)
    with _pytest.raises(RunConflict, match=_re.escape(why)):
        _load_eval_progress(path, "idA")


def test_real_writer_records_pass_the_progress_schema():
    # the validator and the writer agree: every session _eval_sessions actually emits -- fresh and
    # carried, committing and read-only -- passes, so the strict schema never refuses real progress
    from scripts.experiments.qwen_operating_point import _eval_sessions, _progress_record_problem
    prompts = [{"id": i, "prompt": f"p{i}"} for i in range(5)]
    for kinds, drive in ((["commit"] * 40, _seed_drive), (["readonly", "commit"] * 20, _fake_drive)):
        emitted = []
        _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=0, chains=2, deadline=1e18,
                       on_progress=lambda regime, rec, txns, ng: emitted.append((regime, rec, txns)),
                       provenance=_prov("a"), _runner=_FakeRunner(kinds), _drive=drive)
        assert len(emitted) == 7  # 5 fresh + 2 carried chains
        for regime, rec, txns in emitted:
            line = _json.loads(_json.dumps({"regime": regime, "record": rec, "txns": txns}))  # through JSON
            assert _progress_record_problem(line) is None, (regime, rec["ids"])


def _proto(**gen_over):
    from scripts.experiments.qwen_operating_point import _protocol
    base = _protocol(8, fit_harness={"log_only": True}, eval_harness={"freeze_on_alarm": True})
    base["gen"].update(gen_over)
    return base


def test_settings_identity_binds_the_effective_sampling_and_harness_protocol():
    # CODEX-001 #2: temperature/top-k, the seed schedule and the effective fit/eval harness configs are
    # not CLI settings, so they must be bound explicitly or a changed sampler pools under one identity
    from scripts.experiments.qwen_operating_point import _settings_identity
    base = _settings_identity(_fake_args(), "d", "r", protocol=_proto())
    assert base == _settings_identity(_fake_args(), "d", "r", protocol=_proto())
    assert base != _settings_identity(_fake_args(), "d", "r", protocol=_proto(temperature=0.5))
    assert base != _settings_identity(_fake_args(), "d", "r", protocol=_proto(top_k=10))
    for key, value in (("seed", 1), ("seed_offsets", {"eval": 1, "followups": 2}), ("chunk", 16),
                       ("target_fpr", 0.05), ("fit_harness", {"log_only": False}), ("eval_harness", {"freeze_on_alarm": False})):
        assert base != _settings_identity(_fake_args(), "d", "r", protocol={**_proto(), key: value}), key


def test_changed_sampler_is_refused_on_resume_before_any_reuse(tmp_path):
    # CODEX-001 #2 across resume: the run created under one sampler is refused (and preserved) when
    # resumed under another, so neither its completed calibration nor its eval progress can be reused
    from scripts.experiments.qwen_operating_point import RunConflict, _reconcile_run, _settings_identity
    split = _split_of({"fit": [1, 2], "cusum": [3], "dev": [4], "eval": [5, 6]})
    manifest = _manifest_of("corpusA", split)
    ident_a = _settings_identity(_fake_args(), "d", "r", protocol=_proto())
    mode, mid, _, _ = _reconcile_run(str(tmp_path), manifest, split, ident_a, lambda: "m1")
    assert mode == "fresh"
    before = {n: (tmp_path / n).read_bytes() for n in ("run-record.json", "split-manifest.json")}
    for over in ({"temperature": 0.5}, {"top_k": 10}):
        with _pytest.raises(RunConflict):
            _reconcile_run(str(tmp_path), manifest, split, _settings_identity(_fake_args(), "d", "r", protocol=_proto(**over)),
                           lambda: "m2")
    assert {n: (tmp_path / n).read_bytes() for n in before} == before
    assert _reconcile_run(str(tmp_path), manifest, split, ident_a, lambda: "m3")[:2] == ("resume", "m1")


@_pytest.mark.parametrize("meta, ok", [
    ({}, True),                                                                  # nothing yet: will record owner
    ({"calibration_settings_identity": "S"}, True),                              # in progress, same owner
    ({"calibration_settings_identity": "S", "calibration_fit_complete": True, "calibration_cusum_complete": True}, True),
    ({"calibration_settings_identity": "T"}, False),                             # in progress under another identity
    ({"calibration_settings_identity": "T", "calibration_fit_complete": True, "calibration_cusum_complete": True}, False),
    ({"calibration_fit_complete": True, "calibration_cusum_complete": True}, False),  # completed, owner unknown
])
def test_completed_or_partial_calibration_is_reused_only_by_its_owner(meta, ok):
    # CODEX-001 #2 on the completed-calibration reuse path itself (defense in depth behind reconcile)
    from scripts.experiments.qwen_operating_point import RunConflict, _check_calibration_owner
    if ok:
        _check_calibration_owner(meta, "S")
    else:
        with _pytest.raises(RunConflict):
            _check_calibration_owner(meta, "S")


def test_eval_progress_header_is_atomic_and_never_overwritten(tmp_path):
    from scripts.experiments.qwen_operating_point import RunConflict, _append_eval_progress, _init_eval_progress
    path = str(tmp_path / "eval-progress.jsonl")
    prov = {"code_commit": "a" * 40, "code_dirty": False, "torch": "t", "transformers": "x", "device": "cpu"}
    _init_eval_progress(path, "idA", prov)
    assert _json.loads(open(path, encoding="utf-8").read().splitlines()[0]) == {"identity": "idA", "provenance": prov}
    assert sorted(_os.listdir(tmp_path)) == ["eval-progress.jsonl"]  # temp + replace leaves no temp behind
    _append_eval_progress(path, "fresh", _wrec([0], "fresh"), [])
    with _pytest.raises(RunConflict):
        _init_eval_progress(path, "idA", prov)  # an existing log (holding a session) is never clobbered
    assert _fresh_ids(path) == [[0]]


def test_eval_progress_missing_or_damaged_header_is_refused_never_truncated(tmp_path):
    # FABLE-085 #5/#6: the append-boundary repair never creates, empties or de-headers a log, and the
    # loader's refusal names the precise remedy (remove only the progress log; calibration unaffected)
    from scripts.experiments.qwen_operating_point import RunConflict, _append_eval_progress, _load_eval_progress
    path = str(tmp_path / "eval-progress.jsonl")
    with _pytest.raises(RunConflict):  # missing: an append never creates a headerless log
        _append_eval_progress(path, "fresh", _wrec([0], "fresh"), [])
    assert not _os.path.exists(path)

    open(path, "w").close()  # empty
    with _pytest.raises(RunConflict, match="remove ONLY this eval-progress.jsonl") as exc:
        _load_eval_progress(path, "idA")
    assert "calibration" in str(exc.value) and "unaffected" in str(exc.value)
    with _pytest.raises(RunConflict):
        _append_eval_progress(path, "fresh", _wrec([0], "fresh"), [])
    assert _os.path.getsize(path) == 0

    damaged = '{"identity": "id'  # a damaged header and no newline anywhere
    with open(path, "w", encoding="utf-8") as f:
        f.write(damaged)
    with _pytest.raises(RunConflict, match="unreadable identity header"):
        _load_eval_progress(path, "idA")
    with _pytest.raises(RunConflict, match="no complete identity header"):
        _append_eval_progress(path, "fresh", _wrec([0], "fresh"), [])
    assert open(path, encoding="utf-8").read() == damaged  # preserved, not truncated to empty


def _prov(commit):
    return {"code_commit": commit, "code_dirty": False, "torch": "t", "transformers": "x", "device": "cpu"}


def test_eval_provenance_stamped_per_invocation_and_listed(tmp_path, monkeypatch):
    # FABLE-085 #3: an eval resumed across invocations records which invocation produced each complete
    # session, warns (does not refuse) on a provenance change, and lists every contributor with counts
    import scripts.experiments.qwen_operating_point as qop
    from scripts.experiments.qwen_operating_point import (
        _append_eval_progress, _eval_provenances, _eval_sessions, _init_eval_progress, _load_eval_progress,
    )
    prompts = [{"id": i, "prompt": f"p{i}"} for i in range(4)]
    path = str(tmp_path / "eval-progress.jsonl")
    clock = {"t": 0.0}
    monkeypatch.setattr(qop.time, "time", lambda: clock["t"])

    def clock_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        out = _seed_drive(runner, tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, gen=gen)
        clock["t"] += 1.0
        return out

    def append(regime, rec, txns, ng):
        _append_eval_progress(path, regime, rec, txns)

    def run(restore, prov, deadline):
        return _eval_sessions(None, None, None, prompts, hcfg=None, gen=_gen_settings(), seed_base=0, chains=2,
                              deadline=deadline, restore=restore, on_progress=append, provenance=prov,
                              _runner=_FakeRunner(["commit"] * 100), _drive=clock_drive)

    _init_eval_progress(path, "eid", _prov("a"))
    run({"fresh": [], "carried": []}, _prov("a"), deadline=2.5)  # invocation A: 3 fresh sessions, then cut
    quiet: list[str] = []
    same = _load_eval_progress(path, "eid", provenance=_prov("a"), log=quiet.append)
    assert [s["record"]["provenance"]["code_commit"] for s in same["fresh"]] == ["a", "a", "a"]
    assert not any("WARNING" in m for m in quiet)  # same provenance: nothing to warn about

    logs: list[str] = []
    restore = _load_eval_progress(path, "eid", provenance=_prov("b"), log=logs.append)
    assert any("WARNING" in m and "provenance" in m for m in logs)  # changed: warned, NOT refused
    rep = run(restore, _prov("b"), deadline=1e18)  # invocation B completes the screen
    assert [s["provenance"]["code_commit"] for s in rep["fresh"]["sessions"]] == ["a", "a", "a", "b"]
    assert [s["provenance"]["code_commit"] for s in rep["carried"]["sessions"]] == ["b", "b"]
    contributors = _eval_provenances(rep)
    assert [(c["provenance"]["code_commit"], c["sessions"]) for c in contributors] == [
        ("a", {"fresh": 3, "carried": 0}), ("b", {"fresh": 1, "carried": 2})]

    # a deadline-cut (incomplete) group is reported but never counted as a contributing session
    cut = dict(rep)
    cut["carried"] = {**rep["carried"], "sessions": rep["carried"]["sessions"] + [{"incomplete": True, "provenance": _prov("c")}]}
    assert [c["provenance"]["code_commit"] for c in _eval_provenances(cut)] == ["a", "b"]


def test_invocation_provenance_records_code_dependencies_and_device():
    from scripts.experiments.qwen_operating_point import _REPO_ROOT, _invocation_provenance
    prov = _invocation_provenance("cpu")
    assert set(prov) == {"code_commit", "code_dirty", "torch", "transformers", "device"}
    assert prov["device"] == "cpu" and prov["torch"]
    if _os.path.exists(_os.path.join(_REPO_ROOT, ".git")):  # a checkout (or worktree), not a source export
        assert _re.fullmatch(r"[0-9a-f]{40}", prov["code_commit"])
        assert prov["code_dirty"] in (True, False)


def test_followups_record_fixture_identity_and_seed(tmp_path):
    # ASTRA-106 condition 1 / ASTRA-109: the wholesale-rerun follow-up result names the exact fixture
    # BYTES it ran (path + sha256), its seed base, whether those bytes are the fixture DECLARED at run
    # creation, and requested/completed session+turn counts; a missing fixture is skipped yet still
    # names path and seed and is never the declared fixture
    from scripts.experiments.qwen_operating_point import _file_sha256, _run_followups
    fixture = {"purpose": "p", "sessions": [{"id": "s0", "turns": ["a", "b"]}, {"id": "s1", "turns": ["c"]}]}
    fp = tmp_path / "followups.json"
    raw = _json.dumps(fixture).encode("utf-8")
    fp.write_bytes(raw)
    declared = _file_sha256(str(fp))
    assert declared == _hashlib.sha256(raw).hexdigest() and _file_sha256(str(tmp_path / "absent.json")) is None

    def run(path, **kw):
        return _run_followups(None, None, None, _gen_settings(), 4242, str(path), None, deadline=1e18,
                              _runner=_FakeRunner(["commit"] * 8), _drive=_fake_drive, **kw)

    out = run(fp, declared_sha256=declared)
    assert (out["fixture_path"], out["seed0"], out["all_ok"]) == (str(fp), 4242, True)
    assert out["fixture_sha256"] == declared and out["fixture_is_declared"] is True
    assert (out["n_sessions_requested"], out["n_sessions_completed"], out["n_turns_requested"], out["n_turns_completed"]) == (2, 2, 3, 3)
    assert run(fp)["fixture_is_declared"] is False  # nothing declared at run creation -> never "the declared one"

    fp.write_bytes(_json.dumps({**fixture, "purpose": "edited"}).encode("utf-8"))
    edited = run(fp, declared_sha256=declared)
    assert edited["fixture_sha256"] != declared and edited["fixture_is_declared"] is False  # a different attempt

    miss = run(tmp_path / "absent.json", declared_sha256=declared)
    assert miss["skipped"] and miss["fixture_sha256"] is None and miss["fixture_is_declared"] is False
    assert (miss["fixture_path"], miss["seed0"]) == (str(tmp_path / "absent.json"), 4242)


def test_followups_deadline_keeps_requested_vs_completed_accounting():
    from scripts.experiments.qwen_operating_point import _run_followups
    fixture = {"sessions": [{"id": f"s{i}", "turns": ["a", "b"]} for i in range(3)]}
    out = _run_followups(None, None, None, _gen_settings(), 0, "unused", None, deadline=-1.0,
                         _runner=_FakeRunner(["commit"] * 8), _drive=_fake_drive, _fixture=fixture)
    assert out["incomplete"] is True and out["all_ok"] is False
    assert (out["n_sessions_requested"], out["n_sessions_completed"], out["n_turns_requested"], out["n_turns_completed"]) == (3, 0, 6, 0)


def _inv(commit="a" * 40, dirty=False, mode="resume", **extra):
    return {"t_unix": 0, "mode": mode, "provenance": {**_prov(commit), "code_dirty": dirty, **extra}}


_MARKER = {"t_unix": None, "mode": "unrecorded_prior", "provenance": None}


@_pytest.mark.parametrize("invocations, ok, reason", [
    ([_inv(mode="fresh")], True, None),
    ([_inv(mode="fresh"), _inv(), _inv()], True, None),                   # continuations from ONE clean source
    ([_inv(mode="fresh"), _inv(commit="b" * 40)], False, "distinct invocation provenances"),
    ([_inv(mode="fresh"), _inv(torch="other")], False, "distinct invocation provenances"),  # runtime change
    ([_inv(mode="fresh", dirty=True)], False, "not a content identity"),
    ([_inv(mode="fresh", dirty=None)], False, "not a content identity"),  # unknown tree state is not clean
    ([_inv(mode="fresh", commit="unknown")], False, "commit is unknown"),
    ([{"t_unix": 0, "mode": "fresh", "provenance": None}], False, "no recorded provenance"),
    ([], False, "no invocation recorded"),
    # ASTRA-112: a history that does not begin with the run's creation cannot certify earlier work
    ([_inv()], False, "single creating (fresh) invocation"),               # a lone continuation
    ([_inv(), _inv()], False, "single creating (fresh) invocation"),
    ([_MARKER, _inv()], False, "single creating (fresh) invocation"),      # unrecorded prior work
    ([_inv(mode="fresh"), _inv(mode="fresh")], False, "single creating (fresh) invocation"),
])
def test_run_provenance_check_requires_one_clean_identified_source(invocations, ok, reason):
    from scripts.experiments.qwen_operating_point import _run_provenance_check
    out = _run_provenance_check(invocations)
    assert out["ok"] is ok
    if reason is not None:
        assert any(reason in r for r in out["reasons"]), out["reasons"]


def test_record_invocation_accumulates_atomically_and_refuses_unreadable(tmp_path):
    from scripts.experiments.qwen_operating_point import RunConflict, _record_invocation
    first = _record_invocation(str(tmp_path), _prov("a"), "fresh")
    both = _record_invocation(str(tmp_path), _prov("b"), "resume")
    assert [i["mode"] for i in first] == ["fresh"]
    assert [(i["mode"], i["provenance"]["code_commit"]) for i in both] == [("fresh", "a"), ("resume", "b")]
    assert sorted(_os.listdir(tmp_path)) == ["invocations.json"]  # atomic rewrite leaves no temp behind

    path = tmp_path / "invocations.json"
    path.write_text('[{"mode": "fresh"', encoding="utf-8")  # damaged: refused, preserved
    with _pytest.raises(RunConflict):
        _record_invocation(str(tmp_path), _prov("c"), "resume")
    assert path.read_text(encoding="utf-8") == '[{"mode": "fresh"'


def test_unrecorded_history_is_never_one_clean_source_on_any_continuation(tmp_path):
    # ASTRA-112: a continued run whose record is ABSENT or EMPTY (created before it existed, or lost)
    # holds computation from unrecorded invocations. Even a clean current invocation must not make it
    # read as one clean source -- on this continuation or any later one -- and a record that begins
    # with a continuation certifies nothing before it. The intact same-source history stays valid.
    from scripts.experiments.qwen_operating_point import RunConflict, _record_invocation, _run_provenance_check
    clean = {**_prov("a" * 40), "code_dirty": False}

    def run_dir(name, ledger=None):
        d = tmp_path / name
        _os.makedirs(d)
        if ledger is not None:
            (d / "invocations.json").write_text(_json.dumps(ledger), encoding="utf-8")
        return str(d)

    for name, ledger in (("absent", None), ("empty", []), ("stale", [_inv()])):
        d = run_dir(name, ledger)
        first = _record_invocation(d, clean, "resume")
        second = _record_invocation(d, clean, "resume")  # the unknown history persists in the record
        for record in (first, second):
            check = _run_provenance_check(record)
            assert check["ok"] is False, name
            assert any("single creating (fresh) invocation" in r for r in check["reasons"]), (name, check["reasons"])
        if ledger is not None and ledger:
            assert second[0] == ledger[0]  # an existing (stale) history is kept, never rewritten
        else:
            assert second[0]["mode"] == "unrecorded_prior" and second[0]["provenance"] is None

    d = run_dir("intact")  # control: created fresh, continued twice from the same clean source
    _record_invocation(d, clean, "fresh")
    _record_invocation(d, clean, "resume")
    intact = _record_invocation(d, clean, "resume")
    assert [i["mode"] for i in intact] == ["fresh", "resume", "resume"] and _run_provenance_check(intact)["ok"] is True

    d = run_dir("fresh-over-stale", [_inv(mode="fresh")])  # a fresh run never adopts an existing record
    before = (tmp_path / "fresh-over-stale" / "invocations.json").read_bytes()
    with _pytest.raises(RunConflict):
        _record_invocation(d, clean, "fresh")
    assert (tmp_path / "fresh-over-stale" / "invocations.json").read_bytes() == before


@_pytest.mark.parametrize("artifact", ["split-manifest.json", "calibration.ckpt", "calibration-status.json", "invocations.json",
                                       "dev-progress.jsonl", "dev-diagnostic.json", "eval-progress.jsonl", "eval-operating-point.json", "followups-result.json",
                                       "screen-result.json"])
def test_reconcile_never_starts_fresh_over_any_prior_run_artifact(tmp_path, artifact):
    # a directory holding ANY run artifact but no run record is prior work: a fresh run must not
    # overwrite it or adopt a stale invocation record / progress log as its own (ASTRA-098/112)
    from scripts.experiments.qwen_operating_point import _RUN_ARTIFACTS, RunConflict, _reconcile_run
    assert artifact in _RUN_ARTIFACTS
    (tmp_path / artifact).write_text("prior", encoding="utf-8")
    split = _split_of({"fit": [0], "cusum": [1], "dev": [2], "eval": [3]})
    with _pytest.raises(RunConflict):
        _reconcile_run(str(tmp_path), _manifest_of("new", split), split, "idA", lambda: "qwen_x")
    assert (tmp_path / artifact).read_text(encoding="utf-8") == "prior"
    assert not (tmp_path / "run-record.json").exists()


# ---- dev stage: the ASTRA-088/109 paired guarded vs log-only development diagnostic ----


class _CusumRunner:
    """A runner stand-in with the REAL Cusum and TransactionRunner's per-chunk order (transaction.py):
    a chunk that can carry evidence updates the CUSUM (z from the continuous reference when it has >= 8
    values, else the per-chunk z), a guarded runner latches read-only on alarm, then the record is
    APPENDED; a latched chunk is ineligible read-only with all z None. ``skew`` feeds the CUSUM a value
    other than the recorded one, to prove the replay check catches an unobserved change."""

    def __init__(self, *, log_only, k=0.5, h=2.0, cref=None, skew=0.0):
        from plastic.harness.stats import Cusum
        self.log_only, self.k, self.h, self.cref, self.skew = log_only, k, h, cref or [], skew
        self.cusum = Cusum(k, h)
        self.transactions = []
        self.read_only, self.read_only_reason, self.resets = False, None, 0

    def reset(self):
        from plastic.harness.stats import Cusum
        self.cusum = Cusum(self.k, self.h)
        self.read_only, self.read_only_reason = False, None
        self.resets += 1

    def chunk(self, *, user, model, value):
        from plastic.harness.stats import robust_z
        eligible = not self.read_only
        alarm, z = False, {"log_delta_norm": None, "chunk_loss": None}
        if eligible:
            z = {"log_delta_norm": value, "chunk_loss": 0.0}
            z_ld = robust_z(value, self.cref) if len(self.cref) >= 8 else value
            alarm = self.cusum.update(z_ld + self.skew)
            if alarm and not self.log_only:
                self.read_only, self.read_only_reason = True, "cusum_alarm"
        self.transactions.append({
            "decision": {"kind": "commit" if eligible else "readonly"}, "sources": {"user": user, "model": model},
            "eligible": eligible, "accepted": {"delta_norm": 1.0 if eligible else 0.0},
            "signals": {"z": z, "cusum_alarm": alarm, "log_delta_norm": value, "pos_start": 0, "pos_end": 8},
            "read_only": self.read_only, "read_only_reason": self.read_only_reason})


def _zdrive(prompt_values, gen_values, *, raise_on=None):
    """Emit the prompt chunks, then one generation chunk per sampled token; a latched runner produces
    no tokens (the measured ASTRA-088/090 empty-output shape, NOT assumed universal by the driver)."""
    def drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        if raise_on is not None and prompt == raise_on and not runner.log_only:
            raise RuntimeError("boom")
        for v in prompt_values:
            runner.chunk(user=8, model=0, value=v)
        out = []
        for j, v in enumerate(gen_values[:max_new_tokens]):
            if runner.read_only:
                break
            out.append(int(gen.initial_seed()) % 1000 + j)
            runner.chunk(user=0, model=1, value=v)
        return (f"o{len(out)}", out, [1] * (8 * len(prompt_values)))
    return drive


def _dev(prompts, drive, *, restore=None, on_progress=None, deadline=1e18, cref=None, skew=0.0):
    from types import SimpleNamespace
    from scripts.experiments.qwen_operating_point import _dev_sessions
    runners = {"guarded": _CusumRunner(log_only=False, cref=cref, skew=skew), "log_only": _CusumRunner(log_only=True, cref=cref, skew=skew)}
    cal = SimpleNamespace(cusum_reference=cref or [])
    rep = _dev_sessions(None, None, cal, prompts, guarded_hcfg=None, gen=_gen_settings(), seed_base=2000, deadline=deadline,
                        restore=restore, on_progress=on_progress, provenance=_prov("a"), _runners=runners, _drive=drive)
    return rep, runners


def _dev_prompts(n):
    return [{"id": 10 + i, "prompt": f"p{i}"} for i in range(n)]


def test_dev_pair_records_cusum_boundary_latch_outputs_and_matched_seeds():
    # prompt chunks z=2,2 alarm on the SECOND prompt chunk (h=2, k=0.5: 1.5 then 3.0>2 -> alarm, reset)
    rep, runners = _dev(_dev_prompts(3), _zdrive([2.0, 2.0, 0.0], [0.0, 0.0, 0.0]))
    assert rep["complete"] is True and [p["seed"] for p in rep["pairs"]] == [2000, 2001, 2002]
    assert runners["guarded"].resets == 3 and runners["log_only"].resets == 3  # fresh state per arm per prompt
    for p in rep["pairs"]:
        g, lo = p["arms"]["guarded"], p["arms"]["log_only"]
        assert g["seed"] == lo["seed"] == p["seed"]                             # matched seed, only log_only differs
        assert g["cusum_replay_mismatch"] == lo["cusum_replay_mismatch"] == 0  # observed statistic == replay
        assert [c["cusum_s_hi"] for c in lo["chunks"]] == [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # reset after alarm
        assert [c["phase"] for c in lo["chunks"]] == ["prompt"] * 3 + ["generation"] * 3
        assert [c["rel_to_boundary"] for c in lo["chunks"]] == [-3, -2, -1, 0, 1, 2]
        # guarded: the alarm latches during the PROMPT, the rest is read-only with no CUSUM evidence
        assert [c["cusum_alarm"] for c in g["chunks"]] == [False, True, False]
        assert [c["read_only_after"] for c in g["chunks"]] == [False, True, True] and g["chunks"][2]["cusum_z"] is None
        assert (g["outcome"], g["out_ids"], g["read_only_end"]) == ("empty", [], True)
        assert (lo["outcome"], lo["read_only_end"]) == ("eos", False) and len(lo["out_ids"]) == 3
        assert p["outputs_identical"] is False and p["first_divergent_token"] == 0
    s = rep["summary"]
    assert s["n_pairs"] == 3 and s["cusum_replay_mismatch"] == 0 and s["outputs_identical"] == 0
    assert s["arms"]["guarded"]["first_alarm_phase"] == {"prompt": 3, "generation": 0, "mixed": 0, "none": 0}
    assert s["arms"]["guarded"]["latched_before_first_accepted_generation_chunk"] == 3
    assert s["arms"]["guarded"]["outcomes"]["empty"] == 3 and s["arms"]["guarded"]["retained_both_sources"] == 0
    assert s["arms"]["log_only"]["read_only_at_end"] == 0 and s["arms"]["log_only"]["retained_both_sources"] == 3
    assert s["arms"]["guarded"]["first_alarm_rel_to_boundary"] == [-2, -2, -2]  # boundary = no generation chunk


def test_dev_cusum_input_follows_the_continuous_reference_and_replay_catches_skew():
    from plastic.harness.stats import robust_z
    cref = [0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15]
    rep, _ = _dev(_dev_prompts(1), _zdrive([0.3, 0.1], [0.0]), cref=cref)
    lo = rep["pairs"][0]["arms"]["log_only"]
    assert lo["cusum_replay_mismatch"] == 0
    assert lo["chunks"][0]["cusum_z"] == _pytest.approx(robust_z(0.3, cref))  # the continuous-reference z, not the per-chunk z
    skewed, _ = _dev(_dev_prompts(1), _zdrive([0.3, 0.1], [0.0]), cref=cref, skew=0.7)
    assert skewed["summary"]["cusum_replay_mismatch"] > 0  # an unobserved change in the CUSUM input is flagged


def test_dev_exception_is_recorded_as_an_outcome_and_the_pair_completes():
    rep, _ = _dev(_dev_prompts(2), _zdrive([0.0], [0.0, 0.0], raise_on="p1"))
    g = rep["pairs"][1]["arms"]["guarded"]
    assert (g["outcome"], g["error"]) == ("exception", "RuntimeError: boom")
    assert rep["pairs"][1]["arms"]["log_only"]["outcome"] == "eos" and rep["complete"] is True
    assert rep["summary"]["arms"]["guarded"]["outcomes"]["exception"] == 1


def test_dev_resume_through_disk_equals_uninterrupted_and_never_persists_half_a_pair(tmp_path, monkeypatch):
    import scripts.experiments.qwen_operating_point as qop
    from scripts.experiments.qwen_operating_point import (
        _append_eval_progress, _dev_record_problem, _init_eval_progress, _load_eval_progress,
    )
    prompts = _dev_prompts(4)
    base = _zdrive([2.0, 2.0, 0.0], [0.0, 0.5])
    full, _ = _dev(prompts, base)

    def sig(rep):
        return [(p["id"], p["seed"], {a: (t["out_ids"], [c["cusum_s_hi"] for c in t["chunks"]]) for a, t in p["arms"].items()})
                for p in rep["pairs"]]

    clock = {"t": 0.0}
    monkeypatch.setattr(qop.time, "time", lambda: clock["t"])

    def clock_drive(runner, tok, prompt, **kw):
        out = base(runner, tok, prompt, **kw)
        clock["t"] += 1.0
        return out

    path = str(tmp_path / "dev-progress.jsonl")
    _init_eval_progress(path, "dev-id")
    appended = []

    def append(pair):
        appended.append(pair["id"])
        _append_eval_progress(path, "dev", pair, [])

    def load():
        return _load_eval_progress(path, "dev-id", regimes=("dev",), record_problem=_dev_record_problem)["dev"]

    cut, _ = _dev(prompts, clock_drive, restore=[], on_progress=append, deadline=2.5)  # cut inside pair 1
    assert cut["complete"] is False and [x["record"]["id"] for x in load()] == [10]      # only the COMPLETE pair persisted
    resumed, _ = _dev(prompts, clock_drive, restore=load(), on_progress=append)
    assert sig(resumed) == sig(full) and resumed["complete"] is True           # position-based seeds: no shift
    assert appended == [10, 11, 12, 13]                                         # each pair appended exactly once
    assert [x["record"]["id"] for x in load()] == [10, 11, 12, 13]


def test_dev_restore_must_be_the_exact_pinned_prefix():
    from scripts.experiments.qwen_operating_point import RunConflict
    full, _ = _dev(_dev_prompts(2), _zdrive([0.0], [0.0]))
    with _pytest.raises(RunConflict):
        _dev(_dev_prompts(2), _zdrive([0.0], [0.0]), restore=[{"record": full["pairs"][1]}])  # pair 1 at position 0


def _dev_line(mutate=None):
    full, _ = _dev(_dev_prompts(1), _zdrive([2.0, 2.0], [0.0]))
    line = _json.loads(_json.dumps({"regime": "dev", "record": full["pairs"][0], "txns": []}))
    if mutate:
        mutate(line)
    return line


@_pytest.mark.parametrize("mutate, why", [
    (lambda L: L.update(regime="fresh"), "not a dev pair line"),
    (lambda L: L["record"].pop("seed"), "integer id/position/seed"),
    (lambda L: L["record"].update(id=True), "integer id/position/seed"),
    (lambda L: L["record"].update(incomplete=True), "incomplete dev pair"),
    (lambda L: L["record"]["arms"].pop("log_only"), "exactly the guarded and log_only arms"),
    (lambda L: L["record"]["arms"]["log_only"].update(seed=1), "matched seed"),
    (lambda L: L["record"]["arms"]["guarded"].update(outcome="weird"), "known outcome"),
    (lambda L: L["record"]["arms"]["log_only"].update(out_ids=[True]), "integer out_ids"),
    (lambda L: L["record"]["arms"]["log_only"]["chunks"].pop(), "chunks do not match"),
    (lambda L: L["record"]["arms"]["log_only"]["txns"][0].pop("eligible"), "boolean eligibility"),
    (lambda L: L.update(txns=[{"x": 1}]), "transactions outside its arms"),
])
def test_dev_progress_schema_refuses_non_writer_pairs(mutate, why):
    from scripts.experiments.qwen_operating_point import _dev_record_problem
    assert _dev_record_problem(_dev_line()) is None  # the real writer's pair passes
    assert why in (_dev_record_problem(_dev_line(mutate)) or "")


def test_dev_and_eval_logs_never_cross_load(tmp_path):
    from scripts.experiments.qwen_operating_point import (
        RunConflict, _append_eval_progress, _dev_record_problem, _init_eval_progress, _load_eval_progress,
    )
    dev_path, eval_path = str(tmp_path / "d.jsonl"), str(tmp_path / "e.jsonl")
    _init_eval_progress(dev_path, "i")
    _append_eval_progress(dev_path, "dev", _dev_line()["record"], [])
    with _pytest.raises(RunConflict):
        _load_eval_progress(dev_path, "i")  # a dev pair is not an eval session
    _init_eval_progress(eval_path, "i")
    _append_eval_progress(eval_path, "fresh", _wrec([0]), [_wtx()])
    with _pytest.raises(RunConflict):
        _load_eval_progress(eval_path, "i", regimes=("dev",), record_problem=_dev_record_problem)


def test_dev_identity_is_distinct_from_eval_and_binds_both_arms():
    from types import SimpleNamespace
    from scripts.experiments.qwen_operating_point import _eval_identity
    cal = SimpleNamespace(thresholds={}, reference={}, cusum_reference=[], model_signature="s")
    recs = [{"id": 1, "text_sha256": "h", "prompt": "p"}]
    guarded, logonly = {"log_only": False, "freeze_on_alarm": True}, {"log_only": True, "freeze_on_alarm": True}
    dev = _eval_identity(recs, "S", cal, {"stage": "dev", "guarded": guarded, "log_only": logonly})
    assert dev != _eval_identity(recs, "S", cal, guarded)  # the same content can never share an eval identity
    assert dev != _eval_identity(recs, "S", cal, {"stage": "dev", "guarded": guarded, "log_only": {**logonly, "freeze_on_alarm": False}})


def test_cli_requires_an_explicit_stage(monkeypatch):
    # the locked evaluation is never opened by default: running without --stage is a usage error
    import sys
    from scripts.experiments.qwen_operating_point import main
    monkeypatch.setattr(sys, "argv", ["qwen_operating_point.py", "--out", "unused"])
    with _pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
