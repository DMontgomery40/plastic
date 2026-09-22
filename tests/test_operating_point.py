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


def test_screen_verdict_separates_completion_from_pass():
    # ASTRA-092: `complete` means all requested work RAN (fit + both eval regimes + follow-ups, not
    # skipped/deadline-cut); the follow-up all_ok is a quality result folded into PASS, not completion.
    from scripts.experiments.qwen_operating_point import _screen_verdict

    ok_v = {"valid": True, "pass": True}

    # everything ran and passed
    v = _screen_verdict(fit_complete=True, report=_rep(True, True), followups={"all_ok": True}, verdict=ok_v)
    assert v["complete"] is True and v["pass"] is True and v["followups_ok"] is True

    # KEY separation: fully RAN but follow-ups flagged read-only turns -> complete, NOT pass
    v = _screen_verdict(fit_complete=True, report=_rep(True, True), followups={"all_ok": False}, verdict=ok_v)
    assert v["complete"] is True and v["followups_ran"] is True and v["followups_ok"] is False and v["pass"] is False

    # follow-ups skipped (missing fixture): followups_ok null (never False), not complete
    v = _screen_verdict(fit_complete=True, report=_rep(True, True), followups={"skipped": "fixture missing"}, verdict=ok_v)
    assert v["followups_ok"] is None and v["followups_ran"] is False and v["complete"] is False and v["pass"] is False

    # follow-ups deadline-cut: ran but incomplete -> not complete
    v = _screen_verdict(fit_complete=True, report=_rep(True, True), followups={"all_ok": False, "incomplete": True}, verdict=ok_v)
    assert v["followups_ran"] is False and v["complete"] is False and v["pass"] is False

    # an incomplete eval regime -> not complete regardless of follow-ups
    v = _screen_verdict(fit_complete=True, report=_rep(True, False), followups={"all_ok": True}, verdict=ok_v)
    assert v["complete"] is False and v["pass"] is False

    # everything ran but the criterion failed -> complete, not pass
    v = _screen_verdict(fit_complete=True, report=_rep(True, True), followups={"all_ok": True}, verdict={"valid": True, "pass": False})
    assert v["complete"] is True and v["pass"] is False

    # fit incomplete -> not complete
    v = _screen_verdict(fit_complete=False, report=_rep(True, True), followups={"all_ok": True}, verdict=ok_v)
    assert v["complete"] is False


def _fake_args(**over):
    from types import SimpleNamespace
    base = dict(n_fit=8, n_cusum=4, n_dev=4, n_eval=4, max_prompt_tokens=256, max_new_tokens=8, eval_chains=2, smoke=False, exclusions=None)
    base.update(over)
    return SimpleNamespace(**base)


def test_settings_identity_changes_with_each_determining_setting():
    from scripts.experiments.qwen_operating_point import _settings_identity
    base = _settings_identity(_fake_args(), "digestX", "rev1")
    assert base == _settings_identity(_fake_args(), "digestX", "rev1")           # stable
    assert base != _settings_identity(_fake_args(max_new_tokens=16), "digestX", "rev1")  # decoding changed
    assert base != _settings_identity(_fake_args(n_fit=16), "digestX", "rev1")   # counts changed
    assert base != _settings_identity(_fake_args(), "digestY", "rev1")           # checkpoint changed
    assert base != _settings_identity(_fake_args(), "digestX", "rev2")           # dataset revision changed


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
    _append_eval_progress(path, "fresh", {"ids": [0]}, [{"decision": {"kind": "commit"}}])
    _append_eval_progress(path, "carried", {"ids": [1, 2]}, [{"x": 1}, {"x": 2}])
    _append_eval_progress(path, "fresh", {"ids": [3]}, [{"y": 1}])

    restore = _load_eval_progress(path, "idA")
    assert [s["record"]["ids"] for s in restore["fresh"]] == [[0], [3]]       # order preserved per regime
    assert [s["record"]["ids"] for s in restore["carried"]] == [[1, 2]]
    assert restore["carried"][0]["txns"] == [{"x": 1}, {"x": 2}]

    with pytest.raises(RunConflict):  # a different identity is refused, the log preserved
        _load_eval_progress(path, "idB")
    assert _load_eval_progress(path, "idA") is not None  # still usable under the right identity


def test_eval_progress_skips_a_malformed_trailing_line(tmp_path):
    from scripts.experiments.qwen_operating_point import _append_eval_progress, _init_eval_progress, _load_eval_progress
    path = str(tmp_path / "eval-progress.jsonl")
    _init_eval_progress(path, "idA")
    _append_eval_progress(path, "fresh", {"ids": [0]}, [])
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"regime": "fresh", "record": {"ids": [1]')  # a crash mid-append: truncated JSON
    restore = _load_eval_progress(path, "idA")
    assert [s["record"]["ids"] for s in restore["fresh"]] == [[0]]  # the completed one kept, the partial skipped


def test_eval_identity_binds_calibration_content_and_policy():
    from types import SimpleNamespace
    from scripts.experiments.qwen_operating_point import _eval_identity
    cal_a = SimpleNamespace(thresholds={"chunk_loss": 1.0}, cusum_reference=[1, 2, 3], model_signature="qwen:x")
    base = _eval_identity("split1", "settings1", cal_a, {"freeze_on_alarm": True})
    assert base == _eval_identity("split1", "settings1", cal_a, {"freeze_on_alarm": True})  # stable
    cal_b = SimpleNamespace(thresholds={"chunk_loss": 9.0}, cusum_reference=[1, 2, 3], model_signature="qwen:x")
    assert base != _eval_identity("split1", "settings1", cal_b, {"freeze_on_alarm": True})  # calibration content
    assert base != _eval_identity("split2", "settings1", cal_a, {"freeze_on_alarm": True})  # split
    assert base != _eval_identity("split1", "settings1", cal_a, {"freeze_on_alarm": False})  # eval policy
