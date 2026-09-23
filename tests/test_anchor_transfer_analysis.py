"""Read-only analyses over archived sleep_controls runs: the per-turn write table and the anchor storage shift."""

from __future__ import annotations

import json
import os

import pytest

from scripts.experiments.anchor_transfer_analysis import analyze_run, answer_group, storage_shift, turn_write_table


def _tx(i, kind="commit", delta=1.0, surprise=5.0, loss=2.0, reasons=None, requested=None):
    return {"index": i, "decision": {"kind": kind, "reasons": []}, "requested": {"kind": requested or kind, "reasons": reasons or []},
            "signals": {"surprise_max": surprise, "chunk_loss": loss}, "accepted": {"delta_norm": delta}}


def test_turn_write_table_joins_chunks_by_tx_range_and_ranks_turns():
    tx = [_tx(0, delta=2.0), _tx(1, delta=3.0, reasons=["log_only", "would_rollback:chunk_loss_z(6>=6)"]), _tx(2, delta=0.5, surprise=9.0),
          _tx(3, delta=0.5, loss=7.0), _tx(9, delta=100.0)]  # index 9 belongs to no turn and must be ignored
    trace = [{"kind": "chat", "prompt": "My cat is called Marlowe.", "tx_start": 0, "tx_end": 2},
             {"kind": "chat", "prompt": "Remember this: a week has nine days.", "tx_start": 2, "tx_end": 4},
             {"kind": "reset"}, {"kind": "chat", "prompt": "no chunks", "tx_start": 20, "tx_end": 21}]
    rows = turn_write_table(tx, trace)
    assert [r["prompt"][:6] for r in rows] == ["My cat", "Rememb"]
    assert rows[0]["delta_norm_sum"] == 5.0 and rows[0]["flagged"] and not rows[0]["planted"] and rows[0]["rank_by_write"] == 1
    assert rows[1]["delta_norm_sum"] == 1.0 and rows[1]["planted"] and not rows[1]["flagged"] and rows[1]["rank_by_write"] == 2
    assert rows[1]["surprise_max"] == 9.0 and rows[1]["chunk_loss_max"] == 7.0 and rows[1]["chunks"] == 2


def test_storage_shift_keys_probes_by_question_and_answer_and_groups_them():
    """The planted contradictions reuse general questions with a different expected answer; both must survive."""
    def res(q, e, lp, hit, variant="verbatim"):
        return {"question": q, "expected": e, "answer_logprob": lp, "contains": hit, "variant": variant}
    report = {"before": {"recall": {"results": [res("Capital of France?", "Paris", -4.0, True), res("Capital of France?", "Berlin", -12.0, False),
                                                 res("My cat?", "Marlowe", -8.0, False), res("My cat?", "Marlowe", -1.0, True, "paraphrase")]}},
              "after": {"recall": {"results": [res("Capital of France?", "Paris", -3.0, True), res("Capital of France?", "Berlin", -10.0, False),
                                                res("My cat?", "Marlowe", -7.5, False)]}}}
    s = storage_shift(report)
    by = {(p["question"], p["expected"]): p for p in s["probes"]}
    assert by[("Capital of France?", "Paris")]["shift"] == pytest.approx(1.0) and by[("Capital of France?", "Berlin")]["shift"] == pytest.approx(2.0)
    assert by[("My cat?", "Marlowe")]["shift"] == pytest.approx(0.5) and len(s["probes"]) == 3   # the paraphrase row is not a verbatim probe
    assert s["groups"]["planted"] == {"n": 1, "mean_shift": pytest.approx(2.0), "greedy_hits_before": 0, "greedy_hits_after": 0}
    assert s["groups"]["general"]["n"] == 1 and s["groups"]["taught"]["n"] == 1
    assert answer_group("50") == "planted" and answer_group("blue") == "general" and answer_group("Wren") == "taught"


def test_analyze_run_reports_absent_sessions_as_absent(tmp_path):
    run = tmp_path / "some_run"
    run.mkdir()
    out = analyze_run(str(run))
    assert out["teach_turns"] is None and "anchor_storage_shift" not in out


@pytest.mark.skipif(not os.path.exists("docs/research/results/sleep-2026-09-23/final_step250_seed0_exclude/sessions/teach/transactions.jsonl"),
                    reason="archive not present")
def test_archived_seed0_planted_turns_are_not_the_largest_writes():
    """FABLE-179: on the archived seed-0 teach session the four planted turns rank at or below the median write."""
    out = analyze_run("docs/research/results/sleep-2026-09-23/final_step250_seed0_exclude")
    ws = out["write_summary"]
    assert ws["turns"] == 30 and len(ws["planted_ranks"]) == 4 and min(ws["planted_ranks"]) > 5
    st = out["anchor_storage_shift"]["groups"]
    assert st["planted"]["n"] == 4 and st["general"]["n"] == 6 and st["taught"]["n"] == 29
    assert st["taught"]["mean_shift"] < st["general"]["mean_shift"] < st["planted"]["mean_shift"] + 1.0
    json.dumps(out)  # serializable
