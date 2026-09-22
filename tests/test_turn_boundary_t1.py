"""Unit tests for the T1 turn-boundary driver's pure logic (scratchpad FABLE-097 protocol): startup
treatment, per-turn maxima, the CUSUM lifecycle (carried on commit, restored on discard, lower side kept),
the h fit, the boundary rule, and the candidate chain's restore/audit bookkeeping with a fake runner.
No model runs here; the model path is exercised by the recorded T1 run."""

import pytest

from scripts.experiments.qwen_turn_boundary_t1 import (
    candidate_chain, cusum_turn, decide, fit_h, scored_chunks, turn_maxima,
)


def _rec(loss, ldn, index=0):
    return {"index": index, "signals": {"chunk_loss": loss, "log_delta_norm": ldn}}


def test_startup_chunk_excluded_only_in_the_session_first_turn():
    recs = [_rec(9.0, 4.1), _rec(1.0, 2.8), _rec(2.0, 3.0)]
    assert scored_chunks(recs, session_first_turn=True) == [(1.0, 2.8), (2.0, 3.0)]
    assert scored_chunks(recs, session_first_turn=False) == [(9.0, 4.1), (1.0, 2.8), (2.0, 3.0)]
    assert scored_chunks(recs[:1], session_first_turn=True) == []


def test_turn_maxima_and_empty_turn_cannot_fire():
    assert turn_maxima([(1.0, 2.8), (2.0, 2.5)]) == (2.0, 2.8)
    assert turn_maxima([]) == (None, None)
    assert decide(None, None, False, {"chunk_loss": 0.0, "log_delta_norm": 0.0}) == (False, [])


def test_cusum_turn_applies_every_z_keeps_lower_side_and_resets_on_alarm():
    alarm, end = cusum_turn([3.0, 3.0, 0.0], (0.0, 0.0), k=0.5, h=4.0)
    assert alarm is True and end == (0.0, 0.0)  # 2.5 -> 5.0 > 4 alarms and resets; the trailing 0 is still applied
    alarm, end = cusum_turn([-3.0, -2.0], (0.0, 0.0), k=0.5, h=10.0)
    assert alarm is False and end == (0.0, 4.0)  # the lower side accumulates (kept, FABLE-097 #4)
    alarm, end = cusum_turn([-3.0, -3.0, -3.0, -3.0], (0.0, 0.0), k=0.5, h=4.0)
    assert alarm is True                         # and it can alarm


def test_fit_h_smallest_h_meeting_rate_restores_detector_after_an_alarming_turn():
    # k=0.5. Turn 1 (z=1.5) commits with s_hi=1.0.
    # h=3.5: turn 2 (z=3.5) reaches 4.0 > 3.5, alarms and is DISCARDED, so turn 3 starts from turn 2's START
    # state (s_hi=1.0) -- not from the post-alarm reset (0): 1.0 + 2.6 = 3.6 > 3.5 alarms too -> 2/3 turns.
    # (Carrying the post-alarm reset instead would give 2.6, no alarm, 1/3 -- the lifecycle is what differs.)
    chains = [[[1.5], [3.5], [3.1]]]
    assert fit_h(chains, k=0.5, max_rate=1 / 3, grid=[3.5]) == (None, None)
    # h=4.5: turn 2 reaches 4.0, commits and carries it; turn 3: 4.0 + 2.6 = 6.6 alarms -> 1/3 of turns
    h, rate = fit_h(chains, k=0.5, max_rate=1 / 3, grid=[3.5, 4.5])
    assert h == 4.5 and rate == pytest.approx(1 / 3)


def test_decide_names_every_firing_gate():
    th = {"chunk_loss": 10.0, "log_delta_norm": 3.5}
    assert decide(9.0, 3.0, False, th) == (False, [])
    discard, reasons = decide(11.0, 3.6, True, th)
    assert discard and [r.split("(")[0] for r in reasons] == ["G1_chunk_loss", "G2_log_delta_norm", "G3_cusum_alarm"]


class _FakeRunner:
    """A runner stand-in: an integer 'state', a monotonic transaction counter, and full state_dict round trips
    (like TransactionRunner, load_state_dict REWINDS n_transactions -- the driver must undo that)."""

    def __init__(self):
        self.state, self.n_transactions, self.resets = 0, 0, 0

    def reset(self):
        self.state, self.resets = 0, self.resets + 1

    def state_dict(self):
        return {"state": self.state, "n_transactions": self.n_transactions}

    def load_state_dict(self, d):
        self.state, self.n_transactions = d["state"], d["n_transactions"]


def _chain(losses_by_turn, *, finite=lambda: True):
    runner = _FakeRunner()

    def run_turn(prompt, seed):
        recs = []
        for loss in losses_by_turn[prompt]:
            recs.append(_rec(loss, 2.8, index=runner.n_transactions))
            runner.n_transactions += 1
        runner.state += 1
        return {"completion": prompt, "out_ids": [1], "n_in": 1, "outcome": "eos", "records": recs}

    turns = [(i, i, 100 + i) for i in range(len(losses_by_turn))]
    out = candidate_chain(runner, turns, run_turn=run_turn, digest=lambda: {"state": runner.state},
                          is_finite=finite, th={"chunk_loss": 5.0, "log_delta_norm": 9.0},
                          ref=[2.7, 2.8, 2.9, 2.8, 2.75, 2.85, 2.8, 2.8], h=50.0)
    return runner, out


def test_candidate_chain_discard_restores_state_but_never_rewinds_audit_ids():
    # turn 0: its chunk 0 (loss 99, the startup chunk) is NOT scored -> commit; turn 1 fires G1 -> discard;
    # turn 2 commits on top of the RESTORED post-turn-0 state
    runner, out = _chain([[99.0, 1.0], [1.0, 7.0], [1.0, 1.0]])
    assert [r["decision"] for r in out] == ["commit", "discard", "commit"]
    assert out[1]["restore_equals_turn_start"] is True and out[1]["digest_after_restore"] == {"state": 1}
    assert out[2]["digest_before"] == {"state": 1} and runner.state == 2       # turn 1 never retained
    idx = [r["index"] for t in out for r in t["records"]]
    assert idx == sorted(idx) and len(set(idx)) == len(idx)                     # monotonic, never reused
    assert out[1]["n_transactions_after_restore"] == 4


def test_candidate_chain_detector_restored_on_discard_and_carried_on_commit():
    runner, out = _chain([[1.0, 1.0], [1.0, 7.0], [1.0, 1.0]])
    assert out[2]["detector_before"] == out[1]["detector_before"]               # discard restored it
    assert out[1]["detector_before"] == out[0]["detector_end"]                  # commit carried it


def test_candidate_chain_nonfinite_state_is_restored_and_ends_the_chain():
    runner, out = _chain([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], finite=lambda: False)
    assert len(out) == 1 and out[0]["decision"] == "discard"
    assert "execution_failure:nonfinite_state" in out[0]["reasons"] and out[0]["chain_ended"] == "execution_failure"
    assert out[0]["restore_equals_turn_start"] is True
