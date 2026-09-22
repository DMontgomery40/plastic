"""Pure-logic tests for the shock pass-1 driver: prompt resolution from pinned sources, the arm plan, seed
schedule, later-turn divergence, and the guarded-arm intervention summary. No model runs."""

import pytest

from scripts.experiments.qwen_shock_pass1 import (
    SEED_OFFSET, arm_plan, chunk_table, intervention_summary, later_turn_divergence, resolve_turn, turn_seeds,
)

SOURCES = {"dev": ["dev prompt zero", "dev prompt one"], "jbb_benign": ["benign goal"]}


def test_resolve_turn_literal_source_and_template():
    assert resolve_turn({"text": "hello"}, SOURCES) == "hello"
    assert resolve_turn({"source": "dev", "index": 1}, SOURCES) == "dev prompt one"
    assert resolve_turn({"source": "jbb_benign", "index": 0, "template": "Please: {goal}."}, SOURCES) == "Please: benign goal."
    with pytest.raises(KeyError):
        resolve_turn({"source": "missing", "index": 0}, SOURCES)


def test_arm_plan_and_seeds_are_distinct_per_chain_and_turn():
    assert arm_plan({"shock_turn": 1}) == ["N", "D", "O", "G"]
    assert arm_plan({}) == ["N", "G"]
    s0, s1 = turn_seeds(0, 4, 100), turn_seeds(1, 4, 100)
    assert s0 == [100 + SEED_OFFSET + t for t in range(4)]
    assert not set(s0) & set(s1)


def _turn(t, ids):
    return {"turn": t, "out_ids": ids}


def test_later_turn_divergence_aligns_by_turn_index_and_skips_turns_before_the_shock():
    d = [_turn(0, [1]), _turn(1, [9]), _turn(2, [2]), _turn(3, [3])]
    o = [_turn(0, [1]), _turn(2, [2]), _turn(3, [4])]           # the omitted arm has no turn 1
    res = later_turn_divergence(d, o, from_turn=2)
    assert [t["turn"] for t in res["turns"]] == [2, 3]          # turn 2 vs turn 2, turn 3 vs turn 3; nothing positional
    assert res["turns"][0]["ids_differ"] is False and res["first_differing_turn"] == 3
    same = later_turn_divergence(o, [dict(x) for x in o], from_turn=2)
    assert same["first_differing_turn"] is None and all(not t["ids_differ"] for t in same["turns"])


def _rec(kind, reasons, alarm=False, **sig):
    base = {"chunk_loss": 1.0, "log_delta_norm": 2.0, "z": None, "cusum_alarm": alarm}
    base.update(sig)
    return {"pos_start": 0, "pos_end": 8, "sources": {"user": 8, "model": 0}, "decision": {"kind": kind, "reasons": reasons},
            "signals": base, "accepted": {"delta_norm": 0.0 if kind == "rollback" else 5.0}}


def test_intervention_summary_counts_kinds_reasons_and_alarms():
    turns = [{"turn": 0, "records": [_rec("commit", []), _rec("rollback", ["chunk_loss_z(6.10>=6.0)", "cusum_alarm"], alarm=True)]},
             {"turn": 1, "records": [_rec("readonly", ["learning_ineligible"])]}]
    s = intervention_summary(turns)
    assert s["decisions"] == {"commit": 1, "rollback": 1, "readonly": 1}
    assert s["reasons"] == {"chunk_loss_z": 1, "cusum_alarm": 1, "learning_ineligible": 1}
    assert s["cusum_alarms"] == 1


def test_chunk_table_flattens_and_reads_missing_z_and_canary_fields_as_none():
    rows = chunk_table([{"turn": 2, "records": [_rec("commit", [])]}])
    assert rows == [{"turn": 2, "pos_start": 0, "pos_end": 8, "sources": {"user": 8, "model": 0}, "chunk_loss": 1.0,
                     "log_delta_norm": 2.0, "accepted_delta_norm": 5.0, "z_chunk_loss": None, "z_log_delta_norm": None,
                     "cusum_alarm": False, "canary_delta_coherence": None, "canary_alignment": None, "decision": "commit", "reasons": []}]
