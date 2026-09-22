"""Durable-resume tests for calibrate_qwen's fit/CUSUM collection (ASTRA-086 #2).

These drive ``_collect_calibration_records`` with a FAKE runner/drive so the resume control flow is
tested without a model: interrupted-then-resumed must equal an uninterrupted run (records and the
continuous-CUSUM values identical AND in order), a deadline persists progress and raises
CalibrationIncomplete, a checkpoint from a different identity is discarded, and the transition
checkpoint carries no runner state. The exactness of the real runner-state round-trip during a
continuous chat is a separate model-level test (test_qwen_backend.py). Determinism here rests on the
Generator being rebuilt per turn from seed+salt (no RNG carry-over to persist).
"""

import pytest

import plastic.harness.calibrate as cal
from plastic.harness.calibrate import CalibrationIncomplete, _collect_calibration_records

GEN = {"max_new_tokens": 2, "temperature": 0.9, "top_k": 50}


class _FakeCalRunner:
    """A runner stand-in: a clearable transaction log plus a continuity counter that state_dict/
    load_state_dict round-trip, so a wrong CUSUM resume shows up as a different produced value."""

    def __init__(self):
        self.transactions = []
        self._k = 0
        self.resets = 0
        self.loads = []

    def reset(self):
        self.transactions = []
        self._k = 0
        self.resets += 1

    def state_dict(self):
        return {"k": self._k}

    def load_state_dict(self, d):
        self._k = int(d["k"])
        self.loads.append(int(d["k"]))


def _fake_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
    # log_delta_norm depends on BOTH the continuity counter and the seed, so the continuous pass is
    # order/state sensitive; increment continuity so a resumed chat must restore the exact state
    val = 100.0 * runner._k + float(gen.initial_seed())
    runner.transactions.append({
        "signals": {"log_delta_norm": val, "chunk_loss": 1.0},
        "sources": {"user": 4, "model": 4},
        "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 0.1},
    })
    runner._k += 1


def _collect(runner, prompts, cusum, *, deadline=None, checkpoint_path=None, identity="id-A", drive=_fake_drive):
    return _collect_calibration_records(
        runner, None, prompts, cusum, seed=0, gen=GEN, deadline=deadline,
        checkpoint_path=checkpoint_path, identity=identity, drive=drive, log=lambda *_: None,
    )


def _ldn(records):
    return [r["signals"]["log_delta_norm"] for r in records]


def _clock_drive_factory(clock):
    def clock_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        _fake_drive(runner, tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, gen=gen)
        clock["t"] += 1.0  # each completed turn advances the wall clock by one unit
    return clock_drive


def test_resume_equivalence_across_cusum_interruption(tmp_path, monkeypatch):
    fit = [f"f{i}" for i in range(4)]
    cus = [f"c{i}" for i in range(4)]

    # uninterrupted reference
    rec0, fu0, cont0, cu0, ran0 = _collect(_FakeCalRunner(), fit, cus)
    assert ran0 and fu0 == 4 and cu0 == 4
    assert cont0 == [1000.0, 1101.0, 1202.0, 1303.0]  # continuity accumulates across the CUSUM chat

    clock = {"t": 0.0}
    monkeypatch.setattr(cal.time, "time", lambda: clock["t"])
    drive = _clock_drive_factory(clock)
    ckpt = str(tmp_path / "cal.ckpt")

    # first invocation: fit (clock 0->4) completes, CUSUM turns 0,1 run (clock 4->6), turn 2 trips at 6 > 5.5
    r1 = _FakeCalRunner()
    with pytest.raises(CalibrationIncomplete) as ei:
        _collect(r1, fit, cus, deadline=5.5, checkpoint_path=ckpt, drive=drive)
    assert ei.value.phase == "cusum" and ei.value.fit_used == 4 and ei.value.cusum_used == 2

    # second invocation: resume (no deadline). Must EXTEND, restoring the continuity state at turn 2
    r2 = _FakeCalRunner()
    rec1, fu1, cont1, cu1, ran1 = _collect(r2, fit, cus, deadline=None, checkpoint_path=ckpt, drive=drive)
    assert cu1 == 4 and cont1 == cont0            # identical continuous values, in order
    assert _ldn(rec1) == _ldn(rec0)               # identical fit records, in order
    assert r2.loads == [2]                        # resume restored the runner state saved at turn 2
    # (the checkpoint file is removed by calibrate_qwen only after the calibration is durably saved)


def test_resume_equivalence_across_fit_interruption(tmp_path, monkeypatch):
    fit = [f"f{i}" for i in range(4)]
    cus = [f"c{i}" for i in range(4)]
    rec0, fu0, cont0, cu0, _ = _collect(_FakeCalRunner(), fit, cus)

    clock = {"t": 0.0}
    monkeypatch.setattr(cal.time, "time", lambda: clock["t"])
    drive = _clock_drive_factory(clock)
    ckpt = str(tmp_path / "cal.ckpt")

    # deadline 1.5: fit turns 0,1 run (clock 0->2), turn 2 trips at 2 > 1.5 -> cut mid-fit
    with pytest.raises(CalibrationIncomplete) as ei:
        _collect(_FakeCalRunner(), fit, cus, deadline=1.5, checkpoint_path=ckpt, drive=drive)
    assert ei.value.phase == "fit" and ei.value.fit_used == 2 and ei.value.cusum_used == 0

    rec1, fu1, cont1, cu1, _ = _collect(_FakeCalRunner(), fit, cus, deadline=None, checkpoint_path=ckpt, drive=drive)
    assert fu1 == 4 and cu1 == 4
    assert _ldn(rec1) == _ldn(rec0)               # fit records are per-prompt independent, so identical
    assert cont1 == cont0                          # CUSUM ran fresh after fit completed, identical


def test_identity_mismatch_discards_checkpoint(tmp_path):
    fit = ["f0", "f1"]
    cus = ["c0", "c1"]
    ckpt = str(tmp_path / "cal.ckpt")
    # a checkpoint from a DIFFERENT identity with bogus far-along progress must be ignored, not extended
    cal._ckpt_save(ckpt, {
        "identity": "id-OTHER", "phase": "cusum", "records": [{"signals": {"log_delta_norm": 9.0}}],
        "fit_used": 99, "cont": [9.0], "cusum_used": 99, "runner_state": {"k": 99},
        "fit_requested": 2, "cusum_requested": 2,
    })
    rec, fu, cont, cu, ran = _collect(_FakeCalRunner(), fit, cus, checkpoint_path=ckpt, identity="id-A")
    assert fu == 2 and cu == 2 and len(rec) == 2      # started fresh under id-A
    assert cont == [1000.0, 1101.0]                    # the fresh values, not the bogus 9.0s


def test_no_checkpoint_deadline_returns_partial_without_raising(tmp_path, monkeypatch):
    fit = [f"f{i}" for i in range(4)]
    cus = [f"c{i}" for i in range(4)]
    clock = {"t": 0.0}
    monkeypatch.setattr(cal.time, "time", lambda: clock["t"])
    drive = _clock_drive_factory(clock)
    # deadline 1.5 and NO checkpoint_path: partial fit, no exception (the un-checkpointed behavior)
    rec, fu, cont, cu, ran = _collect(_FakeCalRunner(), fit, cus, deadline=1.5, checkpoint_path=None, drive=drive)
    assert fu == 2 and ran is True                     # two fit turns kept; delta signal present
    assert cu == 0 and cont == []                      # the already-passed deadline stops CUSUM at once


def test_transition_checkpoint_has_no_runner_state(tmp_path, monkeypatch):
    fit = [f"f{i}" for i in range(3)]
    cus = [f"c{i}" for i in range(3)]
    clock = {"t": 0.0}
    monkeypatch.setattr(cal.time, "time", lambda: clock["t"])
    drive = _clock_drive_factory(clock)
    ckpt = str(tmp_path / "cal.ckpt")
    # deadline 2.5: fit (3 turns, clock 0->3) completes; CUSUM turn 0 trips at 3 > 2.5 before any turn runs
    with pytest.raises(CalibrationIncomplete):
        _collect(_FakeCalRunner(), fit, cus, deadline=2.5, checkpoint_path=ckpt, drive=drive)
    st = cal._ckpt_load(ckpt)
    assert st["phase"] == "cusum" and st["cusum_used"] == 0 and st["runner_state"] is None
