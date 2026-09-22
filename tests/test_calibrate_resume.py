"""Durable-resume tests for calibrate_qwen's fit/CUSUM collection (ASTRA-086 #2, ASTRA-091).

These drive ``_collect_calibration_records`` with a FAKE runner/drive so the resume control flow is
tested without a model: interrupted-then-resumed must equal an uninterrupted run in BOTH the numeric
fit/CUSUM values AND the record indices (order preserved), a deadline persists progress and raises
CalibrationIncomplete, and an incompatible/corrupt/malformed checkpoint is REJECTED and preserved
(never silently overwritten or restarted). The exactness of the real runner-state round-trip during a
continuous chat is a separate model-level test (test_qwen_backend.py). Determinism here rests on the
Generator being rebuilt per turn from seed+salt (no RNG carry-over to persist).

The fake runner mirrors the real TransactionRunner counter contract (transaction.py): each record
carries a global ``index`` = ``n_transactions``, ``n_transactions`` increments per transaction, and
``reset()`` does NOT clear it (so a fresh process would restart indices unless the counter is
persisted and restored).
"""

import pytest

import plastic.harness.calibrate as cal
from plastic.harness.calibrate import CalibrationCheckpointError, CalibrationIncomplete, _collect_calibration_records

GEN = {"max_new_tokens": 2, "temperature": 0.9, "top_k": 50}


class _FakeCalRunner:
    """A runner stand-in with a continuity counter ``_k`` and a global ``n_transactions`` counter that
    reset() preserves (matching transaction.py), both round-tripped through state_dict/load_state_dict."""

    def __init__(self):
        self.transactions = []
        self._k = 0
        self.n_transactions = 0
        self.resets = 0
        self.loads = []

    def reset(self):
        self.transactions = []
        self._k = 0
        self.resets += 1
        # n_transactions intentionally preserved, exactly as TransactionRunner.reset() does

    def state_dict(self):
        return {"k": self._k, "n_transactions": self.n_transactions}

    def load_state_dict(self, d):
        self._k = int(d["k"])
        self.n_transactions = int(d["n_transactions"])
        self.loads.append(self._k)


def _fake_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
    # log_delta_norm depends on BOTH the continuity counter and the seed, so the continuous pass is
    # order/state sensitive; index/n_transactions give each record a lifetime-independent identity
    val = 100.0 * runner._k + float(gen.initial_seed())
    runner.transactions.append({
        "index": runner.n_transactions,
        "signals": {"log_delta_norm": val, "chunk_loss": 1.0},
        "sources": {"user": 4, "model": 4},
        "decision": {"kind": "commit"}, "eligible": True, "accepted": {"delta_norm": 0.1},
    })
    runner._k += 1
    runner.n_transactions += 1


def _collect(runner, prompts, cusum, *, deadline=None, checkpoint_path=None, identity="id-A", drive=_fake_drive):
    return _collect_calibration_records(
        runner, None, prompts, cusum, seed=0, gen=GEN, deadline=deadline,
        checkpoint_path=checkpoint_path, identity=identity, drive=drive, log=lambda *_: None,
    )


def _ldn(records):
    return [r["signals"]["log_delta_norm"] for r in records]


def _idx(records):
    return [r["index"] for r in records]


def _clock_drive_factory(clock):
    def clock_drive(runner, tok, prompt, *, max_new_tokens, temperature, top_k, gen):
        _fake_drive(runner, tok, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, gen=gen)
        clock["t"] += 1.0  # each completed turn advances the wall clock by one unit
    return clock_drive


def test_resume_equivalence_across_cusum_interruption(tmp_path, monkeypatch):
    fit = [f"f{i}" for i in range(4)]
    cus = [f"c{i}" for i in range(4)]

    rec0, fu0, cont0, cu0, ran0 = _collect(_FakeCalRunner(), fit, cus)
    assert ran0 and fu0 == 4 and cu0 == 4
    assert cont0 == [1000.0, 1101.0, 1202.0, 1303.0]  # continuity accumulates across the CUSUM chat
    assert _idx(rec0) == [0, 1, 2, 3]

    clock = {"t": 0.0}
    monkeypatch.setattr(cal.time, "time", lambda: clock["t"])
    drive = _clock_drive_factory(clock)
    ckpt = str(tmp_path / "cal.ckpt")

    r1 = _FakeCalRunner()
    with pytest.raises(CalibrationIncomplete) as ei:
        _collect(r1, fit, cus, deadline=5.5, checkpoint_path=ckpt, drive=drive)
    assert ei.value.phase == "cusum" and ei.value.fit_used == 4 and ei.value.cusum_used == 2

    r2 = _FakeCalRunner()
    rec1, fu1, cont1, cu1, ran1 = _collect(r2, fit, cus, deadline=None, checkpoint_path=ckpt, drive=drive)
    assert cu1 == 4 and cont1 == cont0            # identical continuous values, in order
    assert _ldn(rec1) == _ldn(rec0) and _idx(rec1) == _idx(rec0)  # identical fit records AND indices
    assert r2.loads == [2]                        # resume restored the runner state saved at turn 2


def test_resume_equivalence_across_fit_interruption(tmp_path, monkeypatch):
    # ASTRA-091 #3: a mid-fit resume must reproduce record INDICES too, not just the numbers
    fit = [f"f{i}" for i in range(4)]
    cus = [f"c{i}" for i in range(4)]
    rec0, fu0, cont0, cu0, _ = _collect(_FakeCalRunner(), fit, cus)
    assert _idx(rec0) == [0, 1, 2, 3]

    clock = {"t": 0.0}
    monkeypatch.setattr(cal.time, "time", lambda: clock["t"])
    drive = _clock_drive_factory(clock)
    ckpt = str(tmp_path / "cal.ckpt")

    with pytest.raises(CalibrationIncomplete) as ei:
        _collect(_FakeCalRunner(), fit, cus, deadline=1.5, checkpoint_path=ckpt, drive=drive)
    assert ei.value.phase == "fit" and ei.value.fit_used == 2 and ei.value.cusum_used == 0

    rec1, fu1, cont1, cu1, _ = _collect(_FakeCalRunner(), fit, cus, deadline=None, checkpoint_path=ckpt, drive=drive)
    assert fu1 == 4 and cu1 == 4
    assert _ldn(rec1) == _ldn(rec0)
    assert _idx(rec1) == [0, 1, 2, 3]             # NOT [0,1,0,1]: the counter was persisted and restored
    assert cont1 == cont0


def test_no_checkpoint_deadline_returns_partial_without_raising(monkeypatch):
    fit = [f"f{i}" for i in range(4)]
    cus = [f"c{i}" for i in range(4)]
    clock = {"t": 0.0}
    monkeypatch.setattr(cal.time, "time", lambda: clock["t"])
    drive = _clock_drive_factory(clock)
    rec, fu, cont, cu, ran = _collect(_FakeCalRunner(), fit, cus, deadline=1.5, checkpoint_path=None, drive=drive)
    assert fu == 2 and ran is True                 # partial fit kept; delta signal present
    assert cu == 0 and cont == []                  # the already-passed deadline stops CUSUM at once


def test_transition_checkpoint_has_no_runner_state(tmp_path, monkeypatch):
    fit = [f"f{i}" for i in range(3)]
    cus = [f"c{i}" for i in range(3)]
    clock = {"t": 0.0}
    monkeypatch.setattr(cal.time, "time", lambda: clock["t"])
    drive = _clock_drive_factory(clock)
    ckpt = str(tmp_path / "cal.ckpt")
    with pytest.raises(CalibrationIncomplete):
        _collect(_FakeCalRunner(), fit, cus, deadline=2.5, checkpoint_path=ckpt, drive=drive)
    st = cal._ckpt_load(ckpt)
    assert st["phase"] == "cusum" and st["cusum_used"] == 0 and st["runner_state"] is None
    assert st["n_transactions"] == 3  # the transition checkpoint carries the fit counter forward


# --------------------------------------------------------------------- rejection / preservation

def _valid_ckpt(**over):
    # a well-formed matching-identity checkpoint: fit done (2 records), 1 of 2 CUSUM turns completed
    base = {
        "identity": "id-A", "phase": "cusum",
        "records": [{"index": 0, "signals": {"log_delta_norm": 1000.0}},
                    {"index": 1, "signals": {"log_delta_norm": 1001.0}}],
        "fit_used": 2, "cont": [1000.0], "cusum_used": 1,
        "runner_state": {"k": 1, "n_transactions": 3}, "n_transactions": 3,
        "fit_requested": 2, "cusum_requested": 2,
    }
    base.update(over)
    return base


def test_valid_cusum_checkpoint_resumes_to_completion(tmp_path):
    ckpt = str(tmp_path / "cal.ckpt")
    cal._ckpt_save(ckpt, _valid_ckpt())
    r = _FakeCalRunner()
    rec, fu, cont, cu, ran = _collect(r, ["f0", "f1"], ["c0", "c1"], checkpoint_path=ckpt, identity="id-A")
    assert cu == 2 and r.loads == [1]             # loaded the carried state at turn 1, finished turn 2
    assert cont == [1000.0, 1101.0]               # extended (100*1 + (0+1001)), not restarted


def test_incompatible_identity_checkpoint_is_rejected_and_preserved(tmp_path):
    # ASTRA-091 #1: a checkpoint for a different model/corpus/seed/config is rejected, not overwritten
    ckpt = str(tmp_path / "cal.ckpt")
    cal._ckpt_save(ckpt, _valid_ckpt(identity="id-OTHER"))
    before = open(ckpt, "rb").read()
    with pytest.raises(CalibrationCheckpointError):
        _collect(_FakeCalRunner(), ["f0", "f1"], ["c0", "c1"], checkpoint_path=ckpt, identity="id-A")
    assert open(ckpt, "rb").read() == before      # incremental work preserved, not clobbered


def test_corrupt_checkpoint_is_rejected_and_preserved(tmp_path):
    ckpt = str(tmp_path / "cal.ckpt")
    with open(ckpt, "wb") as f:
        f.write(b"this is not a torch checkpoint")
    before = open(ckpt, "rb").read()
    with pytest.raises(CalibrationCheckpointError):
        _collect(_FakeCalRunner(), ["f0", "f1"], ["c0", "c1"], checkpoint_path=ckpt, identity="id-A")
    assert open(ckpt, "rb").read() == before


@pytest.mark.parametrize("over, why", [
    ({"phase": "bogus"}, "invalid phase"),
    ({"fit_used": 999}, "fit_used out of range"),
    ({"cusum_used": 999}, "cusum_used out of range"),
    ({"fit_requested": 5}, "inconsistent fit_requested"),
    ({"cusum_requested": 9}, "inconsistent cusum_requested"),
    ({"cusum_used": 1, "runner_state": None}, "mid-CUSUM missing carried state"),
    ({"phase": "fit", "cusum_used": 1}, "fit phase with cusum_used>0"),
    ({"n_transactions": -1}, "negative counter"),
    ({"records": "notalist"}, "malformed records"),
    ({"fit_used": True}, "boolean is not a valid counter"),
])
def test_malformed_checkpoint_is_rejected_and_preserved(tmp_path, over, why):
    # ASTRA-091 #2: validate phase/counters/consistency/carried-state BEFORE any write
    ckpt = str(tmp_path / "cal.ckpt")
    cal._ckpt_save(ckpt, _valid_ckpt(**over))
    before = open(ckpt, "rb").read()
    with pytest.raises(CalibrationCheckpointError):
        _collect(_FakeCalRunner(), ["f0", "f1"], ["c0", "c1"], checkpoint_path=ckpt, identity="id-A")
    assert open(ckpt, "rb").read() == before, f"file must be preserved on rejection ({why})"
