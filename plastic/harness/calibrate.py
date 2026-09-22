"""Calibration: per-signal benign reference windows and thresholds, Fisher diagonal, canary baselines.

Produced once per model by running a benign stream through the transaction
runner in ``log_only`` mode. Thresholds are empirical upper quantiles at the
target false-positive rate, split across the decision signals by a union bound
so the total benign gating rate stays near ``target_fpr``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import Tensor

from plastic.harness.config import HarnessConfig
from plastic.harness.signals import STAT_SIGNALS
from plastic.harness.stats import quantile_threshold

ROLLBACK_DECISION_SIGNALS: tuple[str, ...] = STAT_SIGNALS + ("canary_delta_coherence",)


@dataclass
class Calibration:
    model_signature: str = ""
    n_chunks: int = 0
    reference: dict[str, list[float]] = field(default_factory=dict)
    # log_delta_norm from a continuous benign session; the CUSUM signal is standardized against
    # this (not ``reference``, which is reset-every-N and biases a cross-chunk statistic).
    cusum_reference: list[float] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    achievable_fpr: dict[str, float] = field(default_factory=dict)
    canary_baseline: dict[str, float] = field(default_factory=dict)
    target_fpr: float = 0.01
    created_at_unix: int = 0
    fisher: list[Tensor] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_signature": self.model_signature,
            "n_chunks": self.n_chunks,
            "reference": {k: [float(x) for x in v] for k, v in self.reference.items()},
            "cusum_reference": [float(x) for x in self.cusum_reference],
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            "achievable_fpr": {k: float(v) for k, v in self.achievable_fpr.items()},
            "canary_baseline": {k: float(v) for k, v in self.canary_baseline.items()},
            "target_fpr": self.target_fpr,
            "created_at_unix": self.created_at_unix,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Calibration":
        return cls(
            model_signature=str(d.get("model_signature", "")),
            n_chunks=int(d.get("n_chunks", 0)),
            reference={k: list(v) for k, v in d.get("reference", {}).items()},
            cusum_reference=[float(x) for x in d.get("cusum_reference", [])],
            thresholds={k: float(v) for k, v in d.get("thresholds", {}).items()},
            achievable_fpr={k: float(v) for k, v in d.get("achievable_fpr", {}).items()},
            canary_baseline={k: float(v) for k, v in d.get("canary_baseline", {}).items()},
            target_fpr=float(d.get("target_fpr", 0.01)),
            created_at_unix=int(d.get("created_at_unix", 0)),
        )

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        if self.fisher is not None:
            torch.save([t.detach().cpu() for t in self.fisher], os.path.join(model_dir, "fisher.pt"))

    @classmethod
    def load(cls, model_dir: str) -> "Calibration":
        with open(os.path.join(model_dir, "calibration.json"), "r", encoding="utf-8") as f:
            cal = cls.from_dict(json.load(f))
        fp = os.path.join(model_dir, "fisher.pt")
        if os.path.exists(fp):
            cal.fisher = torch.load(fp, map_location="cpu")
        return cal

    @staticmethod
    def exists(model_dir: str) -> bool:
        return os.path.exists(os.path.join(model_dir, "calibration.json"))


def conformal_threshold(values: list[float], fpr: float, *, side: str = "upper") -> tuple[float, float]:
    """Split-conformal order-statistic threshold and the false-positive rate it can actually deliver.

    With ``n`` exchangeable benign values, the ``ceil((n+1)(1-fpr))``-th smallest value is exceeded by
    a fresh benign value with probability at most ``fpr``; when ``fpr (n+1) < 1`` that is the sample
    maximum and the achievable rate is ``1/(n+1)``, which is reported so nobody claims a rate the
    sample size cannot support.
    """
    vals = sorted(float(v) for v in values if v == v)
    n = len(vals)
    if n < 8:
        raise ValueError(f"need at least 8 values, got {n}")
    if side == "lower":
        lo, achievable = conformal_threshold([-v for v in vals], fpr, side="upper")
        return -lo, achievable
    k = math.ceil((n + 1) * (1.0 - float(fpr)))
    if k > n:
        return vals[-1], 1.0 / (n + 1)
    return vals[k - 1], max(float(fpr), 1.0 / (n + 1))


def thresholds_from_records(records: list[dict[str, Any]], *, target_fpr: float) -> tuple[dict[str, float], dict[str, float]]:
    """Per-signal thresholds with the false-positive budget split by union bound; returns
    (thresholds, achievable_fpr). Coherence and the statistical signals get upper thresholds,
    the poison canary a lower one."""
    upper = [name for name in ROLLBACK_DECISION_SIGNALS if sum(1 for r in records if r.get(name) is not None) >= 8]
    lower = ["canary_delta_poison"] if sum(1 for r in records if r.get("canary_delta_poison") is not None) >= 8 else []
    present = upper + lower
    if not present:
        return {}, {}
    per_signal_fpr = max(1e-5, float(target_fpr) / len(present))
    thresholds: dict[str, float] = {}
    achievable: dict[str, float] = {}
    for name in upper:
        vals = [float(r[name]) for r in records if r.get(name) is not None]
        thresholds[name], achievable[name] = conformal_threshold(vals, per_signal_fpr)
    for name in lower:
        vals = [float(r[name]) for r in records if r.get(name) is not None]
        thresholds[name], achievable[name] = conformal_threshold(vals, per_signal_fpr, side="lower")
    return thresholds, achievable


def calibrated_cusum_h(zs: list[float], *, k: float, h_min: float, target_fpr: float) -> tuple[float, float] | None:
    """Calibrate the CUSUM alarm threshold as a benign run-length, not a walk endpoint.

    The two-sided CUSUM resets to zero every time it alarms, so ``h`` controls the *rate*
    at which the in-control stream alarms — not the peak it reaches. Given a benign z-sequence
    ``zs`` (centered on its own regime, see the caller), we replay it through the real
    ``Cusum(k, h)`` for a grid of candidate ``h``, count alarms, and return the smallest ``h``
    whose benign per-chunk alarm rate is at or below ``target_fpr``, together with that achieved
    rate. Note this is a fitted empirical run-length minimization on a reference sequence, *not* a
    conformal order statistic: unlike the per-chunk thresholds it carries no exchangeability
    guarantee, and the achieved rate is an in-sample fit to ``zs``.

    Two mistakes this avoids. (1) The old heuristic took ``1.25 * peak`` of a never-resetting
    walk, which is not a quantile of anything and let a benign continuous stream alarm within
    tens of chunks. (2) ``zs`` must come from a *continuous* benign session and be standardized
    against a reference from that same continuous regime: a CUSUM fed a signal whose benign mean
    is nonzero ratchets and has no valid ``h``, whatever this function returns.
    """
    from plastic.harness.stats import Cusum

    if not zs:
        return None
    n = len(zs)

    def alarm_rate(h: float) -> float:
        c = Cusum(k, h)
        return sum(1 for z in zs if c.update(z)) / n

    # Upper bound for the search: the peak of a never-resetting walk can never be exceeded
    # by the resetting statistic, so no candidate above it can help.
    s_hi = s_lo = peak = 0.0
    for z in zs:
        s_hi = max(0.0, s_hi + z - k)
        s_lo = max(0.0, s_lo - z - k)
        peak = max(peak, s_hi, s_lo)
    hi = max(float(h_min), peak + 1.0)
    # Smallest h on a fine grid whose benign alarm rate is within target.
    steps = 200
    best_h, best_rate = hi, alarm_rate(hi)
    for i in range(steps + 1):
        h = float(h_min) + (hi - float(h_min)) * i / steps
        rate = alarm_rate(h)
        if rate <= target_fpr:
            best_h, best_rate = h, rate
            break
    # best_rate is the empirical alarm frequency at best_h (alarms / n) on the calibration
    # z-sequence — reported verbatim, including 0.0 when no alarm fired. Zero observed alarms is
    # an observation, not a claim of zero population risk; do not floor it to a resolution
    # convention (that would report a rate the replay never produced).
    return best_h, best_rate


def _continuous_cusum_vals(runner, stream_iter, *, n: int, k_signal: str = "log_delta_norm") -> list[float]:
    """Collect ``n`` chunks of a benign signal from a single continuous (never-reset) session.

    The CUSUM signal must be standardized against a reference from this same continuous regime:
    the per-chunk reset-every-N reference biases it (mature sessions write less than the
    fresh-heavy reference), so the CUSUM would ratchet on benign text. These raw values become
    the calibration's ``cusum_reference`` and the live runner standardizes against them."""
    runner.flush()
    while runner.transactions:
        runner.transactions.pop(0)
    runner.reset()
    vals: list[float] = []
    while len(vals) < n:
        try:
            item = next(stream_iter)
        except StopIteration:
            break  # a finite stream (tests): calibrate on what the continuous pass could gather
        if isinstance(item, tuple):
            runner.feed_physics(item[0], item[1])
        else:
            runner.feed_tokens(list(item), source="user")
        while runner.transactions:
            rec = runner.transactions.pop(0)
            v = rec["signals"].get(k_signal)
            if v is not None:
                vals.append(float(v))
    return vals[:n]


def calibrate_from_runner(
    runner,
    stream: Iterable[Any],
    *,
    n_chunks: int,
    model_signature: str,
    target_fpr: float = 0.01,
    fisher: list[Tensor] | None = None,
    reference_cap: int = 512,
    reset_every: int | None = 32,
) -> Calibration:
    """Run ``stream`` items through ``runner`` (which must be in log-only mode) and build a calibration.

    ``stream`` yields token-id lists (text) or ``(inputs (T, 7), targets (T, 4))`` tuples (physics).
    The runner is reset every ``reset_every`` chunks so the reference distribution covers
    fresh sessions as well as mature ones (a session that has just started has a
    different surprise and write profile from one that has absorbed context).
    """
    if not runner.hcfg.log_only:
        raise ValueError("calibration requires a runner in log_only mode")
    records: list[dict[str, Any]] = []
    since_reset = 0
    stream_iter = iter(stream)
    for item in stream_iter:
        if isinstance(item, tuple):
            runner.feed_physics(item[0], item[1])
        else:
            runner.feed_tokens(list(item), source="user")
        while runner.transactions:
            rec = runner.transactions.pop(0)
            records.append(rec["signals"])
            since_reset += 1
        if reset_every is not None and since_reset >= reset_every:
            runner.flush()
            while runner.transactions:
                records.append(runner.transactions.pop(0)["signals"])
            runner.reset()
            since_reset = 0
        if len(records) >= n_chunks:
            break
    runner.flush()
    while runner.transactions:
        records.append(runner.transactions.pop(0)["signals"])
    if not records:
        raise ValueError("no chunks observed during calibration")
    reference = {
        name: [float(r[name]) for r in records if r.get(name) is not None][-reference_cap:]
        for name in ROLLBACK_DECISION_SIGNALS
    }
    reference = {k: v for k, v in reference.items() if v}
    baseline: dict[str, float] = {}
    for key in ("canary_coherence_before", "canary_poison_before"):
        vals = [float(r[key]) for r in records if r.get(key) is not None]
        if vals:
            baseline[key.replace("_before", "")] = sum(vals) / len(vals)
    thresholds, achievable = thresholds_from_records(records, target_fpr=target_fpr)
    # The CUSUM is a cross-chunk statistic, so it is calibrated on a separate CONTINUOUS benign
    # session with its own reference — not the reset-every-N reference above, which biases the
    # signal and makes the alarm ratchet on benign text. See _continuous_cusum_z.
    cusum_reference: list[float] = []
    if "log_delta_norm" in reference:
        from plastic.harness.stats import robust_z

        n_cont = min(len(records), 256)
        vals = _continuous_cusum_vals(runner, stream_iter, n=n_cont)
        ch = None
        if len(vals) >= 16:  # enough of a continuous session to calibrate a cross-chunk statistic
            zc = [z for z in (robust_z(v, vals) for v in vals) if z is not None]
            ch = calibrated_cusum_h(zc, k=runner.hcfg.cusum_k, h_min=runner.hcfg.cusum_h, target_fpr=target_fpr)
        if ch is not None:
            thresholds["cusum_h"], achievable["cusum_h"] = ch
            # the live runner standardizes the CUSUM signal against this same continuous regime
            cusum_reference = vals
        else:
            # no continuous session available (a short/finite stream): keep the config default
            # threshold and no continuous reference, so the runner falls back to its default CUSUM.
            thresholds["cusum_h"] = float(runner.hcfg.cusum_h)
    return Calibration(
        model_signature=model_signature,
        n_chunks=len(records),
        reference=reference,
        cusum_reference=cusum_reference,
        thresholds=thresholds,
        achievable_fpr=achievable,
        canary_baseline=baseline,
        target_fpr=float(target_fpr),
        created_at_unix=int(time.time()),
        fisher=fisher,
    )


def log_only(cfg: HarnessConfig) -> HarnessConfig:
    return HarnessConfig.from_dict({**cfg.to_dict(), "log_only": True})


# ---------------------------------------------------------------------- model-level entry point
def calibrate_model(
    store,
    model_id: str,
    *,
    data_dir: str | None = None,
    n_chunks: int = 512,
    fisher_chunks: int = 64,
    target_fpr: float = 0.01,
    harness_cfg: HarnessConfig | None = None,
    device: torch.device | str = "cpu",
    seed: int = 0,
    log=print,
) -> Calibration:
    """Build the canary suite (if absent), estimate the Fisher diagonal, and calibrate thresholds
    for ``model_id`` on a benign stream: held-out windows for text (``data_dir/validation.bin``),
    generated episodes for physics. Writes ``canary.json``, ``calibration.json``, ``fisher.pt``.
    """
    from plastic.data.physics import physics_batch
    from plastic.data.text import TokenWindows
    from plastic.harness.canary import CanarySuite
    from plastic.harness.fisher import estimate_fisher_diag
    from plastic.harness.transaction import TransactionRunner

    device = torch.device(device)
    cfg, model, _ = store.load_checkpoint(model_id, device)
    model_dir = store.model_dir(model_id)
    hcfg = log_only(harness_cfg or HarnessConfig(target_fpr=target_fpr))
    L = cfg.chunk
    canary_path = store.canary_path(model_id)
    g = torch.Generator().manual_seed(int(seed))

    if cfg.domain == "text":
        if not data_dir:
            raise ValueError("text calibration needs data_dir with validation.bin")
        heldout_path = os.path.join(data_dir, "validation.bin")
        if os.path.exists(canary_path):
            suite = CanarySuite.load(canary_path)
        else:
            suite = CanarySuite.default_text(heldout_path, vocab_size=cfg.vocab_size, seed=seed)
            suite.save(canary_path)
        windows = TokenWindows(heldout_path, seq_len=L * 4)
        canary_tokens = 3 * 128  # skip the region the coherence canaries were cut from

        def stream():
            # sessions of 4 chunks each, starting after the canary region, non-overlapping
            start = canary_tokens
            while True:
                if start + L * 4 + 1 > len(windows):
                    start = canary_tokens
                seq = torch.from_numpy(windows.data[start : start + L * 4].astype("int64")).tolist()
                start += L * 4
                yield seq

        def fisher_seqs():
            for _ in range(fisher_chunks):
                yield windows.sample(4, g)

        log(f"[calibrate] {model_id}: text, {n_chunks} chunks from {heldout_path}")
        fisher = estimate_fisher_diag(model, fisher_seqs(), chunk=L, n_chunks=fisher_chunks, device=device)
    else:
        if os.path.exists(canary_path):
            suite = CanarySuite.load(canary_path)
        else:
            suite = CanarySuite.default_physics(seed=seed, steps=L)
            suite.save(canary_path)

        def stream():
            while True:
                b = physics_batch(1, seq_len=L * 4, episodes_per_seq=2, mu_range=(0.02, 0.25), nonlinear=False, action_std=0.5, rng=g)
                yield (b.inputs[0], b.target_delta[0])

        def fisher_seqs():
            for _ in range(fisher_chunks):
                b = physics_batch(4, seq_len=L * 4, episodes_per_seq=2, mu_range=(0.02, 0.25), nonlinear=False, action_std=0.5, rng=g)
                yield (b.inputs, b.target_delta)

        log(f"[calibrate] {model_id}: physics, {n_chunks} chunks of generated episodes")
        fisher = estimate_fisher_diag(model, fisher_seqs(), chunk=L, n_chunks=fisher_chunks, device=device)

    # the observation runner carries the Fisher so fisher_update is observed and calibrated
    runner = TransactionRunner(model, cfg, hcfg, calibration=Calibration(fisher=fisher), suite=suite, device=device)
    cal = calibrate_from_runner(
        runner, stream(), n_chunks=n_chunks, model_signature=store.model_signature(model_id),
        target_fpr=target_fpr, fisher=fisher, reset_every=4,
    )
    cal.save(model_dir)
    store.register_model(model_id, {"calibrated_at_unix": cal.created_at_unix, "calibration_chunks": cal.n_chunks})
    log(f"[calibrate] thresholds: " + ", ".join(f"{k}={v:.4g}" for k, v in cal.thresholds.items()))
    return cal


class CalibrationIncomplete(Exception):
    """Raised when a checkpointed calibration stops at its deadline before finishing. Progress is
    persisted to the checkpoint; re-invoking ``calibrate_qwen`` with the same ``checkpoint_path``
    resumes and EXTENDS the collected references rather than restarting."""

    def __init__(self, phase: str, fit_used: int, fit_requested: int, cusum_used: int, cusum_requested: int) -> None:
        self.phase = phase
        self.fit_used, self.fit_requested = fit_used, fit_requested
        self.cusum_used, self.cusum_requested = cusum_used, cusum_requested
        super().__init__(
            f"calibration incomplete in {phase}: fit {fit_used}/{fit_requested}, cusum {cusum_used}/{cusum_requested}"
        )


class CalibrationCheckpointError(Exception):
    """Raised when a checkpoint at ``checkpoint_path`` is present but incompatible (different model,
    corpus, seed or config), unreadable, or malformed. The existing file is left UNTOUCHED so the
    incremental work it may hold is never lost to an overwrite; an explicit fresh run must point at a
    different ``checkpoint_path`` (or remove the stale one deliberately)."""


def _stable_hash(obj: Any) -> str:
    """Order-sensitive digest of a JSON-able structure (lists keep order, dict keys are sorted)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _validate_checkpoint(st: dict[str, Any], path: str, identity: str, fit_requested: int, cusum_requested: int) -> None:
    """Reject an incompatible or malformed checkpoint BEFORE any write, so a stale/corrupt file is
    preserved rather than silently overwritten or trusted. Covers identity, phase, counter types and
    ranges, requested-count consistency, and the carried state a mid-CUSUM resume requires."""
    def bad(msg: str) -> None:
        raise CalibrationCheckpointError(
            f"checkpoint at {path} {msg}; refusing to overwrite it. Use a fresh checkpoint_path "
            f"(or remove the stale file deliberately).")

    if st.get("identity") != identity:
        bad("is for a different model/corpus/seed/config (identity mismatch)")
    if st.get("phase") not in ("fit", "cusum"):
        bad(f"has an invalid phase {st.get('phase')!r}")
    if not isinstance(st.get("records"), list):
        bad("has a malformed records list")
    for key in ("fit_used", "cusum_used", "n_transactions", "fit_requested", "cusum_requested"):
        v = st.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            bad(f"has a non-integer/negative {key}={v!r}")
    if st["fit_requested"] != fit_requested or st["cusum_requested"] != cusum_requested:
        bad(f"has inconsistent requested counts (fit {st['fit_requested']} vs {fit_requested}, "
            f"cusum {st['cusum_requested']} vs {cusum_requested})")
    fit_used, cusum_used, phase = int(st["fit_used"]), int(st["cusum_used"]), st["phase"]
    if fit_used > fit_requested or cusum_used > cusum_requested:
        bad(f"has out-of-range counters (fit {fit_used}/{fit_requested}, cusum {cusum_used}/{cusum_requested})")
    if phase == "fit" and cusum_used != 0:
        bad(f"is in the fit phase but records cusum_used={cusum_used}")
    if phase == "cusum" and fit_used != fit_requested:
        # the CUSUM pass only begins once the fit pass is COMPLETE; a cusum-phase checkpoint with an
        # unfinished fit would silently skip the missing fit turns (ASTRA-092 cross-phase case)
        bad(f"is in the CUSUM phase but the fit pass is incomplete (fit {fit_used}/{fit_requested})")
    if phase == "cusum" and cusum_used > 0 and st.get("runner_state") is None:
        bad("is mid-CUSUM (cusum_used>0) but is missing the carried runner state")


def _calibration_identity(*, model_signature: str, seed: int, gen: dict[str, Any], target_fpr: float,
                          chunk: int, fit_prompts: list[str], cusum_prompts: list[str],
                          corpus_hash: str | None, harness_cfg: HarnessConfig, device: str) -> str:
    """Bind a checkpoint to the exact model, decoding, corpus (ordered), full harness config and compute
    device, so a checkpoint from any different run -- including the same run on another device, whose
    kernels need not match numerically -- is rejected rather than silently extended into an invalid mix."""
    return _stable_hash({
        "model_signature": model_signature, "seed": int(seed), "gen": gen,
        "target_fpr": float(target_fpr), "chunk": int(chunk),
        "fit_prompts": list(fit_prompts), "cusum_prompts": list(cusum_prompts),
        "corpus_hash": corpus_hash, "harness_cfg": harness_cfg.to_dict(), "device": str(device),
    })


def _ckpt_save(path: str, obj: dict[str, Any]) -> None:
    """Atomic write: a full torch.save to a temp file in the same directory, then os.replace, so a
    kill mid-write never leaves a truncated checkpoint at ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _ckpt_load(path: str, *, log=print) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        return torch.load(path, weights_only=False)
    except Exception as e:  # a truncated / unreadable checkpoint is treated as absent, loudly
        log(f"[calibrate] ignoring unreadable checkpoint {path}: {e}")
        return None


def _collect_calibration_records(runner, tok, prompts: list[str], cusum_prompts: list[str], *, seed: int,
                                 gen: dict[str, Any], deadline: float | None, checkpoint_path: str | None,
                                 identity: str, drive, log=print):
    """Run the fit (reset-between) and continuous-CUSUM passes, resuming from a matching checkpoint.

    Returns ``(records, fit_used, cont, cusum_used, cusum_ran)``. The fit pass is per-prompt
    independent (``runner.reset()`` is a genuine clean slate: fresh CUSUM accumulator, history and
    budget), so its checkpoint is just ``(records, fit_used)``. The CUSUM pass is CONTINUOUS, so its
    checkpoint also carries the runner state at the last completed turn (only once ``cusum_used>0``).
    With a ``checkpoint_path`` a deadline persists progress and raises ``CalibrationIncomplete``;
    without one it breaks and returns partial records — the existing un-checkpointed behavior."""
    fit_requested, cusum_requested = len(prompts), len(cusum_prompts)
    st = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        st = _ckpt_load(checkpoint_path, log=log)
        if st is None:  # present but unreadable/corrupt: preserve it, never overwrite blindly
            raise CalibrationCheckpointError(
                f"checkpoint at {checkpoint_path} is unreadable/corrupt; refusing to overwrite it. "
                f"Move it aside or use a fresh checkpoint_path.")
        _validate_checkpoint(st, checkpoint_path, identity, fit_requested, cusum_requested)
    if st is None:
        phase, records, fit_used, cont, cusum_used = "fit", [], 0, [], 0
    else:
        phase = str(st["phase"])
        records, fit_used = list(st["records"]), int(st["fit_used"])
        cont, cusum_used = list(st["cont"]), int(st["cusum_used"])
        # restore the transaction counter so resumed records keep the SAME global indices as an
        # uninterrupted run (reset() deliberately does not clear it, so a fresh process starts at 0)
        runner.n_transactions = int(st["n_transactions"])

    def _chat(prompt: str, salt: int) -> None:
        runner.transactions = []
        g = torch.Generator().manual_seed(int(seed) + salt)
        drive(runner, tok, prompt, max_new_tokens=gen["max_new_tokens"], temperature=gen["temperature"],
              top_k=gen["top_k"], gen=g)

    def _save(*, runner_state=None) -> None:
        if not checkpoint_path:
            return
        _ckpt_save(checkpoint_path, {
            "identity": identity, "phase": phase, "records": records, "fit_used": fit_used,
            "cont": cont, "cusum_used": cusum_used, "runner_state": runner_state,
            "n_transactions": int(runner.n_transactions),
            "fit_requested": fit_requested, "cusum_requested": cusum_requested,
        })

    def _stop(cur_phase: str, *, runner_state=None) -> bool:
        # shared deadline handling: checkpoint+raise when resumable, else signal a partial break
        if deadline is None or time.time() <= deadline:
            return False
        if checkpoint_path:
            _save(runner_state=runner_state)
            raise CalibrationIncomplete(cur_phase, fit_used, fit_requested, cusum_used, cusum_requested)
        return True

    # ---- FIT: a fresh chat per prompt (reset between); each turn's records are independent ----
    if phase == "fit":
        while fit_used < fit_requested:
            if _stop("fit"):
                break
            runner.reset()
            _chat(prompts[fit_used], fit_used)
            records.extend(runner.transactions)
            fit_used += 1
            _save()
        if fit_used == fit_requested:
            phase, cusum_used = "cusum", 0
            _save()

    cusum_ran = any(r["signals"].get("log_delta_norm") is not None for r in records)
    if not cusum_ran:
        return records, fit_used, cont, cusum_used, False

    # ---- CUSUM: one CONTINUOUS multi-turn chat (no reset); resume restores the runner state ----
    if phase == "cusum" and cusum_used > 0:
        runner.load_state_dict(st["runner_state"])  # validated present above; carries n_transactions too
    else:
        runner.reset()  # fresh CUSUM (fit just completed, or resuming the phase-transition checkpoint);
        # reset() preserves n_transactions, so the CONTINUOUS records keep counting on from the fit pass
    while cusum_used < cusum_requested:
        if _stop("cusum", runner_state=(runner.state_dict() if cusum_used > 0 else None)):
            break
        _chat(cusum_prompts[cusum_used], 1000 + cusum_used)
        cont.extend(float(r["signals"]["log_delta_norm"]) for r in runner.transactions
                    if r["signals"].get("log_delta_norm") is not None)
        cusum_used += 1
        _save(runner_state=runner.state_dict())
    return records, fit_used, cont, cusum_used, True


def calibrate_qwen(
    store,
    model_id: str,
    prompts: Iterable[str],
    *,
    target_fpr: float = 0.01,
    max_new_tokens: int = 64,
    temperature: float = 0.9,
    top_k: int = 50,
    seed: int = 0,
    cusum_prompts: Iterable[str] | None = None,
    harness_cfg: HarnessConfig | None = None,
    device: torch.device | str = "cpu",
    deadline: float | None = None,
    checkpoint_path: str | None = None,
    corpus_hash: str | None = None,
    log=print,
) -> "Calibration":
    """Calibrate Qwen harness thresholds on the ACTUAL Session.chat protocol.

    Drives real chats (log_only) over ``prompts`` — the model GENERATES each response (not
    teacher-forced) — so every turn goes through the real protocol: prompt (user source) partial
    flush, model-source generated tokens, assistant closure, final flush. Thresholds are built from
    those real operating-point records; Qwen produces only the reduced decision signals (chunk NLL,
    recurrent-state change), so only those get references/thresholds. The per-chunk reference uses a
    fresh chat per prompt (reset between); the CUSUM reference uses a separate CONTINUOUS multi-turn
    chat (no reset), the same protocol. The calibration is stamped with the ACTUAL loaded checkpoint
    digest so it installs only on the matching model (ASTRA-078).

    Prompt-chunk and generation-chunk counts are recorded SEPARATELY (never pooled). Calibrate on
    prompts DISJOINT from evaluation. This produces thresholds, not a measured intervention-rate or
    safety claim.
    """
    from plastic.backends.qwen import QwenBackend
    from plastic.config import ModelConfig
    from plastic.harness.stats import robust_z
    from plastic.harness.transaction import TransactionRunner
    from plastic.session.runner import _QwenTextIO, drive_chat_turn

    device = torch.device(device)
    rec = store.load_model_record(model_id)
    if rec.get("backend") != "qwen":
        raise ValueError(f"calibrate_qwen requires a qwen model, got backend={rec.get('backend')!r}")
    backend = QwenBackend.load(rec["checkpoint_dir"], device=device)
    cfg = ModelConfig(domain="text", chunk=int(rec.get("chunk", 8)))
    hcfg = log_only(harness_cfg or HarnessConfig(target_fpr=target_fpr))
    runner = TransactionRunner(None, cfg, hcfg, device=device, backend=backend)
    tok = _QwenTextIO(backend)
    prompts = list(prompts)
    cusum_list = list(cusum_prompts) if cusum_prompts is not None else prompts
    gen = {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_k": top_k}
    identity = _calibration_identity(
        model_signature=f"qwen:{backend.checkpoint_digest}", seed=seed, gen=gen, target_fpr=target_fpr,
        chunk=cfg.chunk, fit_prompts=prompts, cusum_prompts=cusum_list, corpus_hash=corpus_hash,
        harness_cfg=hcfg, device=str(device),
    )
    # per-chunk reference (fresh chat per prompt) and continuous-CUSUM reference (no reset), both
    # through the real generation path; with a checkpoint_path a deadline persists progress and
    # raises CalibrationIncomplete so a bounded invocation resumes rather than re-fitting.
    records, fit_used, cont, cusum_used, cusum_ran = _collect_calibration_records(
        runner, tok, prompts, cusum_list, seed=seed, gen=gen, deadline=deadline,
        checkpoint_path=checkpoint_path, identity=identity, drive=drive_chat_turn, log=log,
    )
    if not records:
        raise ValueError("no chunks observed during calibration")
    signals = [r["signals"] for r in records]
    reference = {name: [float(s[name]) for s in signals if s.get(name) is not None] for name in ROLLBACK_DECISION_SIGNALS}
    reference = {k: v for k, v in reference.items() if v}
    thresholds, achievable = thresholds_from_records(signals, target_fpr=target_fpr)

    # CUSUM threshold from the continuous reference collected above (kept separate from the reference)
    cusum_reference: list[float] = []
    if "log_delta_norm" in reference:
        if len(cont) >= 16:
            zc = [z for z in (robust_z(v, cont) for v in cont) if z is not None]
            ch = calibrated_cusum_h(zc, k=runner.hcfg.cusum_k, h_min=runner.hcfg.cusum_h, target_fpr=target_fpr)
            if ch is not None:
                thresholds["cusum_h"], achievable["cusum_h"] = ch
                cusum_reference = cont
            else:
                thresholds["cusum_h"] = float(runner.hcfg.cusum_h)
        else:
            thresholds["cusum_h"] = float(runner.hcfg.cusum_h)

    # prompt-chunk vs generation-chunk denominators kept separate (the prompt flushes before
    # generation, so no chunk mixes the two within a turn)
    prompt_chunks = sum(1 for r in records if r["sources"]["user"] > 0 and r["sources"]["model"] == 0)
    gen_chunks = sum(1 for r in records if r["sources"]["model"] > 0)
    cal = Calibration(
        model_signature=f"qwen:{backend.checkpoint_digest}",
        n_chunks=len(signals),
        reference=reference,
        cusum_reference=cusum_reference,
        thresholds=thresholds,
        achievable_fpr=achievable,
        target_fpr=float(target_fpr),
        created_at_unix=int(time.time()),
    )
    # reached only when collection COMPLETED (an interrupted checkpointed run raises before here, so a
    # partial calibration is never saved and never auto-installed by a Session before resume finishes)
    cal.save(store.model_dir(model_id))
    cusum_complete = (not cusum_ran) or (cusum_used == len(cusum_list))
    store.register_model(model_id, {
        "calibrated_at_unix": cal.created_at_unix, "calibration_chunks": cal.n_chunks,
        "calibration_prompt_chunks": prompt_chunks, "calibration_generation_chunks": gen_chunks,
        "calibration_fit_prompts_used": fit_used, "calibration_fit_prompts_requested": len(prompts),
        "calibration_fit_complete": fit_used == len(prompts),
        "calibration_cusum_prompts_used": cusum_used, "calibration_cusum_prompts_requested": len(cusum_list),
        "calibration_cusum_complete": cusum_complete,
    })
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)  # a completed calibration no longer needs its resume checkpoint
    log(f"[calibrate] qwen {model_id}: {len(signals)} chunks ({prompt_chunks} prompt, {gen_chunks} generation); "
        + "thresholds: " + ", ".join(f"{k}={v:.4g}" for k, v in thresholds.items()))
    return cal


_DECISION_KINDS = ("commit", "rollback", "scale", "project", "readonly")
_INTERVENTIONS = ("rollback", "scale", "project")


def summarize_operating_point(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-source operating-point breakdown of chat transactions, on ELIGIBLE denominators.

    Chunks are split by source: ``prompt`` (user tokens only), ``generation`` (any model-source
    token — the any-model rule), and ``unknown`` (a record missing ``sources``). A chunk with BOTH
    user and model tokens counts as generation but is also tallied as a ``mixed_source`` chat-protocol
    anomaly; an unknown decision kind and a missing source are likewise surfaced in ``anomalies``.

    Eligibility is read from each record's ``eligible`` flag — NEVER inferred from the decision label.
    Per source it reports: total ``chunks``; per-kind counts; ``eligible`` count; ``eligible_interventions``
    (rollback/scale/project among eligible chunks) and ``eligible_intervention_rate`` (``None`` when no
    eligible chunk — the promised eligible-write operating point); ``readonly`` count, ``readonly_rate``
    (among all chunks), and a ``readonly_reasons`` breakdown; ``accepted_change`` (chunks with a nonzero
    accepted state delta); and ``total_interventions`` / ``total_intervention_rate`` (explicitly a
    total-chunk burden, not the eligible operating point). A measurement over recorded decisions, not a
    calibrated-performance or safety claim.
    """
    def _blank() -> dict[str, Any]:
        return {
            "chunks": 0, "eligible": 0, "eligible_interventions": 0, "eligible_intervention_rate": None,
            "readonly": 0, "readonly_rate": None, "readonly_reasons": {}, "accepted_change": 0,
            "total_interventions": 0, "total_intervention_rate": None, **{k: 0 for k in _DECISION_KINDS},
        }

    out: dict[str, Any] = {"prompt": _blank(), "generation": _blank(), "unknown": _blank()}
    anomalies = {"mixed_source": 0, "missing_source": 0, "unknown_kind": 0, "nonfinite_accepted": 0}
    for tx in transactions:
        src_info = tx.get("sources")
        if not isinstance(src_info, dict) or ("user" not in src_info and "model" not in src_info):
            src = "unknown"
            anomalies["missing_source"] += 1
        else:
            u, m = int(src_info.get("user", 0)), int(src_info.get("model", 0))
            if u > 0 and m > 0:  # any-model rule -> generation, but record the mixed chunk as an anomaly
                anomalies["mixed_source"] += 1
                src = "generation"
            elif m > 0:
                src = "generation"
            else:
                src = "prompt"
        rec = out[src]
        rec["chunks"] += 1
        kind = (tx.get("decision") or {}).get("kind")
        if kind in _DECISION_KINDS:
            rec[kind] += 1
        else:
            anomalies["unknown_kind"] += 1
        if bool(tx.get("eligible", False)):
            rec["eligible"] += 1
            if kind in _INTERVENTIONS:
                rec["eligible_interventions"] += 1
        if kind in _INTERVENTIONS:
            rec["total_interventions"] += 1
        if kind == "readonly":  # rec["readonly"] is already the per-kind count above; just add the reason
            reason = tx.get("read_only_reason") or "learning_ineligible"
            rec["readonly_reasons"][reason] = rec["readonly_reasons"].get(reason, 0) + 1
        acc = tx.get("accepted")
        if isinstance(acc, dict):
            d = float(acc.get("delta_norm", 0) or 0)
            if not math.isfinite(d):  # a nonfinite accepted delta is not a valid retention
                anomalies["nonfinite_accepted"] += 1
            elif d > 0:
                rec["accepted_change"] += 1
    for rec in (out["prompt"], out["generation"], out["unknown"]):
        n, e = rec["chunks"], rec["eligible"]
        rec["eligible_intervention_rate"] = (rec["eligible_interventions"] / e) if e else None
        rec["readonly_rate"] = (rec["readonly"] / n) if n else None
        rec["total_intervention_rate"] = (rec["total_interventions"] / n) if n else None
    out["anomalies"] = anomalies
    return out
