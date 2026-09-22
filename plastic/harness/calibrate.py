"""Calibration: per-signal benign reference windows and thresholds, Fisher diagonal, canary baselines.

Produced once per model by running a benign stream through the transaction
runner in ``log_only`` mode. Thresholds are empirical upper quantiles at the
target false-positive rate, split across the decision signals by a union bound
so the total benign gating rate stays near ``target_fpr``.
"""

from __future__ import annotations

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


def calibrated_cusum_h(
    records: list[dict[str, Any]], reference: dict[str, list[float]], *, k: float, h_min: float, target_fpr: float
) -> tuple[float, float] | None:
    """Calibrate the CUSUM alarm threshold as a benign run-length, not a walk endpoint.

    The two-sided CUSUM resets to zero every time it alarms, so ``h`` controls the *rate*
    at which the in-control stream alarms — not the peak it reaches. We replay the benign
    reference z-sequence (of ``log_delta_norm``) through the real ``Cusum(k, h)`` for a grid
    of candidate ``h``, count alarms, and return the smallest ``h`` whose benign per-chunk
    alarm rate is at or below ``target_fpr``, together with that achieved rate. This is the
    same order-statistic discipline the per-chunk thresholds use, so ``cusum_h`` means what
    the rest of the calibration means. (The old heuristic took ``1.25 * peak`` of a
    never-resetting walk, which is not a quantile of anything and let a benign continuous
    stream alarm within tens of chunks.)
    """
    from plastic.harness.stats import Cusum, robust_z

    ref = reference.get("log_delta_norm")
    if not ref:
        return None
    zs: list[float] = []
    for r in records:
        v = r.get("log_delta_norm")
        if v is None:
            continue
        z = robust_z(float(v), ref)
        if z is not None:
            zs.append(float(z))
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
        if h < float(h_min):
            continue
        rate = alarm_rate(h)
        if rate <= target_fpr:
            best_h, best_rate = h, rate
            break
    return best_h, best_rate


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
    for item in stream:
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
    ch = calibrated_cusum_h(records, reference, k=runner.hcfg.cusum_k, h_min=runner.hcfg.cusum_h, target_fpr=target_fpr)
    if ch is not None:
        thresholds["cusum_h"], achievable["cusum_h"] = ch
    return Calibration(
        model_signature=model_signature,
        n_chunks=len(records),
        reference=reference,
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
