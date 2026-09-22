"""Calibration: per-signal benign reference windows and thresholds, Fisher diagonal, canary baselines.

Produced once per model by running a benign stream through the transaction
runner in ``log_only`` mode. Thresholds are empirical upper quantiles at the
target false-positive rate, split across the decision signals by a union bound
so the total benign gating rate stays near ``target_fpr``.
"""

from __future__ import annotations

import json
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


def thresholds_from_records(records: list[dict[str, Any]], *, target_fpr: float) -> dict[str, float]:
    """Upper quantile per decision signal, with the false-positive budget split by union bound."""
    present = [name for name in ROLLBACK_DECISION_SIGNALS if sum(1 for r in records if r.get(name) is not None) >= 20]
    if not present:
        return {}
    per_signal_fpr = max(1e-4, float(target_fpr) / len(present))
    out: dict[str, float] = {}
    for name in present:
        vals = [float(r[name]) for r in records if r.get(name) is not None]
        out[name] = quantile_threshold(vals, per_signal_fpr)
    return out


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
    return Calibration(
        model_signature=model_signature,
        n_chunks=len(records),
        reference=reference,
        thresholds=thresholds_from_records(records, target_fpr=target_fpr),
        canary_baseline=baseline,
        target_fpr=float(target_fpr),
        created_at_unix=int(time.time()),
        fisher=fisher,
    )


def log_only(cfg: HarnessConfig) -> HarnessConfig:
    return HarnessConfig.from_dict({**cfg.to_dict(), "log_only": True})
