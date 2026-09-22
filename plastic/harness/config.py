"""Harness configuration: what is measured, what fires, and how hard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class HarnessConfig:
    enable_rollback: bool = True
    enable_projection: bool = True
    enable_budget: bool = True
    enable_stats: bool = True
    log_only: bool = False

    # canary triggers (uncalibrated fallbacks; calibrated thresholds override)
    canary_delta_max: float = 0.5
    poison_delta_min: float = -0.5

    # robust z fallbacks when no calibrated threshold exists for a signal
    z_rollback: float = 6.0
    z_scale: float = 3.0
    scale_factor: float = 0.25

    # budgets on the complete state delta (Frobenius norm over all layers)
    budget_chunk: float | None = None
    budget_session: float | None = None
    fisher_drift_max: float | None = None

    # projection against the coherence-canary gradient
    project_eps_dot: float = 0.0
    project_eps_cos: float = 0.02
    project_max_removed: float = 0.5

    # sequential change detection on log update norm
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    freeze_on_alarm: bool = True

    history_window: int = 64
    target_fpr: float = 0.01
    learn_from_generation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HarnessConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown HarnessConfig fields: {sorted(unknown)}")
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "HarnessConfig":
        return cls.from_dict(json.loads(s))
