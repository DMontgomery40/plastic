"""The decision function: signals in, one of commit / rollback / scale / project / readonly out.

Ordered checks. Every fired check contributes a reason string with its numbers so
the transaction log and the dashboard can show why. In ``log_only`` mode nothing
blocks; the reasons record what would have fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from plastic.harness.config import HarnessConfig
from plastic.harness.signals import STAT_SIGNALS, ChunkSignals

DecisionKind = Literal["commit", "rollback", "scale", "project", "readonly"]


@dataclass
class Decision:
    kind: DecisionKind
    reasons: list[str] = field(default_factory=list)
    scale: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "reasons": list(self.reasons), "scale": float(self.scale)}


def constraint_violations(
    delta_coherence: float | None,
    delta_poison: float | None,
    fisher_drift: float | None,
    cfg: HarnessConfig,
    thresholds: dict[str, float] | None,
) -> list[str]:
    """The finite limits an accepted state must satisfy: canary coherence may not rise and poison may not fall beyond
    their thresholds, and the Fisher drift from the anchor may not exceed its cap. ``decide`` applies them to the
    proposal; the runner applies them again to a candidate that scaling or projection changed afterwards, because a
    smaller or projected step is not guaranteed to do less damage in a nonlinear model (external review of 6cf4457,
    finding 2)."""
    thresholds = thresholds or {}
    out: list[str] = []
    thr_c = thresholds.get("canary_delta_coherence", cfg.canary_delta_max)
    if delta_coherence is not None and delta_coherence > thr_c:
        out.append(f"canary_coherence(delta={delta_coherence:+.4f}>{thr_c:.4f})")
    thr_p = thresholds.get("canary_delta_poison", cfg.poison_delta_min)
    if delta_poison is not None and delta_poison < thr_p:
        out.append(f"canary_poison(delta={delta_poison:+.4f}<{thr_p:.4f})")
    if cfg.fisher_drift_max is not None and fisher_drift is not None and fisher_drift > cfg.fisher_drift_max:
        out.append(f"fisher_drift({fisher_drift:.4g}>{cfg.fisher_drift_max:.4g})")
    return out


def decide(
    sig: ChunkSignals,
    cfg: HarnessConfig,
    thresholds: dict[str, float] | None,
    *,
    read_only: bool,
) -> Decision:
    if read_only:
        return Decision("readonly", ["session_read_only"])

    rollback: list[str] = []
    scale_reasons: list[str] = []
    scale = 1.0
    thresholds = thresholds or {}

    if cfg.enable_budget and cfg.budget_chunk is not None and sig.delta_norm > cfg.budget_chunk:
        scale = min(scale, cfg.budget_chunk / max(sig.delta_norm, 1e-12))
        scale_reasons.append(f"budget_chunk(delta_norm={sig.delta_norm:.4g}>{cfg.budget_chunk:.4g})")

    if cfg.enable_rollback:
        rollback += constraint_violations(sig.canary_delta_coherence, sig.canary_delta_poison, sig.fisher_drift, cfg, thresholds)

    if cfg.enable_stats:
        for name in STAT_SIGNALS:
            v = sig.value(name)
            if v is None:
                continue
            thr = thresholds.get(name)
            if thr is not None:
                if v > thr:
                    rollback.append(f"{name}({v:.4g}>{thr:.4g})")
                continue
            z = sig.z.get(name)
            if z is None:
                continue
            if z >= cfg.z_rollback:
                rollback.append(f"{name}_z({z:.2f}>={cfg.z_rollback})")
            elif z >= cfg.z_scale:
                scale = min(scale, cfg.scale_factor)
                scale_reasons.append(f"{name}_z({z:.2f}>={cfg.z_scale})")
        if sig.cusum_alarm:
            rollback.append("cusum_alarm")

    want_project = (
        cfg.enable_projection
        and sig.canary_alignment is not None
        and sig.canary_alignment > cfg.project_eps_cos
    )

    if cfg.log_only:
        reasons = ["log_only"] + [f"would_rollback:{r}" for r in rollback] + [f"would_scale:{r}" for r in scale_reasons]
        if want_project:
            reasons.append(f"would_project:canary_alignment({sig.canary_alignment:.3f})")
        return Decision("commit", reasons)
    if rollback:
        return Decision("rollback", rollback)
    if want_project:
        return Decision("project", [f"canary_alignment({sig.canary_alignment:.3f}>{cfg.project_eps_cos})"])
    if scale_reasons:
        return Decision("scale", scale_reasons, scale=float(scale))
    return Decision("commit", [])
