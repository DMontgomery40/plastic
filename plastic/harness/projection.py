"""Half-space projection of a multi-layer state delta against a canary-loss gradient.

With ``g = ∂ score / ∂ S`` (flattened over layers and heads) and ``Δ = S_working − S_committed``,
the first-order change of the canary score is ``⟨g, Δ⟩``. If it exceeds the allowed slack
``max(eps_dot, eps_cos · ‖g‖ · ‖Δ‖)`` the delta is projected onto the boundary:

    Δ ← Δ − ((⟨g, Δ⟩ − allowed) / ‖g‖²) g

This is a local, first-order statement (A-GEM style); the harness still checks the
finite canary delta after the fact and caps the complete state change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ProjectionStats:
    dot_before: float
    dot_after: float
    allowed: float
    removed_ratio: float
    applied: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "dot_before": self.dot_before,
            "dot_after": self.dot_after,
            "allowed": self.allowed,
            "removed_ratio": self.removed_ratio,
            "applied": self.applied,
        }


def flatten_all(ts: list[Tensor]) -> Tensor:
    return torch.cat([t.reshape(-1).float() for t in ts])


def unflatten_like(flat: Tensor, like: list[Tensor]) -> list[Tensor]:
    out, i = [], 0
    for t in like:
        n = t.numel()
        out.append(flat[i : i + n].reshape(t.shape).to(t.dtype))
        i += n
    return out


def project_delta(
    deltas: list[Tensor],
    grads: list[Tensor],
    *,
    eps_dot: float = 0.0,
    eps_cos: float = 0.0,
    tiny: float = 1e-12,
) -> tuple[list[Tensor], ProjectionStats]:
    if len(deltas) != len(grads):
        raise ValueError("deltas and grads must have the same number of layers")
    d = flatten_all(deltas)
    g = flatten_all(grads)
    gn = float(g.norm())
    dn = float(d.norm())
    dot = float((g * d).sum())
    allowed = max(float(eps_dot), float(eps_cos) * gn * dn)
    if gn < tiny or dot <= allowed:
        return [t.clone() for t in deltas], ProjectionStats(dot, dot, allowed, 0.0, False)
    # project along the unit direction so a rescaled gradient gives the same geometric result
    unit = g / gn
    d_new = d - ((dot - allowed) / gn) * unit
    dot_after = float((g * d_new).sum())
    removed = float((d - d_new).norm() / max(dn, tiny))
    return unflatten_like(d_new, deltas), ProjectionStats(dot, dot_after, allowed, min(1.0, removed), True)
