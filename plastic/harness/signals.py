"""Per-chunk signals, all computed from the model or from the session's own statistics.

The signals used for decisions (``STAT_SIGNALS``) are compared against calibrated
thresholds when a calibration exists, and otherwise against robust z-scores over
the session's own history. ``compression_ratio`` is computed for display only.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from plastic.harness.stats import SignalHistory, log_safe, robust_z
from plastic.model.memory import MemorySignals

# signals that participate in threshold / z decisions (values as stored on ChunkSignals)
STAT_SIGNALS: tuple[str, ...] = ("chunk_loss", "surprise_mean", "log_delta_norm", "log_write_norm", "fisher_update")


@dataclass
class ChunkSignals:
    """Signals of one chunk, measured on the PROVISIONAL (post-forward, pre-decision) state.

    These describe the update the policy saw, not what was committed: for a rollback the
    ``delta_norm`` and ``canary_delta_*`` here are the proposed change, while the accepted
    change is zero. The transaction record carries the accepted metrics separately.
    """

    pos_start: int
    pos_end: int
    n_tokens: int
    chunk_loss: float
    # the memory-derived signals are None on a backend whose kernel does not expose them (Qwen has no
    # per-token surprise/write-norm/decay); they are carried as None end-to-end, never zero or NaN.
    surprise_mean: float | None
    surprise_max: float | None
    beta_mean: float | None
    alpha_mean: float | None
    write_norm_sum: float | None
    delta_norm: float
    delta_norm_per_layer: list[float] = field(default_factory=list)
    fisher_update: float | None = None
    fisher_drift: float | None = None
    canary_coherence_before: float | None = None
    canary_coherence_after: float | None = None
    canary_poison_before: float | None = None
    canary_poison_after: float | None = None
    canary_delta_coherence: float | None = None
    canary_delta_poison: float | None = None
    canary_alignment: float | None = None
    compression_ratio: float | None = None
    z: dict[str, float | None] = field(default_factory=dict)
    cusum_alarm: bool = False
    budget_used: float = 0.0
    budget_remaining: float | None = None

    @property
    def log_delta_norm(self) -> float:
        return log_safe(self.delta_norm)

    @property
    def log_write_norm(self) -> float | None:
        return None if self.write_norm_sum is None else log_safe(self.write_norm_sum)

    def value(self, name: str) -> float | None:
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["log_delta_norm"] = self.log_delta_norm
        d["log_write_norm"] = self.log_write_norm
        return d


def summarize_memory_signals(signals: Sequence[MemorySignals]) -> dict[str, float]:
    """Aggregate per-token memory signals of one chunk over layers, heads, and tokens."""
    err = torch.cat([s.err.detach().float().flatten() for s in signals])
    beta = torch.cat([s.beta.detach().float().flatten() for s in signals])
    alpha = torch.cat([s.alpha.detach().float().flatten() for s in signals])
    write = torch.cat([s.write_norm.detach().float().flatten() for s in signals])
    return {
        "surprise_mean": float(err.mean()),
        "surprise_max": float(err.max()),
        "beta_mean": float(beta.mean()),
        "alpha_mean": float(alpha.mean()),
        "write_norm_sum": float(write.sum()),
    }


def delta_norms(deltas: Sequence[Tensor]) -> tuple[float, list[float]]:
    per_layer = [float(d.float().norm()) for d in deltas]
    return float(sum(x * x for x in per_layer) ** 0.5), per_layer


def cosine(a: Sequence[Tensor], b: Sequence[Tensor]) -> float | None:
    fa = torch.cat([x.reshape(-1).float() for x in a])
    fb = torch.cat([x.reshape(-1).float() for x in b])
    na, nb = float(fa.norm()), float(fb.norm())
    if na < 1e-12 or nb < 1e-12:
        return None
    return float((fa * fb).sum() / (na * nb))


def compression_ratio(ids: Sequence[int] | None) -> float | None:
    """zlib ratio of the chunk's token ids as bytes; display only, never a decision input."""
    if not ids:
        return None
    arr = np.asarray(list(ids), dtype=np.int64)
    # token ids can exceed 16 bits — Qwen's native vocab is ~248k, whose chat special tokens overflow
    # uint16 — so pick the byte width that fits. plastic vocabs (<= 8192) stay uint16, preserving the
    # existing display values.
    dtype = "<u2" if int(arr.max()) < 65536 else "<u4"
    raw = arr.astype(dtype).tobytes()
    if len(raw) < 32:
        return None
    return len(zlib.compress(raw, 9)) / len(raw)


def compute_z(
    sig: ChunkSignals,
    *,
    reference: dict[str, Sequence[float]] | None,
    history: dict[str, SignalHistory],
) -> dict[str, float | None]:
    """Robust z per STAT_SIGNAL against the fixed calibration reference, else the session history."""
    out: dict[str, float | None] = {}
    for name in STAT_SIGNALS:
        v = sig.value(name)
        if v is None:
            out[name] = None
            continue
        ref = reference.get(name) if reference else None
        if ref is None or len(ref) < 8:
            ref = history[name].values() if name in history else []
        out[name] = robust_z(float(v), ref)
    return out
