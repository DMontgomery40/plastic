"""Robust online statistics for signal streams.

Norm-like signals are heavy-tailed, so their z-scores are taken on the log.
Reference windows come from calibration on a benign stream and are fixed,
which keeps a slowly poisoned session from moving its own baseline.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any, Literal, Sequence

MIN_REFERENCE = 8


def log_safe(x: float) -> float:
    return math.log(max(float(x), 1e-12))


def robust_z(x: float, reference: Sequence[float]) -> float | None:
    """``(x − median) / (1.4826 · MAD)``; ``None`` with fewer than 8 reference points."""
    ref = [float(v) for v in reference if v == v]
    if len(ref) < MIN_REFERENCE:
        return None
    med = statistics.median(ref)
    mad = statistics.median(abs(v - med) for v in ref)
    return (float(x) - med) / (1.4826 * max(mad, 1e-9))


def quantile_threshold(values: Sequence[float], fpr: float, *, side: Literal["upper", "lower"] = "upper") -> float:
    """Empirical quantile at ``1 − fpr`` (upper) or ``fpr`` (lower), linear interpolation."""
    vals = sorted(float(v) for v in values if v == v)
    if len(vals) < 20:
        raise ValueError(f"need at least 20 values for a quantile threshold, got {len(vals)}")
    q = (1.0 - fpr) if side == "upper" else fpr
    q = min(1.0, max(0.0, q))
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


class Cusum:
    """Two-sided CUSUM on standardized values: alarm when either side exceeds ``h``."""

    def __init__(self, k: float = 0.5, h: float = 5.0) -> None:
        self.k = float(k)
        self.h = float(h)
        self.s_hi = 0.0
        self.s_lo = 0.0
        self.alarms = 0

    def update(self, z: float) -> bool:
        if z != z:  # NaN: ignore
            return False
        self.s_hi = max(0.0, self.s_hi + float(z) - self.k)
        self.s_lo = max(0.0, self.s_lo - float(z) - self.k)
        alarm = False
        if self.s_hi > self.h:
            alarm = True
            self.s_hi = 0.0
        if self.s_lo > self.h:
            alarm = True
            self.s_lo = 0.0
        if alarm:
            self.alarms += 1
        return alarm

    def state(self) -> dict[str, Any]:
        return {"k": self.k, "h": self.h, "s_hi": self.s_hi, "s_lo": self.s_lo, "alarms": self.alarms}

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> "Cusum":
        c = cls(float(d.get("k", 0.5)), float(d.get("h", 5.0)))
        c.s_hi = float(d.get("s_hi", 0.0))
        c.s_lo = float(d.get("s_lo", 0.0))
        c.alarms = int(d.get("alarms", 0))
        return c


class SignalHistory:
    def __init__(self, maxlen: int = 64, values: Sequence[float] | None = None) -> None:
        self.maxlen = int(maxlen)
        self._d: deque[float] = deque(values or [], maxlen=self.maxlen)

    def append(self, x: float) -> None:
        self._d.append(float(x))

    def values(self) -> list[float]:
        return list(self._d)

    def __len__(self) -> int:
        return len(self._d)
