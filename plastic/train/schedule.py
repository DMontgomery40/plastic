"""Learning-rate schedule: linear warmup then cosine decay to a floor."""

from __future__ import annotations

import math


def lr_scale(step: int, *, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Multiplier on the base learning rate at 1-indexed ``step``."""
    if warmup > 0 and step <= warmup:
        return float(step) / float(warmup)
    if total <= warmup:
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup) / float(total - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio + (1.0 - min_ratio) * cosine)
