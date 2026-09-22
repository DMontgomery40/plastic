"""Muon for the 2-D matrices inside the blocks, AdamW for everything else.

``torch.optim.Muon`` accepts only 2-D parameters, so embeddings, heads,
norms, biases, the decay parameter λ, and the gate projections go to AdamW.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


def split_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return (matrix_params, other_params) with no duplicates."""
    matrix: list[nn.Parameter] = []
    other: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        in_blocks = ".blocks." in name or name.startswith("blocks.")
        is_gate = ".W_beta." in name or ".W_alpha." in name
        if p.ndim == 2 and in_blocks and not is_gate:
            matrix.append(p)
        else:
            other.append(p)
    return matrix, other


class SplitOptimizer:
    """Thin wrapper presenting one optimizer interface over Muon + AdamW."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("need at least one optimizer")
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [g for opt in self.optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        states = d["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError("optimizer count mismatch")
        for opt, st in zip(self.optimizers, states):
            opt.load_state_dict(st)

    def set_lr_scale(self, scale: float) -> None:
        """Multiply every group's base learning rate (stored as ``base_lr``) by ``scale``."""
        for g in self.param_groups:
            g["lr"] = g["base_lr"] * float(scale)


def build_optimizer(
    model: nn.Module,
    *,
    lr_matrix: float = 2e-2,
    lr_other: float = 1e-3,
    weight_decay: float = 0.1,
    use_muon: bool = True,
    betas: tuple[float, float] = (0.9, 0.95),
) -> SplitOptimizer:
    matrix, other = split_parameters(model)
    opts: list[torch.optim.Optimizer] = []
    if use_muon and matrix:
        muon = torch.optim.Muon(
            matrix,
            lr=float(lr_matrix),
            weight_decay=float(weight_decay),
            momentum=0.95,
            nesterov=True,
            adjust_lr_fn="match_rms_adamw",
        )
        opts.append(muon)
        adam_params: Iterable[nn.Parameter] = other
    else:
        adam_params = matrix + other
    if list(adam_params):
        opts.append(
            torch.optim.AdamW(
                list(adam_params),
                lr=float(lr_other),
                betas=betas,
                weight_decay=float(weight_decay),
            )
        )
    for opt in opts:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    return SplitOptimizer(opts)
