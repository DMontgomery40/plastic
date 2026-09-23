"""The coordinate-recurrence candidate under the learning contract.

``CoordinateLearner`` wraps ``CoordinateDynamics`` (plastic/model/coordinate.py) as a contract
``Learner``. Two modes for now, mirroring the baselines: ``frozen`` (no lasting update) and
``continued`` (Adam steps on the stream's loss; because the outer loss differentiates through
the inner fast-weight steps, this is meta-training on the stream). The slow update rule of
docs/superpowers/specs/2026-09-23-slow-update-rule.md will be a third mode.

Measurement semantics, per the T1 interface requirements:
- every ``step_mse`` call starts from ``init_state``: the candidate's own W0/theta0, canonical
  carry r = 0, no pending chunk, so nothing transient survives between calls;
- ``adapt=True`` runs the unguarded learner (``commit=True``: proposals are committed at every
  chunk boundary);
- ``adapt=False`` is ``freeze=True``: every fast parameter, coordinates AND decay, stays at its
  initial value while the activation carry advances. This is a different intervention from the
  delta baseline's ``beta_scale=0`` (writes disabled, decay active), and it keeps that label.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor

from plastic.data.mechanisms import MechanismBatch


class CoordinateLearner:
    MODES = ("frozen", "continued")

    def __init__(self, model: Any, *, mode: str = "frozen", lr: float = 1e-3, steps: int = 10, device: torch.device | None = None) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.model = model
        self.mode = mode
        self.lr = float(lr)
        self.steps = int(steps)
        self.device = device or next(model.parameters()).device
        self.last_report: Any = None

    def update_period(self) -> int:
        """Fast parameters change only at chunk boundaries, so a scored episode must be longer
        than one chunk for adaptation to be measurable at all."""
        return int(self.model.cfg.chunk)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def snapshot_slow(self) -> Any:
        return {"params": copy.deepcopy(self.model.state_dict())}

    def restore_slow(self, snapshot: Any) -> None:
        self.model.load_state_dict(snapshot["params"])

    def consume(self, stream: MechanismBatch) -> dict[str, Any]:
        if self.mode == "frozen":
            return {"accepted": None, "mode": self.mode}
        x = stream.inputs.to(self.device)
        y = stream.target_delta.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        losses: list[float] = []
        self.model.train()
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            loss = self.model.loss(x, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        self.model.eval()
        return {"accepted": True, "mode": self.mode, "optimizer": "adam", "lr": self.lr, "loss_first": losses[0], "loss_last": losses[-1], "steps": self.steps, "meta_gradient": self.model.cfg.meta_gradient}

    def fast_signals_summary(self) -> dict[str, Any] | None:
        """Means over the boundaries of the last ``step_mse`` call: what the fast path proposed
        (inner loss before and after the proposed step on the observed chunk, the size of the
        proposed change per layer, the effective step) and the amplitude margin. A support
        diagnostic, never a score of adaptation; adaptation is scored on later targets."""
        rep = self.last_report
        if rep is None or not rep.chunks:
            return None
        chunks = rep.chunks
        stepped = [c for c in chunks if c.stepped]

        def _m(vals: list[Tensor]) -> float | None:
            return float(torch.stack([v.float().mean() for v in vals]).mean()) if vals else None

        return {
            "boundaries": len(chunks),
            "stepped_fraction": len(stepped) / len(chunks),
            "inner_loss_before": _m([c.inner_loss_before for c in stepped]),
            "inner_loss_after": _m([c.inner_loss_after for c in stepped]),
            "dW_norm_by_layer": [float(x) for x in torch.stack([c.dW_norm.float().mean(0) for c in stepped]).mean(0)] if stepped else None,
            "dtheta_norm_by_layer": [float(x) for x in torch.stack([c.dtheta_norm.float().mean(0) for c in stepped]).mean(0)] if stepped else None,
            "eta_by_layer": [float(x) for x in stepped[0].eta] if stepped else None,
            "r_peak_inf": _m([c.r_peak_inf for c in chunks]),
            "bound_peak": _m([c.bound_peak for c in chunks]),
        }

    @torch.no_grad()
    def step_mse(self, batch: MechanismBatch, *, adapt: bool) -> Tensor:
        x = batch.inputs.to(self.device)
        y = batch.target_delta.to(self.device)
        self.model.eval()
        state = self.model.init_state(int(x.shape[0]))
        if adapt:
            pred, _, report = self.model(x, state, mode="chunk", target_delta=y, commit=True)
        else:
            pred, _, report = self.model(x, state, mode="chunk", target_delta=y, freeze=True)
        self.last_report = report
        return (pred - y).pow(2).mean(-1).cpu()
