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
