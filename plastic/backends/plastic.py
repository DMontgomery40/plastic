"""The native ``plastic`` model as a ``Backend``.

A thin, faithful wrapper over today's ``PlasticLM`` / ``PlasticDynamics`` model, its
``SessionState``, and the canary functions — delegating, not reimplementing. It exists so the
transaction runner can be refactored to drive a ``Backend`` (this or ``QwenBackend``) with no
change in behaviour: every method here is exactly what the runner does today inline. Plastic
provides the full signal set; generation is frozen by default (the harness's
``learn_from_generation`` may allow it), which is the opposite default from a pretrained backend.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from plastic.harness.canary import CanarySuite, canary_gradient, score_suite
from plastic.harness.signals import STAT_SIGNALS
from plastic.model.memory import MemorySignals
from plastic.model.state import SessionState


class PlasticBackend:
    def __init__(self, model, model_cfg, *, device: torch.device) -> None:
        self.model = model
        self.cfg = model_cfg
        self.device = device
        self.domain = model_cfg.domain

    # ---- identity / eligibility ----
    def signal_names(self) -> tuple[str, ...]:
        return STAT_SIGNALS  # the native memory exposes every decision signal

    def writes_for_source(self, source: str) -> bool:
        # generated tokens are frozen read-only by default; the harness's learn_from_generation
        # may override. user tokens write. (Qwen differs: its generation writes.)
        return source != "model"

    # ---- state lifecycle ----
    def init_state(self) -> SessionState:
        return self.model.init_state(1, self.device)

    def clone(self, state: SessionState) -> SessionState:
        return state.clone()

    def position(self, state: SessionState) -> int:
        return int(state.pos)

    def state_delta(self, a: SessionState, b: SessionState) -> list[Tensor]:
        return a.s_delta(b)

    def is_finite(self, state: SessionState) -> bool:
        for l in state.layers:
            tensors = [l.h, l.S, l.M, l.conv_ssm, l.conv_mem]
            if l.chunk is not None:
                tensors += [l.chunk.A, l.chunk.Bv, l.chunk.alpha_sum]
            for t in tensors:
                if t is not None and not bool(torch.isfinite(t).all()):
                    return False
        return True

    def state_dict(self, state: SessionState) -> dict[str, Any]:
        return state.state_dict()

    def load_state_dict(self, data: dict[str, Any]) -> SessionState:
        return SessionState.from_state_dict(data).to(self.device)

    # ---- forward ----
    def forward(self, items: list[Any], state: SessionState, *, freeze: bool, beta_scale: float) -> tuple[Tensor, SessionState, list[MemorySignals]]:
        if self.domain == "text":
            x = torch.tensor([items], dtype=torch.long, device=self.device)
        else:
            x = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in items]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out, new_state, signals = self.model(x, state, mode="chunk", freeze=freeze, beta_scale=beta_scale)
        return out[0], new_state, [s.detach() for s in signals]

    # ---- canaries ----
    def score_suite(self, state: SessionState, suite: CanarySuite) -> dict[str, float]:
        return score_suite(self.model, state, suite, device=self.device)

    def canary_gradient(self, state: SessionState, suite: CanarySuite) -> list[Tensor]:
        return canary_gradient(self.model, state, suite, device=self.device)

    def apply_projected(self, working: SessionState, committed: SessionState, projected: list[Tensor]) -> None:
        for layer, base, d in zip(working.layers, committed.layers, projected):
            layer.S = base.S + d.to(base.S.device)
