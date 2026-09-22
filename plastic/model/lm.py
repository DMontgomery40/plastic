"""The two domain models on a shared core.

``PlasticCore`` stacks blocks and threads ``SessionState`` through them.
``PlasticLM`` adds a tied token embedding and head for text; ``PlasticDynamics``
adds a linear embedding of ``[obs, action, reset_flag]`` and a delta-observation
head for the hidden-mu physics domain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from plastic.config import ModelConfig
from plastic.model.block import PlasticBlock
from plastic.model.memory import MemorySignals, Mode
from plastic.model.norm import RMSNorm
from plastic.model.state import SessionState


class PlasticCore(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([PlasticBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)

    def forward(
        self,
        x: Tensor,
        state: SessionState,
        *,
        mode: Mode = "chunk",
        beta_scale: float = 1.0,
        freeze: bool = False,
    ) -> tuple[Tensor, SessionState, list[MemorySignals]]:
        if len(state.layers) != len(self.blocks):
            raise ValueError(f"state has {len(state.layers)} layers, model has {len(self.blocks)}")
        new_layers = []
        signals = []
        for block, layer_state in zip(self.blocks, state.layers):
            x, new_state, sig = block(x, layer_state, mode=mode, beta_scale=beta_scale, freeze=freeze)
            new_layers.append(new_state)
            signals.append(sig)
        y = self.norm_f(x)
        return y, SessionState(layers=new_layers, pos=state.pos + int(x.shape[1])), signals


class _Base(nn.Module):
    cfg: ModelConfig

    def num_params(self) -> int:
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def init_state(self, batch: int, device: torch.device | str | None = None) -> SessionState:
        if device is None:
            device = next(self.parameters()).device
        return SessionState.zeros(self.cfg, batch=batch, device=device)


class PlasticLM(_Base):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.domain != "text":
            raise ValueError("PlasticLM requires domain='text'")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.core = PlasticCore(cfg)
        if cfg.tie_embeddings:
            self.head = None
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            nn.init.normal_(self.head.weight, std=0.02)

    def logits_from_hidden(self, y: Tensor) -> Tensor:
        if self.head is None:
            return F.linear(y, self.embed.weight)
        return self.head(y)

    def forward(
        self,
        tokens: Tensor,
        state: SessionState | None = None,
        *,
        mode: Mode = "chunk",
        beta_scale: float = 1.0,
        freeze: bool = False,
    ) -> tuple[Tensor, SessionState, list[MemorySignals]]:
        if state is None:
            state = self.init_state(int(tokens.shape[0]), tokens.device)
        x = self.embed(tokens)
        y, new_state, signals = self.core(x, state, mode=mode, beta_scale=beta_scale, freeze=freeze)
        return self.logits_from_hidden(y), new_state, signals

    def loss(
        self,
        tokens: Tensor,
        state: SessionState | None = None,
        *,
        beta_scale: float = 1.0,
        freeze: bool = False,
    ) -> Tensor:
        logits, _, _ = self(tokens, state, mode="chunk", beta_scale=beta_scale, freeze=freeze)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )


class PlasticDynamics(_Base):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.domain != "physics":
            raise ValueError("PlasticDynamics requires domain='physics'")
        self.cfg = cfg
        self.embed = nn.Linear(cfg.input_dim, cfg.d_model)
        self.core = PlasticCore(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.obs_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.embed.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        inputs: Tensor,
        state: SessionState | None = None,
        *,
        mode: Mode = "chunk",
        beta_scale: float = 1.0,
        freeze: bool = False,
    ) -> tuple[Tensor, SessionState, list[MemorySignals]]:
        if state is None:
            state = self.init_state(int(inputs.shape[0]), inputs.device)
        x = self.embed(inputs)
        y, new_state, signals = self.core(x, state, mode=mode, beta_scale=beta_scale, freeze=freeze)
        return self.head(y), new_state, signals

    def loss(
        self,
        inputs: Tensor,
        target_delta: Tensor,
        state: SessionState | None = None,
        *,
        beta_scale: float = 1.0,
        freeze: bool = False,
    ) -> Tensor:
        pred, _, _ = self(inputs, state, mode="chunk", beta_scale=beta_scale, freeze=freeze)
        return F.mse_loss(pred, target_delta)


def build_model(cfg: ModelConfig) -> PlasticLM | PlasticDynamics:
    if cfg.domain == "text":
        return PlasticLM(cfg)
    if cfg.domain == "physics":
        return PlasticDynamics(cfg)
    raise ValueError(f"unknown domain {cfg.domain!r}")
