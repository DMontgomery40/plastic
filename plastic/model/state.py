"""Session state: the fast, per-session tensors carried between forward passes.

Per layer: ``h`` (activation state of the selective recurrence, (B, D)), ``S``
(fast-weight memory, (B, H, d_h, d_h)), ``M`` (inner-loop momentum, only for
``rule="chunk"``), and the two short-convolution buffers ``conv_ssm`` and
``conv_mem`` holding the last ``conv_kernel - 1`` inputs of each branch
((B, K-1, D), absent when the kernel is 1). ``pos`` counts tokens or steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from plastic.config import ModelConfig


def _opt(t: Tensor | None, fn) -> Tensor | None:
    return None if t is None else fn(t)


@dataclass
class LayerState:
    h: Tensor
    S: Tensor
    M: Tensor | None = None
    conv_ssm: Tensor | None = None
    conv_mem: Tensor | None = None

    def clone(self) -> "LayerState":
        return LayerState(
            self.h.clone(),
            self.S.clone(),
            _opt(self.M, lambda t: t.clone()),
            _opt(self.conv_ssm, lambda t: t.clone()),
            _opt(self.conv_mem, lambda t: t.clone()),
        )

    def detach(self) -> "LayerState":
        return LayerState(
            self.h.detach(),
            self.S.detach(),
            _opt(self.M, lambda t: t.detach()),
            _opt(self.conv_ssm, lambda t: t.detach()),
            _opt(self.conv_mem, lambda t: t.detach()),
        )

    def to(self, device: torch.device | str) -> "LayerState":
        return LayerState(
            self.h.to(device),
            self.S.to(device),
            _opt(self.M, lambda t: t.to(device)),
            _opt(self.conv_ssm, lambda t: t.to(device)),
            _opt(self.conv_mem, lambda t: t.to(device)),
        )


_OPTIONAL = ("M", "conv_ssm", "conv_mem")


@dataclass
class SessionState:
    layers: list[LayerState] = field(default_factory=list)
    pos: int = 0

    @classmethod
    def zeros(
        cls,
        cfg: ModelConfig,
        batch: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SessionState":
        dh = cfg.head_dim
        K = cfg.conv_kernel
        layers = []
        for _ in range(cfg.n_layers):
            h = torch.zeros(batch, cfg.d_model, device=device, dtype=dtype)
            S = torch.zeros(batch, cfg.n_heads, dh, dh, device=device, dtype=dtype)
            M = torch.zeros_like(S) if cfg.rule == "chunk" else None
            conv_ssm = torch.zeros(batch, K - 1, cfg.d_model, device=device, dtype=dtype) if K > 1 else None
            conv_mem = torch.zeros(batch, K - 1, cfg.d_model, device=device, dtype=dtype) if K > 1 else None
            layers.append(LayerState(h, S, M, conv_ssm, conv_mem))
        return cls(layers=layers, pos=0)

    @property
    def batch(self) -> int:
        return int(self.layers[0].h.shape[0]) if self.layers else 0

    @property
    def device(self) -> torch.device:
        return self.layers[0].h.device

    def clone(self) -> "SessionState":
        return SessionState([layer.clone() for layer in self.layers], self.pos)

    def detach(self) -> "SessionState":
        return SessionState([layer.detach() for layer in self.layers], self.pos)

    def to(self, device: torch.device | str) -> "SessionState":
        return SessionState([layer.to(device) for layer in self.layers], self.pos)

    def state_dict(self) -> dict[str, Tensor | int]:
        out: dict[str, Tensor | int] = {"pos": int(self.pos)}
        for i, layer in enumerate(self.layers):
            out[f"layer{i}.h"] = layer.h.detach().cpu()
            out[f"layer{i}.S"] = layer.S.detach().cpu()
            for name in _OPTIONAL:
                t = getattr(layer, name)
                if t is not None:
                    out[f"layer{i}.{name}"] = t.detach().cpu()
        return out

    @classmethod
    def from_state_dict(cls, d: dict[str, Tensor | int]) -> "SessionState":
        layers: list[LayerState] = []
        i = 0
        while f"layer{i}.h" in d:
            h = d[f"layer{i}.h"]
            S = d[f"layer{i}.S"]
            assert isinstance(h, Tensor) and isinstance(S, Tensor)
            extras = {}
            for name in _OPTIONAL:
                t = d.get(f"layer{i}.{name}")
                extras[name] = None if t is None else t.clone()  # type: ignore[union-attr]
            layers.append(LayerState(h.clone(), S.clone(), **extras))
            i += 1
        pos = int(d.get("pos", 0))  # type: ignore[arg-type]
        return cls(layers=layers, pos=pos)

    def s_delta(self, other: "SessionState") -> list[Tensor]:
        """Per-layer ``S - other.S``."""
        if len(self.layers) != len(other.layers):
            raise ValueError("layer count mismatch")
        return [a.S - b.S for a, b in zip(self.layers, other.layers)]

    def norms(self) -> dict[str, list[float] | float]:
        s_norm = [float(layer.S.flatten(1).norm(dim=-1).mean()) for layer in self.layers]
        h_norm = [float(layer.h.norm(dim=-1).mean()) for layer in self.layers]
        return {
            "s_norm": s_norm,
            "h_norm": h_norm,
            "s_norm_total": float(sum(x * x for x in s_norm) ** 0.5),
            "h_norm_total": float(sum(x * x for x in h_norm) ** 0.5),
        }
