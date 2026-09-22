"""Session state: the fast, per-session tensors carried between forward passes.

Per layer: ``h`` (activation state of the selective recurrence, (B, D)), ``S``
(fast-weight memory, (B, H, d_h, d_h)), ``M`` (inner-loop momentum, only for
``rule="chunk"``), the two short-convolution buffers ``conv_ssm`` and
``conv_mem`` holding the last ``conv_kernel - 1`` inputs of each branch
((B, K-1, D), absent when the kernel is 1), and for ``rule="chunk"`` the
sufficient statistics of the pending (unfinished) chunk so that token-by-token
processing equals whole-chunk processing. ``pos`` counts tokens or steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from plastic.config import ModelConfig


def _opt(t: Tensor | None, fn) -> Tensor | None:
    return None if t is None else fn(t)


@dataclass
class ChunkStats:
    """Pending-chunk statistics for the chunk rule: ``A = Σ β k kᵀ``, ``Bv = Σ β k vᵀ``,
    ``alpha_sum = Σ α`` (per head), and ``count`` tokens accumulated so far."""

    A: Tensor  # (B, H, d, d)
    Bv: Tensor  # (B, H, d, d)
    alpha_sum: Tensor  # (B, H)
    count: int = 0

    @classmethod
    def zeros(cls, batch: int, heads: int, dh: int, device, dtype) -> "ChunkStats":
        return cls(
            torch.zeros(batch, heads, dh, dh, device=device, dtype=dtype),
            torch.zeros(batch, heads, dh, dh, device=device, dtype=dtype),
            torch.zeros(batch, heads, device=device, dtype=dtype),
            0,
        )

    def map(self, fn) -> "ChunkStats":
        return ChunkStats(fn(self.A), fn(self.Bv), fn(self.alpha_sum), self.count)


@dataclass
class LayerState:
    h: Tensor
    S: Tensor
    M: Tensor | None = None
    conv_ssm: Tensor | None = None
    conv_mem: Tensor | None = None
    chunk: ChunkStats | None = None

    def _map(self, fn) -> "LayerState":
        return LayerState(
            fn(self.h),
            fn(self.S),
            _opt(self.M, fn),
            _opt(self.conv_ssm, fn),
            _opt(self.conv_mem, fn),
            None if self.chunk is None else self.chunk.map(fn),
        )

    def clone(self) -> "LayerState":
        return self._map(lambda t: t.clone())

    def detach(self) -> "LayerState":
        return self._map(lambda t: t.detach())

    def to(self, device: torch.device | str) -> "LayerState":
        return self._map(lambda t: t.to(device))


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
            chunk = ChunkStats.zeros(batch, cfg.n_heads, dh, device, dtype) if cfg.rule == "chunk" else None
            layers.append(LayerState(h, S, M, conv_ssm, conv_mem, chunk))
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
            if layer.chunk is not None:
                out[f"layer{i}.chunk.A"] = layer.chunk.A.detach().cpu()
                out[f"layer{i}.chunk.Bv"] = layer.chunk.Bv.detach().cpu()
                out[f"layer{i}.chunk.alpha_sum"] = layer.chunk.alpha_sum.detach().cpu()
                out[f"layer{i}.chunk.count"] = int(layer.chunk.count)
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
            chunk = None
            if f"layer{i}.chunk.A" in d:
                chunk = ChunkStats(
                    d[f"layer{i}.chunk.A"].clone(),  # type: ignore[union-attr]
                    d[f"layer{i}.chunk.Bv"].clone(),  # type: ignore[union-attr]
                    d[f"layer{i}.chunk.alpha_sum"].clone(),  # type: ignore[union-attr]
                    int(d.get(f"layer{i}.chunk.count", 0)),  # type: ignore[arg-type]
                )
            layers.append(LayerState(h.clone(), S.clone(), chunk=chunk, **extras))
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
