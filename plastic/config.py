"""Configuration dataclasses shared by the model, training, and the harness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

Domain = Literal["text", "physics"]
Rule = Literal["delta", "chunk"]
Memory = Literal["linear", "mlp"]
MemoryInput = Literal["ssm_out", "block_in"]


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters. Immutable; serialize with ``to_json``."""

    domain: Domain = "text"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    chunk: int = 64
    conv_kernel: int = 4
    vocab_size: int = 4096
    tie_embeddings: bool = True
    rule: Rule = "delta"
    memory: Memory = "linear"
    memory_input: MemoryInput = "ssm_out"
    mlp_mult: int = 4
    obs_dim: int = 4
    act_dim: int = 2
    ssm_c: float = 8.0
    beta_bias_init: float = 0.0
    alpha_bias_init: float = -4.0
    lam_init: float = 2.197
    chunk_momentum: float = 0.9
    chunk_orthogonalize: bool = True
    chunk_lr: float = 0.1
    mem_hidden_mult: int = 2

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.chunk < 1:
            raise ValueError("chunk must be >= 1")
        if self.conv_kernel < 1:
            raise ValueError("conv_kernel must be >= 1 (1 disables the short convolution)")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.domain not in ("text", "physics"):
            raise ValueError(f"unknown domain {self.domain!r}")
        if self.rule not in ("delta", "chunk"):
            raise ValueError(f"unknown rule {self.rule!r}")
        if self.memory not in ("linear", "mlp"):
            raise ValueError(f"unknown memory {self.memory!r}")
        if self.memory == "mlp" and self.rule != "chunk":
            raise ValueError("memory='mlp' requires rule='chunk'")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def input_dim(self) -> int:
        """Continuous input width for the physics domain: [obs, action, reset_flag]."""
        return self.obs_dim + self.act_dim + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown ModelConfig fields: {sorted(unknown)}")
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ModelConfig":
        return cls.from_dict(json.loads(s))

    def signature_material(self) -> str:
        """Canonical string hashed (with checkpoint bytes) into a model signature."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
