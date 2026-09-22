"""Fast-weight memory: the test-time-training layer.

Reads and writes a per-head matrix memory ``S`` with keys, values, and
queries projected from its input, a learned per-token write rate ``β`` and
forget ``α``. Two inner rules share the interface:

- ``delta``: one gradient step per token (gated delta rule), chunk-parallel
  in training and recurrent at inference; exact per-token write pressure.
- ``chunk``: one mini-batch step per chunk with momentum and optional
  orthogonalization; reads inside a chunk use the chunk-start weights.

``freeze=True`` leaves ``S`` bit-identical (no write and no decay): reads
still happen, so the model keeps using what it has learned without learning
more. ``beta_scale`` scales the write rate only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from plastic.config import ModelConfig
from plastic.model.chunk_rule import chunk_rule_step
from plastic.model.delta import delta_chunk, delta_recurrent
from plastic.model.norm import RMSNorm

Mode = Literal["chunk", "recurrent"]


@dataclass
class MemorySignals:
    """Per-token signals, all shaped (B, H, T).

    ``err`` is the prediction error ‖v − k(αS)‖ (surprise); ``write_norm`` is
    the gradient part of the mutation, β‖e‖ for the delta rule; ``alpha`` the
    per-token retention so the decay part of the mutation, (1−α)‖S‖, can be
    accounted for separately by the harness.
    """

    err: Tensor
    beta: Tensor
    alpha: Tensor
    write_norm: Tensor

    def detach(self) -> "MemorySignals":
        return MemorySignals(self.err.detach(), self.beta.detach(), self.alpha.detach(), self.write_norm.detach())


class FastWeightMemory(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.memory == "mlp":
            raise NotImplementedError("memory='mlp' is scheduled for the adversarial-gate milestone")
        self.cfg = cfg
        D, H = cfg.d_model, cfg.n_heads
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.W_beta = nn.Linear(D, H)
        self.W_alpha = nn.Linear(D, H)
        self.norm_mem = RMSNorm(cfg.head_dim)
        with torch.no_grad():
            nn.init.normal_(self.W_beta.weight, std=0.02)
            nn.init.normal_(self.W_alpha.weight, std=0.02)
            self.W_beta.bias.fill_(cfg.beta_bias_init)
            self.W_alpha.bias.fill_(cfg.alpha_bias_init)

    def project(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        B, T, D = u.shape
        H, dh = self.cfg.n_heads, self.cfg.head_dim
        q = F.normalize(self.W_q(u).view(B, T, H, dh).transpose(1, 2), dim=-1)
        k = F.normalize(self.W_k(u).view(B, T, H, dh).transpose(1, 2), dim=-1)
        v = self.W_v(u).view(B, T, H, dh).transpose(1, 2)
        beta = torch.sigmoid(self.W_beta(u)).transpose(1, 2)  # (B, H, T)
        alpha = torch.exp(-F.softplus(self.W_alpha(u))).transpose(1, 2)  # (B, H, T)
        return q, k, v, beta, alpha

    def forward(
        self,
        u: Tensor,
        S0: Tensor,
        M0: Tensor | None,
        *,
        mode: Mode,
        beta_scale: float = 1.0,
        freeze: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor | None, MemorySignals]:
        """Returns (m (B, T, D), S_new, M_new, signals)."""
        B, T, D = u.shape
        q, k, v, beta, alpha = self.project(u)
        if freeze:
            beta = torch.zeros_like(beta)
            alpha = torch.ones_like(alpha)
        elif beta_scale != 1.0:
            beta = beta * float(beta_scale)

        if self.cfg.rule == "delta":
            if mode == "chunk":
                o, S_new, err = delta_chunk(q, k, v, beta, alpha, S0, chunk=self.cfg.chunk)
            elif mode == "recurrent":
                o, S_new, err = delta_recurrent(q, k, v, beta, alpha, S0)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            M_new = None
            write_norm = beta * err
        else:
            if M0 is None:
                raise ValueError("rule='chunk' requires momentum state M0")
            L = self.cfg.chunk
            S, M = S0, M0
            outs, errs, writes = [], [], []
            for start in range(0, T, L):
                sl = slice(start, min(start + L, T))
                qc, kc, vc = q[:, :, sl], k[:, :, sl], v[:, :, sl]
                outs.append(torch.einsum("bhtk,bhkv->bhtv", qc, S))  # reads at chunk-start weights
                n_tok = int(sl.stop - sl.start)
                if freeze:
                    E = kc @ S - vc
                    errs.append(E.norm(dim=-1))
                    writes.append(torch.zeros(B, self.cfg.n_heads, n_tok, dtype=u.dtype, device=u.device))
                    continue
                alpha_chunk = alpha[:, :, sl].mean(dim=-1)
                S_next, M, err_c = chunk_rule_step(
                    S,
                    M,
                    kc,
                    vc,
                    beta[:, :, sl],
                    alpha_chunk,
                    lr=self.cfg.chunk_lr,
                    momentum=self.cfg.chunk_momentum,
                    orthogonalize=self.cfg.chunk_orthogonalize,
                )
                delta_norm = (S_next - alpha_chunk[..., None, None] * S).flatten(-2).norm(dim=-1) / n_tok
                writes.append(delta_norm.unsqueeze(-1).expand(B, self.cfg.n_heads, n_tok))
                errs.append(err_c)
                S = S_next
            o = torch.cat(outs, dim=2)
            err = torch.cat(errs, dim=2)
            write_norm = torch.cat(writes, dim=2)
            S_new, M_new = S, M

        m = self.norm_mem(o).transpose(1, 2).reshape(B, T, D)
        return m, S_new, M_new, MemorySignals(err=err, beta=beta, alpha=alpha, write_norm=write_norm)
