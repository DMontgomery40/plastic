"""Fast-weight memory: the test-time-training layer.

Reads and writes a per-head matrix memory ``S`` with keys, values, and
queries projected from its input, a learned per-token write rate ``β`` and
forget ``α``. Two inner rules share the interface:

- ``delta``: one gradient step per token (gated delta rule), chunk-parallel
  in training and recurrent at inference; exact per-token write pressure.
- ``chunk``: one mini-batch step per chunk with momentum and optional
  orthogonalization; reads inside a chunk use the chunk-start weights. The
  pending chunk's sufficient statistics live in the layer state, so any
  partition of a token stream into forward calls gives identical results.

Control modes, by contract:

- ``freeze=True`` leaves ``S`` and ``M`` bit-identical: no write, no decay, no accumulation
  of pending statistics (frozen tokens count with neutral retention; a boundary crossed
  while frozen discards the pending statistics). Reads still happen.
- ``beta_scale`` scales the write only. For the delta rule it scales β; for the chunk
  rule it scales the complete update after orthogonalization (and the gradient's
  contribution to momentum), so ``beta_scale=0`` writes nothing while decay continues.

Chunk-rule controls are chunk-level by contract: the ``beta_scale`` in effect when a
chunk completes applies to that whole chunk (statistics are accumulated unscaled), and
a chunk whose boundary is crossed while frozen is never applied, so its pending
statistics are discarded. Frozen tokens inside a chunk count with neutral retention
(α = 1). The harness only changes these controls at chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from plastic.config import ModelConfig
from plastic.model.chunk_rule import chunk_rule_apply, chunk_stats
from plastic.model.delta import delta_chunk, delta_recurrent
from plastic.model.norm import RMSNorm
from plastic.model.state import ChunkStats

Mode = Literal["chunk", "recurrent"]


@dataclass
class MemorySignals:
    """Per-token signals, all shaped (B, H, T).

    ``err`` is the prediction error ‖v − k(αS)‖ (surprise); ``write_norm`` is
    the gradient part of the mutation, β‖e‖ for the delta rule (for the chunk
    rule: the applied update's Frobenius norm, reported on the token that
    completes a chunk and 0 elsewhere, so it is identical under any streaming
    partition); ``alpha`` the per-token
    retention so the decay part of the mutation, (1−α)‖S‖, can be accounted
    for separately by the harness.
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
        chunk0: ChunkStats | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, ChunkStats | None, MemorySignals]:
        """Returns (m (B, T, D), S_new, M_new, chunk_new, signals)."""
        B, T, D = u.shape
        q, k, v, beta, alpha = self.project(u)

        if self.cfg.rule == "delta":
            b_eff, a_eff = beta, alpha
            if freeze:
                b_eff, a_eff = torch.zeros_like(beta), torch.ones_like(alpha)
            elif beta_scale != 1.0:
                b_eff = beta * float(beta_scale)
            if mode == "chunk":
                o, S_new, err = delta_chunk(q, k, v, b_eff, a_eff, S0, chunk=self.cfg.chunk)
            elif mode == "recurrent":
                o, S_new, err = delta_recurrent(q, k, v, b_eff, a_eff, S0)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            m = self.norm_mem(o).transpose(1, 2).reshape(B, T, D)
            return m, S_new, None, None, MemorySignals(err=err, beta=b_eff, alpha=a_eff, write_norm=b_eff * err)

        if mode not in ("chunk", "recurrent"):
            raise ValueError(f"unknown mode {mode!r}")
        if M0 is None or chunk0 is None:
            raise ValueError("rule='chunk' requires momentum M0 and pending chunk statistics chunk0")
        L = self.cfg.chunk
        H = self.cfg.n_heads
        S, M = S0, M0
        A, Bv, alpha_sum, count = chunk0.A, chunk0.Bv, chunk0.alpha_sum, int(chunk0.count)
        outs, errs, writes = [], [], []
        start = 0
        while start < T:
            take = min(L - count, T - start)
            sl = slice(start, start + take)
            qc, kc, vc = q[:, :, sl], k[:, :, sl], v[:, :, sl]
            outs.append(torch.einsum("bhtk,bhkv->bhtv", qc, S))  # reads at chunk-start weights
            errs.append((kc @ S - vc).norm(dim=-1))
            if freeze:
                # frozen tokens contribute neutral retention so a chunk that resumes
                # writing after a frozen stretch does not decay by a truncated average
                alpha_sum = alpha_sum + float(take)
            else:
                dA, dBv = chunk_stats(kc, vc, beta[:, :, sl])
                A, Bv = A + dA, Bv + dBv
                alpha_sum = alpha_sum + alpha[:, :, sl].sum(dim=-1)
            count += take
            if count == L:
                if freeze:
                    writes.append(torch.zeros(B, H, take, dtype=u.dtype, device=u.device))
                else:
                    alpha_chunk = alpha_sum / float(L)
                    S_next, M = chunk_rule_apply(
                        S,
                        M,
                        A,
                        Bv,
                        alpha_chunk,
                        n_tokens=L,
                        lr=self.cfg.chunk_lr,
                        momentum=self.cfg.chunk_momentum,
                        orthogonalize=self.cfg.chunk_orthogonalize,
                        scale=float(beta_scale),
                    )
                    delta_norm = (S_next - alpha_chunk[..., None, None] * S).flatten(-2).norm(dim=-1)
                    w = torch.zeros(B, H, take, dtype=u.dtype, device=u.device)
                    w[:, :, -1] = delta_norm
                    writes.append(w)
                    S = S_next
                A, Bv = torch.zeros_like(A), torch.zeros_like(Bv)
                alpha_sum = torch.zeros_like(alpha_sum)
                count = 0
            else:
                writes.append(torch.zeros(B, H, take, dtype=u.dtype, device=u.device))
            start += take
        o = torch.cat(outs, dim=2)
        err = torch.cat(errs, dim=2)
        write_norm = torch.cat(writes, dim=2)
        beta_eff = torch.zeros_like(beta) if freeze else beta * float(beta_scale)
        alpha_eff = torch.ones_like(alpha) if freeze else alpha
        m = self.norm_mem(o).transpose(1, 2).reshape(B, T, D)
        return m, S, M, ChunkStats(A, Bv, alpha_sum, count), MemorySignals(err=err, beta=beta_eff, alpha=alpha_eff, write_norm=write_norm)
