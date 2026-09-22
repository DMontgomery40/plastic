"""PlasticBlock: selective diagonal recurrence, fast-weight memory, and an MLP.

    u   = SiLU(CausalConv_K(RMSNorm(x)))          short depthwise convolution (kernel K)
    a_t = σ(λ)^{c · σ(W_r u)}                     per-channel input-dependent decay in (0, 1)
    z_t = sqrt(1 − a_t²) ⊙ (σ(W_i u) ⊙ W_in u)
    h_t = a_t ⊙ h_{t−1} + z_t                     activation state
    x  += W_o (h ⊙ SiLU(W_g u))
    u2  = SiLU(CausalConv_K(RMSNorm(x)))          memory input (or the block input when memory_input="block_in")
    m   = Memory(u2)                              fast weights S updated by the inner loop
    x  += W_o2 (m ⊙ SiLU(W_g2 u2))
    x  += MLP(RMSNorm(x))

The short convolutions are what let a key at one position bind to the value
at the next (associative recall); their last K−1 inputs are carried in the
session state so the chunked and recurrent paths agree exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from plastic.config import ModelConfig
from plastic.model.memory import FastWeightMemory, MemorySignals, Mode
from plastic.model.norm import RMSNorm
from plastic.model.scan import scan_chunked, scan_sequential
from plastic.model.state import LayerState


class CausalConv(nn.Module):
    """Depthwise causal convolution with an explicit carry buffer of the last K−1 inputs."""

    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        self.kernel = int(kernel)
        self.conv = nn.Conv1d(dim, dim, kernel_size=self.kernel, groups=dim, bias=True)
        nn.init.normal_(self.conv.weight, std=0.02)
        with torch.no_grad():
            # start as the identity on the current token so the block behaves like a plain
            # residual stack at init and learns to look back
            self.conv.weight[:, 0, -1] += 1.0
            self.conv.bias.zero_()

    def forward(self, u: Tensor, buf: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if self.kernel == 1:
            return F.silu(self.conv(u.transpose(1, 2)).transpose(1, 2)), None
        if buf is None:
            raise ValueError("conv buffer state is required when conv_kernel > 1")
        u_cat = torch.cat([buf, u], dim=1)  # (B, K-1+T, D)
        y = self.conv(u_cat.transpose(1, 2)).transpose(1, 2)  # valid conv -> exactly T outputs
        return F.silu(y), u_cat[:, -(self.kernel - 1) :]


class PlasticBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        self.norm3 = RMSNorm(D)
        self.conv_ssm = CausalConv(D, cfg.conv_kernel)
        self.conv_mem = CausalConv(D, cfg.conv_kernel)
        # selective recurrence
        self.W_in = nn.Linear(D, D, bias=False)
        self.W_r = nn.Linear(D, D)
        self.W_i = nn.Linear(D, D)
        self.lam = nn.Parameter(torch.full((D,), float(cfg.lam_init)))
        self.W_g = nn.Linear(D, D, bias=False)
        self.W_o = nn.Linear(D, D, bias=False)
        # fast-weight memory
        self.memory = FastWeightMemory(cfg)
        self.W_g2 = nn.Linear(D, D, bias=False)
        self.W_o2 = nn.Linear(D, D, bias=False)
        # feed-forward
        self.mlp = nn.Sequential(
            nn.Linear(D, cfg.mlp_mult * D),
            nn.GELU(),
            nn.Linear(cfg.mlp_mult * D, D),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (self.W_in, self.W_r, self.W_i, self.W_g, self.W_o, self.W_g2, self.W_o2):
            nn.init.normal_(lin.weight, std=0.02)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        for lin in (self.mlp[0], self.mlp[2]):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    def _ssm(self, u: Tensor, h0: Tensor, *, mode: Mode) -> tuple[Tensor, Tensor]:
        r = torch.sigmoid(self.W_r(u))
        i = torch.sigmoid(self.W_i(u))
        log_a = -float(self.cfg.ssm_c) * r * F.softplus(-self.lam)
        a = torch.exp(log_a)
        z = torch.sqrt(1.0 - a * a + 1e-6) * (i * self.W_in(u))
        if mode == "chunk":
            # the scan's chunk only sets the (B, n, L, L, D) working-set size; results are
            # identical for any value, so it stays small (memory) and independent of cfg.chunk
            return scan_chunked(a, z, h0, chunk=self.cfg.scan_chunk)
        if mode == "recurrent":
            return scan_sequential(a, z, h0)
        raise ValueError(f"unknown mode {mode!r}")

    def forward(
        self,
        x: Tensor,
        state: LayerState,
        *,
        mode: Mode,
        beta_scale: float = 1.0,
        freeze: bool = False,
    ) -> tuple[Tensor, LayerState, MemorySignals]:
        u, conv_ssm = self.conv_ssm(self.norm1(x), state.conv_ssm)
        h_all, h_last = self._ssm(u, state.h, mode=mode)
        x = x + self.W_o(h_all * F.silu(self.W_g(u)))

        u2, conv_mem = self.conv_mem(self.norm2(x), state.conv_mem)
        mem_in = u2 if self.cfg.memory_input == "ssm_out" else u
        m, S_new, M_new, chunk_new, signals = self.memory(
            mem_in, state.S, state.M, mode=mode, beta_scale=beta_scale, freeze=freeze, chunk0=state.chunk
        )
        x = x + self.W_o2(m * F.silu(self.W_g2(u2)))

        x = x + self.mlp(self.norm3(x))
        new_state = LayerState(h=h_last, S=S_new, M=M_new, conv_ssm=conv_ssm, conv_mem=conv_mem, chunk=chunk_new)
        return x, new_state, signals
