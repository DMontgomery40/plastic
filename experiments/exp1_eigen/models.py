"""Exp 1 models: single-mixer variants of the PlasticBlock, built from the repo's own modules.

Every variant uses the same block skeleton as plastic/model/block.py, but with ONE mixer branch
(so the eigenvalue range of that branch is the only thing that differs inside a family):

    u  = SiLU(CausalConv_K(RMSNorm(x)))           repo CausalConv / RMSNorm
    y  = Mixer(u)
    x += W_o (y * SiLU(W_g u))
    x += MLP(RMSNorm(x))

Mixers
  A  "delta"     repo FastWeightMemory, rule="delta", beta = sigmoid(.) in (0,1)       [published]
  B  "delta2"    identical module, only beta = beta_max * sigmoid(.), beta_max = 2      (Grazzi et al.)
                 beta_max=1 reproduces A bit-for-bit (sanity check).
  E  "nschunk"   repo FastWeightMemory, rule="chunk" (momentum + Newton-Schulz), chunk=1 so the
                 state updates every token (the repo default chunk=64 would not update at all
                 inside the 8-32 token training sequences).
  C  "ssm"       repo selective diagonal recurrence (block._ssm), a = sigma(lam)^{c r} in (0,1],
                 computed with the repo's log-space scan_chunked.
  D  "ssm_neg"   same parameters; a_signed = 2*a - 1 in (-1,1] (Grazzi et al.'s Mamba recipe),
                 z = sqrt(1 - a_signed^2) * (i * W_in u) exactly as the repo, scanned with a
                 sign-aware chunked scan (log|a| cumsum + sign parity) verified against the
                 repo's scan_sequential.
  L  "lstm"      nonlinear control: a 1-layer torch LSTM as the mixer (same block skeleton).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from plastic.config import ModelConfig
from plastic.model.block import CausalConv
from plastic.model.memory import FastWeightMemory
from plastic.model.norm import RMSNorm
from plastic.model.scan import scan_chunked
from plastic.model.state import ChunkStats


# ----------------------------------------------------------------------------- signed scan
def scan_signed_chunked(a: Tensor, u: Tensor, h0: Tensor | None = None, chunk: int = 16) -> tuple[Tensor, Tensor]:
    """h_t = a_t h_{t-1} + u_t for a_t in (-1, 1].  Like plastic.model.scan.scan_chunked, but the
    pairwise decay prod_{r=s+1..t} a_r is formed as  sign * exp(sum log|a_r|),  sign = prod sgn(a_r),
    so negative decays are exact (the repo's log-space scan clamps a to >0 and cannot represent them).
    """
    B, T, D = u.shape
    L = int(chunk)
    pad = (-T) % L
    if pad:
        a = torch.cat([a, torch.ones(B, pad, D, dtype=a.dtype, device=a.device)], 1)
        u = torch.cat([u, torch.zeros(B, pad, D, dtype=u.dtype, device=u.device)], 1)
    n = (T + pad) // L
    a = a.reshape(B, n, L, D)
    u = u.reshape(B, n, L, D)
    log_abs = torch.log(a.abs().clamp_min(1e-30))
    neg = (a < 0).to(a.dtype)
    C = torch.cumsum(log_abs, 2)
    Np = torch.cumsum(neg, 2)  # number of negative factors so far
    diff = C.unsqueeze(3) - C.unsqueeze(2)
    ndiff = Np.unsqueeze(3) - Np.unsqueeze(2)  # negatives in (s, t]
    sign = 1.0 - 2.0 * torch.remainder(ndiff, 2.0)
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=u.device)).view(1, 1, L, L, 1)
    P = torch.exp(diff.masked_fill(~mask, float("-inf"))) * sign
    intra = torch.einsum("bntsd,bnsd->bntd", P, u)
    dec0 = torch.exp(C) * (1.0 - 2.0 * torch.remainder(Np, 2.0))
    h = torch.zeros(B, D, dtype=u.dtype, device=u.device) if h0 is None else h0
    outs = []
    for c in range(n):
        oc = intra[:, c] + dec0[:, c] * h.unsqueeze(1)
        outs.append(oc)
        h = oc[:, -1]
    h_all = torch.stack(outs, 1).reshape(B, T + pad, D)[:, :T]
    return h_all, h_all[:, -1]


# ----------------------------------------------------------------------------- mixers
class DeltaMixer(nn.Module):
    """Wraps the repo's FastWeightMemory; beta_max rescales the write rate (A: 1, B: 2)."""

    def __init__(self, cfg: ModelConfig, beta_max: float = 1.0, force_patch: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.mem = FastWeightMemory(cfg)
        self.beta_max = float(beta_max)
        if force_patch or self.beta_max != 1.0:
            orig = self.mem.project

            def project(u: Tensor):
                q, k, v, beta, alpha = orig(u)
                return q, k, v, beta * self.beta_max, alpha

            self.mem.project = project  # type: ignore[method-assign]
        self.last_beta: Tensor | None = None
        self.last_alpha: Tensor | None = None

    def forward(self, u: Tensor) -> Tensor:
        B, T, D = u.shape
        H, dh = self.cfg.n_heads, self.cfg.head_dim
        S0 = torch.zeros(B, H, dh, dh, dtype=u.dtype, device=u.device)
        M0 = chunk0 = None
        if self.cfg.rule == "chunk":
            M0 = torch.zeros_like(S0)
            chunk0 = ChunkStats.zeros(B, H, dh, u.device, u.dtype)
        mode = "chunk"
        m, _, _, _, sig = self.mem(u, S0, M0, mode=mode, chunk0=chunk0)
        self.last_beta, self.last_alpha = sig.beta.detach(), sig.alpha.detach()
        return m


class SSMMixer(nn.Module):
    """The repo's selective diagonal recurrence (PlasticBlock._ssm), optionally with signed decay."""

    def __init__(self, cfg: ModelConfig, signed: bool = False) -> None:
        super().__init__()
        D = cfg.d_model
        self.cfg = cfg
        self.signed = signed
        self.W_in = nn.Linear(D, D, bias=False)
        self.W_r = nn.Linear(D, D)
        self.W_i = nn.Linear(D, D)
        self.lam = nn.Parameter(torch.full((D,), float(cfg.lam_init)))
        for lin in (self.W_in, self.W_r, self.W_i):
            nn.init.normal_(lin.weight, std=0.02)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        self.last_a: Tensor | None = None

    def decay_and_input(self, u: Tensor) -> tuple[Tensor, Tensor]:
        r = torch.sigmoid(self.W_r(u))
        i = torch.sigmoid(self.W_i(u))
        a = torch.exp(-float(self.cfg.ssm_c) * r * F.softplus(-self.lam))
        if self.signed:
            a = 2.0 * a - 1.0
        z = torch.sqrt(1.0 - a * a + 1e-6) * (i * self.W_in(u))
        return a, z

    def forward(self, u: Tensor) -> Tensor:
        a, z = self.decay_and_input(u)
        self.last_a = a.detach()
        if self.signed:
            h, _ = scan_signed_chunked(a, z, None, chunk=self.cfg.scan_chunk)
        else:
            h, _ = scan_chunked(a, z, None, chunk=self.cfg.scan_chunk)
        return h


class LSTMMixer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(cfg.d_model, cfg.d_model, batch_first=True)

    def forward(self, u: Tensor) -> Tensor:
        return self.lstm(u)[0]


def make_mixer(kind: str, cfg: ModelConfig, beta_max: float | None = None) -> nn.Module:
    if kind == "A":
        return DeltaMixer(cfg, 1.0 if beta_max is None else beta_max)
    if kind == "B":
        return DeltaMixer(cfg, 2.0 if beta_max is None else beta_max, force_patch=True)
    if kind == "E":
        return DeltaMixer(cfg, 1.0)  # cfg.rule must be "chunk"
    if kind == "C":
        return SSMMixer(cfg, signed=False)
    if kind == "D":
        return SSMMixer(cfg, signed=True)
    if kind == "L":
        return LSTMMixer(cfg)
    raise ValueError(kind)


class MixBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str, beta_max: float | None = None) -> None:
        super().__init__()
        D = cfg.d_model
        self.norm1, self.norm3 = RMSNorm(D), RMSNorm(D)
        self.conv = CausalConv(D, cfg.conv_kernel)
        self.mixer = make_mixer(kind, cfg, beta_max)
        self.W_g = nn.Linear(D, D, bias=False)
        self.W_o = nn.Linear(D, D, bias=False)
        self.mlp = nn.Sequential(nn.Linear(D, cfg.mlp_mult * D), nn.GELU(), nn.Linear(cfg.mlp_mult * D, D))
        for lin in (self.W_g, self.W_o, self.mlp[0], self.mlp[2]):
            nn.init.normal_(lin.weight, std=0.02)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        self.K = cfg.conv_kernel

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        buf = torch.zeros(B, self.K - 1, D, dtype=x.dtype, device=x.device) if self.K > 1 else None
        u, _ = self.conv(self.norm1(x), buf)
        y = self.mixer(u)
        x = x + self.W_o(y * F.silu(self.W_g(u)))
        return x + self.mlp(self.norm3(x))


class SeqModel(nn.Module):
    def __init__(self, kind: str, vocab_in: int, n_out: int, *, d_model=64, n_heads=2, n_layers=2,
                 beta_max: float | None = None, conv_kernel: int = 4) -> None:
        super().__init__()
        rule = "chunk" if kind == "E" else "delta"
        chunk = 1 if kind == "E" else 64
        self.cfg = ModelConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, rule=rule, chunk=chunk,
                               conv_kernel=conv_kernel, vocab_size=vocab_in)
        self.kind = kind
        self.embed = nn.Embedding(vocab_in, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList([MixBlock(self.cfg, kind, beta_max) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def eig_stats(self) -> dict:
        """Diagnostics from the last forward: how much of the extended range a variant actually used."""
        out = {}
        for li, b in enumerate(self.blocks):
            m = b.mixer
            if isinstance(m, DeltaMixer) and m.last_beta is not None:
                be = m.last_beta
                out[f"L{li}_beta_mean"] = float(be.mean())
                out[f"L{li}_frac_beta_gt1"] = float((be > 1).float().mean())
                # the eigenvalue of alpha*(I - beta k k^T) along k is alpha*(1-beta)
                ev = m.last_alpha * (1 - be)
                out[f"L{li}_frac_eig_neg"] = float((ev < 0).float().mean())
                out[f"L{li}_min_eig"] = float(ev.min())
            if isinstance(m, SSMMixer) and m.last_a is not None:
                out[f"L{li}_frac_a_neg"] = float((m.last_a < 0).float().mean())
                out[f"L{li}_min_a"] = float(m.last_a.min())
        return out
