"""CoordinateBlock: a selective recurrence whose fast learner changes its own coordinates.

Implements the block of ``docs/research/2026-09-21-plastic-coordinate-recurrence.md``
(fast mechanism only: no slow rule, no harness, no training run). This is an
experimental candidate beside the default ``PlasticBlock``; it does not replace it.

Per layer and head, the canonical carry ``r`` is split into halves ``p, q`` and

    F_W(q) = eps * tanh(B_f tanh(A_f q + a_f) + b_f)      bounded additive couplings;
    G_W(y) = eps * tanh(B_g tanh(A_g y + a_g) + b_g)      both MLPs are fast weights
    E_W(r) = [z1, z2],  z1 = p + F_W(q),  z2 = q + G_W(z1)
    E_W^{-1}(z) = [z1 - F_W(q), q],  q = z2 - G_W(z1)

so ``||E_W(r) - r||_inf <= eps`` and ``||E_W^{-1}(z) - z||_inf <= eps`` for every W.
With ``u_t = RMSNorm(x_t)``:

    a_t = rho * sigmoid(P_a u_t + theta)                  theta fast, rho < 1 fixed
    v_t = tanh(P_v u_t)
    z_t = a_t z_{t-1} + (1 - a_t) v_t,   z_start = E_W(r_start)
    r_t = concat_heads E_W^{-1}(z_t)
    x  += P_o [r_t * SiLU(P_g u_t)];   x += FFN(RMSNorm(x))

The fast parameters ``omega = (W, theta)`` are per sequence and fixed inside a chunk of
``C`` tokens. At each boundary ``CoordinateCore`` replays the chunk functionally from
the saved canonical chunk-start carry (``chunk_loss``), differentiates the domain's
prediction loss on the observed targets with respect to omega only
(``torch.func.grad_and_value``; ``r_start`` stays an independent argument that is NOT
detached from the outer graph) and proposes ``omega' = omega - beta_scale * eta_l * g``
for every layer ``l`` jointly, with ``eta_l = eta_max * sigmoid(eta_logit_l)``.

Boundary contract. The authoritative carry is canonical ``r``: the next chunk encodes it
under the committed chart, so an accepted proposal realizes ``z_next = E_W'(E_W^{-1}(z_end))``
exactly and ``r_end`` is retained unchanged; a rejected proposal keeps ``omega_c`` and the
same ``r_end``. Proposal and commit are separate: ``commit=True`` accepts every proposal
(the unguarded learner used for meta-training); ``commit=False`` returns the proposal and
leaves ``omega_c`` in the state until ``commit_proposal``. A proposal is bound to the model
instance and the exact state version (token) that produced it, so it cannot be committed onto
another session or twice along one lineage. ``ChunkSignals`` describe the proposal;
``AcceptedChange`` describes what was retained.

Resolved ambiguities (all switchable in ``CoordinateConfig``):

- ``eta`` is one scalar per layer (the memo's optional conditioning on the mean support
  representation is not built).
- ``meta_gradient``: ``"full"`` (default) differentiates through the inner gradient;
  ``"first_order"`` detaches ``g`` (eta is still meta-learned through ``eta * g``);
  ``"none"`` detaches the whole step. ``True``/``False`` are aliases of ``"full"``/``"none"``.
- ``commit_rule="fixed_z"`` is the memo's incorrect diagnostic control: it keeps the latent
  fixed and reinterprets it under the new chart, so the canonical carry jumps.
- The observed-target mask is supplied by the domain wrapper; the physics wrapper's default
  excludes the last row of each chunk (its next observation arrives with the next chunk).
  Such a target is never used for an inner update.
- A call with ``freeze=True`` computes no inner gradient; its rows enter the pending chunk
  with mask False and a boundary crossed while frozen discards the pending chunk.
  ``beta_scale=0`` skips the step (omega bit-identical) but still reports the signals.
- Reset: the carry starts at canonical ``r = 0``, i.e. ``z = E_W(0)``; the physics reset
  flag is an input feature, as in ``PlasticDynamics``.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from plastic.model.norm import RMSNorm

Mode = Literal["chunk", "recurrent"]
MetaGradient = Literal["full", "first_order", "none"]
CommitRule = Literal["transport", "fixed_z"]
# (hidden (B, n, D), targets (B, n, ...)) -> per-position loss (B, n)
LossFn = Callable[[Tensor, Tensor], Tensor]
FastParams = dict[str, Tensor]  # one layer's omega; every tensor has a leading batch dim

W_KEYS = ("f_in", "f_in_b", "f_out", "f_out_b", "g_in", "g_in_b", "g_out", "g_out_b")
THETA_KEY = "theta"
V_BOUND = 1.0  # ||v_t||_inf <= 1 because v_t = tanh(.)


@dataclass(frozen=True)
class CoordinateConfig:
    """Architecture of the coordinate-recurrence stack. ``epsilon`` and ``rho`` are fixed
    architectural constants, never learnable. Defaults follow the memo (D=256, H=4, four
    blocks, C=32, coupling hidden 16, FFN expansion 2) and the probe (eps=0.2, rho=0.97)."""

    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    chunk: int = 32
    coupling_hidden: int = 16
    epsilon: float = 0.2
    rho: float = 0.97
    ffn_mult: int = 2
    theta_init: float = 2.0
    coupling_in_std: float | None = None  # None: 1/sqrt(head_dim/2)
    coupling_out_std: float = 0.05  # small nonzero output matrices, per the memo
    eta_max: float = 1.0
    eta_init: float = 0.1
    obs_dim: int = 4
    act_dim: int = 2
    # falsifier switches (the memo's ablation table); defaults are the full model
    fast_updates: bool = True
    adapt_W: bool = True
    adapt_theta: bool = True
    meta_gradient: MetaGradient | bool = "full"
    commit_rule: CommitRule = "transport"

    def __post_init__(self) -> None:
        if self.meta_gradient is True:
            object.__setattr__(self, "meta_gradient", "full")
        elif self.meta_gradient is False:
            object.__setattr__(self, "meta_gradient", "none")
        if self.meta_gradient not in ("full", "first_order", "none"):
            raise ValueError(f"unknown meta_gradient {self.meta_gradient!r}")
        if self.commit_rule not in ("transport", "fixed_z"):
            raise ValueError(f"unknown commit_rule {self.commit_rule!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (the coupling splits each head in halves)")
        if self.n_layers < 1 or self.chunk < 1 or self.coupling_hidden < 1 or self.ffn_mult < 1:
            raise ValueError("n_layers, chunk, coupling_hidden and ffn_mult must be >= 1")
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (0, 1): the amplitude bound needs a uniform cap below one")
        if not self.epsilon > 0.0:
            raise ValueError("epsilon must be > 0")
        if not 0.0 < self.eta_init < self.eta_max:
            raise ValueError("need 0 < eta_init < eta_max")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def input_dim(self) -> int:
        """Physics input width: [obs, action, reset_flag]."""
        return self.obs_dim + self.act_dim + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CoordinateConfig":
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown CoordinateConfig fields: {sorted(unknown)}")
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


# ---------------------------------------------------------------------------
# Coordinates E_W, their inverse, and chart transport


def _heads(r: Tensor, n_heads: int) -> tuple[Tensor, bool]:
    squeeze = r.dim() == 2
    if squeeze:
        r = r.unsqueeze(1)
    B, n, D = r.shape
    return r.reshape(B, n, n_heads, D // n_heads), squeeze


def _unheads(z: Tensor, squeeze: bool) -> Tensor:
    B, n, H, d = z.shape
    z = z.reshape(B, n, H * d)
    return z.squeeze(1) if squeeze else z


def _shear(y: Tensor, w_in: Tensor, b_in: Tensor, w_out: Tensor, b_out: Tensor, eps: float) -> Tensor:
    """eps * tanh(w_out^T tanh(w_in^T y + b_in) + b_out) per batch element and head; |.| <= eps."""
    hid = torch.tanh(torch.einsum("bnhk,bhkj->bnhj", y, w_in) + b_in.unsqueeze(1))
    return eps * torch.tanh(torch.einsum("bnhj,bhjk->bnhk", hid, w_out) + b_out.unsqueeze(1))


def encode(W: FastParams, r: Tensor, *, eps: float, n_heads: int) -> Tensor:
    """z = E_W(r) for r of shape (B, D) or (B, n, D); each half is shifted once by at most eps."""
    rh, squeeze = _heads(r, n_heads)
    p, q = rh.chunk(2, dim=-1)
    z1 = p + _shear(q, W["f_in"], W["f_in_b"], W["f_out"], W["f_out_b"], eps)
    z2 = q + _shear(z1, W["g_in"], W["g_in_b"], W["g_out"], W["g_out_b"], eps)
    return _unheads(torch.cat([z1, z2], dim=-1), squeeze)


def decode(W: FastParams, z: Tensor, *, eps: float, n_heads: int) -> Tensor:
    """r = E_W^{-1}(z), closed form (no solve)."""
    zh, squeeze = _heads(z, n_heads)
    z1, z2 = zh.chunk(2, dim=-1)
    q = z2 - _shear(z1, W["g_in"], W["g_in_b"], W["g_out"], W["g_out_b"], eps)
    p = z1 - _shear(q, W["f_in"], W["f_in_b"], W["f_out"], W["f_out_b"], eps)
    return _unheads(torch.cat([p, q], dim=-1), squeeze)


def transport(W_a: FastParams, W_b: FastParams, z: Tensor, *, eps: float, n_heads: int) -> Tensor:
    """T_ab = E_b o E_a^{-1}: the latent of chart b that represents the same canonical state
    as latent z of chart a. Identities: T_bc(T_ab(z)) = T_ac(z) and T_ba(T_ab(z)) = z."""
    return encode(W_b, decode(W_a, z, eps=eps, n_heads=n_heads), eps=eps, n_heads=n_heads)


def affine_scan(a: Tensor, b: Tensor, z0: Tensor) -> Tensor:
    """All prefixes of z_t = a_t z_{t-1} + b_t for a, b of shape (B, n, D) and z0 (B, D).

    Doubling scan over affine pairs, (a2, b2) o (a1, b1) = (a2 a1, b2 + a2 b1): only
    multiplications and additions, no division by decay products."""
    aa, bb = a, b
    n = a.shape[1]
    off = 1
    while off < n:
        aa, bb = (
            torch.cat([aa[:, :off], aa[:, off:] * aa[:, :-off]], dim=1),
            torch.cat([bb[:, :off], bb[:, off:] + aa[:, off:] * bb[:, :-off]], dim=1),
        )
        off *= 2
    return aa * z0.unsqueeze(1) + bb


def chunk_bound(r_start_inf: Tensor, cfg: CoordinateConfig, n_steps: int) -> Tensor:
    """Memo bound on ||r_t||_inf after n_steps >= 1 steps of one chunk from ||r_start||_inf = R:
    V + rho^n max(R + eps - V, 0) + eps. With n = chunk length it bounds r_end; with n = 1 it
    bounds every position of the chunk."""
    rn = cfg.rho ** int(n_steps)
    return V_BOUND + rn * (r_start_inf + cfg.epsilon - V_BOUND).clamp_min(0.0) + cfg.epsilon


def uniform_bound(r_initial_inf: float, cfg: CoordinateConfig, chunk: int | None = None) -> float:
    """B_C = max(R_initial, V + eps (1 + rho^C) / (1 - rho^C)): holds at every boundary of a
    stream of C-token chunks under arbitrary accepted (transported) fast-weight changes.
    ``chunk=1`` gives the weaker bound that also holds at every position inside chunks."""
    C = cfg.chunk if chunk is None else int(chunk)
    rc = cfg.rho**C
    return max(float(r_initial_inf), V_BOUND + cfg.epsilon * (1.0 + rc) / (1.0 - rc))


# ---------------------------------------------------------------------------
# State, signals, proposals


@dataclass
class PendingChunk:
    """The unfinished chunk: everything the boundary replay needs. Part of the state."""

    r_start: list[Tensor]  # per layer canonical carry at the chunk start, (B, D)
    x: Tensor  # (B, n, D) layer-0 inputs seen so far in this chunk
    targets: Tensor  # (B, n, ...) targets, zeroed where unobserved
    mask: Tensor  # (B, n) bool: target observed and usable by the inner loss

    def _map(self, fn) -> "PendingChunk":
        return PendingChunk([fn(t) for t in self.r_start], fn(self.x), fn(self.targets), fn(self.mask))


def _new_token() -> str:
    return uuid.uuid4().hex


@dataclass
class CoordinateState:
    """Per-session carried state: canonical carry r and committed omega for every layer,
    the pending chunk, and the stream position. ``pos % chunk`` equals the pending length.

    ``token`` identifies this state version. Every forward call and every commit returns a
    state with a fresh token; ``clone``/``detach``/``to`` keep it, so a fork taken at a
    boundary is the same version and may commit that boundary's proposal."""

    r: list[Tensor]  # per layer (B, D)
    omega: list[FastParams]  # per layer committed fast parameters, batch-leading
    pending: PendingChunk | None = None
    pos: int = 0
    token: str = field(default_factory=_new_token)

    def _map(self, fn) -> "CoordinateState":
        return CoordinateState(
            [fn(t) for t in self.r],
            [{k: fn(v) for k, v in om.items()} for om in self.omega],
            None if self.pending is None else self.pending._map(fn),
            self.pos,
            self.token,
        )

    def clone(self) -> "CoordinateState":
        return self._map(lambda t: t.clone())

    def detach(self) -> "CoordinateState":
        return self._map(lambda t: t.detach())

    def to(self, device: torch.device | str) -> "CoordinateState":
        return self._map(lambda t: t.to(device))

    @property
    def batch(self) -> int:
        return int(self.r[0].shape[0])


@dataclass
class ChunkSignals:
    """Signals of one boundary's PROPOSAL (not of what was retained; see AcceptedChange).

    ``inner_loss_after`` re-scores the same observed support under the proposal: a support
    diagnostic, never a score of an earlier prediction. Adaptation is scored on later targets.
    """

    pos: int  # stream position of the boundary
    n_tokens: int
    frozen: bool
    stepped: bool  # an inner step was computed (fast updates on, not frozen, beta_scale != 0)
    n_observed: Tensor  # (B,)
    inner_loss_before: Tensor  # (B,) masked mean loss under omega_c
    inner_loss_after: Tensor  # (B,) same support under the proposed omega'
    dW_norm: Tensor  # (B, L) Frobenius norm of the proposed coupling change
    dtheta_norm: Tensor  # (B, L)
    eta: Tensor  # (L,) effective step beta_scale * eta_l (0 when no step)
    r_start_inf: Tensor  # (B, L) ||r_start||_inf
    r_end_inf: Tensor  # (B, L) ||r_end||_inf of the actual carry
    r_peak_inf: Tensor  # (B, L) max_t ||r_t||_inf over the chunk
    bound_end: Tensor  # (B, L) chunk_bound(r_start_inf, C)
    bound_peak: Tensor  # (B, L) chunk_bound(r_start_inf, 1)


@dataclass
class AcceptedChange:
    """What was retained at a boundary. ``applied=False`` means omega was left unchanged."""

    pos: int
    applied: bool
    dW_norm: Tensor  # (B, L)
    dtheta_norm: Tensor  # (B, L)


@dataclass
class Proposal:
    """A boundary's candidate omega, bound to the state version (``origin_token``) and the
    model instance (``model_uid``) that produced it. ``signals`` describe the raw proposal;
    ``with_omega`` replaces the weights (e.g. after projection) and keeps the binding."""

    pos: int
    omega: list[FastParams]
    signals: ChunkSignals
    origin_token: str = ""
    model_uid: str = ""

    def with_omega(self, omega: list[FastParams]) -> "Proposal":
        return replace(self, omega=list(omega))


@dataclass
class CoordinateReport:
    chunks: list[ChunkSignals]  # one per boundary crossed in the call
    accepted: list[AcceptedChange]  # one per boundary committed in the call (commit=True)
    proposal: Proposal | None = None  # commit=False and the call ended at an unfrozen boundary


def _layer_norms(old: list[FastParams], new: list[FastParams], keys: tuple[str, ...]) -> Tensor:
    out = []
    for o, n in zip(old, new):
        sq = sum((n[k] - o[k]).detach().flatten(1).pow(2).sum(1) for k in keys)
        out.append(torch.as_tensor(sq).sqrt())
    return torch.stack(out, dim=1)


# ---------------------------------------------------------------------------
# Modules


class CoordinateBlock(nn.Module):
    """One layer: slow projections, the learned initial fast weights (W_0, theta_0) and the
    per-layer step size. ``rollout`` is pure in (omega, r0, x) given the slow parameters."""

    def __init__(self, cfg: CoordinateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D, H, h = cfg.d_model, cfg.n_heads, cfg.coupling_hidden
        k = cfg.head_dim // 2
        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        self.P_a = nn.Linear(D, D, bias=False)
        self.P_v = nn.Linear(D, D, bias=False)
        self.P_g = nn.Linear(D, D, bias=False)
        self.P_o = nn.Linear(D, D, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(D, cfg.ffn_mult * D, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult * D, D, bias=False),
        )
        in_std = cfg.coupling_in_std if cfg.coupling_in_std is not None else 1.0 / math.sqrt(k)
        out_std = cfg.coupling_out_std
        self.W0 = nn.ParameterDict(
            {
                "f_in": nn.Parameter(torch.randn(H, k, h) * in_std),
                "f_in_b": nn.Parameter(torch.zeros(H, h)),
                "f_out": nn.Parameter(torch.randn(H, h, k) * out_std),
                "f_out_b": nn.Parameter(torch.zeros(H, k)),
                "g_in": nn.Parameter(torch.randn(H, k, h) * in_std),
                "g_in_b": nn.Parameter(torch.zeros(H, h)),
                "g_out": nn.Parameter(torch.randn(H, h, k) * out_std),
                "g_out_b": nn.Parameter(torch.zeros(H, k)),
            }
        )
        self.theta0 = nn.Parameter(torch.full((D,), float(cfg.theta_init)))
        p = cfg.eta_init / cfg.eta_max
        self.eta_logit = nn.Parameter(torch.tensor(math.log(p / (1.0 - p))))

    def eta(self) -> Tensor:
        """Positive, bounded, meta-learned step size: eta_max * sigmoid(eta_logit)."""
        return self.cfg.eta_max * torch.sigmoid(self.eta_logit)

    def init_fast(self, batch: int) -> FastParams:
        """Per-sequence copies of (W_0, theta_0); connected to the outer graph."""
        out = {key: p.unsqueeze(0).expand(batch, *p.shape).clone() for key, p in self.W0.items()}
        out[THETA_KEY] = self.theta0.unsqueeze(0).expand(batch, -1).clone()
        return out

    def decay(self, omega: FastParams, u: Tensor) -> Tensor:
        """a = rho * sigmoid(P_a u + theta) in (0, rho]."""
        return self.cfg.rho * torch.sigmoid(self.P_a(u) + omega[THETA_KEY].unsqueeze(1))

    def rollout(self, omega: FastParams, r0: Tensor, x: Tensor, *, parallel: bool) -> tuple[Tensor, Tensor]:
        """Process x (B, n, D) from canonical carry r0 (B, D) with omega fixed.

        Returns (x_out (B, n, D), r_all (B, n, D)); r_all[:, -1] is the canonical r_end."""
        cfg = self.cfg
        u = self.norm1(x)
        a = self.decay(omega, u)
        b = (1.0 - a) * torch.tanh(self.P_v(u))
        z0 = encode(omega, r0, eps=cfg.epsilon, n_heads=cfg.n_heads)
        if parallel:
            r_all = decode(omega, affine_scan(a, b, z0), eps=cfg.epsilon, n_heads=cfg.n_heads)
        else:
            z = z0
            rs = []
            for t in range(x.shape[1]):
                z = a[:, t] * z + b[:, t]
                rs.append(decode(omega, z, eps=cfg.epsilon, n_heads=cfg.n_heads))
            r_all = torch.stack(rs, dim=1)
        x = x + self.P_o(r_all * F.silu(self.P_g(u)))
        x = x + self.ffn(self.norm2(x))
        return x, r_all


class CoordinateCore(nn.Module):
    """A stack of CoordinateBlocks with chunk boundaries, inner updates and commits.

    The inner objective is the domain's prediction loss at the stack output, so all
    layers' candidate updates are computed together at a boundary (per the memo)."""

    def __init__(self, cfg: CoordinateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([CoordinateBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.uid = uuid.uuid4().hex  # binds proposals to this instance; not part of state_dict

    def init_state(self, batch: int) -> CoordinateState:
        p = self.norm_f.weight
        r = [torch.zeros(batch, self.cfg.d_model, device=p.device, dtype=p.dtype) for _ in self.blocks]
        return CoordinateState(r=r, omega=[blk.init_fast(batch) for blk in self.blocks], pending=None, pos=0)

    def rollout(
        self, omega: list[FastParams], r_start: list[Tensor], x: Tensor, *, parallel: bool
    ) -> tuple[Tensor, list[Tensor]]:
        h = x
        r_all = []
        for blk, om, r0 in zip(self.blocks, omega, r_start):
            h, ra = blk.rollout(om, r0, h, parallel=parallel)
            r_all.append(ra)
        return self.norm_f(h), r_all

    def chunk_loss(
        self,
        omega: list[FastParams],
        r_start: list[Tensor],
        x: Tensor,
        targets: Tensor,
        mask: Tensor,
        loss_fn: LossFn,
        *,
        parallel: bool = True,
    ) -> tuple[Tensor, tuple[Tensor, list[Tensor]]]:
        """L_c(omega; r_start, observed chunk, Phi): re-encode r_start under omega and roll out.

        Returns (sum over sequences of the masked-mean loss, (per-sequence loss (B,), per-layer
        r_all)). Summing per-sequence means gives each sequence's omega the gradient of its own
        loss; a sequence with no observed target contributes zero."""
        y, r_all = self.rollout(omega, r_start, x, parallel=parallel)
        per_pos = loss_fn(y, targets)
        per_pos = torch.where(mask, per_pos, torch.zeros_like(per_pos))
        count = mask.to(per_pos.dtype).sum(dim=1)
        per_seq = per_pos.sum(dim=1) / count.clamp_min(1.0)
        return per_seq.sum(), (per_seq, r_all)

    def _adapt_keys(self) -> tuple[str, ...]:
        return (W_KEYS if self.cfg.adapt_W else ()) + ((THETA_KEY,) if self.cfg.adapt_theta else ())

    def _propose(
        self,
        omega: list[FastParams],
        pend: PendingChunk,
        r_end: list[Tensor],
        loss_fn: LossFn,
        *,
        parallel: bool,
        beta_scale: float,
        freeze: bool,
        pos: int,
    ) -> Proposal:
        cfg = self.cfg
        keys = self._adapt_keys()
        step = cfg.fast_updates and not freeze and beta_scale != 0.0 and len(keys) > 0
        etas = torch.stack([blk.eta() for blk in self.blocks])  # (L,)
        args = (pend.r_start, pend.x, pend.targets, pend.mask, loss_fn)
        if step:
            fixed = [{k: v for k, v in om.items() if k not in keys} for om in omega]
            adapt = [{k: om[k] for k in keys} for om in omega]

            def f(ad: list[FastParams]):
                full = [{**fx, **a} for fx, a in zip(fixed, ad)]
                return self.chunk_loss(full, *args, parallel=parallel)

            grads, (_, (per_seq, r_all)) = torch.func.grad_and_value(f, argnums=0, has_aux=True)(adapt)
            new_omega = []
            for layer, (om, g) in enumerate(zip(omega, grads)):
                eta_l = etas[layer] * float(beta_scale)
                upd = {}
                for k in keys:
                    gk = g[k].detach() if cfg.meta_gradient == "first_order" else g[k]
                    delta = -eta_l * gk
                    if cfg.meta_gradient == "none":
                        delta = delta.detach()
                    upd[k] = om[k] + delta
                new_omega.append({**om, **upd})
            with torch.no_grad():
                _, (after, _) = self.chunk_loss(new_omega, *args, parallel=parallel)
        else:
            with torch.no_grad():
                _, (per_seq, r_all) = self.chunk_loss(omega, *args, parallel=parallel)
            new_omega = list(omega)
            after = per_seq
        with torch.no_grad():
            r_start_inf = torch.stack([t.abs().amax(dim=-1) for t in pend.r_start], dim=1)
            n = int(pend.x.shape[1])
            signals = ChunkSignals(
                pos=pos,
                n_tokens=n,
                frozen=bool(freeze),
                stepped=bool(step),
                n_observed=pend.mask.sum(dim=1),
                inner_loss_before=per_seq.detach(),
                inner_loss_after=after.detach(),
                dW_norm=_layer_norms(omega, new_omega, W_KEYS),
                dtheta_norm=_layer_norms(omega, new_omega, (THETA_KEY,)),
                eta=(etas.detach() * float(beta_scale)) if step else torch.zeros_like(etas.detach()),
                r_start_inf=r_start_inf,
                r_end_inf=torch.stack([t.detach().abs().amax(dim=-1) for t in r_end], dim=1),
                r_peak_inf=torch.stack([t.detach().abs().amax(dim=(1, 2)) for t in r_all], dim=1),
                bound_end=chunk_bound(r_start_inf, cfg, n),
                bound_peak=chunk_bound(r_start_inf, cfg, 1),
            )
        return Proposal(pos=pos, omega=new_omega, signals=signals, model_uid=self.uid)

    def _retain(
        self, omega: list[FastParams], r: list[Tensor], proposal: Proposal
    ) -> tuple[list[FastParams], list[Tensor], AcceptedChange]:
        new = proposal.omega
        # "applied" means the retained values change; decided on values, so it is invariant
        # under clone/detach/to and under an unchanged projected candidate
        if all(torch.equal(new[i][k], omega[i][k]) for i in range(len(omega)) for k in omega[i]):
            zeros = torch.zeros_like(proposal.signals.dW_norm)
            # a no-op keeps the canonical carry exactly (no fixed_z re-encode). A computed but
            # zero-valued step keeps its differentiable tensors for the outer graph; otherwise
            # the state's own tensors are kept.
            kept = new if proposal.signals.stepped else omega
            return kept, r, AcceptedChange(proposal.pos, False, zeros, zeros.clone())
        if self.cfg.commit_rule == "fixed_z":
            eps, H = self.cfg.epsilon, self.cfg.n_heads
            r = [decode(nw, encode(old, ri, eps=eps, n_heads=H), eps=eps, n_heads=H) for old, nw, ri in zip(omega, new, r)]
        acc = AcceptedChange(
            proposal.pos,
            True,
            _layer_norms(omega, new, W_KEYS),
            _layer_norms(omega, new, (THETA_KEY,)),
        )
        return new, r, acc

    def commit_proposal(self, state: CoordinateState, proposal: Proposal) -> tuple[CoordinateState, AcceptedChange]:
        """Retain a proposal returned by a ``commit=False`` call. With the default transport rule
        the canonical carry is kept exactly and only omega changes.

        The proposal must come from this model instance and from exactly this state version
        (or a clone of it). The returned state has a fresh token, so the same proposal cannot
        be committed twice along one lineage; a rejected proposal simply goes stale."""
        if proposal.model_uid != self.uid:
            raise ValueError("proposal was produced by a different model instance")
        if proposal.origin_token != state.token or state.pos != proposal.pos or state.pending is not None:
            raise ValueError("proposal does not belong to this state version (foreign, stale or already committed)")
        if len(proposal.omega) != len(state.omega) or any(
            a.keys() != b.keys()
            or any(a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or a[k].device != b[k].device for k in a)
            for a, b in zip(proposal.omega, state.omega)
        ):
            raise ValueError("proposal omega does not match the state's layers, keys, shapes, dtype or device")
        omega, r, acc = self._retain(state.omega, state.r, proposal)
        return CoordinateState(r=list(r), omega=list(omega), pending=None, pos=state.pos), acc

    def forward(
        self,
        x: Tensor,
        state: CoordinateState,
        *,
        targets: Tensor,
        target_mask: Tensor,
        loss_fn: LossFn,
        mode: Mode = "chunk",
        beta_scale: float = 1.0,
        freeze: bool = False,
        commit: bool = True,
    ) -> tuple[Tensor, CoordinateState, CoordinateReport]:
        """Process x (B, T, D) with targets (B, T, ...) and target_mask (B, T).

        Returns (normalized hidden (B, T, D), new state, report). Predictions inside a chunk
        use the committed omega; the observed targets of a chunk only influence later chunks."""
        cfg = self.cfg
        if mode not in ("chunk", "recurrent"):
            raise ValueError(f"unknown mode {mode!r}")
        if len(state.r) != len(self.blocks) or len(state.omega) != len(self.blocks):
            raise ValueError(f"state has {len(state.r)} layers, model has {len(self.blocks)}")
        if not (math.isfinite(beta_scale) and beta_scale >= 0.0):
            raise ValueError("beta_scale must be finite and >= 0")
        B, T, _ = x.shape
        if tuple(targets.shape[:2]) != (B, T) or tuple(target_mask.shape) != (B, T):
            raise ValueError("targets and target_mask must be (B, T, ...) and (B, T)")
        C = cfg.chunk
        count0 = 0 if state.pending is None else int(state.pending.x.shape[1])
        if count0 != state.pos % C:
            raise ValueError("state.pos and the pending chunk disagree")
        if not commit and not freeze and T > C - count0:
            raise ValueError("commit=False: a call may reach at most one boundary, at its end")
        parallel = mode == "chunk"
        mask = target_mask.bool()
        if freeze:
            mask = torch.zeros_like(mask)
        tmask = mask.reshape(B, T, *([1] * (targets.dim() - 2)))
        targets = torch.where(tmask, targets, torch.zeros_like(targets))

        r = list(state.r)
        omega = list(state.omega)
        pend = state.pending
        pos = state.pos
        outs = []
        report = CoordinateReport(chunks=[], accepted=[])
        start = 0
        while start < T:
            count = 0 if pend is None else int(pend.x.shape[1])
            take = min(C - count, T - start)
            sl = slice(start, start + take)
            if pend is None:
                pend = PendingChunk(list(r), x[:, sl], targets[:, sl], mask[:, sl])
            else:
                pend = PendingChunk(
                    pend.r_start,
                    torch.cat([pend.x, x[:, sl]], dim=1),
                    torch.cat([pend.targets, targets[:, sl]], dim=1),
                    torch.cat([pend.mask, mask[:, sl]], dim=1),
                )
            y, r_all = self.rollout(omega, r, x[:, sl], parallel=parallel)
            outs.append(y)
            r = [ra[:, -1] for ra in r_all]
            start += take
            pos += take
            if int(pend.x.shape[1]) == C:
                proposal = self._propose(
                    omega, pend, r, loss_fn, parallel=parallel, beta_scale=beta_scale, freeze=freeze, pos=pos
                )
                pend = None
                report.chunks.append(proposal.signals)
                if commit:
                    omega, r, acc = self._retain(omega, r, proposal)
                    report.accepted.append(acc)
                elif not freeze:
                    report.proposal = proposal
        new_state = CoordinateState(r=r, omega=omega, pending=pend, pos=pos)
        if report.proposal is not None:
            report.proposal = replace(report.proposal, origin_token=new_state.token)
        return torch.cat(outs, dim=1), new_state, report


class CoordinateDynamics(nn.Module):
    """Physics wrapper mirroring ``PlasticDynamics``: input rows ``[obs, action, reset_flag]``,
    next-observation delta head, MSE. Embedding and head follow the lm.py conventions."""

    def __init__(self, cfg: CoordinateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.input_dim, cfg.d_model)
        self.core = CoordinateCore(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.obs_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.embed.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_state(self, batch: int) -> CoordinateState:
        return self.core.init_state(batch)

    def _loss_fn(self, y: Tensor, targets: Tensor) -> Tensor:
        return (self.head(y) - targets).pow(2).mean(dim=-1)

    def default_target_mask(self, pos0: int, batch: int, T: int, device: torch.device | str) -> Tensor:
        """True except on the last row of each chunk: that transition's target needs the next
        observation, which arrives only with the next chunk's first row."""
        p = torch.arange(pos0, pos0 + T, device=device)
        return ((p + 1) % self.cfg.chunk != 0).unsqueeze(0).expand(batch, T)

    def forward(
        self,
        inputs: Tensor,
        state: CoordinateState | None = None,
        *,
        mode: Mode = "chunk",
        beta_scale: float = 1.0,
        freeze: bool = False,
        target_delta: Tensor | None = None,
        target_mask: Tensor | None = None,
        commit: bool = True,
    ) -> tuple[Tensor, CoordinateState, CoordinateReport]:
        B, T, _ = inputs.shape
        if state is None:
            state = self.init_state(B)
        x = self.embed(inputs)
        if target_delta is None:
            targets = torch.zeros(B, T, self.cfg.obs_dim, device=inputs.device, dtype=inputs.dtype)
            mask = torch.zeros(B, T, dtype=torch.bool, device=inputs.device)
        else:
            targets = target_delta
            mask = self.default_target_mask(state.pos, B, T, inputs.device) if target_mask is None else target_mask
        y, new_state, report = self.core(
            x,
            state,
            targets=targets,
            target_mask=mask,
            loss_fn=self._loss_fn,
            mode=mode,
            beta_scale=beta_scale,
            freeze=freeze,
            commit=commit,
        )
        return self.head(y), new_state, report

    def loss(
        self,
        inputs: Tensor,
        target_delta: Tensor,
        state: CoordinateState | None = None,
        *,
        beta_scale: float = 1.0,
        freeze: bool = False,
    ) -> Tensor:
        pred, _, _ = self(inputs, state, mode="chunk", beta_scale=beta_scale, freeze=freeze, target_delta=target_delta)
        return F.mse_loss(pred, target_delta)

    def commit_proposal(self, state: CoordinateState, proposal: Proposal) -> tuple[CoordinateState, AcceptedChange]:
        return self.core.commit_proposal(state, proposal)
