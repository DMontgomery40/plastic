"""Gated delta rule: one gradient step per token on ½‖k S − v‖² with write rate β and forget α.

Row-vector convention: ``S`` has shape (d, d), a key reads memory as ``k @ S``
and a write is ``S += β kᵀ e`` where ``e = v − k (α S_prev)`` is the
prediction error, which is also the inner-loop gradient direction.

Two implementations with the same contract:

- ``delta_recurrent``: per-token loop, used at inference.
- ``delta_chunk``: chunk-parallel WY form with an exact unit-lower-triangular
  solve, used for training. Both return the per-token error norm so write
  pressure ``β‖e‖`` is available either way.
"""

from __future__ import annotations

import torch
from torch import Tensor


def delta_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    alpha: Tensor,
    S0: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token recurrence.

    Args:
        q, k, v: (B, H, T, d); q and k are expected L2-normalized.
        beta, alpha: (B, H, T) in (0, 1).
        S0: (B, H, d, d) or None for zeros.

    Returns:
        (o (B, H, T, d), S (B, H, d, d), err (B, H, T)).
    """
    B, H, T, d = q.shape
    S = torch.zeros(B, H, d, d, dtype=q.dtype, device=q.device) if S0 is None else S0
    outs, errs = [], []
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        bt = beta[:, :, t, None, None]
        at = alpha[:, :, t, None, None]
        S = at * S
        e = vt - torch.einsum("bhk,bhkv->bhv", kt, S)
        errs.append(e.norm(dim=-1))
        S = S + bt * torch.einsum("bhk,bhv->bhkv", kt, e)
        outs.append(torch.einsum("bhk,bhkv->bhv", qt, S))
    return torch.stack(outs, dim=2), S, torch.stack(errs, dim=2)


def solve_unit_lower(A: Tensor, Y: Tensor) -> Tensor:
    """Solve ``(I + A) X = Y`` for strictly-lower-triangular ``A`` (batched over leading dims).

    A Neumann/nilpotent product for the inverse is exact only in exact arithmetic:
    with correlated keys (repeated tokens give identical keys) the intermediate
    powers reach 1e17 in fp32 and the result is garbage. Forward substitution is
    exact and stable; ``solve_triangular`` runs on CPU, CUDA, and MPS.
    """
    L = A.shape[-1]
    eye = torch.eye(L, dtype=A.dtype, device=A.device)
    try:
        return torch.linalg.solve_triangular(A + eye, Y, upper=False, unitriangular=True)
    except (RuntimeError, NotImplementedError):
        xs: list[Tensor] = []
        for i in range(L):
            xi = Y[..., i, :]
            if i > 0:
                prev = torch.stack(xs, dim=-2)
                xi = xi - torch.einsum("...j,...jd->...d", A[..., i, :i], prev)
            xs.append(xi)
        return torch.stack(xs, dim=-2)


def delta_chunk(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    alpha: Tensor,
    S0: Tensor | None = None,
    chunk: int = 64,
) -> tuple[Tensor, Tensor, Tensor]:
    """Chunk-parallel form with the same contract as ``delta_recurrent``.

    Any ``T`` is accepted; padding tokens use ``β = 0, α = 1`` so they neither
    write nor decay, and are sliced off.
    """
    B, H, T, d = q.shape
    L = int(chunk)
    if L < 1:
        raise ValueError("chunk must be >= 1")
    pad = (-T) % L
    if pad:
        zeros = torch.zeros(B, H, pad, d, dtype=q.dtype, device=q.device)
        q, k, v = (torch.cat([x, zeros], dim=2) for x in (q, k, v))
        beta = torch.cat([beta, torch.zeros(B, H, pad, dtype=beta.dtype, device=beta.device)], dim=2)
        alpha = torch.cat([alpha, torch.ones(B, H, pad, dtype=alpha.dtype, device=alpha.device)], dim=2)
    n = (T + pad) // L
    q, k, v = (x.reshape(B, H, n, L, d) for x in (q, k, v))
    beta = beta.reshape(B, H, n, L)
    alpha = alpha.reshape(B, H, n, L)

    # log cumulative decay within the chunk: g_t = sum_{r<=t} log alpha_r
    g = torch.cumsum(torch.log(alpha.clamp_min(1e-30)), dim=-1)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2)  # [t, s] = g_t - g_s
    tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
    stril = torch.tril(tril, -1)
    Dm = torch.exp(diff.masked_fill(~tril, float("-inf")))  # inclusive, alpha_{s+1..t}
    Ds = torch.exp(diff.masked_fill(~stril, float("-inf")))  # strict

    kb = k * beta.unsqueeze(-1)
    # A[t, s] = beta_t (k_t . k_s) alpha_{s+1..t} for s < t; solve (I + A) x = y exactly
    A = torch.einsum("bhntd,bhnsd->bhnts", kb, k) * Ds

    gexp = torch.exp(g)  # alpha_{1..t} from chunk start
    w = solve_unit_lower(A, kb * gexp.unsqueeze(-1))  # carries the chunk-start state into pseudo-values
    u = solve_unit_lower(A, v * beta.unsqueeze(-1))

    S = torch.zeros(B, H, d, d, dtype=q.dtype, device=q.device) if S0 is None else S0
    outs, errs = [], []
    for c in range(n):
        qc, kc, vc = q[:, :, c], k[:, :, c], v[:, :, c]
        u_eff = u[:, :, c] - w[:, :, c] @ S  # pseudo-values = beta_t e_t
        Aqk = torch.einsum("bhtd,bhsd->bhts", qc, kc) * Dm[:, :, c]
        o = torch.einsum("bhts,bhsd->bhtd", Aqk, u_eff) + torch.einsum(
            "bhtd,bhdv->bhtv", qc * gexp[:, :, c].unsqueeze(-1), S
        )
        outs.append(o)
        # surprise e_t = v_t - k_t (alpha_t S_{t-1}): the pre-write prediction uses only s < t,
        # computed directly so it does not depend on beta (frozen chunks still report surprise)
        Akk = torch.einsum("bhtd,bhsd->bhts", kc, kc) * Ds[:, :, c]
        pred = torch.einsum("bhts,bhsd->bhtd", Akk, u_eff) + torch.einsum(
            "bhtd,bhdv->bhtv", kc * gexp[:, :, c].unsqueeze(-1), S
        )
        errs.append((vc - pred).norm(dim=-1))
        dec_to_end = torch.exp(g[:, :, c, -1:] - g[:, :, c])  # alpha_{s+1..L}
        S = gexp[:, :, c, -1, None, None] * S + torch.einsum(
            "bhsd,bhsv->bhdv", kc * dec_to_end.unsqueeze(-1), u_eff
        )
    o = torch.stack(outs, dim=2).reshape(B, H, T + pad, d)[:, :, :T]
    err = torch.stack(errs, dim=2).reshape(B, H, T + pad)[:, :, :T]
    return o, S, err
