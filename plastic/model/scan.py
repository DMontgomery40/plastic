"""Selective diagonal recurrence h_t = a_t * h_{t-1} + u_t.

Two implementations with the same contract:

- ``scan_sequential``: the reference, one Python step per token.
- ``scan_chunked``: numerically stable parallel form. Inside a chunk only
  pairwise decay ratios ``exp(C_t - C_s)`` with ``s <= t`` are formed, which
  lie in (0, 1] and can underflow only to "forgotten"; across chunks the
  state is carried sequentially. Never forms ``a^t`` or divides by decays.
"""

from __future__ import annotations

import torch
from torch import Tensor


def scan_sequential(a: Tensor, u: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Reference recurrence.

    Args:
        a: decays in (0, 1), shape (B, T, D).
        u: inputs, shape (B, T, D).
        h0: initial state (B, D) or None for zeros.

    Returns:
        (h_all (B, T, D), h_last (B, D)).
    """
    B, T, D = u.shape
    h = torch.zeros(B, D, dtype=u.dtype, device=u.device) if h0 is None else h0
    out = []
    for t in range(T):
        h = a[:, t] * h + u[:, t]
        out.append(h)
    return torch.stack(out, dim=1), h


def scan_chunked(
    a: Tensor,
    u: Tensor,
    h0: Tensor | None = None,
    chunk: int = 64,
) -> tuple[Tensor, Tensor]:
    """Chunked log-space scan with the same contract as ``scan_sequential``.

    Any ``T`` is accepted; the sequence is padded to a multiple of ``chunk``
    with ``a = 1, u = 0`` (state held constant) and the padding is sliced off.
    """
    B, T, D = u.shape
    L = int(chunk)
    if L < 1:
        raise ValueError("chunk must be >= 1")
    pad = (-T) % L
    if pad:
        a = torch.cat([a, torch.ones(B, pad, D, dtype=a.dtype, device=a.device)], dim=1)
        u = torch.cat([u, torch.zeros(B, pad, D, dtype=u.dtype, device=u.device)], dim=1)
    n = (T + pad) // L
    a = a.reshape(B, n, L, D)
    u = u.reshape(B, n, L, D)

    log_a = torch.log(a.clamp_min(1e-30))
    C = torch.cumsum(log_a, dim=2)  # (B, n, L, D), non-positive
    diff = C.unsqueeze(3) - C.unsqueeze(2)  # (B, n, L, L, D) indexed [t, s]
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=u.device))
    P = torch.exp(diff.masked_fill(~mask.view(1, 1, L, L, 1), float("-inf")))
    intra = torch.einsum("bntsd,bnsd->bntd", P, u)
    decay_from_start = torch.exp(C)

    h = torch.zeros(B, D, dtype=u.dtype, device=u.device) if h0 is None else h0
    outs = []
    for c in range(n):
        oc = intra[:, c] + decay_from_start[:, c] * h.unsqueeze(1)
        outs.append(oc)
        h = oc[:, -1]
    h_all = torch.stack(outs, dim=1).reshape(B, T + pad, D)[:, :T]
    return h_all, h_all[:, -1]
