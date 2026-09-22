"""Multi-query associative recall (MQAR) batches for training and evaluation.

Sequence: ``<bos> k1 v1 k2 v2 ... kn vn  kq1 vq1 kq2 vq2 ...`` where the query
keys are a random permutation of the stored keys, padded with ``<pad>``. The
answer mask marks positions whose next token is a queried value, which is the
only place where recall (not the marginal) is being tested.
"""

from __future__ import annotations

import torch
from torch import Tensor

PAD_ID, BOS_ID = 0, 1


def mqar_batch(
    batch: int,
    *,
    n_pairs: int,
    seq_len: int,
    vocab_size: int,
    key_range: tuple[int, int],
    value_range: tuple[int, int],
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    k_lo, k_hi = key_range
    v_lo, v_hi = value_range
    if k_hi - k_lo < n_pairs:
        raise ValueError("key_range too small for n_pairs distinct keys")
    if v_hi > vocab_size or k_hi > vocab_size:
        raise ValueError("ranges exceed vocab_size")
    need = 1 + 4 * n_pairs
    if need > seq_len:
        raise ValueError(f"seq_len {seq_len} < required {need}")
    toks = torch.full((batch, seq_len), PAD_ID, dtype=torch.long)
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    toks[:, 0] = BOS_ID
    for b in range(batch):
        keys = k_lo + torch.randperm(k_hi - k_lo, generator=rng)[:n_pairs]
        vals = torch.randint(v_lo, v_hi, (n_pairs,), generator=rng)
        pairs = torch.stack([keys, vals], dim=1).reshape(-1)
        toks[b, 1 : 1 + 2 * n_pairs] = pairs
        order = torch.randperm(n_pairs, generator=rng)
        queries = torch.stack([keys[order], vals[order]], dim=1).reshape(-1)
        start = 1 + 2 * n_pairs
        toks[b, start : start + 2 * n_pairs] = queries
        mask[b, start : start + 2 * n_pairs : 2] = True
    return toks, mask


def mqar_accuracy(logits: Tensor, tokens: Tensor, answer_mask: Tensor) -> float:
    """Fraction of answer positions whose argmax next-token prediction is the queried value."""
    pred = logits[:, :-1].argmax(dim=-1)
    target = tokens[:, 1:]
    m = answer_mask[:, :-1]
    if int(m.sum()) == 0:
        return float("nan")
    return float((pred[m] == target[m]).float().mean())
