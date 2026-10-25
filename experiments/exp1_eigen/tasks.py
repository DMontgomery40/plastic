"""State-tracking tasks with per-position labels. Token 0 is BOS at position 0 (shared by all tasks);
labels at position 0 are ignored (-100). Every task returns (tokens (B, T+1), labels (B, T+1)).

parity : x_t in {0,1};  y_t = (sum_{s<=t} x_s) mod 2
mod3   : x_t in {0,1};  y_t = (sum_{s<=t} x_s) mod 3          (cyclic group Z3 word problem, binary input)
s5     : x_t in S5 (120 permutations); y_t = x_1 o x_2 o ... o x_t  (index of the prefix product)
s3     : same with S3 (6 elements) -- smallest non-abelian group, fallback if S5 is out of reach
mqar   : repo's plastic.data.mqar.mqar_batch (separate path in train.py)
"""
from __future__ import annotations

import itertools

import torch

BOS = 0


def _perm_table(n: int):
    perms = list(itertools.permutations(range(n)))
    idx = {p: i for i, p in enumerate(perms)}
    G = len(perms)
    # compose(a, b) = "apply a then b": (a then b)(i) = b[a[i]]
    table = torch.empty(G, G, dtype=torch.long)
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            table[i, j] = idx[tuple(b[a[k]] for k in range(n))]
    ident = idx[tuple(range(n))]
    return table, ident


_TABLES = {}


def group_table(name: str):
    if name not in _TABLES:
        _TABLES[name] = _perm_table({"s5": 5, "s3": 3}[name])
    return _TABLES[name]


TASKS = {
    "parity": dict(vocab_in=3, n_out=2),
    "mod3": dict(vocab_in=3, n_out=3),
    "s5": dict(vocab_in=121, n_out=120),
    "s3": dict(vocab_in=7, n_out=6),
}


def batch(task: str, B: int, T: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if task in ("parity", "mod3"):
        mod = 2 if task == "parity" else 3
        x = torch.randint(0, 2, (B, T), generator=g)
        y = torch.cumsum(x, 1) % mod
        tok = x + 1
    elif task in ("s5", "s3"):
        table, ident = group_table(task)
        G = table.shape[0]
        x = torch.randint(0, G, (B, T), generator=g)
        y = torch.empty_like(x)
        cur = torch.full((B,), ident, dtype=torch.long)
        for t in range(T):
            cur = table[cur, x[:, t]]
            y[:, t] = cur
        tok = x + 1
    else:
        raise ValueError(task)
    tokens = torch.cat([torch.full((B, 1), BOS, dtype=torch.long), tok], 1)
    labels = torch.cat([torch.full((B, 1), -100, dtype=torch.long), y], 1)
    return tokens, labels
