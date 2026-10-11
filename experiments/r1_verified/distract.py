"""Distractor generation shared by retention training / eval / gates (R1).

A distractor *segment* is placed between the lesson block and the queries. Because a foreign valid line is
syntactically identical to a lesson line, the segment is delimited with a new marker token DIST (id 35, an unused id
inside VOCAB_PADDED=40). arm "seg": one DIST at the start of the segment. arm "line": a DIST before every line.
Kinds: struct (valid lines of random OTHER worlds, random names), randline (random tokens in the lesson-line format
elem gen elem SEP), randiid (iid uniform tokens over the element+generator vocab, no structure).
"""
from __future__ import annotations
import random, sys
from pathlib import Path
EXP4 = Path(__file__).resolve().parents[1] / "exp4_compose"
sys.path.insert(0, str(EXP4)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from task import ELEM0, GEN0, N_ELEM_TOK, N_GEN_TOK, SEP, lesson_tokens, make_world  # noqa: E402

DIST = 35
KINDS = ("struct", "randline", "randiid")


def _lines(kind: str, K: int, rng: random.Random) -> list[list[int]]:
    out: list[list[int]] = []
    if kind == "struct":
        while len(out) < K:
            w = make_world(rng); t = lesson_tokens(w, rng)[1:]
            out += [t[i:i + 4] for i in range(0, len(t), 4)]
        rng.shuffle(out); return out[:K]
    for _ in range(K):
        if kind == "randline":
            out.append([rng.randrange(ELEM0, ELEM0 + N_ELEM_TOK), GEN0 + rng.randrange(N_GEN_TOK), rng.randrange(ELEM0, ELEM0 + N_ELEM_TOK), SEP])
        elif kind == "randiid":
            out.append([rng.randrange(ELEM0, GEN0 + N_GEN_TOK) for _ in range(4)])
        else:
            raise ValueError(kind)
    return out


def distractor_tokens(kind: str, K: int, rng: random.Random, arm: str = "seg") -> list[int]:
    """Token list of a K-line distractor segment (empty if K == 0)."""
    if K == 0:
        return []
    lines = _lines(kind, K, rng)
    if arm == "seg":
        return [DIST] + [t for l in lines for t in l]
    if arm == "line":
        return [t for l in lines for t in [DIST] + l]
    if arm == "none":  # ablation: no marker at all
        return [t for l in lines for t in l]
    raise ValueError(arm)
