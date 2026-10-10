"""Exp 4 task: procedurally generated group-action lessons, tested on compositions never shown.

Each episode draws a small group G (Z_n for n in 5..8, S3, D4) as permutations, a random generating pair (g1, g2),
a random injective naming of the |G| elements with element tokens E0..E23 and of the two generators with tokens
G0..G5. The lessons are the full right-Cayley graph: one line "x g y ;" per element x and generator g, where
y = x*g, in random order (2|G| lines, 10-16 lines). A query "Q x w1 .. wd =" asks for x*w1*...*wd; the answer is one
element token. For d >= 2 the composite is never shown in any lesson (lessons show single generator steps only),
and because names are random per episode, nothing outside the episode's lessons determines the answer.

Everything here is model-free and deterministic in its seed.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

# ---------------------------------------------------------------- vocabulary
PAD, BOS, SEP, Q, EQ = 0, 1, 2, 3, 4
N_ELEM_TOK, N_GEN_TOK = 24, 6
ELEM0 = 5
GEN0 = ELEM0 + N_ELEM_TOK  # 29
VOCAB = GEN0 + N_GEN_TOK   # 35
VOCAB_PADDED = 40


def elem_tok(i: int) -> int:
    return ELEM0 + i


def gen_tok(i: int) -> int:
    return GEN0 + i


# ---------------------------------------------------------------- groups as permutation tuples
Perm = tuple[int, ...]


def compose(p: Perm, q: Perm) -> Perm:
    """Right action convention: x*(p q) = apply p then q; as functions on points, (p q)(i) = q(p(i))."""
    return tuple(q[p[i]] for i in range(len(p)))


def closure(gens: list[Perm]) -> list[Perm]:
    n = len(gens[0])
    e = tuple(range(n))
    seen = {e}
    frontier = [e]
    while frontier:
        nxt = []
        for x in frontier:
            for g in gens:
                y = compose(x, g)
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        frontier = nxt
    return sorted(seen)


def cyclic(n: int) -> list[Perm]:
    return closure([tuple((i + 1) % n for i in range(n))])


def s3() -> list[Perm]:
    return sorted(itertools.permutations(range(3)))


def d4() -> list[Perm]:
    r = (1, 2, 3, 0)
    s = (0, 3, 2, 1)
    return closure([r, s])


GROUPS: dict[str, list[Perm]] = {"Z5": cyclic(5), "Z6": cyclic(6), "Z7": cyclic(7), "Z8": cyclic(8), "S3": s3(), "D4": d4()}
assert len(GROUPS["D4"]) == 8 and len(GROUPS["S3"]) == 6


@dataclass(frozen=True)
class World:
    """One episode's group with its naming. ``table[i][j]`` = index of element i * generator j."""
    group: str
    n: int
    gens: tuple[int, int]            # element indices of the two generators
    elem_names: tuple[int, ...]      # element index -> token
    gen_names: tuple[int, int]       # generator slot -> token
    table: tuple[tuple[int, int], ...]

    def act(self, x: int, word: tuple[int, ...]) -> int:
        for j in word:
            x = self.table[x][j]
        return x


def make_world(rng: random.Random, group: str | None = None) -> World:
    group = group or rng.choice(sorted(GROUPS))
    els = GROUPS[group]
    n = len(els)
    idx = {p: i for i, p in enumerate(els)}
    while True:
        a, b = rng.sample(range(n), 2)
        if els[a] == tuple(range(len(els[0]))) or els[b] == tuple(range(len(els[0]))):
            continue
        if len(closure([els[a], els[b]])) == n:
            break
    table = tuple((idx[compose(els[i], els[a])], idx[compose(els[i], els[b])]) for i in range(n))
    names = tuple(elem_tok(t) for t in rng.sample(range(N_ELEM_TOK), n))
    gnames = tuple(gen_tok(t) for t in rng.sample(range(N_GEN_TOK), 2))
    return World(group, n, (a, b), names, gnames, table)  # type: ignore[arg-type]


def relabel(world: World, rng: random.Random) -> World:
    """The content null: the same group, generators, token set and line count, but every element token is moved to
    another element by a random derangement, so each lesson line still reads "x g y" in the same format and the
    lessons still describe a valid Cayley graph, but which name goes where is wrong for every element."""
    n = world.n
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(i != j for i, j in enumerate(perm)):
            break
    names = tuple(world.elem_names[perm[i]] for i in range(n))
    return World(world.group, n, world.gens, names, world.gen_names, world.table)


def lesson_tokens(world: World, rng: random.Random) -> list[int]:
    lines = [(x, j) for x in range(world.n) for j in range(2)]
    rng.shuffle(lines)
    out = [BOS]
    for x, j in lines:
        out += [world.elem_names[x], world.gen_names[j], world.elem_names[world.table[x][j]], SEP]
    return out


def lesson_lines(world: World, toks: list[int]) -> list[tuple[int, int, int]]:
    body = toks[1:]
    return [(body[i], body[i + 1], body[i + 2]) for i in range(0, len(body), 4)]


@dataclass(frozen=True)
class Query:
    start: int                 # element index
    word: tuple[int, ...]      # generator slots
    answer: int                # element index

    @property
    def depth(self) -> int:
        return len(self.word)


def make_query(world: World, rng: random.Random, depth: int) -> Query:
    x = rng.randrange(world.n)
    w = tuple(rng.randrange(2) for _ in range(depth))
    return Query(x, w, world.act(x, w))


def query_tokens(world: World, q: Query, layout: str = "old") -> list[int]:
    """Prompt tokens; the answer is predicted at the LAST prompt token.
    layout "old" (exp 4): ``Q x w1..wd =`` predicts at the EQ token.
    layout "aligned" (exp 4b): ``Q x w1..wd`` predicts at the last generator token wd, exactly as a lesson line
    ``x g y ;`` predicts its result y at the generator token g (depth 1 is then the lesson line's prefix ``x g``)."""
    body = [Q, world.elem_names[q.start]] + [world.gen_names[j] for j in q.word]
    if layout == "old":
        return body + [EQ]
    if layout == "aligned":
        return body
    raise ValueError(layout)


# ---------------------------------------------------------------- model-free baselines (per query)
def baseline_predictions(world: World, lesson: list[int], q: Query) -> dict[str, int]:
    """Token predictions of model-free baselines. ``oracle``: composes the lesson lines (exact by construction).
    ``one_hop``: knows every lesson line but does not compose (answers x*w1). ``copy_start``: answers the start
    element (right whenever the word reduces to the identity). ``last_line``: copies the result of the last lesson
    line (a template copier: the most recent shown answer)."""
    lines = lesson_lines(world, lesson)
    look = {(x, g): y for x, g, y in lines}
    x = world.elem_names[q.start]
    cur = x
    for j in q.word:
        cur = look[(cur, world.gen_names[j])]
    return {"oracle": cur,
            "one_hop": look[(x, world.gen_names[q.word[0]])],
            "other_gen": look[(x, world.gen_names[1 - q.word[0]])],
            "copy_start": x,
            "last_line": lines[-1][2]}
