"""The rule world for the text learning contract (Thread T4).

Six named string operators over short word lists, taught to a chat model through worked situations and tested
on ordered compositions never shown. Everything here is deterministic and model-free, so the split, the
rendering and the poison are unit-testable; the model enters only in ``plastic/eval/text_learner.py``.

Spec: docs/superpowers/specs/2026-09-23-text-rule-contract.md
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field, replace
from typing import Callable

# Operator names are marker-prefixed capitals the checkpoint has never seen used this way.
OPERATORS: dict[str, tuple[str, Callable[[list[str]], list[str]]]] = {
    "#R": ("reverse the list", lambda w: list(reversed(w))),
    "#S": ("swap the first two words", lambda w: [w[1], w[0]] + w[2:] if len(w) >= 2 else list(w)),
    "#D": ("drop the first word", lambda w: w[1:]),
    "#K": ("keep only the last word", lambda w: w[-1:]),
    "#T": ("repeat the last word once more", lambda w: w + w[-1:]),
    "#U": ("write every word in capitals", lambda w: [x.upper() for x in w]),
}

# False definitions for the poisoned stream: consistent, learnable, and wrong on every input.
FALSE_DEFINITIONS: dict[str, tuple[str, Callable[[list[str]], list[str]]]] = {
    "#R": ("drop the first word", lambda w: w[1:]),
    "#S": ("keep only the last word", lambda w: w[-1:]),
    "#D": ("reverse the list", lambda w: list(reversed(w))),
    "#K": ("swap the first two words", lambda w: [w[1], w[0]] + w[2:] if len(w) >= 2 else list(w)),
    "#T": ("drop the first word", lambda w: w[1:]),
    "#U": ("reverse the list", lambda w: list(reversed(w))),
}

# Decorations: fixed tokens inserted around a copied list. The step-250 checkpoint learns these in context from a few
# worked examples (results/text-rules-2026-09-23/probe_decorations_step250.*), where it learns none of the transforming
# operators above; ordered pairs change the output, so composition is testable.
DECORATIONS: dict[str, tuple[str, Callable[[list[str]], list[str]]]] = {
    "#P": ("write the word please before the list", lambda w: ["please"] + list(w)),
    "#Q": ("write the word thanks after the list", lambda w: list(w) + ["thanks"]),
    "#B": ("put the list in square brackets", lambda w: ["["] + list(w) + ["]"]),
    "#W": ("repeat the whole list twice", lambda w: list(w) + list(w)),
    "#H": ("write the word here before the list", lambda w: ["here"] + list(w)),
}
FALSE_DECORATIONS: dict[str, tuple[str, Callable[[list[str]], list[str]]]] = {
    "#P": ("write the word thanks after the list", lambda w: list(w) + ["thanks"]),
    "#Q": ("write the word please before the list", lambda w: ["please"] + list(w)),
    "#B": ("repeat the whole list twice", lambda w: list(w) + list(w)),
    "#W": ("put the list in square brackets", lambda w: ["["] + list(w) + ["]"]),
    "#H": ("write the word thanks after the list", lambda w: list(w) + ["thanks"]),
}

OPERATOR_SETS: dict[str, tuple[dict[str, tuple[str, Callable]], dict[str, tuple[str, Callable]]]] = {
    "transform": (OPERATORS, FALSE_DEFINITIONS),
    "decorate": (DECORATIONS, FALSE_DECORATIONS),
}


def operator_table(rule_set: str) -> dict[str, tuple[str, Callable]]:
    if rule_set not in OPERATOR_SETS:
        raise KeyError(f"unknown rule set {rule_set!r}; expected one of {tuple(OPERATOR_SETS)}")
    return OPERATOR_SETS[rule_set][0]


def false_table(rule_set: str) -> dict[str, tuple[str, Callable]]:
    return OPERATOR_SETS[rule_set][1]


VOCAB = (
    "apple pear plum grape lemon melon cherry peach mango olive fig date kiwi lime "
    "river stone cloud field forest meadow valley island harbor bridge tower garden "
    "hammer needle basket candle mirror ladder bucket kettle pillow saddle wagon anchor"
).split()

RULE_PREFACE = (
    "We are practicing a notation. Each rule name like #R stands for one fixed operation on a list of words. "
    "When two rule names are written together, the rightmost rule is applied first. Answer with the resulting list only."
)


def preface(*, stated: bool = True, definitions: dict[str, tuple[str, Callable]] | None = None, rule_set: str = "transform") -> str:
    """The first-turn preface. With ``stated`` the six rules are spelled out (a stated rule beats examples alone for
    small models; the T4 sources memo); the poisoned stream states its false definition the same way, so the lesson
    is consistent between the stated rule and the worked outcomes."""
    if not stated:
        return RULE_PREFACE
    table = definitions or operator_table(rule_set)
    rules = "; ".join(f"{op} means {text}" for op, (text, _) in table.items())
    return RULE_PREFACE + " The rules: " + rules + "."


def apply(ops: tuple[str, ...], words: list[str], *, definitions: dict[str, tuple[str, Callable]] | None = None, rule_set: str = "transform") -> list[str]:
    """Apply a composition written left to right, evaluating the rightmost operator first (the preface's rule)."""
    table = definitions or operator_table(rule_set)
    out = list(words)
    for op in reversed(ops):
        if op not in table:
            raise KeyError(f"unknown operator {op!r}")
        out = table[op][1](out)
    return out


def all_pairs(rule_set: str = "transform") -> list[tuple[str, str]]:
    return [p for p in itertools.permutations(operator_table(rule_set), 2)]


def all_compositions(rule_set: str = "transform") -> list[tuple[str, ...]]:
    """Every single operator and every ordered pair: the candidate set for the first-situation choice score, the same
    for trained and held-out compositions so both are read against one chance level."""
    return [(op,) for op in operator_table(rule_set)] + [tuple(p) for p in all_pairs(rule_set)]


def split_pairs(*, n_heldout: int = 18, seed: int = 0, min_reversed_heldout: int = 6, rule_set: str = "transform") -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Training compositions (every single plus the training pairs) and held-out ordered pairs. Constraints (the T4
    sources memo, CFQ/COGS-style low atom divergence): every operator appears in a training pair in BOTH positions,
    and at least ``min_reversed_heldout`` held-out pairs are the reverse order of a training pair, so operator order
    is tested. Default 12 training pairs / 18 held-out. Deterministic in ``seed``."""
    ops = operator_table(rule_set)
    pairs = all_pairs(rule_set)
    if not 0 < n_heldout < len(pairs) - len(ops):
        raise ValueError("n_heldout must leave every operator a training pair")
    rng = random.Random(seed)
    for _ in range(5000):
        held = set(rng.sample(pairs, n_heldout))
        train_pairs = [p for p in pairs if p not in held]
        firsts = {p[0] for p in train_pairs}
        seconds = {p[1] for p in train_pairs}
        reversed_heldout = sum(1 for p in held if (p[1], p[0]) in set(train_pairs))
        if firsts == set(ops) and seconds == set(ops) and reversed_heldout >= min(min_reversed_heldout, n_heldout):
            singles: list[tuple[str, ...]] = [(op,) for op in ops]
            return singles + [tuple(p) for p in train_pairs], [tuple(p) for p in sorted(held)]
    raise RuntimeError("could not find a covering split")


def sample_words(rng: random.Random, n: int) -> list[str]:
    return rng.sample(VOCAB, n)


@dataclass(frozen=True)
class Situation:
    ops: tuple[str, ...]
    words: tuple[str, ...]
    answer: tuple[str, ...]

    @property
    def prompt(self) -> str:
        return f"Apply {' '.join(self.ops)} to: {' '.join(self.words)}"

    @property
    def answer_text(self) -> str:
        return " ".join(self.answer)


@dataclass(frozen=True)
class Episode:
    """One composition, several worked situations, one chat session."""
    ops: tuple[str, ...]
    situations: tuple[Situation, ...]
    poisoned: bool = False
    rule_set: str = "transform"
    definitions_text: tuple[tuple[str, str], ...] | None = None  # (op, stated definition) overrides for a poisoned episode
    stated: bool = True  # False: the preface names the task but no definitions; the rule can come only from the worked examples

    def messages(self, *, stated: bool | None = None) -> list[dict[str, str]]:
        """The chat rendering: the preface (with the stated rules, unless the episode is unstated) rides in the first user
        turn, since the checkpoint has no system role; a consistent poisoned episode states its false definition."""
        stated = self.stated if stated is None else stated
        base = operator_table(self.rule_set)
        table = None
        if self.definitions_text:
            table = dict(base)
            for op, text in self.definitions_text:
                table[op] = (text, base[op][1])
        head = preface(stated=stated, definitions=table, rule_set=self.rule_set)
        out: list[dict[str, str]] = []
        for i, s in enumerate(self.situations):
            user = f"{head}\n\n{s.prompt}" if i == 0 else s.prompt
            out.append({"role": "user", "content": user})
            out.append({"role": "assistant", "content": s.answer_text})
        return out


@dataclass
class RuleBatch:
    episodes: list[Episode]
    split_tag: str  # "train" | "heldout"
    poisoned: bool = False
    poisoned_operator: str | None = None
    rule_set: str = "transform"
    format_only: bool = False
    stated: bool = True
    poison_kind: str | None = None  # "consistent" (false definition stated, false answers) or "inconsistent" (true definition stated, false answers)
    name_map: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] | None = None  # the content null: each name's answers follow another composition

    @property
    def situations(self) -> int:
        return sum(len(e.situations) for e in self.episodes)


def make_episode(ops: tuple[str, ...], *, n_situations: int, rng: random.Random, n_words: tuple[int, int] = (4, 5),
                 definitions: dict[str, tuple[str, Callable]] | None = None, rule_set: str = "transform", stated: bool = True) -> Episode:
    sits = []
    for _ in range(n_situations):
        words = sample_words(rng, rng.randint(*n_words))
        sits.append(Situation(ops, tuple(words), tuple(apply(ops, words, definitions=definitions, rule_set=rule_set))))
    return Episode(ops, tuple(sits), poisoned=definitions is not None, rule_set=rule_set, stated=stated)


def rule_batch(compositions: list[tuple[str, ...]], *, episodes: int, n_situations: int, seed: int, split_tag: str,
               rule_set: str = "transform", n_words: tuple[int, int] = (4, 5), stated: bool = True) -> RuleBatch:
    """``episodes`` episodes cycling over ``compositions`` in order, with fresh word lists from ``seed``. The word
    lists do not depend on ``stated``, so stated and unstated batches from one seed differ only in the preface."""
    rng = random.Random(seed)
    eps = [make_episode(compositions[i % len(compositions)], n_situations=n_situations, rng=rng, rule_set=rule_set, n_words=n_words, stated=stated)
           for i in range(episodes)]
    return RuleBatch(eps, split_tag, rule_set=rule_set, stated=stated)


POISON_KINDS = ("consistent", "inconsistent")


def poison_batch(batch: RuleBatch, *, operator: str, kind: str = "consistent") -> RuleBatch:
    """The same episodes with one operator's answers computed from its false definition wherever it occurs: a lesson
    that is wrong on every input (the T1 property). ``consistent`` also states the false definition in the preface,
    so the stated rule and the worked outcomes agree; ``inconsistent`` keeps the true definitions stated, so the
    preface and the outcomes disagree. In an unstated batch no definition is stated either way and the two kinds
    render the same. Episodes not involving the operator are unchanged."""
    if kind not in POISON_KINDS:
        raise ValueError(f"unknown poison kind {kind!r}; expected one of {POISON_KINDS}")
    false = false_table(batch.rule_set)
    if operator not in false:
        raise KeyError(operator)
    table = dict(operator_table(batch.rule_set))
    table[operator] = false[operator]
    eps = []
    for e in batch.episodes:
        if operator not in e.ops:
            eps.append(e)
            continue
        sits = tuple(Situation(s.ops, s.words, tuple(apply(s.ops, list(s.words), definitions=table))) for s in e.situations)
        stated_false = ((operator, false[operator][0]),) if kind == "consistent" else None
        eps.append(Episode(e.ops, sits, poisoned=True, rule_set=batch.rule_set, definitions_text=stated_false, stated=e.stated))
    return replace(batch, episodes=eps, poisoned=True, poisoned_operator=operator, poison_kind=kind)


def shuffle_answers(batch: RuleBatch, *, seed: int = 0) -> RuleBatch:
    """The format-only control: the same lessons with the answers permuted across the situations of each episode,
    so the chat shape, the vocabulary and the answer lengths are kept while every answer is wrong for its input.
    A lasting update that gains as much from this stream as from the true one learned format, not rules."""
    rng = random.Random(seed)
    eps = []
    for e in batch.episodes:
        answers = [s.answer for s in e.situations]
        if len(answers) > 1:
            perm = list(range(len(answers)))
            while any(i == j for i, j in enumerate(perm)):  # a derangement: no situation keeps its own answer
                rng.shuffle(perm)
            answers = [answers[i] for i in perm]
        sits = tuple(Situation(s.ops, s.words, tuple(a)) for s, a in zip(e.situations, answers))
        eps.append(Episode(e.ops, sits, poisoned=False, rule_set=e.rule_set, definitions_text=e.definitions_text, stated=e.stated))
    return replace(batch, episodes=eps, poisoned=False, poisoned_operator=None, poison_kind=None, format_only=True)


def name_permutation(compositions: list[tuple[str, ...]], *, seed: int, rule_set: str) -> dict[tuple[str, ...], tuple[str, ...]]:
    """A fixed reassignment of compositions within each arity such that every composition's output changes on every
    input (commuting twins such as ``#P #Q`` / ``#Q #P`` never stand in for each other)."""
    probes = [["apple", "pear", "plum", "fig"], ["river", "stone", "cloud"]]
    def sig(c: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(" ".join(apply(c, w, rule_set=rule_set)) for w in probes)
    rng = random.Random(seed)
    mapping: dict[tuple[str, ...], tuple[str, ...]] = {}
    comps = sorted(set(compositions))
    for arity in sorted({len(c) for c in comps}):
        group = [c for c in comps if len(c) == arity]
        for _ in range(10000):
            perm = group[:]
            rng.shuffle(perm)
            if all(sig(a) != sig(b) for a, b in zip(group, perm)):
                break
        else:
            raise RuntimeError(f"no output-changing reassignment for the arity-{arity} compositions")
        mapping.update(zip(group, perm))
    return mapping


def permute_names(batch: RuleBatch, *, seed: int = 0, mapping: dict[tuple[str, ...], tuple[str, ...]] | None = None) -> RuleBatch:
    """The content null (OPUS-LEAD-006): every episode keeps its composition's name and its inputs, but its answers are
    computed with a different composition of the same arity, one fixed reassignment per stream, so each name is
    consistently paired with the wrong decoration while copying, answer shapes and lengths are kept. A lasting update
    that raises the correct outputs' first-situation choice as much from this stream as from the true one is not storing
    which decoration a name means. With stated rules the preface contradicts the answers; use it with unstated rules."""
    if mapping is None:
        mapping = name_permutation([e.ops for e in batch.episodes], seed=seed, rule_set=batch.rule_set)
    eps = []
    for e in batch.episodes:
        target = mapping[e.ops]
        sits = tuple(Situation(s.ops, s.words, tuple(apply(target, list(s.words), rule_set=e.rule_set))) for s in e.situations)
        eps.append(replace(e, situations=sits, poisoned=False))
    return replace(batch, episodes=eps, poisoned=False, poisoned_operator=None, poison_kind=None,
                   name_map=tuple(sorted((k, v) for k, v in mapping.items() if k in {e.ops for e in batch.episodes})))


def normalize_output(text: str) -> str:
    """Exact-match normalization: collapsed whitespace, trailing punctuation removed; case is preserved because #U
    is a case operator."""
    words = text.replace(",", " ").split()
    return " ".join(w.strip(".!?") for w in words)


def exact(answer: str, reply: str) -> bool:
    return normalize_output(reply) == normalize_output(answer)
