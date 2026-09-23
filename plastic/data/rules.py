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

VOCAB = (
    "apple pear plum grape lemon melon cherry peach mango olive fig date kiwi lime "
    "river stone cloud field forest meadow valley island harbor bridge tower garden "
    "hammer needle basket candle mirror ladder bucket kettle pillow saddle wagon anchor"
).split()

RULE_PREFACE = (
    "We are practicing a notation. Each rule name like #R stands for one fixed operation on a list of words. "
    "When two rule names are written together, the rightmost rule is applied first. Answer with the resulting list only."
)


def apply(ops: tuple[str, ...], words: list[str], *, definitions: dict[str, tuple[str, Callable]] | None = None) -> list[str]:
    """Apply a composition written left to right, evaluating the rightmost operator first (the preface's rule)."""
    table = definitions or OPERATORS
    out = list(words)
    for op in reversed(ops):
        if op not in table:
            raise KeyError(f"unknown operator {op!r}")
        out = table[op][1](out)
    return out


def all_pairs() -> list[tuple[str, str]]:
    return [p for p in itertools.permutations(OPERATORS, 2)]


def split_pairs(*, n_heldout: int, seed: int) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Training compositions (every single plus the training pairs) and held-out ordered pairs, chosen so that
    every operator appears in at least one training pair in each position. Deterministic in ``seed``."""
    pairs = all_pairs()
    if not 0 < n_heldout < len(pairs) - len(OPERATORS):
        raise ValueError("n_heldout must leave every operator a training pair")
    rng = random.Random(seed)
    for _ in range(1000):
        held = set(rng.sample(pairs, n_heldout))
        train_pairs = [p for p in pairs if p not in held]
        firsts = {p[0] for p in train_pairs}
        seconds = {p[1] for p in train_pairs}
        if firsts == set(OPERATORS) and seconds == set(OPERATORS):
            singles: list[tuple[str, ...]] = [(op,) for op in OPERATORS]
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

    def messages(self, *, preface: str = RULE_PREFACE) -> list[dict[str, str]]:
        """The chat rendering: the preface rides in the first user turn (the checkpoint has no system role)."""
        out: list[dict[str, str]] = []
        for i, s in enumerate(self.situations):
            user = f"{preface}\n\n{s.prompt}" if i == 0 else s.prompt
            out.append({"role": "user", "content": user})
            out.append({"role": "assistant", "content": s.answer_text})
        return out


@dataclass
class RuleBatch:
    episodes: list[Episode]
    split_tag: str  # "train" | "heldout"
    poisoned: bool = False
    poisoned_operator: str | None = None

    @property
    def situations(self) -> int:
        return sum(len(e.situations) for e in self.episodes)


def make_episode(ops: tuple[str, ...], *, n_situations: int, rng: random.Random, n_words: tuple[int, int] = (3, 4),
                 definitions: dict[str, tuple[str, Callable]] | None = None) -> Episode:
    sits = []
    for _ in range(n_situations):
        words = sample_words(rng, rng.randint(*n_words))
        sits.append(Situation(ops, tuple(words), tuple(apply(ops, words, definitions=definitions))))
    return Episode(ops, tuple(sits), poisoned=definitions is not None)


def rule_batch(compositions: list[tuple[str, ...]], *, episodes: int, n_situations: int, seed: int, split_tag: str) -> RuleBatch:
    """``episodes`` episodes cycling over ``compositions`` in order, with fresh word lists from ``seed``."""
    rng = random.Random(seed)
    eps = [make_episode(compositions[i % len(compositions)], n_situations=n_situations, rng=rng) for i in range(episodes)]
    return RuleBatch(eps, split_tag)


def poison_batch(batch: RuleBatch, *, operator: str) -> RuleBatch:
    """The same episodes with one operator taught by its false definition wherever it occurs: a consistent lesson
    that is wrong on every input (the T1 property). Episodes not involving the operator are unchanged."""
    if operator not in FALSE_DEFINITIONS:
        raise KeyError(operator)
    table = dict(OPERATORS)
    table[operator] = FALSE_DEFINITIONS[operator]
    eps = []
    for e in batch.episodes:
        if operator not in e.ops:
            eps.append(e)
            continue
        sits = tuple(Situation(s.ops, s.words, tuple(apply(s.ops, list(s.words), definitions=table))) for s in e.situations)
        eps.append(Episode(e.ops, sits, poisoned=True))
    return replace(batch, episodes=eps, poisoned=True, poisoned_operator=operator)


def normalize_output(text: str) -> str:
    """Exact-match normalization: collapsed whitespace, trailing punctuation removed; case is preserved because #U
    is a case operator."""
    words = text.replace(",", " ").split()
    return " ".join(w.strip(".!?") for w in words)


def exact(answer: str, reply: str) -> bool:
    return normalize_output(reply) == normalize_output(answer)
