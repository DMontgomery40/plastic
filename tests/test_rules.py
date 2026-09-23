"""The rule world: operator definitions, composition order, the covering split, the poison's properties, rendering."""

from __future__ import annotations

import random

import pytest

from plastic.data.rules import (
    FALSE_DEFINITIONS,
    OPERATORS,
    VOCAB,
    Episode,
    all_pairs,
    apply,
    exact,
    make_episode,
    normalize_output,
    poison_batch,
    rule_batch,
    split_pairs,
)

W = ["apple", "pear", "plum"]


def test_operators_match_their_stated_definitions():
    assert apply(("#R",), W) == ["plum", "pear", "apple"]
    assert apply(("#S",), W) == ["pear", "apple", "plum"]
    assert apply(("#D",), W) == ["pear", "plum"]
    assert apply(("#K",), W) == ["plum"]
    assert apply(("#T",), W) == ["apple", "pear", "plum", "plum"]
    assert apply(("#U",), W) == ["APPLE", "PEAR", "PLUM"]


def test_composition_applies_the_rightmost_operator_first():
    assert apply(("#S", "#R"), W) == ["pear", "plum", "apple"]          # R then S
    assert apply(("#R", "#S"), W) == ["plum", "apple", "pear"]          # S then R
    assert apply(("#K", "#R"), W) == ["apple"] and apply(("#R", "#K"), W) == ["plum"]
    with pytest.raises(KeyError):
        apply(("#Z",), W)


def test_split_is_disjoint_covering_and_deterministic():
    train, held = split_pairs(n_heldout=8, seed=0)
    assert split_pairs(n_heldout=8, seed=0) == (train, held)
    singles = [c for c in train if len(c) == 1]
    train_pairs = [c for c in train if len(c) == 2]
    assert len(singles) == len(OPERATORS) and len(held) == 8 and len(train_pairs) == len(all_pairs()) - 8
    assert not set(held) & set(train_pairs)
    assert {p[0] for p in train_pairs} == set(OPERATORS) and {p[1] for p in train_pairs} == set(OPERATORS)
    assert split_pairs(n_heldout=8, seed=1)[1] != held
    with pytest.raises(ValueError):
        split_pairs(n_heldout=29, seed=0)


def test_poison_is_consistent_learnable_and_wrong_on_every_input():
    rng = random.Random(3)
    for op, (text, fn) in FALSE_DEFINITIONS.items():
        assert text != OPERATORS[op][0]                                   # a different rule, not a corrupted one
        for _ in range(50):
            words = rng.sample(VOCAB, rng.randint(3, 4))
            assert fn(words) != OPERATORS[op][1](words), (op, words)      # false on every input tried
    batch = rule_batch([("#R",), ("#S", "#R"), ("#U",)], episodes=6, n_situations=4, seed=0, split_tag="train")
    bad = poison_batch(batch, operator="#R")
    assert bad.poisoned and bad.poisoned_operator == "#R" and not batch.poisoned
    for e_good, e_bad in zip(batch.episodes, bad.episodes):
        assert e_good.ops == e_bad.ops and [s.words for s in e_good.situations] == [s.words for s in e_bad.situations]
        if "#R" in e_good.ops:
            assert e_bad.poisoned and all(sb.answer != sg.answer for sg, sb in zip(e_good.situations, e_bad.situations))
        else:
            assert e_bad is e_good


def test_rendering_puts_the_preface_in_the_first_turn_only_and_is_deterministic():
    e = make_episode(("#S", "#R"), n_situations=3, rng=random.Random(7))
    msgs = e.messages()
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3
    assert msgs[0]["content"].startswith("We are practicing a notation") and "Apply #S #R to:" in msgs[0]["content"]
    assert not msgs[2]["content"].startswith("We are practicing") and msgs[2]["content"].startswith("Apply #S #R to:")
    assert msgs[1]["content"] == e.situations[0].answer_text
    assert make_episode(("#S", "#R"), n_situations=3, rng=random.Random(7)) == e
    b = rule_batch([("#R",), ("#K",)], episodes=4, n_situations=2, seed=5, split_tag="heldout")
    assert [ep.ops for ep in b.episodes] == [("#R",), ("#K",), ("#R",), ("#K",)] and b.situations == 8


def test_exact_normalizes_whitespace_and_punctuation_but_keeps_case():
    assert exact("plum pear apple", " plum  pear apple. ")
    assert exact("APPLE PEAR", "APPLE, PEAR")
    assert not exact("APPLE PEAR", "apple pear")
    assert normalize_output("a, b.") == "a b"
