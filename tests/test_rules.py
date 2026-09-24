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


def test_split_is_disjoint_covering_order_testing_and_deterministic():
    train, held = split_pairs()
    assert split_pairs() == (train, held) and split_pairs(n_heldout=18, seed=0) == (train, held)
    singles = [c for c in train if len(c) == 1]
    train_pairs = [c for c in train if len(c) == 2]
    assert len(singles) == len(OPERATORS) and len(held) == 18 and len(train_pairs) == 12
    assert not set(held) & set(train_pairs)
    assert {p[0] for p in train_pairs} == set(OPERATORS) and {p[1] for p in train_pairs} == set(OPERATORS)
    assert sum(1 for p in held if (p[1], p[0]) in set(train_pairs)) >= 6     # operator order is tested
    assert split_pairs(n_heldout=18, seed=1)[1] != held
    small_train, small_held = split_pairs(n_heldout=8, seed=0)
    assert len(small_held) == 8 and len([c for c in small_train if len(c) == 2]) == 22
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
            assert "#R means drop the first word" in e_bad.messages()[0]["content"]     # the stated rule matches the false outcomes
        else:
            assert e_bad is e_good


def test_rendering_puts_the_preface_in_the_first_turn_only_and_is_deterministic():
    e = make_episode(("#S", "#R"), n_situations=3, rng=random.Random(7))
    msgs = e.messages()
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3
    assert msgs[0]["content"].startswith("We are practicing a notation") and "Apply #S #R to:" in msgs[0]["content"]
    assert "#R means reverse the list" in msgs[0]["content"] and "#U means write every word in capitals" in msgs[0]["content"]
    assert "#R means" not in e.messages(stated=False)[0]["content"]
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


def test_decoration_set_is_a_second_rule_world_with_its_own_split_and_poison():
    from plastic.data.rules import DECORATIONS, FALSE_DECORATIONS, operator_table, split_pairs as sp

    assert set(operator_table("decorate")) == set(DECORATIONS) == {"#P", "#Q", "#B", "#W", "#H"}
    assert apply(("#B", "#P"), W, rule_set="decorate") == ["[", "please", "apple", "pear", "plum", "]"]
    assert apply(("#P", "#B"), W, rule_set="decorate") == ["please", "[", "apple", "pear", "plum", "]"]      # order changes the output
    train, held = sp(n_heldout=8, seed=0, min_reversed_heldout=4, rule_set="decorate")
    assert len([c for c in train if len(c) == 1]) == 5 and len(held) == 8 and len([c for c in train if len(c) == 2]) == 12
    rng = random.Random(1)
    for op, (text, fn) in FALSE_DECORATIONS.items():
        assert text != DECORATIONS[op][0]
        for _ in range(30):
            words = rng.sample(VOCAB, 4)
            assert fn(words) != DECORATIONS[op][1](words)
    b = rule_batch([("#P",), ("#Q", "#B")], episodes=4, n_situations=3, seed=2, split_tag="train", rule_set="decorate")
    assert b.rule_set == "decorate" and "#P means write the word please before the list" in b.episodes[0].messages()[0]["content"]
    assert "#R means" not in b.episodes[0].messages()[0]["content"]
    bad = poison_batch(b, operator="#P")
    assert bad.episodes[0].poisoned and "#P means write the word thanks after the list" in bad.episodes[0].messages()[0]["content"]
    assert bad.episodes[0].situations[0].answer == tuple(list(bad.episodes[0].situations[0].words) + ["thanks"])
    assert bad.episodes[1] is b.episodes[1]


def test_unstated_batches_state_no_definition_anywhere_and_share_their_word_lists_with_the_stated_ones():
    from plastic.data.rules import shuffle_answers

    stated = rule_batch([("#P",), ("#Q", "#B")], episodes=4, n_situations=3, seed=2, split_tag="train", rule_set="decorate")
    unstated = rule_batch([("#P",), ("#Q", "#B")], episodes=4, n_situations=3, seed=2, split_tag="train", rule_set="decorate", stated=False)
    assert unstated.stated is False and stated.stated is True
    for es, eu in zip(stated.episodes, unstated.episodes):
        assert [s.words for s in es.situations] == [s.words for s in eu.situations] and [s.answer for s in es.situations] == [s.answer for s in eu.situations]
        assert not eu.stated and all(" means " not in m["content"] for m in eu.messages())
        assert eu.messages()[0]["content"].startswith("We are practicing a notation") and "Apply #" in eu.messages()[0]["content"]
        assert " means " in es.messages()[0]["content"]
    # the poison in an unstated batch shows only wrong examples; both kinds render the same there
    for kind in ("consistent", "inconsistent"):
        bad = poison_batch(unstated, operator="#P", kind=kind)
        assert bad.poison_kind == kind and bad.episodes[0].poisoned and not bad.episodes[0].stated
        assert all(" means " not in m["content"] for m in bad.episodes[0].messages())
        assert bad.episodes[0].situations[0].answer == tuple(list(bad.episodes[0].situations[0].words) + ["thanks"])
    assert poison_batch(unstated, operator="#P", kind="consistent").episodes[0].messages() == poison_batch(unstated, operator="#P", kind="inconsistent").episodes[0].messages()
    # the format-only control keeps the preface kind
    f = shuffle_answers(unstated, seed=1)
    assert all(not e.stated for e in f.episodes) and f.stated is False and f.poison_kind is None


def test_inconsistent_poison_states_the_true_definition_over_false_answers():
    import pytest

    b = rule_batch([("#P",), ("#Q", "#B")], episodes=4, n_situations=3, seed=2, split_tag="train", rule_set="decorate")
    con = poison_batch(b, operator="#P", kind="consistent")
    inc = poison_batch(b, operator="#P", kind="inconsistent")
    assert con.poison_kind == "consistent" and inc.poison_kind == "inconsistent" and b.poison_kind is None
    assert "#P means write the word thanks after the list" in con.episodes[0].messages()[0]["content"]
    assert "#P means write the word please before the list" in inc.episodes[0].messages()[0]["content"]      # the true definition is stated
    assert inc.episodes[0].situations[0].answer == con.episodes[0].situations[0].answer                      # over the same false answers
    assert inc.episodes[0].situations[0].answer == tuple(list(b.episodes[0].situations[0].words) + ["thanks"])
    assert inc.episodes[1] is b.episodes[1]
    with pytest.raises(ValueError):
        poison_batch(b, operator="#P", kind="wrong")


def test_shuffle_answers_keeps_format_and_derangement_destroys_every_answer():
    from plastic.data.rules import shuffle_answers

    b = rule_batch([("#P", "#B")], episodes=3, n_situations=4, seed=9, split_tag="train", rule_set="decorate")
    f = shuffle_answers(b, seed=1)
    assert f.format_only and not f.poisoned and f.rule_set == "decorate"
    for e_true, e_fmt in zip(b.episodes, f.episodes):
        assert [s.words for s in e_true.situations] == [s.words for s in e_fmt.situations]
        assert sorted(s.answer for s in e_true.situations) == sorted(s.answer for s in e_fmt.situations)   # same answers, moved
        assert all(st.answer != sf.answer for st, sf in zip(e_true.situations, e_fmt.situations))        # none in place
        assert e_fmt.messages()[0]["content"].startswith("We are practicing")
    assert shuffle_answers(b, seed=1) == f


def test_permute_names_is_the_content_null():
    """Each name keeps its inputs and answer shape but its answers follow another composition of the same arity, one
    fixed reassignment per stream, and never a commuting twin (whose answers would be unchanged)."""
    from plastic.data.rules import apply, name_permutation, permute_names, rule_batch

    comps = [("#P",), ("#Q",), ("#B",), ("#W",), ("#H",), ("#P", "#Q"), ("#Q", "#P"), ("#B", "#W"), ("#H", "#B")]
    batch = rule_batch(comps, episodes=2 * len(comps), n_situations=3, seed=4, split_tag="train", rule_set="decorate", stated=False)
    out = permute_names(batch, seed=0)
    m = dict(out.name_map)
    assert set(m) == set(comps) and all(len(k) == len(v) for k, v in m.items())
    for e, f in zip(batch.episodes, out.episodes):
        assert f.ops == e.ops and [s.words for s in f.situations] == [s.words for s in e.situations]
        for s in f.situations:
            assert list(s.answer) == apply(m[e.ops], list(s.words), rule_set="decorate")
            assert list(s.answer) != apply(e.ops, list(s.words), rule_set="decorate")
    assert m[("#P", "#Q")] != ("#Q", "#P") and m[("#Q", "#P")] != ("#P", "#Q")
    assert not out.poisoned and not out.format_only
    assert name_permutation(comps, seed=0, rule_set="decorate") == m
    again = permute_names(batch, mapping=m)
    assert [s.answer for e in again.episodes for s in e.situations] == [s.answer for e in out.episodes for s in e.situations]


def test_an_untrained_template_copier_solves_every_later_decoration_situation():
    """The external review of 6cf4457 (finding 3), reproduced on the archive's held-out split and evaluation seed: a
    copier that reads no operator name and keeps nothing across episodes is exact on every situation after the first,
    so only the first situation can show rule knowledge. The transforming set is not solved this way."""
    from plastic.data.rules import rule_batch, split_pairs, template_baseline

    _, held = split_pairs(n_heldout=8, seed=0, rule_set="decorate", min_reversed_heldout=6)
    batch = rule_batch(held, episodes=16, n_situations=8, seed=100, split_tag="heldout", rule_set="decorate", n_words=(4, 5), stated=False)
    tpl = template_baseline(batch)
    assert all(row[0] == 0.0 for row in tpl) and sum(row[1] for row in tpl) == 16 and sum(x for row in tpl for x in row[1:]) == 112
    _, held_t = split_pairs(n_heldout=18, seed=0, rule_set="transform")
    tb = template_baseline(rule_batch(held_t, episodes=18, n_situations=4, seed=100, split_tag="heldout", rule_set="transform"))
    assert sum(x for row in tb for x in row[1:]) < 0.5 * 18 * 3
