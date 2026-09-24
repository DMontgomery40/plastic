"""The text learner's model-free parts: answer-span extraction, span scoring, and the verify decision on fakes."""

from __future__ import annotations

import torch

from plastic.eval.text_learner import MODES, answer_spans, span_scores


def test_answer_spans_finds_each_assistant_turn():
    labels = [-100, -100, 5, 6, 2, -100, -100, 7, 2]
    assert answer_spans(labels) == [(2, 5), (7, 9)]
    assert answer_spans([-100, -100]) == [] and answer_spans([3, 4]) == [(0, 2)]


def test_span_scores_uses_the_previous_position_logits_and_reports_teacher_forced_exact():
    V = 10
    ids = [0, 1, 5, 6, 2, 1, 7, 2]
    labels = [-100, -100, 5, 6, 2, -100, 7, 2]
    logits = torch.full((len(ids), V), -5.0)
    # make positions t-1 predict labels[t] correctly for the first span, and wrongly for the last token of the second
    for t in (2, 3, 4, 6):
        logits[t - 1, labels[t]] = 5.0
    logits[6, 3] = 5.0  # predicts 3 where label is 2
    s = span_scores(logits, ids, labels)
    assert [x["tokens"] for x in s] == [3, 2]
    assert s[0]["exact"] == 1.0 and s[0]["nll"] < 0.01
    assert s[1]["exact"] == 0.0 and s[1]["nll"] > 1.0


def test_modes_are_the_contract_baselines_plus_propose_and_verify():
    assert MODES == ("frozen", "continued", "in_context", "replay_verify")


def _verifier(adapt: bool):
    from plastic.eval.text_learner import TextRuleLearner

    L = TextRuleLearner.__new__(TextRuleLearner)
    L.tol_exact, L.tol_nll, L.verify_adapt, L.mode = 0.05, 0.1, adapt, "replay_verify"
    return L


def test_verifier_v2_checks_the_first_situation_and_v1_does_not():
    """The v1 verifier scored held-in material with the fast path frozen, where this model's exact is 0 before and after
    (a vacuous check, FABLE-41B-206). v2 scores with adaptation on and adds the first-situation exact."""
    v1, v2 = _verifier(False), _verifier(True)
    assert v1.verifier_version == "v1" and v2.verifier_version == "v2"
    v2.mode = "continued"
    assert v2.verifier_version is None  # the baselines do not verify, so no verifier is named for them
    v2.mode = "replay_verify"
    good = [[{"exact": 1.0, "nll": 0.2}, {"exact": 1.0, "nll": 0.1}], [{"exact": 1.0, "nll": 0.3}, {"exact": 1.0, "nll": 0.1}]]
    poisoned = [[{"exact": 0.0, "nll": 0.4}, {"exact": 1.0, "nll": 0.1}], [{"exact": 0.0, "nll": 0.5}, {"exact": 1.0, "nll": 0.1}]]
    before, after = v2._summ(good), v2._summ(poisoned)
    assert before["exact_first"] == 1.0 and after["exact_first"] == 0.0 and after["exact"] == 0.5
    c2 = {c["name"]: c for c in v2._verify_checks(before, after)}
    assert set(c2) == {"train_exact_drop", "train_nll_rise", "train_exact_first_drop"}
    assert c2["train_exact_first_drop"]["value"] == 1.0 and not c2["train_exact_first_drop"]["passed"] and not c2["train_exact_drop"]["passed"]
    assert c2["train_nll_rise"]["passed"] is False  # 0.175 -> 0.275
    c1 = {c["name"]: c for c in v1._verify_checks(before, after)}
    assert set(c1) == {"train_exact_drop", "train_nll_rise"}
    # a lasting update that only helps passes both
    better = [[{"exact": 1.0, "nll": 0.1}, {"exact": 1.0, "nll": 0.05}], [{"exact": 1.0, "nll": 0.1}, {"exact": 1.0, "nll": 0.05}]]
    assert all(c["passed"] for c in v2._verify_checks(before, v2._summ(better)))


def _order_learner(sampling: str, *, passes: int = 1, steps: int = 20):
    from plastic.eval.text_learner import TextRuleLearner

    L = TextRuleLearner.__new__(TextRuleLearner)
    L.sampling, L.passes, L.steps, L.verify_seed = sampling, passes, steps, 7
    return L


def test_passes_visit_every_episode_once_per_pass_and_draws_reproduce_the_archived_order():
    """The archived continued / replay_verify runs drew 20 episodes with replacement from Random(7): on the 17-episode
    archive stream that trains 11 compositions and leaves #P, #P #Q, #Q #W, #Q #H, #H #P and #H #B untrained
    (OPUS-LEAD-001). Full passes train every episode the same number of times."""
    order = _order_learner("passes", passes=2).training_order(17)
    assert sorted(order) == sorted(list(range(17)) * 2) and order[:17] != list(range(17))
    assert sorted(order[:17]) == list(range(17)) and sorted(order[17:]) == list(range(17))
    draws = _order_learner("draws").training_order(17)
    assert draws == [10, 4, 12, 1, 2, 3, 11, 1, 16, 6, 1, 2, 13, 13, 2, 7, 2, 13, 1, 3]
    assert len(set(draws)) == 11 and sorted(set(range(17)) - set(draws)) == [0, 5, 8, 9, 14, 15]
    from plastic.eval.text_contract import TextContractSpec, contract_split

    train, _ = contract_split(TextContractSpec(n_heldout=8, min_reversed_heldout=6, split_seed=0, rule_set="decorate"))
    assert [" ".join(train[i]) for i in (0, 5, 8, 9, 14, 15)] == ["#P", "#P #Q", "#Q #W", "#Q #H", "#H #P", "#H #B"]


def test_learner_rejects_unknown_sampling_and_zero_passes():
    import pytest
    from plastic.eval.text_learner import TextRuleLearner

    with pytest.raises(ValueError):
        TextRuleLearner(None, sampling="shuffled")
    with pytest.raises(ValueError):
        TextRuleLearner(None, passes=0)


class _CharTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, None

    def __call__(self, text, add_special_tokens=False):
        class _R:
            pass

        r = _R()
        r.input_ids = [3 + (ord(c) % 250) for c in text]
        return r


def _tiny_backend():
    import pytest

    pytest.importorskip("transformers")
    from plastic.backends.ttt_lm import modeling_ttt as M

    torch.manual_seed(0)
    cfg = M.TTTConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, ttt_layer_type="mlp",
                      pre_conv=True, share_qk=True, use_gate=True, ttt_base_lr=0.1, mini_batch_size=16, max_position_embeddings=4096)
    model = M.TTTForCausalLM(cfg).eval()

    class _B:
        pass

    b = _B()
    b.model, b.tokenizer, b.device = model, _CharTokenizer(), torch.device("cpu")
    return b


def test_choice_ranks_every_composition_output_and_padding_does_not_move_a_rows_score():
    from plastic.backends.ttt_lm.backend import encode_conversation
    from plastic.data.rules import rule_batch
    from plastic.eval.text_learner import TextRuleLearner

    be = _tiny_backend()
    L = TextRuleLearner(be, mode="frozen")
    batch = rule_batch([("#P",), ("#B", "#Q")], episodes=2, n_situations=1, seed=3, split_tag="train", rule_set="decorate", stated=False)
    items = L.choice(batch)
    assert [it["ops"] for it in items] == ["#P", "#B #Q"]
    # 25 compositions, 23 distinct outputs: #P #Q / #Q #P and #H #Q / #Q #H commute (a prefix word and a suffix word)
    assert all(it["candidates"] == 23 and 1 <= it["rank"] <= 23 and it["choice"] == (1.0 if it["rank"] == 1 else 0.0) for it in items)
    # a row scored inside a padded batch equals the row scored alone (the model is causal; the pad sits after the row)
    user = {"role": "user", "content": "Apply #P to: apple pear"}
    rows = [encode_conversation(be.tokenizer, [user, {"role": "assistant", "content": t}]) for t in ("please apple pear", "apple pear thanks and a much longer tail")]
    together = L._answer_logprobs(rows)
    alone = [L._answer_logprobs([r])[0] for r in rows]
    for (x, n), (y, m) in zip(together, alone):
        assert n == m and abs(x - y) < 1e-3


def test_continued_training_records_the_compositions_it_actually_trained():
    from plastic.data.rules import poison_batch, rule_batch
    from plastic.eval.text_learner import TextRuleLearner

    be = _tiny_backend()
    comps = [("#P",), ("#Q",), ("#B",), ("#P", "#Q")]
    stream = rule_batch(comps, episodes=4, n_situations=2, seed=1, split_tag="train", rule_set="decorate", stated=False)
    full = TextRuleLearner(be, mode="continued", sampling="passes", passes=1, lr=1e-3)
    snap = full.snapshot_slow()
    rec = full.consume(stream)
    assert rec["compositions_trained"] == 4 and rec["untrained"] == [] and rec["steps"] == 4 and rec["poisoned_steps"] == 0
    assert any((p.detach().cpu() - snap[n]).abs().max() > 0 for n, p in full._slow_params()), "the update moved W0"
    full.restore_slow(snap)
    rec_p = full.consume(poison_batch(stream, operator="#P"))
    assert rec_p["poisoned_steps"] == 2 and rec_p["compositions_trained"] == 4
    full.restore_slow(snap)
    few = TextRuleLearner(be, mode="continued", sampling="draws", steps=2, lr=1e-3)
    rec_d = few.consume(stream)
    assert rec_d["steps"] == 2 and rec_d["compositions_trained"] + len(rec_d["untrained"]) == 4 and rec_d["sampling"] == "draws"
    full.restore_slow(snap)
