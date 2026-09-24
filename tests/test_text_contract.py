"""The text contract on a fake learner: fixed-seed measurements repeat exactly, a lasting update shows up on held-out
compositions and vanishes on revert, the poisoned stream is the false-definition stream, and acceptance is a pair."""

from __future__ import annotations

import copy

from plastic.data.rules import OPERATORS, RuleBatch
from plastic.eval.text_contract import TextContractSpec, measure, run_text_contract
from plastic.eval.text_contract import split_pairs


class FakeLearner:
    """Knows a set of compositions "well" (low nll, exact) and everything else badly; ``consume`` learns the stream's
    compositions when accepting; with ``adapt`` the fast path halves the nll from the second situation on."""

    def __init__(self, *, accept_poison: bool):
        self.known: set[tuple[str, ...]] = set()
        self.accept_poison = accept_poison
        self.consumed: list[RuleBatch] = []

    def snapshot_slow(self):
        return copy.deepcopy(self.known)

    def restore_slow(self, snapshot):
        self.known = copy.deepcopy(snapshot)

    def consume(self, stream: RuleBatch):
        self.consumed.append(stream)
        if stream.poisoned and not self.accept_poison:
            return {"accepted": False, "reason": "verify failed"}
        self.known |= {e.ops for e in stream.episodes}
        return {"accepted": True}

    def score(self, batch: RuleBatch, *, adapt: bool):
        out = []
        for e in batch.episodes:
            base = 0.5 if e.ops in self.known else 4.0
            row = []
            for i, s in enumerate(e.situations):
                nll = base * (0.5 if (adapt and i >= 1) else 1.0)
                row.append({"exact": 1.0 if nll < 1.0 else 0.0, "nll": nll, "tokens": len(s.answer)})
            out.append(row)
        return out


def test_measure_is_deterministic_and_tracks_the_split():
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=4)
    split = split_pairs(n_heldout=8, seed=0)
    L = FakeLearner(accept_poison=True)
    m1, m2 = measure(L, spec, split, 0), measure(L, spec, split, 0)
    assert m1 == m2
    assert set(m1["transfer"]) == {" ".join(c) for c in split[1]} and m1["transfer_mean"]["adapt"]["situations"] == 8 * spec.situations_per_episode
    assert m1["speed"]["curve"][0] == 1.0 and all(v == 0.5 for v in m1["speed"]["curve"][1:]) and m1["speed"]["situations_to_half"] == spec.probe_situations
    assert m1["forgetting"]["adapt"]["situations"] == len(split[0]) * spec.situations_per_episode


def test_contract_shows_gain_revert_and_the_poison_decision():
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6)
    # a learner that "learns" the stream's compositions; the stream only contains TRAINING compositions, so held-out
    # transfer must not move, while forgetting (training compositions) improves
    L = FakeLearner(accept_poison=False)
    r = run_text_contract(L, spec, seed=0)
    assert r["contract"] == "text_rules" and r["revert"]["ok"]
    assert r["transfer"]["delta_nll"] == 0.0 and r["forgetting"]["delta_nll"] < 0
    assert [d["stream"] for d in r["decisions"]] == ["clean", "poisoned", "poisoned_sequential", "corrective", "format_only"]
    assert r["decisions"][1]["accepted"] is False and r["decisions"][2]["accepted"] is False
    assert r["acceptance"] == {"accepted_good": 1.0, "refused_bad": 2 / 3, "n_good": 2, "n_bad": 3}   # the fake accepts the format-only stream
    assert r["decisions"][4]["stream"] == "format_only" and L.consumed[5].format_only
    assert r["format_only"]["true_stream_gain_exact"] == 0.0 and "learned format" in r["format_only"]["note"]
    # consume order: clean, the clean-again control, the poison on top of the clean state, the poison from the snapshot,
    # the corrective stream, the format-only control
    assert L.consumed[1] is L.consumed[0]
    assert L.consumed[2].poisoned and L.consumed[2].poisoned_operator == spec.poison_operator and L.consumed[3].poisoned
    assert any(e.poisoned for e in L.consumed[2].episodes) and not L.consumed[0].poisoned
    assert r["sequential_poison"]["accepted"] is False and r["sequential_poison"]["harm_nll"] == 0.0
    assert r["compute"]["situations_consumed"] == 6 * L.consumed[0].situations
    # the learner ends at its pre-stream state
    assert L.known == set()


def test_contract_records_an_accepted_poison_as_refused_bad_zero():
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6)
    r = run_text_contract(FakeLearner(accept_poison=True), spec, seed=1)
    assert r["acceptance"]["refused_bad"] == 0.0 and r["acceptance"]["accepted_good"] == 1.0 and r["acceptance"]["n_bad"] == 3
    assert set(r["split"]["heldout"][0]) <= set(OPERATORS)
    assert r["sequential_poison"]["accepted"] is True


def test_spec_carries_unstated_rules_and_the_poison_kind_into_every_batch():
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6, rule_set="decorate", poison_operator="#P",
                            stated_rules=False, poison_kind="inconsistent")
    L = FakeLearner(accept_poison=False)
    r = run_text_contract(L, spec, seed=0)
    assert r["stream"]["stated_rules"] is False and r["stream"]["poison_kind"] == "inconsistent"
    assert all(not b.stated for b in L.consumed) and all(not e.stated for b in L.consumed for e in b.episodes)
    assert L.consumed[2].poison_kind == "inconsistent" and L.consumed[2].poisoned and L.consumed[0].poison_kind is None  # [1] is the clean-again control
    split = split_pairs(n_heldout=8, seed=0, rule_set="decorate", min_reversed_heldout=4)
    m = measure(L, spec, split, 0)
    assert m["transfer_mean"]["adapt"]["situations"] == 8 * spec.situations_per_episode
    stated = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6, rule_set="decorate", poison_operator="#P")
    r2 = run_text_contract(FakeLearner(accept_poison=False), stated, seed=0)
    assert r2["stream"]["stated_rules"] is True and r2["stream"]["poison_kind"] == "consistent"


def test_the_learner_held_in_material_must_be_the_contract_split():
    """The report script once built the learner's held-in material from a split with a different reversed-pair minimum
    than the contract's, so the verifier scored six held-out compositions (fresh inputs, read-only) as held-in
    material (FABLE-41B-211). The contract now owns the split and refuses a learner whose material differs."""
    import pytest
    from plastic.eval.text_contract import contract_split

    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6, rule_set="decorate", poison_operator="#P", min_reversed_heldout=4)
    train, held = contract_split(spec)
    other_train, other_held = split_pairs(n_heldout=8, seed=0, rule_set="decorate", min_reversed_heldout=6)
    assert held != other_held and set(other_train) & set(held), "the two minima give different splits, which is what made the leak possible"
    L = FakeLearner(accept_poison=False)
    L.train_compositions = other_train
    with pytest.raises(ValueError):
        run_text_contract(L, spec, seed=0)
    L.train_compositions = train
    r = run_text_contract(L, spec, seed=0)
    assert [tuple(c) for c in r["split"]["train"]] == train and r["spec"]["min_reversed_heldout"] == 4


def test_sequential_arm_can_be_switched_off_and_the_shape_is_the_t1_one():
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6, sequential_poison=False, sequential_clean=False)
    L = FakeLearner(accept_poison=False)
    r = run_text_contract(L, spec, seed=0)
    assert [d["stream"] for d in r["decisions"]] == ["clean", "poisoned", "corrective", "format_only"]
    assert r["sequential_poison"] is None and r["consume_records"]["poisoned_sequential"] is None
    assert r["acceptance"]["n_bad"] == 2 and r["compute"]["situations_consumed"] == 4 * L.consumed[0].situations


def test_transfer_reports_the_first_situation_separately_from_the_rest():
    """FABLE-202's measurement prior: the first situation has no example before it, so its score is what the slow
    parameters carry; averaging it with the later situations blends adaptation into 'transfer'."""
    from plastic.eval.text_contract import _summarize

    scores = [[{"exact": 0.0, "nll": 4.0, "tokens": 3}, {"exact": 1.0, "nll": 0.5, "tokens": 3}, {"exact": 1.0, "nll": 0.4, "tokens": 3}],
              [{"exact": 0.0, "nll": 3.0, "tokens": 3}, {"exact": 0.0, "nll": 1.0, "tokens": 3}, {"exact": 1.0, "nll": 0.2, "tokens": 3}]]
    m = _summarize(scores)
    assert m["exact"] == 0.5 and m["exact_first"] == 0.0 and m["exact_after_first"] == 0.75
    assert [round(b["exact"], 2) for b in m["by_situation"]] == [0.0, 0.5, 1.0] and m["by_situation"][0]["n"] == 2
    assert m["nll_after_first"] == (0.5 + 0.4 + 1.0 + 0.2) / 4
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=4)
    r = run_text_contract(FakeLearner(accept_poison=False), spec, seed=0)
    assert "delta_exact_first" in r["transfer"] and "delta_exact_after_first" in r["transfer"]


def test_report_script_arguments_become_the_contract_spec_and_its_defaults_match_the_archive_split():
    """The report script once carried its own split parameters beside the contract's (FABLE-41B-211). Its command line
    now becomes one spec, and the defaults reproduce the archived seed-0 decorate split."""
    from plastic.eval.text_contract import contract_split
    from scripts.experiments.text_contract_report import build_parser, spec_from_args

    args = build_parser().parse_args(["--checkpoint", "ck", "--out", "o"])
    spec = spec_from_args(args)
    assert spec.min_reversed_heldout == 6 and spec.stated_rules and spec.poison_kind == "consistent" and spec.sequential_poison
    assert args.verify_episodes == 34 and args.verifier == "v2" and args.sampling == "passes" and args.passes == 1
    assert spec.sequential_clean and spec.choice_episodes_per_composition == 2
    # the archived reports' settings, as the script's docstring names them
    old = build_parser().parse_args(["--checkpoint", "ck", "--out", "o", "--sampling", "draws", "--steps", "20", "--verify-episodes", "6",
                                     "--no-sequential-clean", "--choice-episodes", "0"])
    old_spec = spec_from_args(old)
    assert old.sampling == "draws" and old.steps == 20 and old.verify_episodes == 6
    assert not old_spec.sequential_clean and old_spec.choice_episodes_per_composition == 0 and contract_split(old_spec) == contract_split(spec)
    assert [" ".join(c) for c in contract_split(spec)[1]] == ["#B #H", "#B #P", "#B #Q", "#H #Q", "#P #B", "#P #H", "#Q #P", "#W #B"]
    args = build_parser().parse_args(["--checkpoint", "ck", "--out", "o", "--unstated-rules", "--poison-kind", "inconsistent",
                                      "--no-sequential-poison", "--verify-episodes", "18", "--verifier", "v1", "--min-reversed-heldout", "4"])
    spec = spec_from_args(args)
    assert not spec.stated_rules and spec.poison_kind == "inconsistent" and not spec.sequential_poison and spec.min_reversed_heldout == 4
    assert args.verify_episodes == 18 and args.verifier == "v1"
    assert contract_split(spec)[1] != contract_split(spec_from_args(build_parser().parse_args(["--checkpoint", "ck", "--out", "o"])))[1]


class RecordingLearner(FakeLearner):
    """Records the slow state at the start of every consume, so the arms' starting points can be checked."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.states: list[set] = []

    def consume(self, stream: RuleBatch):
        self.states.append(copy.deepcopy(self.known))
        rec = super().consume(stream)
        if rec["accepted"]:
            self.known.add(("consumed", len(self.states)))  # every accepted consume leaves a distinct mark, so a missing restore shows
        return rec


def test_the_sequential_poison_has_a_clean_again_control_outside_the_acceptance_pair():
    """The poisoned stream is the clean stream with one operator's answers changed, so a refusal on the sequential arm is
    attributable to the poison only against the same clean stream consumed again from the same post-clean state
    (OPUS-LEAD-001). The control starts where the sequential poison starts, and neither changes the acceptance pair."""
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6)
    L = RecordingLearner(accept_poison=False)
    r = run_text_contract(L, spec, seed=0)
    assert L.consumed[1] is L.consumed[0] and not L.consumed[1].poisoned and L.consumed[2].poisoned
    assert L.states[1] == L.states[2], "the clean-again control and the sequential poison start from the same post-clean state"
    assert L.states[1] != L.states[0]
    assert r["sequential_control"]["accepted"] is True and r["consume_records"]["clean_again"]["accepted"] is True
    assert [d["stream"] for d in r["decisions"]] == ["clean", "poisoned", "poisoned_sequential", "corrective", "format_only"]
    assert r["acceptance"]["n_good"] == 2 and r["acceptance"]["n_bad"] == 3
    off = run_text_contract(FakeLearner(accept_poison=False), TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6, sequential_clean=False), seed=0)
    assert off["sequential_control"] is None and off["consume_records"]["clean_again"] is None


class ChoiceLearner(FakeLearner):
    """Adds a first-situation choice: a known composition ranks its correct output first, an unknown one does not."""

    def choice(self, batch: RuleBatch):
        out = []
        for e in batch.episodes:
            known = e.ops in self.known
            out.append({"ops": " ".join(e.ops), "choice": 1.0 if known else 0.0, "margin": 1.5 if known else -2.0, "rank": 1 if known else 7,
                        "correct_logp": -3.0 if known else -9.0, "candidates": 23, "top": " ".join(e.ops)})
        return out


def test_choice_is_measured_on_training_and_held_out_compositions_where_the_questions_are_decided():
    spec = TextContractSpec(n_heldout=8, eval_episodes_per_composition=1, stream_episodes=6, choice_episodes_per_composition=2, rule_set="decorate", poison_operator="#P")
    L = ChoiceLearner(accept_poison=False)
    r = run_text_contract(L, spec, seed=0)
    ch = r["choice"]
    assert ch["before"]["train"]["accuracy"] == 0.0 and ch["before"]["heldout"]["accuracy"] == 0.0
    # the fake learns the six streamed compositions: training choice rises only for those, held-out stays at zero
    trained = {" ".join(e.ops) for e in L.consumed[0].episodes}
    after = ch["after"]["train"]["by_composition"]
    assert all(after[k]["accuracy"] == (1.0 if k in trained else 0.0) for k in after)
    assert ch["after"]["heldout"]["accuracy"] == 0.0 and ch["after"]["train"]["n"] == 2 * len(r["split"]["train"])
    assert abs(ch["before"]["train"]["chance_mean"] - 1 / 23) < 1e-12 and len(ch["before"]["train"]["items"]) == ch["before"]["train"]["n"]
    # the choice score costs a forward pass per candidate, so it rides only on the named measurements; the revert check
    # stays prediction-level through the per-item exact and nll records
    assert ch["reverted"] is None and ch["after_correction"] is None and ch["after_poison"]["train"]["items"] and ch["after_format"]["train"]["items"] and r["revert"]["ok"]
    assert ch["paired"]["after_vs_before"]["train"]["n"] == 2 * len(r["split"]["train"]) and ch["paired"]["after_vs_before"]["heldout_novel"]["mean_change"] == 0.0
    assert ch["at"] == list(spec.choice_at) and ch["after_clean_again"]["train"]["items"]
    assert r["sequential_control"]["choice_train_delta"] == 0.0 and r["sequential_poison"]["choice_heldout_delta"] == 0.0
    # per-item records ride along with every measurement
    assert len(r["items"]["after"]["heldout"]) == len(r["split"]["heldout"]) and len(r["items"]["after"]["heldout"][0]["exact"]) == spec.situations_per_episode


def test_revert_check_sees_a_single_moved_choice_item():
    """The revert check is prediction-level: one item's changed margin is a gap even when the means would hide it."""
    from plastic.eval.contract import _max_gap
    from plastic.eval.text_contract import choice_summary

    a = [{"ops": "#P", "choice": 1.0, "margin": 1.0, "rank": 1, "correct_logp": -3.0, "candidates": 25, "top": "#P"},
         {"ops": "#Q", "choice": 0.0, "margin": -1.0, "rank": 2, "correct_logp": -5.0, "candidates": 25, "top": "#P"}]
    b = [dict(a[0], margin=1.2), dict(a[1], margin=-1.2)]
    sa, sb = choice_summary(a), choice_summary(b)
    assert sa["margin_mean"] == sb["margin_mean"]
    assert _max_gap({"c": sa}, {"c": sb}) > 0.1


def test_held_out_pairs_that_repeat_a_trained_answer_are_named():
    """#P #Q and #Q #P write the same answer on every input (a prefix word and a suffix word commute), as do #H #Q and
    #Q #H; the archive split holds out #H #Q and #Q #P while training their twins, so two of its eight held-out pairs are
    not new behaviour (OPUS-LEAD-002). The contract names them and keeps a novel-only reading."""
    from plastic.eval.text_contract import contract_split, output_duplicates

    spec = TextContractSpec(n_heldout=8, min_reversed_heldout=6, split_seed=0, rule_set="decorate", poison_operator="#P", stream_episodes=6)
    train, held = contract_split(spec)
    assert output_duplicates(train, held, "decorate") == {"#H #Q": "#Q #H", "#Q #P": "#P #Q"}
    r = run_text_contract(ChoiceLearner(accept_poison=False), TextContractSpec(**{**spec.__dict__, "eval_episodes_per_composition": 1}), seed=0)
    assert r["split"]["heldout_output_duplicates"] == {"#H #Q": "#Q #H", "#Q #P": "#P #Q"} and len(r["split"]["heldout_novel"]) == 6
    assert set(r["choice"]["heldout_novel_accuracy"]) == {"before", "after", "after_clean_again", "after_poison_sequential"}
    assert output_duplicates([("#R", "#S")], [("#S", "#R")], "transform") == {}


def test_paired_margins_pair_items_by_position_and_refuse_mismatched_material():
    import pytest
    from plastic.eval.text_contract import choice_summary, paired_margins

    it = lambda ops, m: {"ops": ops, "choice": 1.0 if m > 0 else 0.0, "margin": m, "rank": 1 if m > 0 else 2, "correct_logp": -3.0, "candidates": 23, "top": ops}
    a = {"train": choice_summary([it("#P", -1.0), it("#Q", -2.0)]), "heldout": choice_summary([it("#B #H", -3.0), it("#H #Q", -1.0)])}
    b = {"train": choice_summary([it("#P", 0.5), it("#Q", -2.5)]), "heldout": choice_summary([it("#B #H", -2.0), it("#H #Q", -1.0)])}
    groups = {"train": {"#P", "#Q"}, "heldout_novel": {"#B #H"}, "heldout_duplicate": {"#H #Q"}}
    p = paired_margins(a, b, groups)
    assert p["train"] == {"mean_change": 0.5, "improved": 1, "worsened": 1, "n": 2}
    assert p["heldout_novel"] == {"mean_change": 1.0, "improved": 1, "worsened": 0, "n": 1}
    assert p["heldout_duplicate"] == {"mean_change": 0.0, "improved": 0, "worsened": 0, "n": 1}
    assert paired_margins(a, None, groups) is None
    swapped = {"train": choice_summary([it("#Q", 0.5), it("#P", -2.5)]), "heldout": b["heldout"]}
    with pytest.raises(ValueError):
        paired_margins(a, swapped, groups)


def test_the_name_permuted_content_null_is_an_optional_bad_stream_with_its_own_choice_reading():
    spec = TextContractSpec(n_heldout=8, min_reversed_heldout=6, eval_episodes_per_composition=1, stream_episodes=17, rule_set="decorate",
                            poison_operator="#P", stated_rules=False, name_permuted_control=True)
    L = ChoiceLearner(accept_poison=False)
    r = run_text_contract(L, spec, seed=0)
    assert r["decisions"][-1]["stream"] == "name_permuted" and r["acceptance"]["n_bad"] == 4
    assert L.consumed[-1].name_map and r["name_permuted"]["map"] and r["consume_records"]["name_permuted"]["accepted"] is True
    assert r["choice"]["after_names"]["train"]["items"] and r["choice"]["paired"]["names_vs_before"]["train"]["n"] == 2 * 17
    off = run_text_contract(ChoiceLearner(accept_poison=False), TextContractSpec(**{**spec.__dict__, "name_permuted_control": False}), seed=0)
    assert off["name_permuted"] is None and off["acceptance"]["n_bad"] == 3
