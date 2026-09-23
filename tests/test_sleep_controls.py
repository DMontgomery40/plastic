import pytest
"""The matched-controls experiment's pure pieces: probe construction and per-group recall counting."""

from scripts.experiments.sleep_controls import FACTS_BOUNDARY, FACTS_POISON, FACTS_ROLLED, FACTS_TAUGHT, GENERAL, build_probes, group_counts, study_set


def test_fact_lists_are_well_formed_disjoint_and_probe_unseen_phrasings():
    probes = build_probes()
    assert {g: len(ps) for g, ps in probes.items()} == {"taught": 24, "boundary": 2, "rolled": 2, "poison": 4, "general": 7}
    questions = [p.question for ps in probes.values() for p in ps]
    poison_qs = {p.question for p in probes["poison"]}  # a poison probe shares its question with a general control on purpose
    assert all(questions.count(q) == (2 if q in poison_qs else 1) for q in questions)
    for fact in FACTS_TAUGHT + FACTS_BOUNDARY + FACTS_ROLLED + FACTS_POISON:
        stmt, q, a, para, unseen = fact
        assert a.strip() and len({q, para, unseen}) == 3          # three distinct phrasings
        assert a.lower() in stmt.lower()                            # the statement states the answer
        assert all(unseen != u for u, _ in study_set(*fact))        # the unseen phrasing is never taught, not even in the study set
    taught_answers = {a.lower() for _, _, a, _, _ in FACTS_TAUGHT + FACTS_BOUNDARY}
    rolled_answers = {a.lower() for _, _, a, _, _ in FACTS_ROLLED}
    general_answers = {a.lower() for _, a, _ in GENERAL}
    poison_answers = {a.lower() for _, _, a, _, _ in FACTS_POISON}
    assert not (taught_answers & rolled_answers) and not (taught_answers & general_answers) and not (poison_answers & general_answers)
    general_qs = {q for q, _, _ in GENERAL}
    assert all(q in general_qs for _, q, _, _, _ in FACTS_POISON)  # every planted contradiction has a true-answer control


def test_group_counts_attribute_verbatim_and_paraphrase_results_and_ignore_strays():
    probes = build_probes()
    taught = probes["taught"][0]
    rolled = probes["rolled"][0]
    results = [
        {"question": taught.question, "contains": True, "variant": "verbatim"},
        {"question": taught.paraphrase, "contains": False, "variant": "paraphrase"},
        {"question": rolled.question, "contains": False, "variant": "verbatim"},
        {"question": rolled.paraphrase, "contains": True, "variant": "paraphrase"},
        {"question": "who dis?", "contains": True, "variant": "verbatim"},
    ]
    c = group_counts(results, probes)
    assert c["taught"] == {"n": 1, "recalled": 1, "n_paraphrase": 1, "recalled_paraphrase": 0}
    assert c["rolled"] == {"n": 1, "recalled": 0, "n_paraphrase": 1, "recalled_paraphrase": 1}
    assert c["boundary"] == {"n": 0, "recalled": 0, "n_paraphrase": 0, "recalled_paraphrase": 0}
    assert c["general"]["n"] == 0


def test_study_set_teaches_each_fact_several_ways_including_the_question_and_its_paraphrase():
    stmt, q, a, p, _unseen = FACTS_TAUGHT[0]
    items = study_set(stmt, q, a, p)
    users = [u for u, _ in items]
    assert len(items) >= 5 and q in users and p in users and stmt in users
    assert all(a in reply or reply == stmt for _, reply in items)  # every teacher-side reply states the answer (or restates the fact)
    assert len(set(users)) == len(users)


def test_experiment_and_eval_parsers_expose_the_sampling_temperatures():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "scripts.experiments.sleep_controls", "--help"], capture_output=True, text=True, check=True).stdout
    assert "--teach-temperature" in out and "--dream-temperature" in out and "--prompt-loss-weight" in out and "--augment" in out
    assert "--poison" in out and "--facts" in out
    out = subprocess.run([sys.executable, "-m", "scripts.train.eval_ttt_chat", "--help"], capture_output=True, text=True, check=True).stdout
    assert "--temperature" in out and "--top-k" in out and "--skip-nll" in out


def test_arm_configs_gate_the_gated_arms_and_open_the_ungated_control():
    """The ungated control consumes every turn (rolled-back and flagged); the gated arms use the product selection.
    The pinned replay revision reaches each arm; an empty override follows the Hub's main."""
    import argparse

    from plastic.sleep import SMOLTALK_REVISION
    from scripts.experiments.sleep_controls import arm_config

    base = dict(target="w0", steps=5, lr=1e-4, seq_len=256, batch_size=2, replay_ratio=0.5, session_loss="all", prompt_loss_weight=1.0,
                dream_temperature=0.7, dream_token_weighting="uniform", replay_rows=8, heldout_rows=4, device="cpu", max_new_tokens=16, seed=3,
                replay_revision=SMOLTALK_REVISION)
    args = argparse.Namespace(**base)
    for arm in ("anchor", "replay", "distill", "dream"):
        cfg = arm_config(arm, args)
        cfg.validate()
        assert (cfg.method, cfg.provenance, cfg.flagged_policy, cfg.replay_revision) == (arm, "accepted", "exclude", SMOLTALK_REVISION)
    cfg = arm_config("ungated", args)
    cfg.validate()
    assert (cfg.method, cfg.provenance, cfg.flagged_policy) == ("replay", "all", "include")
    assert arm_config("replay", argparse.Namespace(**{**base, "replay_revision": ""})).replay_revision is None


def test_ceiling_statements_teach_everything_or_only_the_probed_fact():
    """The ceiling arm's two modes: the protocol teaches every fact before each probe (in-context recall under session
    load); single mode teaches only the probed fact (can the model use one taught fact at all). Unknown probes fail loudly."""
    from plastic.sleep.recall import RecallProbe
    from scripts.experiments.sleep_controls import FACTS_BOUNDARY, FACTS_TAUGHT, ceiling_statements

    facts = FACTS_TAUGHT[:3] + FACTS_BOUNDARY[:1]
    probe = RecallProbe(FACTS_TAUGHT[1][1], FACTS_TAUGHT[1][2], FACTS_TAUGHT[1][4])
    assert ceiling_statements(facts, probe, "all") == facts
    assert ceiling_statements(facts, probe, "single") == [FACTS_TAUGHT[1]]
    boundary_probe = RecallProbe(FACTS_BOUNDARY[0][1], FACTS_BOUNDARY[0][2], FACTS_BOUNDARY[0][4])
    assert ceiling_statements(facts, boundary_probe, "single") == [FACTS_BOUNDARY[0]]
    with pytest.raises(ValueError, match="no taught statement"):
        ceiling_statements(facts, RecallProbe("Who?", "nobody"), "single")
    with pytest.raises(ValueError, match="unknown ceiling mode"):
        ceiling_statements(facts, probe, "some")
