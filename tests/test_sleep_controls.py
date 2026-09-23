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
