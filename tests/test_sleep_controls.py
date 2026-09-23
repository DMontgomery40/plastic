"""The matched-controls experiment's pure pieces: probe construction and per-group recall counting."""

from scripts.experiments.sleep_controls import FACTS_BOUNDARY, FACTS_ROLLED, FACTS_TAUGHT, GENERAL, build_probes, group_counts, study_set


def test_fact_lists_are_well_formed_and_disjoint():
    probes = build_probes()
    assert {len(probes[g]) for g in ("taught", "boundary", "rolled", "general")} == {6, 2, 2, 5}
    questions = [p.question for ps in probes.values() for p in ps]
    assert len(set(questions)) == len(questions)  # a question belongs to one group only
    for ps in probes.values():
        for p in ps:
            assert p.answer.strip() and p.question.strip() and p.paraphrase and p.paraphrase != p.question
    taught_answers = {a.lower() for _, _, a, _ in FACTS_TAUGHT + FACTS_BOUNDARY}
    rolled_answers = {a.lower() for _, _, a, _ in FACTS_ROLLED}
    general_answers = {a.lower() for _, a, _ in GENERAL}
    assert not (taught_answers & rolled_answers) and not (taught_answers & general_answers)  # contamination and locality stay separable


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
    stmt, q, a, p = FACTS_TAUGHT[0]
    items = study_set(stmt, q, a, p)
    users = [u for u, _ in items]
    assert len(items) >= 5 and q in users and p in users and stmt in users
    assert all(a in reply or reply == stmt for _, reply in items)  # every teacher-side reply states the answer (or restates the fact)
    assert len(set(users)) == len(users)
