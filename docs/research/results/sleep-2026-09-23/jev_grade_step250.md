# Jev-graded recount of every archived recall reply (final checkpoint runs)

Grader: typesafe.ai Jev, two yes/no judgments per reply (`plastic/sleep/grade.py`): *asserts* (the reply answers the
question by asserting the expected answer; mentioning the word, listing it among alternatives or a garbled sentence
that contains it does not count) and *contradicts*. Verdict `asserts` when P(asserts) >= 0.5 and P(contradicts) < 0.5.
Reply text was sent to the typesafe.ai API. 1,664 replies from the five final-checkpoint runs, 712,002 tokens, 141 s,
0 errors. Containment is the archive's original score (`recall.score_reply`).

Overall: both hit 133; containment only 66 (containment hits Jev rejects); Jev only 18; unclear 63; both miss 1,384.

What the disagreements are. Containment-only hits are mostly wrong-unit or garbled general answers ("water boils at
100°F (38°C)" for expected 100 °C; "The Moon is the largest object in the universe" for expected Earth), the
containment artifacts already named in the notes ("the birds and the bees", the four-season list), and, in the
single-fact ceiling, replies that use the taught word as a topic without answering ("Teal is a beautiful color…").
Jev-only hits are mostly bare affirmations ("That's right.", "That's correct.") that Jev credited as asserting the
expected reagent: a grader error in the model's favour, so Jev-only counts are not added to any recall number. Two
Jev-only hits are real answers containment missed ("2 + 2 = 4" for expected "four"; a list whose first item is
typewriters).

Groups: `taught` here includes the two boundary facts (26 probes per variant), `general` the locality controls, `poison` the planted contradictions. Ten of the 18 Jev-only rows are bare affirmations.

Reading for the study's conclusions: no taught fact is asserted by any sleep child under either grader (both graders
agree on 0/24 in every sleep arm); the single-fact ceiling falls from 23/26 verbatim and 19/26 unseen phrasing (containment, taught + boundary) to 16/26 and 13/26 under Jev, because replies that use the taught word as a topic without answering are not credited; the general-knowledge locality counts were inflated by wrong-unit answers. The archive's tables
keep containment as recorded; this file is the semantic recount beside them.

| Run | Arm | Group | Variant | n | containment hits | Jev asserts | both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed0_ceiling_single | ceiling | taught | paraphrase | 26 | 19 | 13 | 12 |
| seed0_ceiling_single | ceiling | taught | verbatim | 26 | 23 | 16 | 16 |
| seed0_exclude | anchor | general | paraphrase | 7 | 3 | 3 | 2 |
| seed0_exclude | anchor | general | verbatim | 7 | 3 | 1 | 1 |
| seed0_exclude | anchor | poison | paraphrase | 4 | 1 | 0 | 0 |
| seed0_exclude | anchor | poison | verbatim | 4 | 2 | 2 | 2 |
| seed0_exclude | anchor | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed0_exclude | anchor | taught | verbatim | 28 | 0 | 0 | 0 |
| seed0_exclude | ceiling | taught | paraphrase | 26 | 2 | 2 | 2 |
| seed0_exclude | ceiling | taught | verbatim | 26 | 3 | 3 | 3 |
| seed0_exclude | distill | general | paraphrase | 7 | 4 | 2 | 2 |
| seed0_exclude | distill | general | verbatim | 7 | 1 | 1 | 1 |
| seed0_exclude | distill | poison | paraphrase | 4 | 1 | 0 | 0 |
| seed0_exclude | distill | poison | verbatim | 4 | 1 | 1 | 1 |
| seed0_exclude | distill | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed0_exclude | distill | taught | verbatim | 28 | 0 | 1 | 0 |
| seed0_exclude | dream | general | paraphrase | 7 | 4 | 2 | 2 |
| seed0_exclude | dream | general | verbatim | 7 | 3 | 1 | 1 |
| seed0_exclude | dream | poison | verbatim | 4 | 1 | 1 | 1 |
| seed0_exclude | dream | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed0_exclude | dream | taught | verbatim | 28 | 0 | 1 | 0 |
| seed0_exclude | floor | general | paraphrase | 7 | 3 | 3 | 3 |
| seed0_exclude | floor | general | verbatim | 7 | 3 | 1 | 1 |
| seed0_exclude | floor | poison | paraphrase | 4 | 1 | 0 | 0 |
| seed0_exclude | floor | poison | verbatim | 4 | 1 | 1 | 1 |
| seed0_exclude | floor | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed0_exclude | floor | taught | verbatim | 28 | 0 | 0 | 0 |
| seed0_exclude | replay | general | paraphrase | 7 | 4 | 2 | 2 |
| seed0_exclude | replay | general | verbatim | 7 | 3 | 2 | 2 |
| seed0_exclude | replay | poison | paraphrase | 4 | 1 | 0 | 0 |
| seed0_exclude | replay | poison | verbatim | 4 | 1 | 1 | 1 |
| seed0_exclude | replay | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed0_exclude | replay | taught | verbatim | 28 | 0 | 0 | 0 |
| seed0_exclude | ungated | general | paraphrase | 7 | 4 | 3 | 3 |
| seed0_exclude | ungated | general | verbatim | 7 | 3 | 1 | 1 |
| seed0_exclude | ungated | poison | paraphrase | 4 | 1 | 0 | 0 |
| seed0_exclude | ungated | poison | verbatim | 4 | 1 | 1 | 1 |
| seed0_exclude | ungated | taught | paraphrase | 28 | 1 | 1 | 0 |
| seed0_exclude | ungated | taught | verbatim | 28 | 0 | 2 | 0 |
| seed0_include | distill | general | paraphrase | 7 | 4 | 3 | 3 |
| seed0_include | distill | general | verbatim | 7 | 3 | 2 | 2 |
| seed0_include | distill | poison | verbatim | 4 | 1 | 1 | 1 |
| seed0_include | distill | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed0_include | distill | taught | verbatim | 28 | 0 | 2 | 0 |
| seed0_include | dream | general | paraphrase | 7 | 4 | 2 | 2 |
| seed0_include | dream | general | verbatim | 7 | 3 | 1 | 1 |
| seed0_include | dream | poison | verbatim | 4 | 1 | 1 | 1 |
| seed0_include | dream | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed0_include | dream | taught | verbatim | 28 | 0 | 1 | 0 |
| seed0_include | replay | general | paraphrase | 7 | 4 | 4 | 4 |
| seed0_include | replay | general | verbatim | 7 | 2 | 1 | 1 |
| seed0_include | replay | poison | paraphrase | 4 | 1 | 0 | 0 |
| seed0_include | replay | poison | verbatim | 4 | 1 | 1 | 1 |
| seed0_include | replay | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed0_include | replay | taught | verbatim | 28 | 0 | 2 | 0 |
| seed1_include | anchor | general | paraphrase | 7 | 2 | 1 | 1 |
| seed1_include | anchor | general | verbatim | 7 | 3 | 2 | 2 |
| seed1_include | anchor | poison | paraphrase | 4 | 1 | 1 | 1 |
| seed1_include | anchor | poison | verbatim | 4 | 1 | 1 | 1 |
| seed1_include | anchor | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed1_include | anchor | taught | verbatim | 28 | 0 | 0 | 0 |
| seed1_include | distill | general | paraphrase | 7 | 4 | 4 | 4 |
| seed1_include | distill | general | verbatim | 7 | 3 | 4 | 3 |
| seed1_include | distill | poison | paraphrase | 4 | 0 | 1 | 0 |
| seed1_include | distill | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed1_include | distill | taught | verbatim | 28 | 0 | 0 | 0 |
| seed1_include | floor | general | paraphrase | 7 | 3 | 3 | 3 |
| seed1_include | floor | general | verbatim | 7 | 3 | 1 | 1 |
| seed1_include | floor | poison | paraphrase | 4 | 1 | 0 | 0 |
| seed1_include | floor | poison | verbatim | 4 | 1 | 1 | 1 |
| seed1_include | floor | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed1_include | floor | taught | verbatim | 28 | 0 | 0 | 0 |
| seed1_include | replay | general | paraphrase | 7 | 5 | 5 | 4 |
| seed1_include | replay | poison | verbatim | 4 | 1 | 1 | 1 |
| seed1_include | replay | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed1_include | replay | taught | verbatim | 28 | 0 | 0 | 0 |
| seed1_include | ungated | general | paraphrase | 7 | 5 | 5 | 4 |
| seed1_include | ungated | poison | verbatim | 4 | 1 | 1 | 1 |
| seed1_include | ungated | taught | paraphrase | 28 | 1 | 0 | 0 |
| seed1_include | ungated | taught | verbatim | 28 | 0 | 0 | 0 |
| seed2_include | anchor | general | paraphrase | 7 | 3 | 2 | 2 |
| seed2_include | anchor | general | verbatim | 7 | 3 | 2 | 2 |
| seed2_include | anchor | poison | paraphrase | 4 | 2 | 1 | 1 |
| seed2_include | anchor | poison | verbatim | 4 | 2 | 2 | 2 |
| seed2_include | anchor | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed2_include | anchor | taught | verbatim | 28 | 0 | 0 | 0 |
| seed2_include | distill | general | paraphrase | 7 | 4 | 2 | 2 |
| seed2_include | distill | general | verbatim | 7 | 2 | 0 | 0 |
| seed2_include | distill | poison | verbatim | 4 | 1 | 1 | 1 |
| seed2_include | distill | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed2_include | distill | taught | verbatim | 28 | 0 | 0 | 0 |
| seed2_include | dream | general | paraphrase | 7 | 3 | 1 | 1 |
| seed2_include | dream | general | verbatim | 7 | 3 | 1 | 1 |
| seed2_include | dream | poison | verbatim | 4 | 1 | 1 | 1 |
| seed2_include | dream | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed2_include | dream | taught | verbatim | 28 | 0 | 0 | 0 |
| seed2_include | floor | general | paraphrase | 7 | 3 | 3 | 3 |
| seed2_include | floor | general | verbatim | 7 | 3 | 1 | 1 |
| seed2_include | floor | poison | paraphrase | 4 | 1 | 0 | 0 |
| seed2_include | floor | poison | verbatim | 4 | 1 | 1 | 1 |
| seed2_include | floor | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed2_include | floor | taught | verbatim | 28 | 0 | 0 | 0 |
| seed2_include | replay | general | paraphrase | 7 | 4 | 4 | 3 |
| seed2_include | replay | general | verbatim | 7 | 2 | 1 | 1 |
| seed2_include | replay | poison | paraphrase | 4 | 2 | 1 | 1 |
| seed2_include | replay | poison | verbatim | 4 | 1 | 1 | 1 |
| seed2_include | replay | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed2_include | replay | taught | verbatim | 28 | 0 | 0 | 0 |
| seed2_include | ungated | general | paraphrase | 7 | 4 | 4 | 3 |
| seed2_include | ungated | general | verbatim | 7 | 2 | 1 | 1 |
| seed2_include | ungated | poison | verbatim | 4 | 1 | 1 | 1 |
| seed2_include | ungated | taught | paraphrase | 28 | 0 | 0 | 0 |
| seed2_include | ungated | taught | verbatim | 28 | 0 | 0 | 0 |
