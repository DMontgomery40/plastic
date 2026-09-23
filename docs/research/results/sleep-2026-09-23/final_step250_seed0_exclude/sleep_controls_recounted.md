# Seed 0 on the final checkpoint, recounted

The table in `sleep_controls.md` was written by the code running at launch (`ccfa446+dirty`), whose group attribution keyed probes by question alone, so the four planted contradictions that reuse general-knowledge questions were counted under `general` (ASTRA-195). This table recounts every arm's saved per-probe results with the corrected attribution (`group_counts` from 011e696). The selected-turn column is derived here from the saved harvest fields (30 accepted teaching turns; the ungated control also selects the 2 completed rolled-back turns); the running code did not record `selected_turns` yet. Seed 0's gated arms ran under the product rule `flagged_policy=exclude`.

| Arm | selected text turns (accepted online / flagged / excluded) | taught (p = unseen phrasing) | boundary | rolled (contamination) | poison (uptake) | general (locality) | held-out NLL mean → | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| floor | n/a | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 1/4) | 3/7 (p 3/7) |  |  |
| ceiling | n/a | 3/24 (p 2/24) | 0/2 (p 0/2) | 0/0 (p 0/0) | 0/0 (p 0/0) | 0/0 (p 0/0) |  |  |
| anchor | 1 (30 / 29 / 29) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 2/4 (p 1/4) | 3/7 (p 3/7) | 1.612 → 1.603 | accepted |
| replay | 1 (30 / 29 / 29) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 1/4) | 3/7 (p 4/7) | 1.612 → 1.466 | accepted |
| distill | 1 (30 / 29 / 29) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 1/4) | 1/7 (p 4/7) | 1.612 → 1.467 | accepted |
| dream | 1 (30 / 29 / 29) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 0/4) | 3/7 (p 4/7) | 1.612 → 1.460 | rejected |
| ungated | 32 (30 / 29 / 0) | 0/24 (p 1/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 1/4) | 3/7 (p 4/7) | 1.612 → 1.483 | accepted |

Floor baseline hits: the fresh model already answers the France and Earth general probes and, by containment, one planted contradiction (its reply to the Moon/Earth question contains the word 'Moon'), so poison 1/4 at the floor is the baseline, not uptake.
