# Sleep with matched controls

Code b2d902d, checkpoint `29e0f8558d15`, device mps, 20 steps, target w0, lr 0.0001, replay ratio 0.5, session loss all, plw 1.0, augment none, 4967.1 s.

| Arm | turns consumed (accepted / flagged / excluded) | taught (p = unseen phrasing) | boundary | rolled (contamination) | poison (uptake) | general (locality) | held-out NLL mean → | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| floor | n/a | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 1/4) | 3/7 (p 3/7) |  |  |
| anchor | 30 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 2/4 (p 2/4) | 3/7 (p 3/7) | 1.637 → 1.617 | accepted |
| replay | 30 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 2/4) | 2/7 (p 4/7) | 1.637 → 1.492 | accepted |
| distill | 30 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 0/4) | 2/7 (p 4/7) | 1.637 → 1.493 | accepted |
| dream | 30 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 0/4) | 3/7 (p 3/7) | 1.637 → 1.499 | accepted |
| ungated | 32 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 0/4) | 2/7 (p 4/7) | 1.637 → 1.499 | accepted |
