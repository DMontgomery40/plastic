# Sleep with matched controls

Code 469211f, checkpoint `29e0f8558d15`, device mps, 20 steps, target w0, lr 0.0001, replay ratio 0.5, session loss all, plw 1.0, augment none, 5022.3 s.

| Arm | turns consumed (accepted / flagged / excluded) | taught (p = unseen phrasing) | boundary | rolled (contamination) | poison (uptake) | general (locality) | held-out NLL mean → | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| floor | n/a | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 1/4) | 3/7 (p 3/7) |  |  |
| anchor | 30 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 1/4) | 3/7 (p 2/7) | 1.463 → 1.444 | accepted |
| replay | 30 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 0/4) | 0/7 (p 5/7) | 1.463 → 1.341 | accepted |
| distill | 30 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 0/4 (p 0/4) | 3/7 (p 4/7) | 1.463 → 1.372 | accepted |
| dream | 30 selected (30 accepted online, 29 flagged, 0 excluded) | n/a | n/a | n/a | n/a | n/a |  | rejected |
| ungated | 32 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 1/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 0/4) | 0/7 (p 5/7) | 1.463 → 1.344 | accepted |
