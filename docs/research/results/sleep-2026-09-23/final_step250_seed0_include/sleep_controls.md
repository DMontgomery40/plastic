# Sleep with matched controls

Code b926e4d, checkpoint `29e0f8558d15`, device mps, 20 steps, target w0, lr 0.0001, replay ratio 0.5, session loss all, plw 1.0, augment none, 3856.8 s.

| Arm | turns consumed (accepted / flagged / excluded) | taught (p = unseen phrasing) | boundary | rolled (contamination) | poison (uptake) | general (locality) | held-out NLL mean → | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay | 30 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 1/4) | 2/7 (p 4/7) | 1.612 → 1.485 | accepted |
| distill | 30 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 0/4) | 3/7 (p 4/7) | 1.612 → 1.474 | accepted |
| dream | 30 selected (30 accepted online, 29 flagged, 0 excluded) | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/4 (p 0/4) | 3/7 (p 4/7) | 1.612 → 1.460 | rejected |
