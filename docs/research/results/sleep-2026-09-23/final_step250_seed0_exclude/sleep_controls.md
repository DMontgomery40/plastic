# Sleep with matched controls

Code ccfa446+dirty, checkpoint `29e0f8558d15`, device mps, 20 steps, target w0, lr 0.0001, replay ratio 0.5, session loss all, plw 1.0, augment none, 10214.0 s.

| Arm | taught (p = unseen phrasing) | boundary | rolled (contamination) | poison (uptake) | general (locality) | held-out NLL mean → | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| floor | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 0/0 (p 1/4) | 4/11 (p 3/7) |  |  |
| ceiling | 3/24 (p 2/24) | 0/2 (p 0/2) | 0/0 (p 0/0) | 0/0 (p 0/0) | 0/0 (p 0/0) |  |  |
| anchor | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 0/0 (p 1/4) | 5/11 (p 3/7) | 1.612 → 1.603 | accepted |
| replay | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 0/0 (p 1/4) | 4/11 (p 4/7) | 1.612 → 1.466 | accepted |
| distill | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 0/0 (p 1/4) | 2/11 (p 4/7) | 1.612 → 1.467 | accepted |
| dream | 0/24 (p 0/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 0/0 (p 0/4) | 4/11 (p 4/7) | 1.612 → 1.460 | rejected |
| ungated | 0/24 (p 1/24) | 0/2 (p 0/2) | 0/2 (p 0/2) | 0/0 (p 1/4) | 4/11 (p 4/7) | 1.612 → 1.483 | accepted |
