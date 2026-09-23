# Sleep with matched controls

Checkpoint `229e97b67875`, device mps, 10 steps, 1632.2 s.

| Arm | taught | boundary | rolled (contamination) | general (locality) | held-out NLL mean → | status |
| --- | --- | --- | --- | --- | --- | --- |
| floor | 0/6 (p 0/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/5 (p 1/5) |  |  |
| ceiling | 5/6 (p 5/6) | 0/2 (p 0/2) | 0/0 (p 0/0) | 0/0 (p 0/0) |  |  |
| anchor | 0/6 (p 1/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/5 (p 1/5) | 1.766 → 1.766 | accepted |
| replay | 0/6 (p 0/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 2/5 (p 1/5) | 1.766 → 1.615 | accepted |
| distill | 0/6 (p 0/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/5 (p 1/5) | 1.766 → 1.627 | accepted |
| ungated | 0/6 (p 0/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 2/5 (p 1/5) | 1.766 → 1.620 | accepted |
