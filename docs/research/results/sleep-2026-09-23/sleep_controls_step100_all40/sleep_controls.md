# Sleep with matched controls

Checkpoint `5d7f0b9c6c1a`, device mps, 40 steps, target all, lr 5e-05, 928.4 s.

| Arm | taught | boundary | rolled (contamination) | general (locality) | held-out NLL mean → | status |
| --- | --- | --- | --- | --- | --- | --- |
| floor | 0/6 (p 0/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 0/5 (p 1/5) |  |  |
| replay | 1/6 (p 1/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 2/5 (p 2/5) | 1.683 → 1.629 | accepted |
| ungated | 1/6 (p 1/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 3/5 (p 2/5) | 1.683 → 1.628 | accepted |
