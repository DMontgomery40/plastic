# Sleep with matched controls

Checkpoint `5d7f0b9c6c1a`, device mps, 40 steps, target w0, lr 0.0001, 892.7 s.

| Arm | taught | boundary | rolled (contamination) | general (locality) | held-out NLL mean → | status |
| --- | --- | --- | --- | --- | --- | --- |
| floor | 0/6 (p 0/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 0/5 (p 1/5) |  |  |
| replay | 1/6 (p 1/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/5 (p 3/5) | 1.683 → 1.480 | rejected |
| distill | 0/6 (p 0/6) | 0/2 (p 0/2) | 0/2 (p 0/2) | 1/5 (p 1/5) | 1.683 → 1.491 | rejected |
