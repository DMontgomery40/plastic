# Text rule contract report

Code 9ae91dd+dirty, checkpoint digest `29e0f8558d15`, device mps, rule set `decorate`, seed 0, 17 training compositions, 8 held-out pairs, 8 situations per episode, stream 17 episodes, lasting update target w0, lr 0.0001, 20 steps, poison operator `#P` (consistent), rules UNSTATED in every preface, 8 held-out chat rows. Exact is teacher-forced (`exact_tf`). Every mode reloads the checkpoint.

| Mode | Held-out exact (adapt) before → after | of which first situation Δ / later Δ | Held-out nll (adapt) before → after | Held-out nll (no adapt) Δ | Speed area before → after | Forgetting nll Δ | Poison harm (nll) | Sequential poison: accepted, harm (nll) | Correction residual | Format-only gain exact (true) | Revert | Accepted good / refused bad | Verifier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 0.74 → 0.74 | +0.00 / +0.00 | 0.417 → 0.417 | +0.000 | 0.05 → 0.05 | +0.000 | +0.000 | None, +0.000 | +0.000 | +0.00 (+0.00) | ok | None / None (n 0/0) | n/a |
| continued | 0.74 → 0.88 | +0.00 / +0.15 | 0.417 → 0.155 | -0.242 | 0.05 → 0.02 | -0.294 | +0.014 | True, -0.015 | -0.009 | -0.05 (+0.13) | ok | 1.0 / 0.0 (n 2/3) | n/a |
| replay_verify | 0.74 → 0.88 | +0.00 / +0.15 | 0.417 → 0.155 | -0.242 | 0.05 → 0.02 | -0.294 | +0.014 | False, +0.000 | -0.009 | -0.05 (+0.13) | ok | 1.0 / 0.3333333333333333 (n 2/3) | v2 |
