# Text rule contract report

Code 65225ee (exported snapshot), checkpoint digest `29e0f8558d15`, device mps, rule set `decorate`, seed 0, 17 training compositions, 8 held-out pairs, 8 situations per episode, stream 17 episodes, lasting update target w0, lr 0.0001, 20 steps, poison operator `#P` (consistent), rules UNSTATED in every preface, 6 verify episodes, 8 held-out chat rows, lasting-update sampling draws (20 draws), 2 choice items per composition. Exact is teacher-forced (`exact_tf`). Every mode reloads the checkpoint.

| Mode | Held-out exact (adapt) before → after | of which first situation Δ / later Δ | Held-out nll (adapt) before → after | Held-out nll (no adapt) Δ | Speed area before → after | Forgetting nll Δ | Poison harm (nll) | Sequential poison: accepted, harm (nll) | Correction residual | Format-only gain exact (true) | Revert | Accepted good / refused bad | Verifier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay_verify | 0.74 → 0.88 | +0.00 / +0.15 | 0.417 → 0.155 | -0.242 | 0.05 → 0.02 | -0.294 | +0.014 | False, +0.000 | -0.009 | -0.05 (+0.13) | ok | 1.0 / 0.3333333333333333 (n 2/3) | v2 |

Coverage, first-situation choice (chance is 1/candidates: 23 distinct outputs on the decoration set) and the sequential arms:

| Mode | Clean stream: compositions trained (steps, poisoned steps) | First-situation choice, held-out: before → after (novel pairs only) | First-situation choice, training: before → after | Clean again: accepted, verify first-situation items changed | Poison on top: accepted, verify first-situation items changed | Poison on top vs clean again: choice held-out / training Δ |
| --- | --- | --- | --- | --- | --- | --- |
| replay_verify | 11/17 (20, 0) | 0.00 → 0.00 (0.00 → 0.00) | 0.06 → 0.12 | True, none | False, #W 1→0 | +0.00 / -0.09 |

Paired first-situation margin change per item (mean nats; improved / worsened of n), by group:

| Mode | Comparison | Training compositions | Held-out, novel | Held-out, repeats a trained answer |
| --- | --- | --- | --- | --- |
| replay_verify | clean update vs before | +8.98 (31/3 of 34) | +3.59 (9/3 of 12) | +15.84 (4/0 of 4) |
| replay_verify | clean again vs after clean | +0.81 (21/13 of 34) | +2.04 (10/2 of 12) | +0.28 (2/2 of 4) |
| replay_verify | poison on top vs after clean | +0.00 (0/0 of 34) | +0.00 (0/0 of 12) | +0.00 (0/0 of 4) |
| replay_verify | poison from snapshot vs before | +7.60 (29/5 of 34) | +2.22 (8/4 of 12) | +15.11 (4/0 of 4) |
| replay_verify | format-only vs before | +7.58 (31/3 of 34) | +5.45 (12/0 of 12) | +15.99 (4/0 of 4) |
