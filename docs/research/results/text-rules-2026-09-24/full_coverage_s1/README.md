# Text rule contract report

Code 65225ee (exported snapshot), checkpoint digest `29e0f8558d15`, device mps, rule set `decorate`, seed 1, 17 training compositions, 8 held-out pairs, 8 situations per episode, stream 17 episodes, lasting update target w0, lr 0.0001, 20 steps, poison operator `#P` (consistent), rules UNSTATED in every preface, 34 verify episodes, 8 held-out chat rows, lasting-update sampling passes (1 pass), 2 choice items per composition. Exact is teacher-forced (`exact_tf`). Every mode reloads the checkpoint.

| Mode | Held-out exact (adapt) before → after | of which first situation Δ / later Δ | Held-out nll (adapt) before → after | Held-out nll (no adapt) Δ | Speed area before → after | Forgetting nll Δ | Poison harm (nll) | Sequential poison: accepted, harm (nll) | Correction residual | Format-only gain exact (true) | Revert | Accepted good / refused bad | Verifier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continued | 0.71 → 0.87 | +0.00 / +0.18 | 0.391 → 0.104 | -0.219 | 0.05 → 0.01 | -0.305 | +0.042 | True, +0.005 | +0.011 | +0.07 (+0.16) | ok | 1.0 / 0.0 (n 2/3) | n/a |
| replay_verify | 0.71 → 0.87 | +0.00 / +0.18 | 0.391 → 0.104 | -0.219 | 0.05 → 0.01 | -0.305 | +0.042 | True, +0.005 | +0.011 | +0.07 (+0.16) | ok | 1.0 / 0.0 (n 2/3) | v2 |

Coverage, first-situation choice (chance is 1/candidates: 23 distinct outputs on the decoration set) and the sequential arms:

| Mode | Clean stream: compositions trained (steps, poisoned steps) | First-situation choice, held-out: before → after (novel pairs only) | First-situation choice, training: before → after | Clean again: accepted, verify first-situation items changed | Poison on top: accepted, verify first-situation items changed | Poison on top vs clean again: choice held-out / training Δ |
| --- | --- | --- | --- | --- | --- | --- |
| continued | 17/17 (17, 0) | 0.00 → 0.00 (0.00 → 0.00) | 0.06 → 0.06 | True, n/a | True, n/a | +0.00 / +0.06 |
| replay_verify | 17/17 (17, 0) | 0.00 → 0.00 (0.00 → 0.00) | 0.06 → 0.06 | True, #P 0→1, #W 1→0, #P 0→1, #B 1→0 | True, #B 1→0 | +0.00 / +0.06 |

Paired first-situation margin change per item (mean nats; improved / worsened of n), by group:

| Mode | Comparison | Training compositions | Held-out, novel | Held-out, repeats a trained answer |
| --- | --- | --- | --- | --- |
| continued | clean update vs before | +9.92 (31/3 of 34) | +8.52 (12/0 of 12) | +18.64 (4/0 of 4) |
| continued | clean again vs after clean | +0.11 (16/18 of 34) | +1.90 (9/3 of 12) | -2.82 (1/3 of 4) |
| continued | poison on top vs after clean | -0.87 (13/21 of 34) | -1.33 (2/10 of 12) | -5.57 (0/4 of 4) |
| continued | poison from snapshot vs before | +7.17 (29/5 of 34) | +2.31 (7/5 of 12) | +13.20 (4/0 of 4) |
| continued | format-only vs before | +7.87 (32/2 of 34) | +7.83 (12/0 of 12) | +13.75 (4/0 of 4) |
| replay_verify | clean update vs before | +9.92 (31/3 of 34) | +8.52 (12/0 of 12) | +18.64 (4/0 of 4) |
| replay_verify | clean again vs after clean | +0.11 (16/18 of 34) | +1.90 (9/3 of 12) | -2.82 (1/3 of 4) |
| replay_verify | poison on top vs after clean | -0.87 (13/21 of 34) | -1.33 (2/10 of 12) | -5.57 (0/4 of 4) |
| replay_verify | poison from snapshot vs before | +7.17 (29/5 of 34) | +2.31 (7/5 of 12) | +13.20 (4/0 of 4) |
| replay_verify | format-only vs before | +7.87 (32/2 of 34) | +7.83 (12/0 of 12) | +13.75 (4/0 of 4) |
