# Text rule contract report

Code a8b12a6+dirty, checkpoint digest `29e0f8558d15`, device mps, rule set `decorate`, seed 0, 17 training compositions, 8 held-out pairs, 8 situations per episode, stream 17 episodes, lasting update target w0, lr 0.0001, 20 steps, poison operator `#P`, 8 held-out chat rows. Exact is teacher-forced (`exact_tf`). Every mode reloads the checkpoint.

| Mode | Held-out exact (adapt) before → after | of which first situation Δ / later Δ | Held-out nll (adapt) before → after | Held-out nll (no adapt) Δ | Speed area before → after | Forgetting nll Δ | Poison harm (nll) | Correction residual | Format-only gain exact (true) | Revert | Accepted good / refused bad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continued | 0.67 → 0.87 | +0.00 / +0.22 | 0.387 → 0.120 | -0.247 | 0.05 → 0.01 | -0.282 | +0.019 | -0.003 | +0.09 (+0.20) | ok | 1.0 / 0.0 (n 2/2) |
| replay_verify | 0.67 → 0.87 | +0.00 / +0.22 | 0.387 → 0.120 | -0.247 | 0.05 → 0.01 | -0.282 | +0.019 | -0.003 | +0.09 (+0.20) | ok | 1.0 / 0.0 (n 2/2) |

**Correction (FABLE-41B-211, 2026-09-24).** The held-out pairs listed in this run's `log.txt` `[setup]` line are the report script's split (reversed-pair minimum 4), not the contract's (minimum 6, recorded in the JSON `split`); the contract measured on `#B #H, #B #P, #B #Q, #H #Q, #P #B, #P #H, #Q #P, #W #B`. The learner's held-in verification material was built from the script's split, so the replay_verify checks scored six of the contract's held-out compositions (with fresh inputs, read-only) alongside training ones. The transfer, speed, forgetting, correction and revert numbers are unaffected (they use the JSON split). The verify numbers are a mixed held-in/held-out score, and the note "never the held-out compositions" in those records is wrong for this run. Fixed in the code after this run: the contract owns the split and refuses mismatched held-in material.
