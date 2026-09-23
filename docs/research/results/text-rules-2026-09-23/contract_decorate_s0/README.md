# Text rule contract report

Code 1c89f1c+dirty, checkpoint digest `29e0f8558d15`, device mps, rule set `decorate`, seed 0, 17 training compositions, 8 held-out pairs, 8 situations per episode, stream 17 episodes, lasting update target w0, lr 0.0001, 20 steps, poison operator `#P`, 8 held-out chat rows. Exact is teacher-forced (`exact_tf`). Every mode reloads the checkpoint.

| Mode | Held-out exact (adapt) before → after | Held-out nll (adapt) before → after | Held-out nll (no adapt) Δ | Speed area before → after | Forgetting nll Δ | Poison harm (nll) | Correction residual | Revert | Accepted good / refused bad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen | 0.67 → 0.67 | 0.387 → 0.387 | +0.000 | 0.05 → 0.05 | +0.000 | +0.000 | +0.000 | ok | None / None (n 0/0) |
| continued | 0.67 → 0.87 | 0.387 → 0.120 | -0.247 | 0.05 → 0.01 | -0.282 | +0.019 | -0.003 | ok | 1.0 / 0.0 (n 2/1) |
| in_context | 0.67 → 0.55 | 0.387 → 0.278 | -0.008 | 0.05 → 0.03 | -0.071 | -0.001 | -0.002 | ok | None / None (n 0/0) |
| replay_verify | 0.67 → 0.87 | 0.387 → 0.120 | -0.247 | 0.05 → 0.01 | -0.282 | +0.019 | -0.003 | ok | 1.0 / 0.0 (n 2/1) |
