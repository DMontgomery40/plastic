# Coordinate ablation on the mechanism testbed

Each variant is trained from scratch on the training combinations (gaussian policy) with the same data seed, then scored under the learning contract with no lasting update. `adapt` is the held-out-combination MSE with fast updates on; `no-adapt` is the same inputs with the fast path off. The off-intervention differs by model and is labelled: full: fast parameters frozen (freeze=True); no_fast: fast parameters frozen (freeze=True); decay_only: fast parameters frozen (freeze=True); coords_only: fast parameters frozen (freeze=True); no_meta: fast parameters frozen (freeze=True); fixed_z: fast parameters frozen (freeze=True); delta_baseline: writes disabled (beta_scale=0).

`speed mean` is the adapting error as a fraction of the no-adapt error over the first probe steps (lower is faster); `half at step` is the first step at which it drops below one half. `η per layer` is the learned inner step size. `inner loss before → after` re-scores the observed chunk under the proposed step (a support diagnostic, not an adaptation score); `‖ΔW‖, ‖Δθ‖` are the proposed changes per layer, averaged over held-out boundaries.

| variant | outer loss | params | train loss (last) | s/step | impulse: adapt / no-adapt | hold: adapt / no-adapt | release: adapt / no-adapt | train dist: adapt / no-adapt | speed mean | half at step | η per layer | inner loss before → after | ‖ΔW‖, ‖Δθ‖ per layer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | all | 409095 | 0.0637 | 0.19 | 0.1561 / 0.1561 | 0.0621 / 0.0624 | 0.0433 / 0.0433 | 0.0549 / 0.0553 | 1.000 | 64 | 0.276, 0.339, 0.386 | n/a | n/a |
| no_fast | all | 409095 | 0.0588 | 0.07 | 0.1534 / 0.1534 | 0.0716 / 0.0716 | 0.0443 / 0.0443 | 0.0538 / 0.0538 | 1.000 | 64 | 0.100, 0.100, 0.100 | n/a | n/a |
| decay_only | all | 409095 | 0.0583 | 0.17 | 0.1542 / 0.1543 | 0.0761 / 0.0761 | 0.0444 / 0.0444 | 0.0507 / 0.0507 | 1.000 | 64 | 0.180, 0.362, 0.358 | n/a | n/a |
| coords_only | all | 409095 | 0.0660 | 0.18 | 0.1536 / 0.1536 | 0.0653 / 0.0657 | 0.0475 / 0.0475 | 0.0496 / 0.0498 | 0.999 | 64 | 0.263, 0.351, 0.372 | n/a | n/a |
| no_meta | all | 409095 | 0.0530 | 0.13 | 0.1588 / 0.1588 | 0.0669 / 0.0670 | 0.0460 / 0.0460 | 0.0509 / 0.0510 | 1.000 | 64 | 0.100, 0.100, 0.100 | n/a | n/a |
| fixed_z | all | 409095 | 0.0567 | 0.19 | 0.1601 / 0.1603 | 0.0660 / 0.0668 | 0.0509 / 0.0509 | 0.0587 / 0.0592 | 0.999 | 64 | 0.256, 0.369, 0.392 | n/a | n/a |
| delta_baseline | all | 897660 | 0.0329 | 0.91 | 0.1426 / 0.3241 | 0.0423 / 0.1309 | 0.0344 / 0.1127 | 0.0399 / 0.2128 | 0.390 | 1 | n/a | n/a | n/a |

`outer loss` names the meta-training objective: `all` is the mean over every step of the episode (a sequence-model objective under which the first chunk can never benefit from a fast update); `post_boundary` is the mean over steps after the first boundary only, the query loss adaptation could have improved. Read `all` rows as a lower bound on what the fast path was asked to do.

Reading guide (from the memo's falsification table): if `decay_only` matches `full`, timescale adaptation explains the gain and the nonlinear coordinates are not earning their place. If `no_meta` matches `full`, meta-training is not necessary. If `delta_baseline` matches `full` at matched compute, coupling learning to the dynamics is not useful. Parameters, seconds per step and state memory are reported because equal parameter counts alone are insufficient. One seed unless stated; no lasting-learning claim is made here.
