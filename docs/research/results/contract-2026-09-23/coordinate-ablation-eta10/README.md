# Coordinate ablation on the mechanism testbed

Each variant is trained from scratch on the training combinations (gaussian policy) with the same data seed, then scored under the learning contract with no lasting update. `adapt` is the held-out-combination MSE with fast updates on; `no-adapt` is the same inputs with the fast path off. The off-intervention differs by model and is labelled: full: fast parameters frozen (freeze=True); no_fast: fast parameters frozen (freeze=True); decay_only: fast parameters frozen (freeze=True).

`speed mean` is the adapting error as a fraction of the no-adapt error over the first probe steps (lower is faster); `half at step` is the first step at which it drops below one half. `η per layer` is the learned inner step size. `inner loss before → after` re-scores the observed chunk under the proposed step (a support diagnostic, not an adaptation score); `‖ΔW‖, ‖Δθ‖` are the proposed changes per layer, averaged over held-out boundaries.

| variant | outer loss | params | train loss (last) | s/step | impulse: adapt / no-adapt | hold: adapt / no-adapt | release: adapt / no-adapt | after step 16: adapt / no-adapt (held-out mean) | train dist: adapt / no-adapt | speed mean | half at step | η per layer | inner loss before → after | ‖ΔW‖, ‖Δθ‖ per layer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | post_boundary | 409095 | 0.0693 | 0.18 | 0.1192 / 0.1298 | 0.0386 / 0.0641 | 0.0543 / 0.0596 | 0.0527 / 0.0711 | 0.0478 / 0.0583 | 0.673 | 23 | 2.960, 4.761, 5.136 | 0.0984 → 0.0481 | 0.176/0.0239, 0.322/0.0472, 0.356/0.0604 |
| no_fast | post_boundary | 409095 | 0.0718 | 0.06 | 0.1240 / 0.1240 | 0.0666 / 0.0666 | 0.0684 / 0.0684 | 0.0627 / 0.0627 | 0.0660 / 0.0660 | 1.000 | 64 | 1.000, 1.000, 1.000 | n/a | n/a |
| decay_only | post_boundary | 409095 | 0.0642 | 0.15 | 0.1218 / 0.1220 | 0.0516 / 0.0531 | 0.0708 / 0.0708 | 0.0535 / 0.0543 | 0.0499 / 0.0500 | 0.997 | 64 | 3.469, 7.275, 6.922 | 0.1118 → 0.1067 | 0/0.0433, 0/0.0861, 0/0.0648 |

`outer loss` names the meta-training objective: `all` is the mean over every step of the episode (a sequence-model objective under which the first chunk can never benefit from a fast update); `post_boundary` is the mean over steps after the first boundary only, the query loss adaptation could have improved. Read `all` rows as a lower bound on what the fast path was asked to do.

Reading guide (from the memo's falsification table): if `decay_only` matches `full`, timescale adaptation explains the gain and the nonlinear coordinates are not earning their place. If `no_meta` matches `full`, meta-training is not necessary. If `delta_baseline` matches `full` at matched compute, coupling learning to the dynamics is not useful. Parameters, seconds per step and state memory are reported because equal parameter counts alone are insufficient. One seed unless stated; no lasting-learning claim is made here.
