# Learning-contract results (23 September 2026)

Reports produced by `scripts/experiments/transfer_contract.py` under the contract defined in
[the testbed and contract spec](../../../superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md).
Each subdirectory holds the generated `README.md`, one JSON per baseline mode, and a
`manifest.json` with the checkpoint digest, model signature, execution commit, split identity
and adaptation window.

Contract version `2026-09-23.2`: every scored row holds exactly one 64-step episode from a
fresh state; a learner declares how often its fast parameters can change and the contract
refuses a spec under which no update boundary falls inside a scored episode; the experience
stream (16 episodes packed into one row) is checked to contain only training combinations
under a training policy, and measurement worlds are checked to be disjoint from it. The first
version of this page reported numbers from a contract that packed four 16-step episodes per
scored row; those numbers are superseded and are not reproduced here.

Every measurement is taken from a fresh state with the stream removed. Deltas are after minus
before on identical inputs; negative is improvement. Acceptance is a pair of rates
(accepted-good, refused-bad); `n/a` means no decision was recorded, not zero.

## `phys_mps_3k` (the before reference)

The existing physics checkpoint (3 layers, d_model 256, chunk 64, delta-rule memory writing
at every token), trained on the single hidden-friction environment, scored on the mechanism
testbed with its three baseline modes. One seed, one stream of 16 episodes (1024 steps), 10
Adam steps at 1e-3 for continued training. CPU. No new mechanism is evaluated here; this is
what the contract says about the checkpoint we already have, and it is the reference every
later learner is compared against.

| held-out pairings, before any stream | with fast writes | writes disabled |
|---|---|---|
| impulse policy | 0.364 | 0.530 |
| hold policy | 0.068 | 0.360 |
| release policy | 0.110 | 0.165 |
| training distribution (gaussian) | 0.250 | 0.474 |

Adaptation speed on held-out worlds: the adapting error averages 0.39 of the writes-disabled
error over the 64-step episode and drops below one half at step 2 (the delta rule writes at
every token, so 63 update boundaries fall inside each episode).

| mode | transfer Δ impulse | transfer Δ hold | transfer Δ release | forgetting Δ | poison harm vs clean | poison harm vs start | correction residual | revert |
|---|---|---|---|---|---|---|---|---|
| frozen (fast weights only) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ok |
| continued training (Adam) | −0.102 | +0.221 | −0.029 | −0.077 | −0.011 | +0.019 | +0.004 | ok |
| everything in context | −0.000 | −0.001 | −0.001 | −0.000 | +0.000 | −0.001 | +0.001 | ok |

What this shows, and only this:

- The fast weights do real within-episode work on mechanism pairs the checkpoint never saw:
  under every held-out policy the adapting error is below the writes-disabled error, by a
  factor of five under held pushes. That is temporary adaptation, and the contract
  deliberately does not count it as learning.
- Ten Adam steps on one 1024-step stream move the checkpoint in both directions: better under
  impulses and releases, worse under held pushes, and better on the training distribution.
  Averaged over the held-out policies, the clean stream made transfer worse (0.181 to 0.211).
  One seed and one stream; no claim beyond "the stream carries something a slow update can
  use, and plain continued training does not use it well".
- Against its own start, the poisoned stream hurt transfer (+0.019) and the corrective
  stream did not repair it (+0.035 after correction). Against the clean arm the poison
  reading is negative, because the clean arm itself was harmful on this checkpoint; both
  readings are reported for that reason. Continued training accepts everything, so its pair
  of rates is (1, 0); that is the control the acceptance function has to beat.
- The everything-in-context baseline changes nothing on this checkpoint. That is a property
  of the checkpoint, not of the baseline: it was trained to open its forget gate on the reset
  flag, so a prepended stream of earlier episodes is discarded by construction. A learner
  that is meant to use cross-episode context must be trained to keep it.
- The revert ablation holds in every mode: restoring the pre-stream parameters reproduces the
  before measurements to within 1e-6.

Not shown: any lasting, transferable learning. Nothing here has an acceptance function; the
coordinate block and the slow update rule will be the first learners scored against this
reference. The coordinate ablation (the memo's falsifier table) runs under the same contract
and will be archived beside this report when it completes.

## Coordinate ablation, whole-episode objective (`coordinate-ablation-all/`)

The memo's falsifier table (`docs/research/2026-09-21-plastic-coordinate-recurrence.md`,
"What would falsify the contribution") on the mechanism testbed, produced by
`scripts/experiments/coordinate_ablation.py`. Every variant trained from scratch for 1500
Adam steps (batch 16, one 64-step episode per row, chunk 16 so three fast-update boundaries
fall inside each episode, gradient clip 1.0, data seed 1, model seed 0), then scored under the
contract with no lasting update. One seed. CPU.

**The outer loss in this row set is the whole-episode mean MSE.** That is a sequence-model
objective: the first chunk of every episode can never benefit from a fast update, so the
gradient reaching the fast path is diluted by steps only the slow weights can serve. A second
row set under the post-boundary objective (the mean over steps after the first boundary, the
loss adaptation could have improved) is the corrected comparison and is reported separately
when it lands. Read this table as what the fast path did when it was asked less than it
should have been.

| variant | params | s/step | train loss | impulse adapt / off | hold | release | training dist. | speed | η per layer |
|---|---|---|---|---|---|---|---|---|---|
| full | 409k | 0.19 | 0.064 | 0.156 / 0.156 | 0.062 / 0.062 | 0.043 / 0.043 | 0.055 / 0.055 | 1.00 | 0.28, 0.34, 0.39 |
| no_fast | 409k | 0.07 | 0.059 | 0.153 / 0.153 | 0.072 / 0.072 | 0.044 / 0.044 | 0.054 / 0.054 | 1.00 | (off) |
| decay_only | 409k | 0.17 | 0.058 | 0.154 / 0.154 | 0.076 / 0.076 | 0.044 / 0.044 | 0.051 / 0.051 | 1.00 | 0.18, 0.36, 0.36 |
| coords_only | 409k | 0.18 | 0.066 | 0.154 / 0.154 | 0.065 / 0.066 | 0.048 / 0.048 | 0.050 / 0.050 | 1.00 | 0.26, 0.35, 0.37 |
| no_meta | 409k | 0.13 | 0.053 | 0.159 / 0.159 | 0.067 / 0.067 | 0.046 / 0.046 | 0.051 / 0.051 | 1.00 | (fixed 0.10) |
| fixed_z | 409k | 0.19 | 0.057 | 0.160 / 0.160 | 0.066 / 0.067 | 0.051 / 0.051 | 0.059 / 0.059 | 1.00 | 0.26, 0.37, 0.39 |
| delta_baseline | 898k | 0.91 | 0.033 | 0.143 / 0.324 | 0.042 / 0.131 | 0.034 / 0.113 | 0.040 / 0.213 | 0.39 | (per-token) |

"off" is each model's own no-adaptation control: fast parameters frozen (coordinates and
decay) for the coordinate variants, writes disabled with decay active for the delta baseline.
`speed` is the adapting error as a fraction of the off error over the episode; 1.00 means the
fast path removed nothing.

What this shows, and only this:

- Under the whole-episode objective the coordinate block's fast path changes nothing: on and
  off agree to three decimals for every switch, including the deliberately incorrect
  fixed-latent commit, which can only matter if a proposal moves the state. The learned step
  sizes grew from 0.10 to about 0.3 per layer, so meta-training was not switching the fast
  path off; its proposals simply do not move the predictions.
- The same objective did not stop the delta baseline: its per-token writes cut held-out error
  by a factor of two to three within an episode and it trains to half the loss. Dilution of
  the meta-gradient is therefore not a sufficient explanation on its own; the remaining
  candidates are the size of the coordinate block's inner gradient (the coupling starts near
  the identity) and three boundaries per episode with each chunk's late target dropped, and
  the post-boundary row set tests the objective directly.
- The delta baseline has 2.2 times the parameters and 4.8 times the step cost; the memo asks
  for parameter, state-memory and wall-clock matching, so this is not yet a matched
  comparison in its favour either.
- No falsifier from the memo is called on this row set. The call waits for the post-boundary
  objective, and if the fast path still does nothing there, the next question is the
  mechanism's gradient scale, not the recurrence's competence.
