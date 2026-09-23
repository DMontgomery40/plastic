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
| impulse policy | 0.365 | 0.530 |
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
