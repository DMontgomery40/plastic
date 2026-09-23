# Learning-contract results (23 September 2026)

Reports produced by `scripts/experiments/transfer_contract.py` under the contract defined in
[the testbed and contract spec](../../../superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md).
Each subdirectory holds the generated `README.md`, one JSON per baseline mode, and a
`manifest.json` with the checkpoint digest, model signature and execution commit.

Every measurement is taken from a fresh state with the stream removed. Deltas are after minus
before on identical inputs; negative is improvement. Acceptance is a pair of rates
(accepted-good, refused-bad); `n/a` means no decision was recorded, not zero.

## `phys_mps_3k` (the before reference)

The existing physics checkpoint (3 layers, d_model 256, chunk 64), trained on the single
hidden-friction environment, scored on the mechanism testbed with its three baseline modes.
One seed, one stream of 16 episodes (256 steps), 10 Adam steps at 1e-3 for continued training.
CPU. No new mechanism is evaluated here; this is what the contract says about the checkpoint we
already have, and it is the reference every later learner is compared against.

| held-out combinations, before any stream | with adaptation | writes disabled |
|---|---|---|
| impulse policy | 0.296 | 0.437 |
| hold policy | 0.097 | 0.355 |
| release policy | 0.104 | 0.241 |
| training distribution (gaussian) | 0.212 | 0.408 |

Adaptation speed on held-out worlds: the adapting error is on average 0.58 of the
writes-disabled error over the first eight steps and drops below one half at step 2.

| mode | transfer Δ impulse | transfer Δ hold | transfer Δ release | forgetting Δ | poison harm | correction residual | revert |
|---|---|---|---|---|---|---|---|
| frozen (fast weights only) | 0 | 0 | 0 | 0 | 0 | 0 | ok |
| continued training | −0.062 | +0.159 | −0.009 | −0.007 | +0.019 | +0.011 | ok |
| everything in context | +0.000 | −0.001 | +0.000 | −0.000 | +0.000 | +0.000 | ok |

What this shows, and only this:

- The fast weights do real within-episode work on mechanisms the checkpoint never saw: on
  every held-out policy the adapting error is well below the writes-disabled error. That is
  temporary adaptation, and the contract deliberately does not count it as learning.
- Ten steps of plain continued training on 256 steps of experience change the checkpoint in
  both directions: better under impulses, worse under held pushes, slightly better on the
  training distribution. One seed and one stream; no claim beyond "the stream carries
  something a slow update can use, and plain SGD does not use it well".
- The poisoned stream harms clean transfer relative to the clean stream, and a corrective
  clean stream removes about half of the harm. Continued training accepts everything, so its
  pair of rates is (1, 0); that is the control the acceptance function has to beat.
- The everything-in-context baseline changes nothing on this checkpoint. That is a property of
  the checkpoint, not of the baseline: it was trained to open its forget gate on the reset
  flag, so a prepended stream of earlier episodes is discarded by construction. A learner
  that is meant to use cross-episode context must be trained to keep it.
- The revert ablation holds in every mode: restoring the pre-stream parameters reproduces the
  before measurements to within 1e-6.

Not shown: any lasting, transferable learning. Nothing here has an acceptance function; the
coordinate block (T2) and the slow update rule (T3) will be the first learners scored against
this reference.
