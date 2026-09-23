# Mechanism testbed and the learning contract

Spec, 23 September 2026. Status: specified and implemented (`plastic/data/mechanisms.py`,
`plastic/eval/contract.py`, `scripts/experiments/transfer_contract.py`, tests); the first
report, on the existing physics checkpoint, is archived under
`docs/research/results/contract-2026-09-23/`. The benchmark for this work is the contract
report itself. No learning result is claimed here.

## Research readiness note

**Task and constraints.** Build the environment and the measurement that let any learner in
this repository be scored on *learning* in David's sense (23 September): durable, transferable
change in competence, judged after the conversation and the temporary state are gone, with
facts left to retrieval. The measurement must also score safety in both directions, because
an acceptance function that refuses everything passes every poison check (Astra, 20:4x UTC).
No model training beyond CPU smoke tests in this thread; the first report runs the existing
physics checkpoint as a "before" reference.

**Local sources read.** `AGENTS.md`; `docs/research/README.md`; `current-status.md`; the
coordinate memo (`2026-09-21-plastic-coordinate-recurrence.md`, physics falsifiers and
controls); `plastic/data/physics.py`; `plastic/train/loop.py` (`evaluate_physics`);
`plastic/model/lm.py` (`PlasticDynamics`); `plastic/session/runner.py`
(`physics_episode`, the base/frozen/adaptive per-step comparison); `tests/test_data.py`;
the team's coordination record for 23 September and the amended mechanism proposal (coupled
coordinate recurrence plus a slow update rule, judged after reset).

**Primary sources checked (2026-09-23, abstract level unless noted).**
- Lopez-Paz and Ranzato, *Gradient Episodic Memory for Continual Learning*,
  [arXiv:1706.08840](https://arxiv.org/abs/1706.08840): defines ACC, backward transfer
  (performance on earlier tasks after later training, relative to when they were learned) and
  forward transfer. The contract's *forgetting* measure is BWT with MSE in place of accuracy;
  *transfer* is the FWT idea applied to held-out combinations with no independent model.
- IBM, *Scalable Evaluation and Neural Models for Compositional Generalization*,
  [arXiv:2511.02667](https://arxiv.org/abs/2511.02667): standard framing of compositional
  generalization as prediction on unseen combinations of known concepts; used only for the
  split design (held-out combinations, every concept seen in training).
- *Predictive learning enables compositional representations* (bioRxiv, Sept 2025,
  [preprint](https://www.biorxiv.org/content/10.1101/2025.09.26.678731v1.full.pdf)): an RNN
  predicting frames in a world with independent latent factors and per-factor dynamics learned
  modular representations; combinations of intervention types were withheld in its
  compositional test. Abstract and figure text only. It is the nearest precedent for a
  mechanism library with held-out combinations; it measures within-training generalization,
  not lasting change after reset.
- The coordinate memo's physics section: "A recurrent activation can infer mu without gradient
  updates; therefore lower MSE or a decodable mu probe alone does not prove that fast learning
  matters. Use freeze, reset, and fast-state-swap controls on matched activation states." The
  contract adopts freeze and reset as its `adapt=False` and after-reset measurements.
- In-Place TTT ([arXiv:2604.06169](https://arxiv.org/html/2604.06169v1)) and TTT-NTP
  ([arXiv:2606.21803](https://arxiv.org/pdf/2606.21803)): current TTT evaluation practice is
  ablation of state size, chunk size and inner objective on held-out loss; none evaluates
  after resetting state and reverting weights. Abstract level.
- Exposure and edit-evaluation literature: via the reassessment memo
  (`2026-09-23-reassessment-concepts-not-phrases.md`), not re-read.

**Closest known mechanism.** Meta-learned system identification for neural state-space models
([arXiv:2501.06167](https://arxiv.org/abs/2501.06167), cited in the coordinate memo) measures
rapid *within-episode* adaptation to a new system. GEM-style continual-learning metrics measure
retention across tasks for ordinary gradient training.

**The specific distinction.** This contract measures a lasting change produced by a learner's
own consolidation from an experience stream, scored *after* the activation state and fast
parameters are cleared and the stream removed, with a revert ablation (restore the pre-stream
slow parameters; the gain must vanish) and a retrieval/in-context baseline at matched compute.
No source above combines those three. That is a bounded search and not a priority claim.

**Unresolved assumptions.** (1) That a 2-D point mass with six additive mechanisms is rich
enough for "structure" to be distinguishable from "parameters": a learner could treat every
world as one 7-parameter regression. The held-out combination split and the held-out
intervention policies are the defence, and the first report will show whether the existing
checkpoint already generalizes across combinations (if so, the testbed is too easy and needs a
nonlinear or hierarchical mechanism). (2) That MSE on next-observation delta is the right
outcome; it is the checkpoint's training objective, so it is the honest first choice.
(3) That a structured target bias is a fair "wrong lesson"; it is learnable and false in every
world, which is the property needed.

**Falsifying check for the testbed itself.** If a frozen learner (no consolidation) and a
continued-training learner score the same on transfer after reset, the stream carries nothing
a slow update can use at this scale, and the testbed must change before any mechanism is
judged on it.

## What is built

### `plastic/data/mechanisms.py`

A 2-D point mass with the same input row as `plastic/data/physics.py`
(`[obs(4), action(2), reset(1)]`) and the same target (next-observation delta, 4), so
`PlasticDynamics`, the session runner and `phys_mps_3k` consume it unchanged.

Base dynamics with a fixed, known damping `BASE_DAMPING = 0.1`:

```
a_eff   = c · a
vel'    = (1 − (0.1 + μ)) · vel + a_eff + g − k · pos + ω · R90(vel)
pos'    = pos + vel'
wall:     if |pos'_x| > L: pos'_x ← sign · (2L − |pos'_x|), vel'_x ← −vel'_x
target  = [pos' − pos, vel' − vel]
```

Six hidden mechanisms, each with one sampled parameter, inactive values in brackets:

| name | parameter | range | inactive |
| --- | --- | --- | --- |
| `drag` | extra damping μ | [0.05, 0.25] | 0 |
| `field` | constant force g (2-vector, random direction) | magnitude [0.05, 0.20] | 0 |
| `spring` | restoring k toward the origin | [0.02, 0.15] | 0 |
| `coupling` | axis rotation rate ω | [−0.30, 0.30] | 0 |
| `gain` | actuator gain c | [0.50, 2.00] | 1 |
| `wall` | reflecting wall at |x| = L | [2.0, 6.0] | ∞ |

A *world* is a set of active mechanisms with sampled parameters. A *combination* is the set of
names. `split_combinations(k, n_heldout, seed)` returns disjoint train and held-out
combinations of size `k` such that every mechanism name appears in at least one training
combination: only the pairing is new at test.

*Intervention policies* generate the action sequence: `gaussian` (training), and the held-out
`impulse` (rare large kicks), `hold` (one constant push for the episode) and `release` (one
kick at step 0, then nothing). A learner never sees the held-out policies during a stream.

`mechanism_batch(...)` packs several episodes per sequence with reset flags, exactly as
`physics_batch` does; `MechanismEnv` is the per-step reference used by the tests.
`poison_stream(batch, bias)` adds a constant bias to the x-velocity delta whenever the x
action is positive: a consistent, learnable lesson that is false in every world.

### `plastic/eval/contract.py`

A `Learner` protocol with four methods: `snapshot_slow`, `restore_slow`, `consume(stream)`
(the lasting update; returns a record with `accepted`), and `step_mse(batch, adapt)` (per-step
MSE from a fresh state; `adapt=False` freezes the fast path, so it measures what the slow
parameters carry on their own).

`run_contract(learner, spec, seed)` produces one report with, for each of *before*, *after*
a clean stream, *after revert*, *after a poisoned stream* and *after a correcting clean stream*:

1. **transfer**: MSE on held-out combinations under each held-out intervention policy, with
   and without fast adaptation;
2. **speed**: on held-out worlds, at each within-episode step, the adapting error as a
   fraction of the writes-disabled error on the same inputs (1.0 means the fast path has
   removed nothing yet); reported as the mean over the first `probe_steps` steps and the step
   at which it first drops below one half;
3. **forgetting**: MSE on the training distribution (BWT with MSE);
4. **correction**: transfer after the poisoned stream (harm) and after the correcting stream;
5. **revert**: the largest absolute difference between the reverted and the before
   measurements; the report flags the run when it exceeds `revert_tolerance`.

Every batch used for measurement is generated from fixed seeds, so before and after are
compared on identical inputs. Every number carries its denominator. Compute is recorded as
tokens processed by `consume` and by measurement, and as parameter count. Acceptance is
summarized by `acceptance_rates(records)` as a pair, accepted-good and refused-bad, with
counts; an empty side is `None`, never zero.

`DynamicsLearner` wraps `PlasticDynamics` in three modes that are the contract's baselines:
`frozen` (no lasting update; with `adapt=True` this is the TTT-only baseline), `continued`
(Adam steps on the stream's loss, the continued-training baseline; the optimizer is recorded
in the report) and `in_context` (the stream is prepended as context at measurement time; the
everything-in-context baseline, with its extra tokens counted).

The correction arm starts from the reverted pre-stream snapshot, so the report gives two
readings of harm: poison minus the clean arm (`harm`, which includes the clean gain the
poisoned learner forwent) and poison minus its own start (`harm_vs_before`, the damage
alone). The same pair is reported after the corrective stream.

### Not built here

Any new learning mechanism. The coordinate block (T2) and the slow rule (T3) will implement
`Learner` and be scored by this contract. No training run beyond CPU smoke tests.

## Falsifiers and what the first report must show

- Frozen versus continued: if equal on transfer after reset, the testbed is uninformative at
  this scale (see above).
- Continued training must show a positive revert gap only if it changed anything; a revert
  gap above tolerance is a bug in the learner's snapshot, not a result.
- The in-context baseline sets the bar: a lasting update that does not beat it at matched
  compute is not earning its place.
- Poisoned stream: continued training should be harmed (that is the point of the control);
  a learner with an acceptance function should refuse it, and the pair of rates must show
  the clean stream was accepted.
