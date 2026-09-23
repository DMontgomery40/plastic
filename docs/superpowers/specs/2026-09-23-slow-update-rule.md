# A lasting update for the coordinate learner

Specification, 23 September 2026. **Proposed, not implemented or measured.**
The target is a lasting change in transferable predictive competence. The decisive
measurement is improvement on unseen mechanism combinations after clearing temporary
state and removing the experience stream, followed by loss of that improvement when
the pre-stream weights are restored. This specification extends the
[coordinate recurrence](../../research/2026-09-21-plastic-coordinate-recurrence.md)
and uses the [mechanism testbed contract](2026-09-23-mechanism-testbed-and-contract.md).
It does not resume the earlier fact-recall experiments or release a hosted model.

## 1. Decision and research readiness

Implement a **bounded, groupwise meta-learned gradient step** on accumulated observed
episodes. It updates the coordinate initialization, decay initialization and two
transition projections. Its training objective judges the resulting model on later,
separate episodes, including episodes with temporary learning disabled. The external
harness can refuse the entire lasting update. This is a concrete first hypothesis;
ordinary continued training is an essential competing explanation.

Local sources read: the [briefing](../../research/README.md), current status, original
architecture and coordinate memos, literature and safety reviews, calibration replay
audit, actual model/state code, fast-state Fisher estimator, projection helper, Sleep
gate, and the initial mechanism testbed and evaluation implementation. Relevant
corrections: writes-disabled is not frozen; an accepted proposal is not an effective
learner; fast-state Fisher data is not a slow-weight metric; a local projection is not
a finite-update guarantee. The physics adapter here consumes **7 inputs**, including
the reset flag, and predicts 4 observation deltas. The coordinate memo's older
6-input parameter count does not describe this adapter.

Primary methods checked on **2026-09-23**, including the indicated equations and
assumptions rather than only abstracts:

| Source and version | Material checked; consequence for this specification |
| --- | --- |
| [Meta-SGD](https://arxiv.org/abs/1707.09835v2), 2017-09-28 | §3, equations 2–3 and Algorithm 1 learn an initialization and update coefficients from post-update test loss. Positive coefficients shared by parameter group below are a restricted variant, not a new optimizer or an exact reproduction. |
| [TTT-E2E](https://arxiv.org/html/2512.23675v2), 2025-12-31 | §2.1–2.3 differentiate future predictive losses through inner gradient steps. Keep that path for the coordinate fast learner; within-stream gains alone do not answer the lasting-update question. |
| [Nested Learning](https://arxiv.org/html/2512.24695v1), 2025-12-31 | §7.1, equations 70–74, distinguishes update frequencies and learned initializations. Multiple timescales are established prior art; their usefulness here must be measured. |
| [Meta-learning neural system identification](https://arxiv.org/html/2501.06167v1), 2025-01-10 | §III, context/target split and inner/outer updates: closest application family for rapid adaptation of neural state-space models. |
| [Meta-learning for reference tracking](https://arxiv.org/html/2605.22513v1), 2026-05-21 | §III equations 5–13 and appendix assumptions: related newer iMAML method uses a regularized inner optimum and Hessian conditions. We use a finite unroll, not its implicit-gradient or convergence claims. |
| [A-GEM](https://arxiv.org/abs/1812.00420v2), 2019-01-09 | §4, equations 6–11 distinguish multiple gradient constraints from an average constraint. Sequential projections require rechecking every constraint. |
| [EWC](https://arxiv.org/html/1612.00796v2), 2017-01-25 | §2, equation 3 uses importance-weighted parameter protection. The empirical sensitivity metric below is a constraint heuristic, not a posterior or a protection theorem. |
| [SEAL](https://arxiv.org/html/2506.10943v2), 2025-09-18 | Algorithm 1 and limitations judge self-edits using downstream feedback. This is an alternative update-learning route; the present supervised simulator supplies observed outcomes and uses differentiable updates. |

The distinction under test is **reset-aware lasting learning coupled to the nonlinear
coordinate recurrence**, against the same recurrence with conventional updates and
against simpler decay adaptation. This bounded search does not establish priority.
The open assumptions are that observed outcomes are informative, the testbed separates
competing learners, and higher-order gradients remain practical. No result establishes
general reasoning, autonomous discovery of truth, or adversarial safety.

Three choices are deliberately separated:

1. **Main candidate:** learned group rates, reset support objective, exact finite
   meta-gradient, fast coordinate learner, external lasting-update decision.
2. **Required baseline:** fixed-rate continued training on the identical support and
   parameter subset; compare both with and without the same external constraints.
3. **Later alternative:** transfer terminal fast weights into the initialization, or
   distill a fast teacher. These may be compared separately. They are not silently
   substituted for the rule specified here.

## 2. Parameters, state and timescales

Write the entire slow checkpoint as `Phi = (U, V, psi, eta)`. `U` is the allowlisted
online-mutable subset; `V` is the remaining model state. All groups are named in an
explicit manifest, with tensor names, shapes, dtypes and ordering. Reject missing,
extra, overlapping or shape-mismatched entries; do not select parameters by a broad
substring match.

| Group, separately for each layer | Offline training | Lasting update from experience | Within an episode |
| --- | --- | --- | --- |
| `chart_init`: all `A_f, B_f, a_f, b_f, A_g, B_g, a_g, b_g` defining shared `W0` | Train | Update | Fast copies `W` may update |
| `decay_init`: shared `theta0` | Train | Update | Fast copy `theta` may update |
| `transition_gate`: `P_a` | Train | Update | Fixed |
| `transition_value`: `P_v` | Train | Update | Fixed |
| Embedding, prediction head, `P_g`, `P_o`, FFNs and norm scales (`V`) | Train | Fixed | Fixed |
| Slow rate logits `psi`, four per layer | Meta-train | Fixed | Fixed |
| Fast rate parameters `eta` from the coordinate memo | Meta-train | Fixed | Used for fast updates |
| Architectural `epsilon`, `rho`, dimensions, chunk length | Fixed configuration | Fixed | Fixed |

The initial four-layer design has 16 learned slow-rate scalars. Rates can suppress a
group toward zero, but this is a learned **static allocation**, not an episode-specific
per-weight importance selector. Compare it with a shared fixed rate before claiming
the allocation matters. `P_a` changes input-dependent decay and `P_v` changes what is
written into the recurrence. Updating only `W0` would not isolate these effects.

Three clocks are independent:

- Fast transactions occur at observed chunk boundaries, with `Phi` fixed.
- One lasting proposal occurs after **8 complete new episodes**, never mid-episode.
  A window is consumed once, accepted or refused; the next window is disjoint.
- Offline meta-training learns the initial checkpoint and rates. It is a separate
  measured cost, not an invisible online operation.

Eight is a starting configuration, not a discovered optimum or an independence
guarantee. Repeated episodes can repeat false evidence. Incomplete or all-masked
episodes do not advance the count. A finite stream with fewer than eight complete
episodes returns `insufficient_experience`; it does not fabricate a zero-loss update.
At stream completion, explicitly discard and report any undersized remainder.

## 3. Data boundaries and causal prediction

Keep these data roles disjoint by episode identity, seed and world realization:

| Role | Permitted use |
| --- | --- |
| Support `B` | Eight completed experience episodes; produces the lasting gradient. Only observed inputs, targets, masks and reset flags enter the learner. |
| Offline query `Q` | Separate later episodes from development combinations; trains `Phi` and `psi`. It is training data, not the final transfer test. |
| Protection pool `R` | Fixed trusted development trajectories used for slow sensitivity and damage probes. Outcomes were collected before screening. |
| Benefit pool `S` | Separate trusted development trajectories used only for the acceptance decision. Never differentiated to generate a lasting proposal. |
| Final test `T` | Sealed combination and intervention-policy splits, absent from every preceding role. Scored only after the run's settings and decisions are fixed. |

For the initial simulator fixture use 32 episodes in `R` and 32 in `S`, with independent
seeds and training combinations/policies only. Use separate seed namespaces from the
contract's existing evaluation generators. Store identities and hashes. Both pools
are reusable screening data and can be overfit by repeated selection; report every
attempt. Their successful scores are not held-out evidence. Different applications
need their own trusted feedback; this design assumes it and does not manufacture an
oracle from a model's agreement with itself.

Within offline meta-training, hold combinations out of each support window for its
query, using only the development combination set. The final test's held-out pairs
and intervention policies never enter meta-training, selection or calibration.
Using a final result to choose the next configuration makes that split development
data; a subsequent confirmatory result needs a new untouched split.

The learner must not read `MechanismBatch.worlds`, its `poisoned` flag, hidden
parameters, or clean/poison labels for any update or decision. Those are evaluator
metadata. A physics target arrives only with the next observation. In a future text
adapter, use observed next-token labels and masks; self-generated targets must be
identified separately. Record losses on the predictions made **before** observing
their targets. Do not rescore earlier predictions using updated weights.

## 4. Exact lasting proposal

For each episode `e`, start with canonical state zero and fast parameters initialized
from the current `Phi`: `omega = (W0, theta0)`, `z = E_W0(0)`. Roll through observed
inputs with **all fast updates and fast decay disabled**. Normal activation evolution
continues. Disabling fast learning does not detach `W0` or `theta0` from differentiation.

Let `m_et` be the observed-target mask and `ell_et` the mean squared error across the
four delta coordinates. Define:

```text
L_e(Phi) = sum_t m_et * ell_et(Phi) / sum_t m_et
L_B(Phi) = mean over the eight valid episodes of L_e(Phi)
g_g      = partial L_B / partial U_g
alpha_g  = alpha_max * sigmoid(psi_g)
delta_raw_g = -alpha_g * s_g^2 * g_g
```

Equal episode weighting prevents long episodes alone from deciding the update;
report both valid episodes and target-coordinate counts. The scale
`s_g = max(RMS(U_root_g), 0.01)` is fixed at the audit's root checkpoint. Start with
`alpha_max = 1` and `alpha_g = 0.01`. These are uncalibrated fixture settings, to be
recorded and tested, not efficacy recommendations. No momentum, weight decay, replay
of previously consumed support windows or hidden optimizer state is used online.

The proposal therefore relearns from actual accumulated outcomes under a reset
model. It does **not** average the terminal fast weights. Fast learning contributes
through the jointly trained representation and the post-update adaptation objective
below. If this turns out to be ordinary continued training with a useful learning
rate, report that result.

Let `q_g = delta_g / s_g`. Apply the uniform magnitude cap in §6 to the raw proposal.
One proposal uses one support-gradient step. Projection and backtracking may choose a
smaller candidate; they may not take extra support steps or retune `psi` after looking
at screening losses. Save raw and selected deltas separately.

## 5. Offline objective and gradient path

A meta-training sample contains two consecutive, disjoint eight-episode support
windows. Starting from the meta-checkpoint, apply the differentiable proposal and
magnitude cap after each window. After each proposal, reset completely and score a
separate query set. Queries are never added to online support during that sample.
Use the mean of:

```text
J_k = 0.5 * mean_query_loss(fast_frozen, Phi_k)
    + 0.5 * mean_query_loss(fast_adapting, Phi_k)
J   = mean(J_1, J_2)
```

The first term requires competence carried by the slow checkpoint; the second rewards
its ability to learn a new episode. In the adapting term, use the coordinate memo's
functional observed-target loss, true fast gradients and prospective boundaries.
Reset fast parameters and activations independently for every query episode. Queries
after the second window include both fresh worlds and worlds from the first window's
distribution, with new trajectories, to expose interference.

Differentiate `J` through **both slow steps**, their support gradients, the learned
rates, chart initializations, encodes/decodes, fast steps in the adapting query, and
activation carries within each episode. Use functional parameter dictionaries and
graph-building gradients. No `.data` assignment, optimizer step that breaks the
graph, detached fast initialization, or first-order replacement is the main method.
The maximum unroll is two lasting windows, recorded as a truncation horizon; there
is no truncation inside it. Longer-history claims require longer-history tests.

The immutable normalization/constraint reference is supplied as fixed input to this
derivative, including in finite-difference checks. Refreshing it is a separately
versioned offline preparation step, never part of an online stream. Initial offline
training can use `M = I`; before guarded evaluation, estimate and freeze the metric
below and record whether the checkpoint was additionally meta-trained with that
metric. Do not equate an untrained-rate smoke test with a trained learner.

The external canary projection, finite screening, acceptance branch and ledger are
outside this meta-gradient. Train the unguarded differentiable learner first; evaluate
the actual guarded trajectory separately. A detached-gradient version is a named
ablation. An implicit-gradient solver would require a different derivation and tests,
not a silent memory-saving replacement.

## 6. Magnitudes, sensitivity and the harness decision

### Fixed metric and budgets

Estimate a **slow-parameter** diagonal sensitivity at `Phi_root`, from separate
per-episode gradients on `R`, with a reset, fast-frozen forward for every episode:

```text
f_i     = mean_e (s_group(i) * partial_i L_e(Phi_root))^2
fbar_i  = f_i / max(mean_i f_i, 1e-12)
M_i     = 1 + fbar_i
N(q)    = sqrt(mean_i M_i * q_i^2)
```

For MSE this is empirical squared-gradient sensitivity; call it that in artifacts.
It is not a true Fisher/KL metric. A likelihood-based variant must specify its
likelihood and estimator. The existing [Fisher implementation](../../../plastic/harness/fisher.py)
estimates sensitivity of fast `S` state, so its saved arrays are incompatible here.
The `1` floor keeps zero-sensitivity directions subject to a norm budget.

Starting fixture budgets are `N(delta/s) <= 0.01`, every group's `RMS(q_g) <= 0.01`,
`N((U_candidate-U_root)/s) <= 0.05`, and cumulative **accepted path length**
`sum N(actual_step/s) <= 0.10`. These dimensionless caps depend on this
parameterization; they are not behavioral distances or calibrated safety limits.
Scale the entire concatenated proposal by one factor to meet per-step, per-group and
remaining-path limits. Do not clip groups independently after projection.

No fast reset, resume, successful canary test or rejected attempt replenishes the
lasting budget. An evaluation revert restores the whole original ledger; an online
rollback must remain an audited event and must not create fresh budget. Resource
accounting counts every attempted step, including rejected candidates and probes.

### Future probes and projection

Define eight protection groups by partitioning the fixed `R` episodes deterministically
into groups of four. Each group supplies two losses: teacher-forced next-delta MSE and
four-step open-loop observation MSE. For the latter, start after a four-observation
prefix, reuse recorded actions, feed the model's predicted next observation back as
input, and compare with the already recorded trajectory. Use valid starting positions
only and record their count; insufficient trajectory length is unavailable, not zero.

Every probe starts from reset and replays its fixed prefix using the **candidate's**
parameters. Its fast weights stay frozen throughout. Differentiate through prefix,
initial chart encoding and future rollout when computing canary gradients. Do not
reuse an old latent state under a new representation or probe only an encode/decode
identity. These are counterfactual future-of-prefix probes on previously observed
data, not access to outcomes that have not occurred.

For each of the 16 losses `Q_j`, form `h_j = s * grad_U Q_j(U_current)`. Starting
with the normalized raw proposal `q`, enforce `dot(h_j,q) <= 0` by at most ten
cyclic projection sweeps in the diagonal `M` metric:

```text
q <- q - max(0, dot(h_j,q)) / dot(h_j, M^-1 h_j) * M^-1 h_j
```

Use the exact denominator when nonzero. A zero gradient contributes no restriction;
nonfinite values or a numerically unsatisfied degenerate constraint fail the attempt.
After every full sweep, check **all** constraints. Stop when every violation is at
most `1e-8 * max(1, norm(h_j)*norm(q))`; otherwise reject after ten sweeps. This is a
bounded feasibility procedure, not a claim of the nearest projection. The existing
[single-halfspace helper](../../../plastic/harness/projection.py) is not sufficient
for the multi-constraint rule without these rechecks.

Apply one uniform scale for the magnitude/path caps, then try at most eight candidates
at factors `1, 1/2, ..., 1/128`. The first passing factor wins; no search for the best
screen score. For each candidate, cast to its persisted dtype, compute the actual
representable delta, and recheck finite values, every tangent constraint, group/step,
root-displacement and remaining-path budgets. A nonzero raw proposal that rounds to
no change returns `no_change`, never a beneficial acceptance.

### Finite decision

Evaluate every actual candidate on every required protection loss and the separate
benefit pool `S`, from reset. Let
`tau_j = 1e-6 + 0.01 * max(Q_j(Phi_root), 1e-6)` and
`nu = 1e-6 * max(1, L_S(Phi_root))`. The initial guarded fixture accepts only if:

```text
all Q_j(candidate) <= Q_j(current) + tau_j
all Q_j(candidate) <= Q_j(root)    + tau_j
L_S(candidate) <= L_S(current) - nu
all state, identity, finite-value and budget checks pass
```

The root comparison prevents repeated small tolerated losses from accumulating
unchecked. The benefit score is episode-mean, fast-frozen next-delta MSE. These
thresholds are explicit experimental settings, not a policy-level false-positive
rate. `R` and `S` must cover legitimate development regime changes; otherwise the
guard may block useful learning. Missing, empty or stale probes make guarded mode
unavailable. Exploratory `log_only` may still run and records what the guard would
do, but must be labeled unprotected.

The current [Sleep gate](../../../plastic/sleep/ttt.py) is a separate damage screen.
Passing it does not implement this lasting-update rule, its benefit condition or
the final transfer contract. A tangent condition is only a first-order heuristic;
even passing the finite probes says nothing guaranteed about unseen behavior.

## 7. Atomic commit, reset and protocol

A lasting commit is allowed only at a quiescent episode boundary. Build and test a
functional candidate isolated from the live model. Recheck its parent checkpoint
identity immediately before committing all allowed tensors and the ledger atomically.
On a mismatch or error, commit nothing. Frozen tensors must be byte-identical.
Changing `P_a` or `P_v` changes how a past trajectory would have evolved; the chart
transport identity cannot reconstruct that counterfactual history.

After a commit, the next episode therefore starts from full reset under the new
checkpoint. Other active sessions remain pinned to their old signature; never silently
reinterpret their state. Integrating multi-session publication is later work.

Reset clears all activation/convolution state, canonical and latent carries, pending
chunk targets/statistics, fast optimizer state, retrieval/context caches and fast
parameters. It recreates fast parameters from the **current candidate's** `W0/theta0`,
not from the pre-stream checkpoint. The slow budget and root reference survive reset.
Final evaluation has no access to consumed support, including indirectly via a loader,
closure or cached teacher. It may observe the new task's causal input history.

Implement the four methods of the [evaluation protocol](../../../plastic/eval/contract.py):

- `snapshot_slow()` returns a deep, immutable snapshot of all model tensors, rates,
  configuration/signature, root/metric/probe identities, mutable buffer/cursor state,
  ledger and learner RNG needed to reproduce future behavior. Data can be addressed
  by verified immutable references. No tensor alias may mutate the snapshot.
- `restore_slow(snapshot)` validates identities, restores that whole state and clears
  transients. The external evaluator's cumulative cost ledger is not rolled back.
- `consume(stream)` splits complete episodes, forms disjoint windows, proposes and
  records transactions. Its aggregate `accepted` is `None` with no valid decisions,
  otherwise whether any window committed; this is a **stream-level** compatibility
  field. Return individual transaction records for transaction-level rates.
- `step_mse(batch, adapt=...)` returns per-step coordinate-mean MSE from fresh state
  per episode. `adapt=False` freezes all coordinate fast parameters; `adapt=True`
  uses observed targets only after their predictions. It never changes slow tensors,
  budgets, support buffers or screening data. Restore local RNG after a read-only
  measurement; charge its real cost to the separate evaluator ledger.

The adapting path must locally enable gradients for fast proposals even when called
by an evaluator under `no_grad`; discard its temporary graph after the measurement.
The main meta-training path instead preserves that graph for the outer derivative.
Do not put a blanket `no_grad` wrapper around the coordinate learner's inner update.

Each transaction records parent/root/candidate identities; manifest and data hashes;
valid counts; group rates and scales; raw/projected/actual norms; budgets before/after;
projection sweeps and backtrack factor; current/root/candidate probe losses; decision
and reason; device/dtype; and detailed resource counters. `signals` describe the raw
proposal; `accepted` describes retained mutation. On rejection retained delta is zero,
but the proposal and its counterfactual measurements remain recorded.

Required reasons include `accepted`, `insufficient_experience`, `no_change`,
`budget_exhausted`, `invalid_numeric`, `probe_unavailable`, `projection_failed`,
`damage`, `no_screen_benefit`, and `parent_changed`. No new public API or UI is part
of this specification.

## 8. What the reset and revert experiment must show

Use the same held-out inputs and evaluator randomness before and after. For each
world/policy, report raw MSE and denominators, then paired differences:

```text
G_reset  = loss(before, fast_frozen) - loss(after, fast_frozen)
G_adapt  = loss(before, fast_adapting) - loss(after, fast_adapting)
R_error  = abs(loss(reverted) - loss(before))
G_revert = loss(reverted, fast_frozen) - loss(after, fast_frozen)
```

A positive `G_reset` is the primary lasting-transfer signal. Revert must reproduce
the pre-stream model within the declared device tolerance, and `G_revert` must
recover that measured gain. Revert equality without an initial positive gain proves
only snapshot correctness. Improvement only in `G_adapt` is improved learnability,
a useful but different result. Include prediction-level revert error, not only a
scalar mean that could hide offsetting mistakes.

Report adaptation curves in absolute MSE as well as normalized form, area, and the
threshold-crossing count with a **not reached** flag. Dividing by each model's own
first-step loss can reward a worse initial prediction; a curve score alone is not
enough. Report forgetting on the pre-stream distribution with both fast modes.

For poison and correction, save a fresh pre-poison snapshot. Measure
`loss(after_poison)-loss(pre_poison)` as harm and
`loss(after_correction)-loss(pre_poison)` as residual harm. Also compare correction
with a clean-only arm started from the same pre-poison snapshot. A poison arm that
starts before a beneficial update must not call its deficit versus the clean-trained
arm actual damage from poisoning.

Report both clean-stream acceptance and poisoned-stream refusal, with counts and
invalid decisions separated. Do not label an update beneficial solely because its
stream was nominally clean. Separately evaluate each saved **uncommitted candidate**
on the final contract, after decisions are sealed, to classify measured benefit,
harm, no detectable effect and invalid outcomes using declared effect thresholds.
Give utility-conditioned accepted-good/refused-bad rates beside the provenance rates.
Never feed those final labels back into that run's guard. Missing sides are `None`.
The candidate probes are additional measured evaluation cost.

Required controls include reject-all, unguarded accept-all, matched guarded and
unguarded candidates, and a fast-only learner. Rejection alone is not evidence of
protection; the unguarded candidate must cause measured harm for that claim. If the
guard prevented harm, correction may be unnecessary; also test correction from the
same genuinely damaged checkpoint in an explicitly separate arm. Repeated poisoning,
beneficial shifts and budget exhaustion must be visible in longer sequences.

The current contract draft provides the four-method interface and first measures.
The exact reset semantics, complete snapshots, prediction-level revert check,
transaction denominators, poison-start comparison and cost breakdown above are
integration requirements, not claims that the draft already implements them.

## 9. Matched comparisons and resource accounting

Start every paired arm from the same compatible trained checkpoint. The existing
`phys_mps_3k` is an initial baseline reference; its tensors are not a trained
CoordinateBlock checkpoint. Report architecture-specific pretraining separately.

| Arm | What is held constant or isolated |
| --- | --- |
| No lasting update; fast enabled/disabled | Same initial model; establishes temporary adaptation versus lasting change |
| Fixed-rate SGD on the same allowed groups | Same episodes, reduction, steps, scale and caps; isolates learned allocation/objective |
| Continued training of all weights | Same observations and separately matched wall time; optimizer and step count explicit (the current baseline code uses Adam) |
| Replay/importance-protected continued training | Same access to trusted data and screening resources; avoids giving only the candidate retention help |
| Full slow rule with fast `W` frozen or fast `theta` frozen | Isolates nonlinear coordinate learning and decay learning, with the same slow rule |
| Slow `W0` fixed, or slow `P_a/P_v` fixed | Separates changed initialization from changed representation/transition matrices |
| Detached versus exact meta-gradients | Same architecture and data; report meta-training time as well as online time |
| Everything in context / retrieval | Same available experience; count storage and repeated prefill. Context access is explicit and does not pass the stream-removed durability criterion |
| Simple system-identification predictor | Fit the toy system's available parametric family from the same observations; exposes an unnecessarily easy benchmark |

Do not claim all resource dimensions are exactly matched in one run. Publish a table
of total parameters, offline trainable parameters, online mutable parameters, learned
rates, and per-session fast state. Record persistent model/metric/root/snapshot bytes,
support and reference-pool bytes, optimizer bytes, peak device allocation and process
memory separately. Shared immutable storage and per-session storage are separate.

For wall time, synchronize the device at interval boundaries and include ingestion,
all support backward passes, projection gradients, rejected/backtracked probes,
serialization/reset, and subsequent predictions. Report these components plus total
end-to-end time. Separate one-time metric preparation, offline meta-training and
hyperparameter search from per-stream costs; do not hide any of them. Count processed
input positions, observed target coordinates, backward/Hessian-vector work and probe
calls; “tokens consumed” alone omits repeated optimization work. Report FLOPs only if
actually measured or explicitly estimated with a stated method.

Give both an equal-data/steps comparison and a wall-time-limited comparison on the
same device, with equal development search allowance. A context baseline may spend
its budget on more context; a conventional optimizer may spend it on more steps.
It is a result if either wins. No CPU smoke test supports an MPS/CUDA throughput or
one-hour training claim.

## 10. Implementation tests and falsification

Tests precede the corresponding implementation. Use the existing Python stack and
small synthetic fixtures; no paid training is needed to validate these contracts.

| Contract family | Required cases |
| --- | --- |
| Update scope | Every allowed group can receive gradients; frozen tensor identity; missing/extra/aliased manifest entries rejected |
| Causality and partitions | Ragged masks, delayed targets, partial final chunk, episode resets, different input partitions; no query/final/hidden-label access |
| Exact gradients | Float64 directional finite differences through two slow updates, learned rates and at least two fast updates; reset initialization remains connected; checks away from clipping kinks |
| Constraints | Zero/tiny/collinear/conflicting canary gradients, later projection reviolating an earlier constraint, nonfinite values, all-zero sensitivity, dtype rounding and per-group/root/path boundaries |
| Atomicity | Accept/reject/error/stale-parent transitions; rejection leaves all persistent model state unchanged; budgets charged only for actual retained mutation |
| Lifecycle | Deep snapshot, fork, restore, reset, insufficient buffer, empty probes, consumed-window deduplication; resetting fast state cannot replenish slow budget |
| Measurement | Deterministic prediction-level revert, no cached-stream access, read-only probes, unavailable crossing counts, mixed transaction decisions, absent/invalid rate denominators |
| Resources | Rejected attempts and candidate evaluations counted; context prefill and metric preparation counted; evaluator ledger survives model revert |

Keep these possible outcomes explicit in the report:

| Observation | What fails or survives |
| --- | --- |
| Gain disappears when temporary state or context is cleared | No lasting competence shown |
| Revert does not recover pre-stream predictions | Persistence/evaluation defect; do not interpret a learning effect |
| Reset gain survives but ordinary continued training matches it | Durable learning may exist; the specialized slow rule has no demonstrated advantage |
| Decay-only matches the complete coordinate learner | No demonstrated benefit from nonlinear fast coordinates |
| Only adaptation speed improves | Learned initialization helps future adaptation; direct frozen-fast transfer remains unproved |
| Benefit screening passes but sealed transfer does not improve | Screening is not an adequate proxy for the intended outcome |
| Guard refuses poison and useful updates alike | Reject-all behavior, not a successful learning/safety tradeoff |
| Poison candidates were harmless without the guard | This attack set does not establish protective efficacy |
| Parametric identification solves all held-out cases | Testbed does not yet discriminate the architectural hypothesis |
| Accurate finite gradients but no useful result at measured cost | Mathematical implementation succeeds; practical hypothesis fails at that scale |

Release an evidence package with checkpoint/config and split identities, immutable
before/after/reverted tensors or retrievable manifests, paired per-world measurements,
all decision records, seed-level uncertainty and the resource table. Report inference
by world/episode rather than treating correlated timesteps as independent samples.
Exploratory single-seed results are useful when labeled; they are not replication.
This document supplies an implementable proposal and its failure conditions, not a
claim that any of those conditions have already been passed.
