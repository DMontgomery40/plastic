# Fast-weight coordinates for a selective recurrence

Research proposal, 2026-09-21. This goes beyond the existing architecture memo; it is not a claim of established novelty or demonstrated task performance. Only the mathematical CPU probe described below has been run. No training run, deployment, or production-backbone change is part of this proposal.

## Decision and intended contribution

Replace the separate associative-memory branch with **gradient-trained recurrent dynamics**. A small nonlinear fast network defines invertible coordinates in which activation evolution is a selective linear recurrence. Learning changes those coordinates and the recurrence's decay rates. At a transaction boundary, change coordinates while preserving the represented activation state exactly.

The intended contribution is the combination of:

1. A nonlinear fast learner that changes the recurrence itself, rather than supplying an additive retrieved value.
2. An exact state-preservation contract for accepted, projected, and rejected updates.
3. Bounded-displacement coordinates that prevent activation-amplitude blow-up even under arbitrary sequences of accepted weight changes.

Neither nonlinear TTT, invertible coordinates, meta-learned system identification, nor the chain rule is new. Whether this particular coupling earns architectural novelty requires comparison with the nearby work and empirical falsification below.

The constraints are 1–10M trainable parameters, one shared stack for text and hidden-friction physics, ordinary PyTorch, and a one-hour L4/A10G training target. The user confirmed the comparison candidate is `2026-09-21-architecture-memo.md`.

## Why the existing proposal does not answer the architectural question

Its RG-LRU plus Gated DeltaNet is a useful baseline, but its memory update is the established linear delta rule. Replacing that matrix with a TTT-MLP would satisfy nonlinear inner learning without establishing a new coupling.

There are also contract problems to avoid carrying forward:

- Setting beta to zero leaves alpha decay active. That is not frozen fast state.
- Beta times residual norm measures the gradient correction; the total mutation also contains `(alpha - 1) S`.
- Projecting lower-layer fast weights after provisional writes can leave upper-layer activations influenced by the rejected weights.
- A half-space projection offers a local, first-order statement, not finite-step canary protection.

These observations concern the memo; this research task does not modify its implementation.

One literature qualification matters: [Liu et al.](https://arxiv.org/html/2602.21204v4) analyze KV-binding inner objectives and a linear bias-free terminal layer, with history-dependent features in their representation. Their result does not establish that every nonlinear gradient-trained recurrent dynamics model is fixed-feature linear attention. Nonlinearity alone also does not establish usefulness or novelty.

## The block

Use four blocks with width D=256 and H=4 heads of width d=64. Within every chunk, all fast parameters are fixed. All layers consider their candidate updates together at the boundary.

For one head, split canonical activation state r into p,q of width d/2. Define two bounded additive coupling functions:

```
F_W(q) = epsilon * tanh(B_f tanh(A_f q + a_f) + b_f)
G_W(y) = epsilon * tanh(B_g tanh(A_g y + a_g) + b_g)

z_1 = p + F_W(q)
z_2 = q + G_W(z_1)
E_W(r) = [z_1, z_2]
```

Both layers of both coupling MLPs are fast weights. Hidden width 16 is the starting configuration. The inverse needs no solve:

```
q = z_2 - G_W(z_1)
p = z_1 - F_W(q)
E_W^{-1}(z) = [p, q]
```

Initialize the coupling output matrices to small nonzero values, not all-zero input and output matrices. Learn the initial fast weights W_0 in the outer loop. Epsilon is fixed, initially 0.1–0.2; it controls amplitude, not an optimizer learning rate.

With u_t=RMSNorm(x_t), the full-width selective recurrence is:

```
a_t = rho * sigmoid(P_a u_t + theta_c)   # theta_c is fast, rho < 1 is fixed
v_t = tanh(P_v u_t)
z_t = a_t * z_{t-1} + (1 - a_t) * v_t
r_t = concat_heads E_{W_c}^{-1}(z_t)
x_t = x_t + P_o [r_t * SiLU(P_g u_t)]
x_t = x_t + FFN(RMSNorm(x_t))
```

P_a,P_v,P_g,P_o and the FFN are slow weights. Gates depend on the current layer input, not that layer's recurrent state. The fast parameters are omega=(W,theta). At a boundary whose canonical carry is r_start, initialize latent carry as z_start=E_W(r_start). The actual canonical transition is therefore

```
T_omega(r,u) = E_W^{-1}(a_omega(u) * E_W(r)
                           + (1 - a_omega(u)) * v(u)).
```

This is linear in latent z while W is held fixed, and nonlinear in both the canonical state and the trainable fast weights. It has no separate KV memory.

Fast theta is necessary to change timescales. At corresponding fixed points, smooth conjugacy alone preserves local eigenvalues. A chart-only model should not be claimed to identify a changed friction eigenvalue through conjugacy alone. Adaptive theta changes those eigenvalues; adapting W can change which nonlinear state coordinates they act on.

The intended benefit is context-dependent transition structure: information can be retained or transformed differently after learning. The risk is equally concrete: fast theta might do all the useful work and make the nonlinear map redundant.

## Exact boundary semantics

The authoritative activation carry is canonical r, not the chart-specific z. Given a current terminal latent state z_end and an accepted proposal omega_prime,

```
r_end = E_W^{-1}(z_end)
z_next = E_W_prime(r_end)
```

Consequently decoding z_next with the accepted chart gives exactly r_end, up to arithmetic error. Writing T_ab=E_b composed with inverse(E_a),

```
T_bc(T_ab(z)) = T_ac(z)
T_ba(T_ab(z)) = z.
```

These identities hold for coordinate changes without intervening observations. They do not say learning commutes with processing inputs.

All predictions and recurrent activations inside a chunk use committed omega_c. The proposal is evaluated only at the boundary. On rejection, retain omega_c and the actual r_end reached while processing the chunk. On acceptance or projection, retain that same r_end and use the selected omega next. No candidate fast weights have influenced upper-layer activations, and no chunk replay is required merely to undo a rejected candidate.

This is a **prospective dynamics update**. It does not reconstruct what the activation would have been had the new weights processed the preceding history. That stronger promise is generally impossible. For h_t=a h_{t-1}+x_t, h_0=0, the histories (0,1) and (2,0) both end at h_2=1 when a=.5; replay at a=.25 ends at 1 and .5. The old compressed state cannot distinguish them.

Rejecting learning also does not erase the observed chunk from activation context. If an application needs content deletion rather than learning rejection, it needs separate replay/redaction semantics.

An all-state abort restores the pre-chunk snapshot and cursor. A learning rejection advances the cursor and activation carry while keeping fast weights frozen. Keep these operations distinct in the session API. Snapshot/fork state includes all layers' omega and canonical r, pending observed targets, stream position, reset state, budget ledger, and any optimizer state if subsequently introduced. Start with plain inner SGD and no momentum. Reset via canonical r=0 and z=E_W(0), not unconditional z=0. A frozen mode performs no fast-parameter decay or mutation.

## Genuine inner learning and meta-training

Use the actual prediction objective, as in the E2E TTT direction: text next-token cross-entropy or next-observation MSE. The embedding and prediction-head/loss adapter are domain-specific; the blocks, inner update, state representation, and transaction code are identical.

Define a functional chunk loss

```
L_c(omega; r_start, observed_chunk, Phi)
```

that re-encodes r_start using its omega argument and rolls out the chunk. Phi denotes slow parameters. Differentiate with respect to omega while treating canonical r_start as an independent argument:

```
g_c = partial_omega L_c(omega_c; r_start, data_c, Phi)
omega_candidate = omega_c - eta_c * g_c
```

Eta is a positive, bounded, meta-learned scalar per layer, optionally conditioned on the mean support representation. The outer loop differentiates future predictive losses through the gradient step, the evolving nonlinear weights, and activation carry. No first-order stop-gradient approximation is part of the main proposal.

`torch.func.grad_and_value(..., argnums=0, has_aux=True)` is a suitable functional implementation. Holding r_start independent in the inner partial derivative must not detach it from the outer graph. The auxiliary terminal canonical state remains connected to the outer graph and is retained unchanged by the proposal. The checked probe exercises this through three updates.

At inference, rebuild a differentiable support replay from the saved canonical chunk-start state and actual observed inputs to obtain the gradient; ordinary inference can otherwise run without retaining a graph. Replay is for computing the inner gradient, not replacing the actual terminal carry with a counterfactual carry.

Causality is explicit. At a text chunk boundary, update only from targets already observed; exclude its last next-token target if that token has not arrived. Never retrospectively report an updated model's score as the score of an earlier prediction. For physics, consume a transition's target only after the environment supplies the next observation. A common observed-target mask implements both cases. Score adaptation on future targets. Self-generated tokens, if later used as support, are pseudo-labels and must be identified as such.

Meta-train the unguarded differentiable learner first. The external guard remains an explicit inference-time decision; it is not a differentiable safety approximation. Testing with guards active is necessary because they alter the adaptation trajectory.

## Boundedness under changing charts

Stable eigenvalues of individual charts do not imply stability under switching. For example, the two matrices

```
F1 = [[.25, 1.  ], [0. , .75]]
F2 = [[.25, 0.  ], [1. , .75]]
```

are each diagonalizable with eigenvalues .25,.75, but the spectral radius of F2 F1 is about 1.603069. Unrestricted learned conjugacies can therefore explode under alternating updates.

The bounded additive couplings above supply a different invariant. For every W, regardless of its parameter magnitudes,

```
||E_W(r) - r||_infinity <= epsilon
||E_W^{-1}(z) - z||_infinity <= epsilon.
```

Each half of the vector is shifted only once. With ||v_t||_infinity<=V and a_t<=rho<1, a chunk of C steps has per-coordinate carry product P_i<=rho^C. If ||r_start||<=R and R>=V,

```
||r_end|| <= V + rho^C (R-V) + epsilon (1+rho^C).
```

The proof is the convex-combination bound on the affine recurrence, plus at most epsilon displacement at encode and decode. For arbitrary R<V, a valid bound is `V + rho^C max(R+epsilon-V,0) + epsilon`. Thus, for fixed chunk size C, all boundaries obey

```
B_C = max(R_initial, V + epsilon * (1+rho^C)/(1-rho^C)).
```

A weaker uniform bound, including within chunks and arbitrary positive chunk lengths, uses C=1. These bounds require an explicit uniform rho cap; an uncapped gate merely below one does not suffice. Epsilon and rho are fixed architectural constraints, not freely growing fast weights.

This elementary bound prevents unbounded activation amplitude, including under arbitrary accepted parameter changes. It does not prove contraction, bounded parameter sensitivity, well-conditioned derivatives, good predictions, or adversarial safety. Near-one rho makes the weaker bound loose. Steep tanh networks can still produce large local derivatives. Measure this and use outer gradient clipping; a stronger Lipschitz constraint is a separate ablation, not a claimed property of this proposal.

## External transaction harness

At the boundary, retain canonical r_end. Define each canary score as a future rollout from a fixed canonical starting state:

```
Q_j(omega; r_ref) = canary_loss_j(rollout(omega, E_W(r_ref))).
```

Use multiple reference states, including r_end and domain-appropriate clean references. Freeze fast weights for the complete canary rollout and discard all probe activations. A reference-friction canary can reject legitimate adaptation to another friction regime, so calibrate on held-out benign regime changes as well as attacks.

The correct gradient includes the chart's effect on the initial latent state:

```
g_j = partial_omega Q_latent
       + J_omega E_W(r_ref)^T * partial_z_start Q_latent.
```

This is ordinary chain rule, not a new derivative. Its necessity is testable. For E_w(r)=r+w and latent z_next=a*z, the canonical next state is a*r+(a-1)*w; its correct derivative is a-1. Holding the old latent state fixed incorrectly gives -1. A canary that only reads the current decoded state has exactly zero gradient after encode/decode cancellation. It cannot detect changed future dynamics.

Let delta be the concatenated fast-parameter proposal, including all chart weights and decay logits. For one canary, project with

```
delta_safe = delta - max(0, dot(g,delta)) / (dot(g,g)+tiny) * g.
```

This is approximate when `tiny` is nonzero; use the exact denominator above a chosen nonzero-gradient threshold if requiring an exact tangent half-space. For multiple canaries, solve the small intersection projection, or iterate with constraint rechecks. A single sweep can re-violate earlier constraints. If projection, backtracking, or constraint verification fails within a fixed attempt budget, reject.

After projecting, cap the full concatenated parameter step norm and cumulative accepted path length. Evaluate the actual finite proposal on every required canary before committing. Compare both with the previous state and an immutable reference; only comparing consecutive steps permits tolerance-sized cumulative drift. The cap includes every fast mutation; the first implementation has no hidden momentum or parameter decay.

Additionally record a function-space diagnostic: mean squared difference between old and proposed future predictions on fixed canonical-state probes. It detects parameterization artifacts in raw norm budgets, but a finite probe set is not a global behavioral guarantee. Avoid claiming all acceptance decisions are reparameterization invariant while retaining a raw parameter norm cap or ordinary parameter-space SGD.

The guard's decisions and ledger live outside the model graph. Canary gradients are computed in a separate probe graph. External nondifferentiability makes the policy explicit; it does not prevent black-box or surrogate attacks.

## Size and execution target

For D=256,H=4, head coupling hidden width 16, two RMSNorms, four bias-free D-by-D projections, and a two-layer FFN of expansion 2 with no biases:

- Slow block matrices: 8 D^2 = 524,288 parameters.
- Norm scales: 512.
- Fast chart initializations: 8,576 including coupling biases.
- Fast decay initializations: 256.
- Learned per-layer chunk-rate projection and bias: 257.
- Total: 533,889 parameters per block.

Four blocks, tied 4096-token embeddings/head, and a final RMSNorm give **3,184,388 trainable parameters**. Four blocks carry 35,328 fast scalar values plus 1,024 canonical activation values per sequence, about 142 KiB in fp32 before snapshots, support buffers, and autograd storage. These are arithmetic counts for the specified design, not a measurement of an integrated model.

For physics, replace the embedding with Linear(6,256) for `[position, velocity, action]`, and the head with Linear(256,4), optionally predicting observation deltas through the head adapter. With biases in these two adapters, the same stack has 2,138,632 trainable parameters. Hidden mu is never an input. Use the same external episode-reset semantics for both domains.

Affine pairs compose associatively:

```
(a2,b2) o (a1,b1) = (a2*a1, b2 + a2*b1).
```

A doubling scan uses only multiplication, addition, concatenation, tanh, sigmoid and matmuls, with no division by tiny decay products. Cost is O(C D log C) per chunk; C is fixed, so work is linear in sequence length. Decode all within-chunk states in a batch. Nonlinear fast updates remain sequential across chunk boundaries. Use fp32 recurrent state and fast updates initially.

Start with C=32,T=256,B=8. Time the complete meta-training step, including higher-order gradients, before choosing a token budget. A proposed 8M-token run with 10 minutes reserved for startup and evaluation needs about 2,667 training tokens/second to fit one hour. That is a go/no-go requirement, not a throughput prediction. Reduce the planned dataset/experiment or reject the budget claim if measured throughput misses it. Multiple ablations and seeds are additional runs; the full research study is not promised in one GPU-hour.

MPS suitability is unverified in this session: installed torch reports MPS built but unavailable. No CUDA timing was performed. The primitive set avoids specialized kernels, but actual device tests, including higher-order backward, are required.

## What would falsify the contribution

Run matched parameter, state-memory, and wall-clock comparisons; report all three because equal parameter counts alone are insufficient:

| Variant | Question |
|---|---|
| Existing RG-LRU + GDN memo baseline | Does the proposal improve anything against the original candidate? |
| Selective recurrence + ordinary TTT-MLP | Is coupling learning to dynamics useful? |
| Same coordinate model, fast updates disabled | Are test-time updates doing useful work? |
| Frozen W, adaptive theta | Does simple timescale adaptation explain the gain? |
| Adaptive W, frozen theta | Do nonlinear coordinates contribute separately? |
| Full model, no meta-gradient through updates | Is meta-training necessary? |
| Full model, incorrect fixed-z commit | Does state transport prevent the predicted discontinuity? |
| Full model, correct versus fixed-z canary gradients | Does the guard evaluate the intended intervention? |

The incorrect variants are diagnostic controls, not acceptable deployment alternatives.

Text: held-out BPE next-token loss, adaptation curves after distribution changes, MQAR measured at explicit positions relative to chunk boundaries, and loss after deliberately adverse support chunks. Include an in-chunk recall evaluation: the fast weights cannot learn until the next boundary, so short-span retrieval rests on activation state. Do not hide this latency cost.

Physics: next-observation MSE after unseen mu changes, adaptation speed, and held-out action rollouts. Include the analytic or least-squares scalar friction estimator available for the toy simulator. A recurrent activation can infer mu without gradient updates; therefore lower MSE or a decodable mu probe alone does not prove that fast learning matters. Use freeze, reset, and fast-state-swap controls on matched activation states.

Safety: compare benign loss/adaptation against held-out attack damage at matched rejection rates and budgets. Include repeated subthreshold poisoning, attacks targeting the canary blind spots, and legitimate mu/topic switches. Count candidate probes and rejected attempts in latency. Robustness does not follow from a finite canary set or from the bounded-state proof.

Abandon the architectural claim if frozen coordinates plus adaptive theta match the full model, or if an ordinary TTT-MLP is as good at matched compute. Retain the transaction semantics as an engineering result only. A 3M-parameter success would support a toy mechanism claim, not scaling to frontier language models.

An independent alternative considered was storing exact decaying polynomial moments of a nonlinear learner's full historical objective. It has an attractive sufficient-statistic invariant, but a quadratic network lifts into fixed polynomial features, creating a strong direct-regression baseline and a weaker answer to the requested departure. It is not the recommended first experiment.

## Nearest prior art and claim boundary

- [Sun et al., TTT](https://arxiv.org/abs/2407.04620): gradient-updated fast models and meta-training are prior work.
- [TTT-E2E](https://arxiv.org/abs/2512.23675): predictive task loss as the inner objective is prior work.
- [FlowDMD](https://arxiv.org/abs/2306.17396): invertible coupling-flow coordinates around linear latent dynamics are prior work.
- [Meta-Learning for Physically-Constrained Neural System Identification](https://arxiv.org/abs/2501.06167): gradient-based meta-learning for rapid adaptation of neural state-space models is prior work.
- [MetaKoopman](https://arxiv.org/abs/2607.26345): meta-learned online operator adaptation also exists in a Bayesian form.
- [Modular TTT](https://arxiv.org/abs/2608.07110): swapping nonlinear fast networks and learning-rule components is already an explicit design space.

The proposed departure is a selective sequence block whose nonlinear fast learner changes its own recurrent coordinates, whose transactions preserve canonical carry, and whose bounded chart displacements retain an amplitude invariant under arbitrary adaptation. I found no exact match in a bounded primary-source search; that is not proof of priority. The strongest publishable claim would require the ablations above to show that this coupling improves adaptation or retention per unit compute beyond its known constituents.

## Evidence produced in this task

Reproduce the mathematical probe from the project environment:

```
.venv/bin/python docs/research/probes/check_plastic_coordinates.py
```

The saved run used torch 2.14.0, CPU, float64. It checks an eight-dimensional head, not the proposed full model:

| Check | Observed result |
|---|---:|
| Encode/decode maximum error | 1.11e-16 |
| Transport preservation, composition, round trip | 0 in this run |
| Parallel versus sequential recurrence error | 1.11e-16 |
| Parallel versus sequential fast-gradient error | 3.47e-18 |
| Three-update outer directional derivative relative error | 3.26e-10 |
| Learned learning-rate derivative relative error | 3.36e-8 |
| Zero-horizon canary gradient | 0 |
| Future-input perturbation effect on earlier outputs | 0 |
| Changed nonlinear chart effect on future state | 0.05657 maximum absolute change |
| Unrestricted conjugacy switching spectral radius | 1.603069 |

The probe also checks activation amplitudes over 100 large randomly changing bounded charts. These calculations validate identities and a differentiable path, not learning quality, adversarial robustness, fp32/MPS parity, inference throughput, or the one-hour training target. The code is a research probe, not an integrated model or harness implementation.

Repository verification at closeout: `.venv/bin/python -m pytest` reported 42 passed and one failure in `tests/test_chunk_rule.py::test_newton_schulz_orthogonalizes` (the test's lower singular-value bound). This research task did not modify that separate implementation or test. The standalone coordinate probe passed; the repository-wide suite was not green. `git diff --check` passed for tracked changes; the new research artifacts were also checked directly for trailing whitespace.
