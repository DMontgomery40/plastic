# Architecture memo: one fused TTT×SSM block for text and physics

Verified with throwaway code in `scratchpad/advisor/check_scan_and_delta.py` (chunked scan, chunk-parallel gated delta rule, stability eigenvalues, MPS) and `scratchpad/advisor/block_timing.py` (full 4-layer stack, fwd+bwd timing on MPS). Numbers quoted below come from those runs. Both files are lift-able reference implementations, not throwaway math.

## Summary of the recommendation

One block type, stacked 4–6 times, identical for text and physics:

1. **Selective diagonal recurrence (SSM branch)**: an RG-LRU-style gated linear recurrence with per-channel input-dependent decay. Activation-level state `h ∈ R^D`, forgets on a learned timescale, stable by construction, computed with a chunked log-space scan that fixes the current backbone's broken closed form.
2. **TTT memory (fast-weight branch)**: a per-head matrix memory `S ∈ R^{d_h×d_h}` updated by the *gated delta rule*, i.e. exact online gradient descent on the associative loss `½‖S k − v‖²` with a per-token learned learning rate `β_t` and forget gate `α_t`. This is TTT-Linear in Sun et al.'s sense with mini-batch size 1 plus Gated DeltaNet's forgetting; the test-time-regression view (Wang et al. 2025) and Yang et al. (2024) both describe it that way. Queries/keys/values are computed from the SSM output, so the recurrence is the featurizer for the memory. Chunk-parallel (WY form) for training, per-token recurrent for inference; the two paths agree to 4e-7.
3. **MLP**.

The inner-loop gradient has a closed form, so meta-training through the inner loop needs only first-order autograd (no `create_graph`, no `torch.func`), which is why it runs on MPS today. The harness wraps the layer at chunk granularity, treating `{h, S}` per layer as the transactional session state, and reads write pressure directly from the delta rule: with `‖k_t‖ = 1`, the per-token update is `β_t · ‖v_t − k_tᵀS_{t−1}‖`, which is simultaneously the gradient norm and the update norm. No text heuristics anywhere.

A second update rule, `rule="chunk"` (Sun et al. mini-batch TTT at chunk level with Titans-style momentum and optional Newton–Schulz/Muon orthogonalization à la LaCT), is specified as a switch behind the same interface. It is worth having because Muon-orthogonalized updates have a *fixed Frobenius norm per chunk*, which is a built-in write budget. It is not the default because per-token recall is weaker at small chunk sizes.

## A. The block

Notation: `D = 256`, heads `H = 4`, `d_h = 64`, chunk `L = 64`, sequence `T` (multiple of `L`; pad otherwise). Row-vector convention: a key `k ∈ R^{1×d_h}` reads memory as `k S`.

### A.1 SSM branch (selective diagonal recurrence)

```
u_t   = RMSNorm(x_t)                                  (D)
r_t   = σ(W_r u_t + b_r)        recurrence gate       (D)
i_t   = σ(W_i u_t + b_i)        input gate            (D)
log a_t = −c · r_t · softplus(−λ)   c = 8, λ ∈ R^D learned   ⇒ a_t = σ(λ)^{c r_t} ∈ (0,1)
z_t   = sqrt(1 − a_t²) ⊙ (i_t ⊙ W_in u_t)
h_t   = a_t ⊙ h_{t−1} + z_t                           (D)   ← state 1
x_t  += W_o ( h_t ⊙ SiLU(W_g u_t) )
```

This is Griffin/Hawk's RG-LRU: `|a_t| < 1` always, the `sqrt(1−a²)` factor keeps `‖h‖` bounded independent of the decay, and the gates are cheap. Per-channel diagonal is deliberate (matches the repo's "diagonal SSM" identity and keeps the state small).

**Parallel computation (the bug fix).** The repo computes `s_t = a^t · cumsum(u_k / a^k)`, which divides by underflowing powers; at init (`a ≈ 1e-4`) the state is exactly zero after 4 tokens and even at `a = 0.9` the error versus a sequential loop is 0.83 over 256 tokens. Replace with a chunked scan that only ever forms *pairwise* decay ratios inside a 64-token chunk, in log space:

```
C_t = Σ_{r≤t} log a_r                     (cumsum within chunk; ≤ 0)
P[t,s] = exp(C_t − C_s)  for s ≤ t, else 0    ∈ (0,1]  — never overflows, underflow = "forgotten"
h_chunk = P z_chunk + exp(C_t) · h_{chunk start}
```

Across the `T/L` chunks the carry is a Python loop (16–32 steps). Verified: max error 1.9e-5 (CPU) / 2.3e-5 (MPS) versus the sequential reference, including decays as small as 1e-3; gradients flow to `a` and `z`. Cost `O(T·L·D)`.

### A.2 TTT memory branch (gated delta rule, fed by the SSM)

```
u_t = RMSNorm(x_t)                                                   (after the SSM residual)
q_t = normalize(W_q u_t)_h,  k_t = normalize(W_k u_t)_h,  v_t = (W_v u_t)_h      per head, d_h = 64, ‖q‖=‖k‖=1
β_t = σ(w_β · u_t + b_β)               write learning rate  ∈ (0,1)   per head   (Titans' θ_t)
α_t = exp(−softplus(w_α · u_t + b_α))  forget/decay        ∈ (0,1)   per head   (Mamba-2/GDN style)

e_t = v_t − k_t (α_t S_{t−1})                     prediction error = the inner-loop gradient direction
S_t = α_t S_{t−1} + β_t k_tᵀ e_t                  = α_t S_{t−1}(I − β_t k_tᵀ k_t) + β_t k_tᵀ v_t    ← state 2
m_t = q_t S_t                                     read after the token's own write
x_t += W_o2 ( concat_h RMSNorm(m_t) ⊙ SiLU(W_g2 u_t) )
x_t += MLP(RMSNorm(x_t))
```

`S_t = α S_{t−1} − β ∇_S ½‖k S − v‖²|_{S=αS_{t−1}}` — one gradient step per token on the associative loss, with an input-dependent learning rate and weight decay. That is the definition of a TTT layer with a linear memory; Sun et al.'s TTT-Linear with mini-batch 1 is the `α ≡ 1` case (their reconstruction target is `v − k`; using it here is a one-line option: `v_t ← v_t − k_t` before the update and `m_t ← m_t + q_t` after).

**Why this parameterization.**
- *Linear memory, not MLP.* The gradient is closed-form, so the outer loop backprops through the inner loop with ordinary first-order autograd; on MPS that is the difference between "works today" and "works if second-order autograd of every op is implemented". It also gives exact per-token write pressure. An MLP memory (Titans/TTT-MLP/LaCT) is the documented extension: same interface, `torch.func`-based inner step, second-order outer loop (the parent verified MPS handles one second-order step).
- *Per-token GD, not mini-batch.* With `‖k‖ = 1`, `β ∈ (0,1)`, `α ∈ (0,1]`, each step is a contraction along `k` (eigenvalues of `α(I − βkkᵀ)` are `α(1−β)` and `α`), so the state is bounded for any input. Mini-batch TTT sums 64 gradients evaluated at the chunk-start weights; with correlated keys the update matrix `αI − Σβ_s k_s k_sᵀ` had eigenvalues down to −56 in the experiment (unstable) unless mean-scaled (then `[0.10, 0.99]`, stable but each key is written at `β/64` strength, which is why single-shot recall is weak). Sun et al.'s code divides `eta` by the mini-batch size for exactly this reason.
- *Recall.* One write at `β = 1` stores a key–value pair exactly (relative error 6e-8 in the check), which is what makes MQAR-style in-context recall work.
- *Normalization/gating placement* follows Gated DeltaNet and TTT: L2-normalized `q,k`; RMSNorm on the memory read before the residual (Sun et al. apply LN to the reconstruction, `XQ + LN(Z)`); SiLU output gate; no bias on projections.

**Chunk-parallel form (training).** Within a chunk with `γ_t = Π_{r≤t} α_r` and `S_0` the state at chunk start:

```
A[t,s] = β_t (k_t·k_s) γ_t/γ_s        for s < t (strictly lower), else 0
T      = (I + A)^{-1}                  computed as Π_{j=0}^{5} (I + (−A)^{2^j})   (A is nilpotent; 6 matmuls of 64×64)
u      = T (β ⊙ V) − T (β ⊙ γ ⊙ K) S_0            pseudo-values (rows)
m_t    = γ_t q_t S_0 + Σ_{s≤t} (γ_t/γ_s)(q_t·k_s) u_s      (inclusive causal mask)
S_L    = γ_L S_0 + Σ_s (γ_L/γ_s) k_sᵀ u_s
```

This is Gated DeltaNet's WY representation (`T = [I + strictLower(diag(β)KKᵀ)]⁻¹diag(β)` in their notation) with the decay ratios folded in. Verified against the per-token loop: output and state error 4e-7 on CPU and MPS; gradients reach `q`, `β`, `α`. `torch.linalg.solve_triangular` is avoided on purpose (no MPS kernel guarantees); the nilpotent product is all matmuls. Forward at `B=8, T=1024, H=4, d_h=64`: 10.8 ms on MPS.

**Shapes** (`B` batch, `T` tokens): `u` `(B,T,256)`; `q,k,v` `(B,4,T,64)`; `β,α` `(B,4,T)`; `S` `(B,4,64,64)`; `h` `(B,256)`; chunk internals `(B,4,T/64,64,64)`.

**Parameter count** (measured, `D=256`, `V=4096`): 1.18 M per block; embedding + untied head 2.10 M; 4 layers = 6.83 M. Tie the head to the embedding → 5.8 M at 4 layers, ~8.1 M at 6 layers. State per layer: `S` 4×64×64 = 16 K floats + `h` 256 floats.

### A.3 The alternative rule (`rule="chunk"`)

Same `(q,k,v,β,α,state) → (m, state)` interface, different inner step, applied once per chunk:

```
E = (K S_c) − V  (errors at chunk-start weights), Δ_c = (1/L) Kᵀ diag(β) E      (mean-scaled mini-batch gradient)
optionally Δ_c ← NewtonSchulz(Δ_c) · η_fixed           (LaCT/Muon: ‖Δ_c‖_F is then input-independent)
M_{c+1} = η_c M_c − Δ_c        (Titans momentum, η_c = σ(w_η · mean(u)))
S_{c+1} = ᾱ_c S_c + M_{c+1}   (ᾱ_c per-chunk decay)
reads within the chunk: m_t = q_t S_c − (1/L) Σ_{s≤t} β_s (q_t·k_s) E_s      (Sun et al.'s dual form; causal)
```

Keep it as an experiment switch, second priority. Its selling point for this repo is the fixed-norm write: an adversary can only choose the *direction* of a chunk's write, never its size, which turns "write budget" from a harness rule into an architectural invariant.

## B. Meta-training (outer loop)

**What is trained.** All slow weights: embeddings/heads, SSM projections and `λ`, memory projections `W_q,W_k,W_v,W_o2,W_g2`, the gate projections `w_β,w_α`, norms, MLPs. The fast state `{h, S}` is *not* a parameter; it starts at zero for every training sequence and is produced by the forward pass. The gates learn "when to write and when to forget" purely from next-token loss, which is the Titans/TTT-E2E recipe in miniature.

**Backprop through the inner loop.** The chunk-parallel form is a differentiable function of the slow weights; ordinary `loss.backward()` is the meta-gradient. Full BPTT within a training sequence of `T = 1024` (16 chunks): no truncation, no detach; measured 966 ms per fwd+bwd+AdamW step at `B=8, T=1024`, 4 layers, 6.8 M params on the Mac (8.5 K tok/s, fp32, no `torch.compile`). Detached state carry across consecutive sequences of one document is an option for a later long-context curriculum; do not start there.

**Data so the fast weights are actually used.**
- Text: TinyStoriesV2 *train* split (the local 22 MB valid file is 5.5 M tokens, too small for more than ~2 epochs of a 6 M-param model; pull ~50–100 M tokens of the train split inside the HF job). Concatenate stories with `<eos>` into 1024-token sequences so entities recur across chunk boundaries.
- Synthetic MQAR mixed in at ~20 % of batches: reserved key/value token ranges, `n` pairs followed by queries, sequence 512–1024, accuracy reported by number of pairs. This is the standard probe on which DeltaNet-family layers beat pure SSMs and is the cleanest evidence the memory is doing work.
- Physics: separate checkpoint, same block code (Section E).

**Optimizer.** Muon for 2-D matrices (`torch.optim.Muon`, 2-D only, confirmed) with AdamW for embeddings, norms, `λ`, biases and gate vectors — the standard Muon split; `lr` 2e-2 Muon (with `adjust_lr_fn="match_rms_adamw"`) / 1e-3 AdamW, 500-step warmup, cosine to 10 %, weight decay 0.1 on matrices, grad-clip 1.0. If Muon misbehaves on the fast-weight projections, plain AdamW 1e-3 is a fine fallback at this scale.

**Gate initialization.** `b_β` so `σ(b_β) ≈ 0.5` (DeltaNet default), `b_α` so `α ≈ 0.98` at init (`exp(−softplus(−4))`), `λ` so `σ(λ) ≈ 0.9`. Bad init here is the usual reason fast weights look useless.

**Compute.** 8.5 K tok/s on MPS ≈ 30 M tok/h locally (enough for the physics model and for smoke runs). Pure-PyTorch on an L4/A10G should land at 40–80 K tok/s → 150–300 M tokens/hour, i.e. one pass over 100 M TinyStories tokens in 20–40 min. Expect ≈1.6–1.9 nats/token at 6 M params; the current backbone plateaus at 3.2–4.3 because the recurrence is broken.

**Loss.** Plain next-token CE (text) / MSE (physics). No auxiliary "use the memory" loss; the ablation in F is the check.

## C. Session semantics and the harness

**Session state** (all per layer `ℓ`, per head where applicable):

```
committed = { h[ℓ]: (D,),  S[ℓ]: (H,d_h,d_h),  M[ℓ] (rule=chunk only),  pos: int }
working   = same structure, advanced token by token; the object generation reads from
pending   = token ids since the last chunk boundary (< L)
harness   = { write_pressure_history, canary_delta_history, budget_used, decisions[] }
meta      = { session_id, parent_session_id, root_session_id, model_id, model_signature, domain, created_at, ... }
```

The inner learning-rate schedule is *not* state: `β_t, α_t` are functions of the input through the slow weights.

**Chunk transaction** (identical for text and physics):

1. Tokens (or `[obs, action]` steps) append to `pending`; each is processed through the *recurrent* path against `working`, so outputs and generation use the freshest memory. Per token the layer emits `surprise_t = ‖e_t‖`, `β_t`, `α_t`, `‖ΔS_t‖_F = β_t‖e_t‖` (per head/layer).
2. At `len(pending) == L`: compute chunk signals — `Δ[ℓ] = S_working[ℓ] − S_committed[ℓ]`, `‖Δ‖_F` per layer and total (update norm), mean/max surprise, chunk NLL, robust MAD-z of each against the session's history, canary score (below), canary alignment `cos(Δ, g_canary)`, budget remaining.
3. Decide: `commit` (`committed := working`), `rollback` (`working := committed`, then reprocess the chunk with `β ≡ 0` — the SSM state still advances and the memory is read but not written: "read the content, refuse it as training signal"), or `scale` (recompute the chunk with `β ← s·β`, `s ∈ (0,1)`), or `project` (Section C.2). Log the decision with all signals; that log is the dashboard's transaction timeline.
4. Clear `pending`, advance `pos`.

Rollback therefore costs one extra chunk forward; the reprocessed outputs replace the provisional ones only for state — text already generated in that chunk is not retracted (document this; it is the same "learning refused, inference continued" semantics as the paper's §9.3).

**Fork** = deep-copy `committed` + harness history into a new session directory with `parent_session_id`; `pending` is dropped (fork at a chunk boundary). `model_signature = sha256(config ‖ base checkpoint)` guards loading, as now.

**Canary probe.** A fixed token sequence per domain (text: a few sentences; physics: a canonical action sequence at a reference μ). `score(state) = mean NLL` (or MSE) of the canary when run from a *scratch copy* of `working` with `β ≡ 0` (read-only), state discarded afterwards. Cost: one canary forward per chunk.

**Canary gradient and projection (multi-layer).** Run the same read-only canary forward with `S[ℓ]` marked `requires_grad`; `g[ℓ] = ∂score/∂S[ℓ]`, a list of `(H,d_h,d_h)` tensors. Flatten `Δ = [Δ[1..L]]` and `g = [g[1..L]]` into two vectors and apply the existing half-space rule: if `⟨g, Δ⟩ > ε_dot + ε_cos‖g‖‖Δ‖` then `Δ ← Δ − ((⟨g,Δ⟩ − ε)/‖g‖²) g`, then `S_working[ℓ] := S_committed[ℓ] + Δ[ℓ]`. First-order guarantee that the chunk's write does not raise the canary score. The existing `spfw.py` already operates on `GradList`; it moves over unchanged, now applied to *state deltas* instead of adapter gradients. `h[ℓ]` is never projected (it is activation state and re-derivable).

**Write budget.** Per-chunk cap `‖Δ‖_F ≤ B_chunk` (scale down), per-session cumulative cap `Σ‖Δ‖_F ≤ B_session` (then `β ≡ 0` for the rest of the session, i.e. the session becomes read-only). With `rule="chunk"` + Muon the per-chunk cap is automatically tight.

**What replaces the regex gate.** Every pre-commit signal is a function of the model: chunk NLL (OOD), surprise statistics, `‖Δ‖`, the model's own `β_t` (it was trained to be low on things not worth learning), canary delta and alignment, MAD-z of each against session history, and a CUSUM on `‖Δ‖` for slow drift. Thresholds are calibrated on a benign stream to a target false-positive rate, stored with the model, and shown in the UI. Compression ratio can stay as a *display* signal; it must not gate.

## D. The learned write gate, meta-trained against a red team

`β_t` is the model's per-token answer to "should I learn from this?". In Titans/TTT-E2E it is trained only to minimize LM loss; here the proposal is to also train it to minimize *canary damage under attack* while keeping in-context learning intact.

**Objective.** For a slow-weight vector `Φ`:

```
L(Φ) = CE_benign(Φ) + λ_rec · CE_MQAR(Φ) + λ_adv · Damage(Φ, x_adv(Φ))
Damage = score_canary(state after processing [benign prefix ‖ x_adv]) − score_canary(state after prefix)
```

`Damage` is differentiable in `Φ` through the chunk-parallel form (the state is a differentiable function of the input and `Φ`), so `∂Damage/∂Φ` pushes `w_β` toward low `β` on adversarial tokens and pushes `W_k,W_v` toward writes that do not move the canary.

**The attacker `x_adv(Φ)`.** Two tiers, both cheap:
1. Sampled families (no optimization): random-token blobs, single-token repeats, shuffled benign text, "topic-switch" segments, high-surprise synthetic segments. These give the gate a broad prior.
2. Gradient attacker: `K = 5–10` steps of projected gradient ascent on the *continuous embeddings* of a 64-token suffix, maximizing `Damage` subject to `‖δ‖ ≤ ρ` and a benign-looking constraint `NLL(x_adv) ≤ τ` (the model's own perplexity; this is the honest replacement for "entropy above a threshold"). Snap to nearest tokens at the end and re-validate through the *token* path — this fixes the current red team's bug where the validated payload is not the optimized one.

Alternate: 9 benign steps, 1 adversarial step (cost ≈ 1.5–2× baseline). Stop-gradient the attacker's inner steps (treat `x_adv` as a constant for the outer step; no need to differentiate through the attack).

**Minimal honest experiment.** Two runs, same seed and data: baseline (`λ_adv = 0`) vs gate-hardened. Report:
- LM loss and MQAR accuracy (must not regress beyond noise).
- Held-out *adaptive* attacker (fresh 50-step PGD per model, several seeds, and a black-box token-swap attacker): max achievable canary damage, fraction of attacks exceeding the rollback threshold, and how much of the remaining damage the harness (rollback + projection) removes.
- AUROC of `β_t` as a detector of attack tokens vs benign tokens; histogram of `β` on both.
- Transfer: attacks optimized against the baseline replayed against the hardened model.

Feasible at this scale: the attacker is a few chunk forwards on a 6 M-param model; the full experiment is well under an hour on one GPU.

**What is novel, what is not.** Not novel: input-dependent inner learning rates (TTT, Titans, DeltaNet's `β`), the delta rule, canaries, A-GEM projection, adversarial training, PGD in embedding space. Novel as far as I can tell: (i) training the TTT write gate *for update safety* against an adaptive attacker rather than only for LM loss, and measuring it as a detector; (ii) using the delta rule's exact per-token `β‖e‖` as a free, un-gameable-by-magnitude write-pressure signal; (iii) canary-gradient projection applied to a TTT layer's *state delta* rather than to adapter gradients, with the fixed-norm (Muon) variant as an architectural write budget. Modest, but real, and the experiment above is the honest way to claim it. Expect partial robustness: an adaptive attacker will still find directions the gate has not seen; report that number rather than hide it.

## E. Physics domain (hidden-μ system identification)

Input per step `x_t = [obs_t (4), action_t (2), reset_flag (1)] → Linear(7 → D)`; head `Linear(D → 4)` predicting `obs_{t+1} − obs_t` (delta prediction; MSE). Same blocks, same state, same harness, same session store (`domain: "physics"`).

**Training sequences.** `T = 512` steps built from several episodes with μ resampled per episode and `reset_flag = 1` on the first step of each; random actions as now. The forget gate `α_t` has to learn to open at resets, the memory `S` has to absorb μ-specific dynamics within an episode. This is precisely "the only way to do well is to infer μ into the plastic state" from the current README, now with a mechanism that can do it.

**Metrics.**
- Three-way MSE per step as now: base (fresh state, `β ≡ 0` everywhere → pure slow-weight predictor), session-start-no-update (loaded state, `β ≡ 0`), adaptive (loaded state, learning on). Gate-off is the honest analog of "no updates".
- μ-probe: ridge regression from a fixed-dimensional readout of the state to μ across many sessions/episodes. Use `[h[ℓ]; S[ℓ] applied to 16 fixed probe keys]` per layer (≈ 5 K dims), report held-out `R²`. A probe from `h` alone versus `h + S` shows how much of μ lives in the fast weights.
- μ-switch: change μ mid-episode without a reset flag; report steps until the adaptive MSE re-converges. This is the cleanest visual of "learning at test time" for the dashboard.

## F. Risks and failure modes

- **Inner-loop instability.** Mitigated by construction: unit keys, `β ∈ (0,1)`, `α ∈ (0,1]`. Keep `v` unnormalized but RMSNorm the read. If `rule="chunk"` is used, mean-scale the gradient (verified stable) or orthogonalize; never sum-scale.
- **MPS.** Everything used is matmul/einsum/cumsum/exp/log/masked_fill/tril: verified on MPS with gradients. Pitfalls hit during verification: in-place writes into a preallocated output tensor break autograd (build lists and `stack`); avoid `solve_triangular`/`linalg.inv` (use the nilpotent product); keep the scan in fp32 (bf16 is fine for projections/MLP). `torch.compile` on MPS is optional at best; do not depend on it.
- **LaCT's "small chunks are bad".** That is a hardware-utilization argument (below 5 % of peak at 16–64 tokens) plus "bigger state helps". Irrelevant to correctness at toy scale; relevant to design: prefer `H=2, d_h=128` (32 K state per layer) over `H=4, d_h=64` if MQAR saturates, and keep `rule="chunk"` available for the large-chunk regime.
- **Memorization at tiny scale.** 5.5 M tokens of valid split will overfit a 6 M-param model in a few epochs; use ≥ 50 M tokens of train split, hold out the valid file, report held-out loss only.
- **Fast weights ignored.** The classic failure: the model learns to set `β ≈ 0` and rely on the SSM. Detection: (a) evaluate with `β ≡ 0` forced — the held-out loss gap is the value of the memory; (b) MQAR accuracy versus number of pairs and versus SSM-only ablation; (c) histogram of learned `β` (if it collapses to 0, lower `b_β` init or raise the MQAR mix). Every checkpoint should ship these three numbers.
- **Gate collapse under adversarial training.** `λ_adv` too large drives `β → 0` everywhere (perfectly safe, useless). The MQAR term in the objective is the guardrail; sweep `λ_adv`.
- **Harness/model inconsistency after projection or rollback.** Outputs already produced within the chunk used the unprojected provisional state. Accept and document; optionally re-run the chunk after projection when `‖Δ_proj − Δ‖/‖Δ‖ > 0.1`.
- **Two code paths.** The recurrent (inference) and chunk-parallel (training) implementations must agree; keep the equivalence test (`< 1e-5` on random inputs, both devices) as a permanent unit test. Same for the scan.
- **Unsure:** whether the SSM-fed-memory composition beats the plain stacked variant (SSM block then memory block with separate residuals) at this scale. Cheap ablation; keep the composition behind one flag.

## G. Names (ranked)

1. **`plastic`** — the repo already calls the learnable state "plastic weights"; short, importable (`import plastic`), reads as both "the architecture" and "the thing the harness controls". GitHub: `DMontgomery40/plastic`. Tagline carries the keywords: "plastic: a tiny test-time-training state-space model with a transactional safety harness".
2. **`engram`** — a memory trace written by experience; fits the delta-rule memory and the session/branch story. More common as a project name elsewhere.
3. **`ttt-ssm`** (package `ttt_ssm`) — purely descriptive, maximally discoverable, no identity.

Whichever is chosen, keep "test-time training" and "state space model" in the repo description and README title line.
