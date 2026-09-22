# plastic: one fused test-time-training state-space model with a transactional safety harness

Design spec, 2026-09-21. Status: approved for implementation under the assumptions in section 12 (David could not answer questions during the session; every judgment call is recorded there and is reversible).

Companion documents (all in `docs/research/`):

- `2026-09-21-ttt-ssm-literature.md`: TTT layers, Titans, LaCT, TTT-E2E, Gated DeltaNet, Mamba-3, the 2026 delta-rule wave, with equations and dates.
- `2026-09-21-inference-time-learning-safety.md`: attacks on and defenses for models that learn at inference, robust online statistics, and a six-layer regex-free safety stack.
- `2026-09-21-tooling-hf-jobs-torch.md`: Hugging Face Jobs, torch 2.12 on MPS, reference implementations.
- `2026-09-21-architecture-memo.md`: the block equations, verified on CPU and MPS, with timing and parameter counts.

## 1. Why this rewrite

The repository grew as two unrelated toys that share a FastAPI process and a dashboard shell:

- `ttt/`: a random-init "monitor" model with a hashed vocabulary, a separately trained tiny LM whose per-session "context net" is a linear adapter fine-tuned on each chat turn, a heuristic gate (regexes for jailbreak phrases and base64 blobs), a canary rollback, an A-GEM style projection, and a red team.
- `ttt_ssm_nano/`: a three-matrix diagonal SSM predicting a 2D point mass with hidden friction mu, updated online with Muon and rolled back on loss regression, persisted as branchable sessions.

Neither half is a test-time-training layer in the sense of Sun et al. (2024) or anything after it: nothing is meta-trained through an inner loop, so the fast weights have no learned semantics, and the adapter is fine-tuned on a model that was never trained to be adapted. The audit also found real defects:

1. The SSM backbone's closed-form scan (`s_t = a^t cumsum(u_k / a^k)`) underflows. At init the hidden state is exactly zero after 4 tokens; at a = 0.9 it differs from a sequential recurrence by 0.83 over 256 tokens, at a = 0.5 by 2.9. Every text result in the repo was produced on this backbone, which is why trained LMs plateau at 3.2 to 4.3 nats on TinyStories.
2. The BPE pretokenizer and the run-id parser use double-escaped regexes in raw strings, so the tokenizer never splits on whitespace (each line is one BPE segment) and run timestamps are never parsed.
3. The red team optimizes soft tokens over hashed ids, decodes them to strings like `tok123`, and validates that text, which re-hashes to different ids. The validated payload is not the optimized payload.
4. The monitor model is untrained, so its "OOD" loss threshold is calibrated on noise.
5. The Muon fallback reshapes 1-D parameters to (n, 1) and "orthogonalizes" them, which is just normalization.
6. The chat path rebuilds the optimizer per message and mixes two near-identical event dataclasses.

The fix is not incremental. This spec replaces both halves with one architecture, one session store, one harness, one API, and one dashboard.

## 2. Goals and non-goals

Goals:

- One model family, `plastic`, that is honestly a TTT layer: fast weights updated by an inner-loop gradient step at inference, meta-trained end to end through that inner loop, composed with a selective diagonal recurrence (SSM). The same block stack serves two domains, text (next-token prediction) and physics (hidden-mu system identification), differing only in embedding and head.
- A transactional safety harness wrapped around the inner loop at chunk granularity, whose every signal is computed from the model (loss, gradient geometry, canary probes, robust statistics). No string heuristics anywhere.
- Branchable, persisted sessions ("git for plastic weights") for both domains, with the same store, the same fork semantics, and the same transaction log.
- A red team that attacks the real token path, and an experiment that meta-trains the model's own write gate against that red team.
- Training runs on Hugging Face Jobs; development and smoke runs on Apple Silicon (MPS).
- A test suite where none existed.
- A dashboard that is one product, not two bolted together.

Non-goals:

- Competing with production models. Target scale is 5 to 8M parameters, TinyStories, minutes to an hour of GPU time.
- Fused Triton kernels. Everything is plain PyTorch that runs on MPS and CUDA.
- Preserving on-disk artifacts, API paths, or dashboard code from the old halves. Nothing has shipped; nothing is kept for compatibility. Old artifacts are regenerated.

## 3. Naming

Repo and package: **`plastic`**. Rationale: the codebase already calls learnable inference-time state "plastic weights"; the word names both the architecture (weights that change at test time) and the thing the harness governs. "Test-time training" and "state space model" stay in the repo description, README title line, and GitHub topics for discoverability. GitHub repo `DMontgomery40/ttt_ssm_eval` is renamed to `DMontgomery40/plastic` in the final milestone (GitHub redirects the old URL); the local remote is updated; the local directory is not moved during the session.

Language rules: no "legacy", "v1", "backward compatibility", or schema-version fields. Compatibility is guarded only by `model_signature = sha256(config JSON || checkpoint bytes)` on session load.

## 4. Architecture

### 4.1 Block

Notation: model width D = 256, heads H = 4, head dim d_h = 64, chunk L = 64, sequence T (multiple of L). Row-vector convention: a key `k ∈ R^{1×d_h}` reads memory `S ∈ R^{d_h×d_h}` as `k S`. One block, stacked N = 4 (text) or 3 (physics) times:

**SSM branch (selective diagonal recurrence, RG-LRU form).**

```
u_t     = RMSNorm(x_t)
r_t     = σ(W_r u_t + b_r)                      recurrence gate
i_t     = σ(W_i u_t + b_i)                      input gate
log a_t = −c · r_t · softplus(−λ)               c = 8, λ ∈ R^D learned  ⇒ a_t = σ(λ)^{c r_t} ∈ (0,1)
z_t     = sqrt(1 − a_t²) ⊙ (i_t ⊙ W_in u_t)
h_t     = a_t ⊙ h_{t−1} + z_t                   activation state, (D,)
x_t    += W_o (h_t ⊙ SiLU(W_g u_t))
```

Computed with a chunked log-space scan: within a chunk, `P[t,s] = exp(C_t − C_s)` for s ≤ t where `C_t = Σ_{r≤t} log a_r`, so only pairwise decay ratios in (0, 1] are ever formed; the chunk carry is a T/L-step loop. Verified max error 2e-5 against the sequential reference including decays of 1e-3, on CPU and MPS, with gradients to `a` and `z`. This replaces the broken closed form.

**Memory branch (fast weights, fed by the SSM output).**

```
u_t = RMSNorm(x_t)
q_t = normalize(W_q u_t)_h,  k_t = normalize(W_k u_t)_h,  v_t = (W_v u_t)_h      per head
β_t = σ(w_β · u_t + b_β)                write rate ∈ (0,1), per head        (the learned "should I learn from this")
α_t = exp(−softplus(w_α · u_t + b_α))   forget ∈ (0,1), per head
```

Inner rule `delta` (default): one gradient step per token on the associative loss `½‖k S − v‖²`, with forgetting:

```
e_t = v_t − k_t (α_t S_{t−1})                    prediction error = inner-loop gradient direction
S_t = α_t S_{t−1} + β_t k_tᵀ e_t                 = α_t S_{t−1}(I − β_t k_tᵀ k_t) + β_t k_tᵀ v_t
m_t = q_t S_t                                    read after the token's own write
```

This is TTT-Linear (Sun et al.) with mini-batch 1 plus Gated DeltaNet's decay; `S_t = α S_{t−1} − β ∇_S ½‖kS − v‖²` evaluated at `αS_{t−1}`. With unit keys and β, α in (0,1) each step is a contraction along k, so the state is bounded for any input. The per-token write pressure `‖ΔS_t‖_F = β_t ‖e_t‖` is exact and free.

Training uses the chunk-parallel WY form (nilpotent Neumann inverse, no `solve_triangular`, all matmuls); inference uses the per-token recurrence. The two agree to 4e-7 on CPU and MPS and the equivalence is a permanent test.

Inner rule `chunk` (experiment switch, same interface): mini-batch TTT once per chunk at chunk-start weights, mean-scaled gradient, Titans-style momentum `M`, per-chunk forget, optional Newton-Schulz orthogonalization of the update (LaCT/Atlas). With orthogonalization the per-chunk write has a fixed Frobenius norm, which turns the write budget into an architectural invariant. `memory ∈ {linear, mlp}`; the 2-layer MLP memory uses `torch.func` for the inner gradient and a second-order outer loop (verified feasible on MPS). Sum-scaled mini-batch gradients are unstable (eigenvalues to −56 with correlated keys) and are never used.

**Output and MLP.**

```
x_t += W_o2 (concat_h RMSNorm(m_t) ⊙ SiLU(W_g2 u_t))
x_t += MLP(RMSNorm(x_t))                          GELU, 4× width
```

Flag `memory_input ∈ {ssm_out, block_in}` (default `ssm_out`); the stacked alternative is a cheap ablation.

**Sizes.** Per block 1.18M parameters; text model with tied embeddings (V = 4096) and 4 blocks about 5.8M; physics model with 3 blocks about 3.6M. State per block: `S` 4×64×64 and `h` 256 floats.

**Gate initialization.** `σ(b_β) ≈ 0.5`, `α ≈ 0.98` at init, `σ(λ) ≈ 0.9`. Every checkpoint ships three numbers: held-out loss with β forced to 0 (the value of the memory), MQAR accuracy versus pairs, and the histogram of learned β.

### 4.2 Domains

Text: byte-level BPE (V = 4096, whitespace-preserving pretokenizer, fixed), tied embedding and head, next-token cross-entropy.

Physics: per step `x_t = [obs_t (4), action_t (2), reset_flag (1)] → Linear(7 → D)`; head `Linear(D → 4)` predicting `obs_{t+1} − obs_t`; MSE. Environment is the existing 2D point mass with hidden friction mu (linear and nonlinear modes), mu resampled per episode, `reset_flag = 1` on the first step of each episode, several episodes per training sequence so the forget gate learns to open at resets and the memory absorbs mu within an episode.

### 4.3 Model registry

`artifacts/models/<model_id>/` holds `config.json`, `checkpoint.pt`, `tokenizer.json` (text only), `train_log.jsonl`, `eval.json` (the three checkpoint numbers plus held-out loss), and `calibration.json` (section 6.5). `artifacts/models/index.json` lists them with status, domain, and lineage (`parent_model_id` for sleep candidates).

## 5. Training (outer loop)

- Fast state `{h, S, M}` starts at zero per training sequence and is produced by the forward pass; it is not a parameter. The outer loss is ordinary next-token CE (text) or MSE (physics), backpropagated through every inner step. With rule `delta` this is first-order autograd through the WY form; no truncation within a sequence.
- Text data: TinyStories train split (50 to 100M tokens, mounted inside the HF job from `hf://datasets/roneneldan/TinyStories`), stories concatenated with `<eos>` into T = 1024 sequences so entities recur across chunk boundaries; the local 22MB valid file is held out. MQAR synthetic recall sequences (reserved key/value token ranges, n pairs then queries) mixed at 20% of batches.
- Physics data: generated on the fly; T = 512, 4 to 8 episodes per sequence.
- Optimizer: `torch.optim.Muon` for 2-D matrices (`adjust_lr_fn="match_rms_adamw"`, lr 2e-2, weight decay 0.1 set explicitly) and AdamW (lr 1e-3) for embeddings, norms, λ, biases, and gate vectors; 500-step warmup, cosine to 10%, grad-clip 1.0. AdamW-only is the documented fallback.
- Compute: measured 8.5K tok/s on MPS (B = 8, T = 1024, 4 blocks, fp32, no compile); expected 40 to 80K tok/s on an L4/A10G, so one pass over 100M tokens in 20 to 40 minutes. Target 1.6 to 1.9 nats held out.
- Launch: a PEP 723 script under `scripts/hf_jobs/` run with `hf jobs uv run --flavor l4x1 --timeout 2h --secrets HF_TOKEN -v hf://buckets/DMontgomery40/plastic-runs:/out -v hf://datasets/roneneldan/TinyStories:/data`, checkpoints and metrics written to the bucket and synced back with `hf sync`. The `plastic` package is installed in the job by cloning the repo at a pinned commit. The local `huggingface_hub` is upgraded to 1.32 first (`wait`, `ls`, local-directory volumes). Every launch is also runnable locally with `--device mps` for smoke tests.

## 6. Sessions and the harness

### 6.1 Session state

Per layer: `h` (D,), `S` (H, d_h, d_h), `M` (rule `chunk` only). Plus `pos`. Three copies live in a session directory:

- `committed`: last accepted state.
- `working`: advanced token by token; generation reads from it.
- `pending`: inputs since the last chunk boundary (< L).

The inner learning-rate schedule is not state: β and α are functions of the input through the slow weights.

`artifacts/sessions/<session_id>/`: `meta.json` (session_id, parent_session_id, root_session_id, model_id, model_signature, domain, created/updated timestamps, harness config, environment config for physics), `committed.pt`, `working.pt`, `pending.json`, `transactions.jsonl` (one record per chunk decision with every signal), `trace.jsonl` (text: prompts and completions for sleep; physics: episodes), `harness_state.json` (histories, budget used, CUSUM state). `artifacts/sessions/index.json` is the lineage registry.

Fork copies `committed` and harness state into a new session with `parent_session_id`; `pending` is dropped (fork at a chunk boundary). Reset restores zero state.

### 6.2 Chunk transaction

Identical for text and physics:

1. Inputs append to `pending`; each is processed through the recurrent path against `working`. The layer emits per token: `‖e_t‖` (surprise), β_t, α_t, `‖ΔS_t‖_F = β_t ‖e_t‖` per head and layer.
2. At `len(pending) == L` compute chunk signals (6.3), then decide (6.4): `commit` (`committed := working`), `rollback` (`working := committed`, reprocess the chunk with β ≡ 0, so the SSM state advances and the memory is read but not written: content is read, refused as training signal), `scale` (reprocess with β ← s·β), or `project` (6.4). Log the decision and all signals to `transactions.jsonl`.
3. Clear `pending`, advance `pos`.

Text already generated within a rolled-back chunk is not retracted; this is documented as "learning refused, inference continued".

### 6.3 Signals (all from the model)

Per chunk, per layer and total:

- Chunk loss (NLL or MSE) under `working`; the OOD signal.
- Surprise statistics: mean and max `‖e_t‖`; mean β_t (the model's own gate); mean α_t.
- Update norm `‖Δ‖_F`, `Δ[ℓ] = S_working[ℓ] − S_committed[ℓ]`; log-transformed for statistics.
- Fisher-weighted update size `Σ_i F_i Δ_i²` and cumulative drift from the session anchor `Σ_i F_i (S_i − S_anchor,i)²`, with diagonal Fisher over state entries estimated on the benign calibration stream (Elastic TTT / EWC style; the alignment-collapse result says first-order projection alone leaks through curvature).
- Canary suite score before and after the chunk: a fixed per-domain probe set run read-only (β ≡ 0) from a scratch copy of the state. Two sets: a coherence set whose loss must not rise, and a private poison set (garbage and recorded attack payloads) whose loss must not fall. `ΔL_C > τ_C` or `ΔL_P < −τ_P` is the rollback trigger, matching the TTT-guardrails perplexity-shift detector.
- Canary gradient alignment `cos(Δ, g_C)` where `g_C[ℓ] = ∂score_C/∂S[ℓ]` from a read-only canary forward with `S` marked differentiable.
- Robust z-scores (median, 1.4826·MAD) of each stream against a fixed benign reference window from calibration and against session history; two-sided CUSUM (k = 0.5, h = 4 to 5 in z units) on log‖Δ‖ for slow drift.
- Budget remaining: per-chunk cap on `‖Δ‖_F` and per-session cumulative cap.
- Display-only: compression ratio of the chunk text. It never gates.

### 6.4 Policy

Ordered checks, each producing a reason string with the numbers:

1. Budget: if `‖Δ‖_F > B_chunk` scale to the cap; if cumulative exceeds `B_session`, the session becomes read-only (β ≡ 0) until reset.
2. Projection: if `⟨g_C, Δ⟩ > ε_dot + ε_cos ‖g_C‖ ‖Δ‖` project `Δ ← Δ − ((⟨g_C, Δ⟩ − ε)/‖g_C‖²) g_C` over the flattened multi-layer state delta, then `S_working := S_committed + Δ`. The existing half-space code moves over unchanged and now acts on state deltas. `h` is never projected.
3. Rollback: canary trigger, or robust z of chunk loss, surprise, or `‖Δ‖` beyond the calibrated quantile, or CUSUM alarm (alarm also freezes learning until a full canary pass succeeds).
4. Scale: intermediate z-scores scale β by a calibrated factor instead of blocking (learn scaling from the paper).
5. Otherwise commit.

### 6.5 Calibration

`plastic calibrate <model_id>` runs a held-out benign stream with the harness in log-only mode, stores per-signal empirical quantiles at a target false-positive rate (default 1% of chunks gated, split across signals by union bound), the Fisher diagonal, the CUSUM reference, and the canary baselines in `calibration.json`. Thresholds are shown in the UI next to the live values. Recorded red-team payloads are replayed against the operating point and the detection rate is stored.

### 6.6 What replaces the gate

Nothing pattern-matches text. The pre-commit decision uses chunk loss, surprise, the model's own β, update norm, Fisher drift, canary delta and alignment, robust z, and CUSUM. The words "gate" and "regex" do not appear in the new code except in this spec.

## 7. Red team and the learned write gate

Attacker (`plastic redteam`): projected gradient ascent on the continuous embeddings of a 64-token suffix after a benign prefix, maximizing canary damage `score_C(after) − score_C(before)` subject to `‖δ‖ ≤ ρ` and a benign-looking constraint `NLL(x_adv) ≤ τ` (the model's own perplexity). Snap to nearest tokens, then validate through the token path so the reported payload is the tested payload. Plus sampled families: random-token blobs, single-token repeats, shuffled benign text, topic switches, recorded high-surprise segments. Results are stored under `artifacts/redteam/<run_id>/` and feed the poison canary set and calibration replay.

Experiment (`plastic train --adversarial`): add `λ_adv · Damage(Φ, x_adv(Φ))` and `λ_rec · CE_MQAR` to the outer loss; `Damage` is differentiable in the slow weights through the chunk-parallel form; alternate 9 benign steps and 1 adversarial step with the attack treated as constant. Report against a fresh adaptive attacker: max canary damage, fraction of attacks over the rollback threshold, harness residual after rollback and projection, AUROC of β as an attack-token detector, LM loss and MQAR accuracy (must not regress), and transfer of baseline-optimized attacks. Claimed novelty is limited to: training the write gate for update safety against an adaptive attacker and measuring it as a detector; using the delta rule's exact per-token `β‖e‖` as write pressure; canary-gradient projection on a TTT layer's state delta; fixed-norm orthogonalized writes as an architectural budget. Partial robustness is expected and reported.

## 8. Sleep

`plastic sleep <model_id>`: harvest `trace.jsonl` from sessions of that model, mix with a core corpus sample, train blocks only (embeddings and head frozen) for a few hundred steps at a small learning rate, and accept the candidate only if the coherence canary loss does not rise and the poison canary loss does not fall beyond calibrated tolerances. Register as a child model with `parent_model_id`. Sessions reset fast state at the next turn.

## 9. Package layout

```
plastic/
  __init__.py
  config.py            ModelConfig, HarnessConfig, TrainConfig (dataclasses, JSON round-trip)
  model/
    scan.py            chunked log-space selective scan + sequential reference
    delta.py           gated delta rule: recurrent and chunk-parallel (WY) paths
    chunk_rule.py      mini-batch TTT rule: momentum, Newton-Schulz, linear or MLP memory
    memory.py          FastWeightMemory (rule switch), per-token signal outputs
    block.py           PlasticBlock
    lm.py              PlasticLM (text), PlasticDynamics (physics)
    state.py           SessionState: per-layer tensors, zero/clone/serialize
  tokenizer/bpe.py
  data/
    text.py            corpus scan, encode, sequence packing, held-out split
    mqar.py
    physics.py         HiddenMuEnv, episode/sequence generators
  train/
    loop.py            outer loop for both domains, logging, checkpoints, eval numbers
    optim.py           Muon + AdamW split
    adversarial.py     attacker-in-the-loop objective
  harness/
    signals.py         write pressure, surprise, Fisher, compression (display)
    canary.py          canary suites, read-only scoring, canary gradient
    projection.py      multi-layer half-space projection on state deltas
    stats.py           robust z, CUSUM, quantile calibration
    policy.py          decision function with reasons
    transaction.py     chunk transaction runner (commit/rollback/scale/project)
    calibrate.py
  session/
    store.py           unified artifact store: models, sessions, runs, redteam, index, fork, signature guard
    runner.py          feed text or physics steps, run transactions, generate
  redteam/attack.py
  sleep/consolidate.py
  api/
    app.py, routers/{health,models,sessions,train,redteam,physics}.py
  cli.py               `plastic {train,calibrate,session,chat,physics,redteam,sleep,serve,demo}`
scripts/hf_jobs/       PEP 723 launch scripts (train_text.py, train_physics.py, eval.py)
tests/                 pytest
dashboard/             React
docs/
```

Removed entirely: `ttt/`, `ttt_ssm_nano/`, `.archive/`, `run_monitor.py`, `examples/`, `ttt_eval.egg-info/` (untracked), the old `artifacts/` contents (regenerated by `plastic demo`). `pyproject.toml` becomes `plastic` with `uv`-managed dependencies and a `plastic` console script.

## 10. API and dashboard

API (FastAPI, `plastic serve`, port 13579):

- `GET /api/health`
- `GET /api/models`, `GET /api/models/{id}`, `POST /api/models/{id}/calibrate`
- `GET /api/sessions` (both domains, lineage summary), `POST /api/sessions` (model_id, domain, harness config, mu for physics), `GET /api/sessions/{id}` (meta, transactions, harness state, state summaries), `POST /api/sessions/{id}/fork`, `POST /api/sessions/{id}/reset`, `POST /api/sessions/{id}/chat` (text), `POST /api/sessions/{id}/episode` (physics: run N steps with random or supplied actions; returns per-step three-way error and transactions), `GET /api/sessions/{id}/state` (per-layer norms, β histogram, S singular values for the weights view)
- `GET /api/train/jobs`, `POST /api/train` (local job), `GET /api/train/{id}`, `POST /api/train/{id}/cancel`
- `GET /api/redteam`, `POST /api/redteam` (run attack against a model or session), `GET /api/redteam/{id}`
- `POST /api/sleep`

Dashboard (React 18, TypeScript, Vite, Tailwind, Recharts, Zustand): one store covering models, sessions, and jobs; one `fetchJson`; no mock data (explicit empty states with the command to run). Tabs:

1. Sessions: lineage tree and table for both domains; create, fork, reset.
2. Session: transaction timeline (commit, rollback, scale, project), per-chunk signal panel with calibrated thresholds, canary suite before/after, write pressure and CUSUM chart, budget meter, per-layer state summary (β histogram, ‖S‖, singular values, drift from anchor).
3. Chat: text session console with per-chunk decisions inline.
4. Physics: mu system-ID view (trajectory, three-way error, mu-probe readout, mu-switch replay).
5. Train: local jobs and HF Jobs status with loss and the three checkpoint numbers.
6. Red Team: run and inspect attacks; damage vs. constraint plot; detector AUROC.
7. Architecture: block diagram from the live config, safety stack explanation.

Design floor (from the user's legibility rules): no emoji, no text under 11px, body 14px, antialiasing only under `min-resolution: 2dppx`, all colors from defined tokens, no opacity for de-emphasis, no grain textures.

## 11. Tests

`tests/` (pytest, run on CPU; MPS cases when available):

- `test_scan.py`: chunked scan equals sequential reference (random, tiny decays, T not multiple of L via padding), gradients flow, no NaN.
- `test_delta.py`: recurrent equals chunk-parallel (outputs and state, both devices), one-shot recall at β = 1, state boundedness under random and adversarial inputs (property test), write-pressure identity `‖ΔS_t‖ = β_t ‖e_t‖`.
- `test_chunk_rule.py`: mean-scaled stability, orthogonalized write has fixed norm, MLP memory second-order step runs.
- `test_block_model.py`: shapes, parameter count, β-off ablation changes outputs, physics and text models share block code.
- `test_tokenizer.py`: whitespace segmentation, round-trip, special tokens.
- `test_data.py`: sequence packing, MQAR generator correctness, physics episode reset flags.
- `test_harness.py`: state-transition matrix over policy outcomes (commit, rollback, scale, project, read-only); projection invariant `⟨g, Δ⟩ ≥ −ε` after projection; budget invariants; CUSUM alarm on injected drift and silence on benign; calibration quantiles reproduce target FPR on the calibration stream.
- `test_store.py`: create, fork, signature mismatch refused, transaction log append, round-trip of state.
- `test_redteam.py`: validated payload ids equal optimized ids; NLL constraint honored.
- `test_api.py`: FastAPI TestClient contract for every route on a generated demo artifact set.
- `test_train_smoke.py`: 20 outer steps on synthetic data decrease loss for both domains.

Dashboard: `tsc`, `vite build`, and vitest for the store and API client.

Verification gate before any milestone is called done: `uv run pytest`, `npm run build`, and the CLI demo (`plastic demo`) producing a text session and a physics session with transactions.

## 12. Decisions and assumptions made without the user

1. Name `plastic` (advisor's first choice; "ttt-ssm" was the descriptive alternative). One-line rename if disliked.
2. Default inner rule is the per-token gated delta rule (linear memory) because it is stable by construction, exact in write pressure, first-order to meta-train, and verified on MPS; the chunk rule with Muon and the MLP memory are implemented behind the same interface as the "TTT-MLP/LaCT" path rather than as the default.
3. Memory q/k/v are computed from the SSM output (composition), with the stacked variant behind a flag.
4. The physics environment is kept as a first-class domain because it is the cleanest controlled test that fast weights infer latent context.
5. The "harmful holdout" of the safety literature becomes a poison canary set (garbage plus recorded attacks) whose loss must not fall, since a TinyStories model has no notion of harmful content.
6. Chat sessions do not retract text generated inside a rolled-back chunk.
7. Sleep is kept minimal (blocks only, canary-gated acceptance); the auditor idea from the old README is dropped.
8. The GitHub repo rename happens in the final milestone; nothing is pushed.
9. Old artifacts are deleted and regenerated; the old trained models are on the broken backbone and have no value.
10. huggingface_hub is upgraded in the user's pyenv environment because the 1.15 CLI lacks `wait` and local-directory volumes.

## 13. Milestones

- M1 Core model: scan, delta (both paths), chunk rule, memory, block, LM, physics model, state; tests; local smoke train on MPS showing decreasing loss and non-collapsed β.
- M2 Data and training: BPE fix, corpus pipeline, MQAR, physics episodes, outer loop, optimizer split, HF Jobs launcher; first text model on an L4; physics model locally; checkpoint eval numbers.
- M3 Sessions and harness: store, transactions, signals, canary suites, projection, budget, stats, calibration, policy, runner for both domains, fork; tests.
- M4 Red team, adversarial write-gate experiment, sleep.
- M5 API and CLI, `plastic demo`.
- M6 Dashboard rebuild.
- M7 README, CLAUDE.md, cleanup of the old packages, repo rename, final verification, screenshots.
