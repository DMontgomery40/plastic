# M3 Sessions and Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persisted, branchable sessions for both domains whose fast state is advanced through chunk transactions governed by a harness whose every signal comes from the model: chunk loss, surprise, β, update norm, Fisher drift, coherence and poison canaries, canary-gradient alignment, robust statistics, CUSUM, and budgets. Decisions: commit, rollback, scale, project, read-only.

**Architecture:** `TransactionRunner` wraps a model and a `SessionState` triple (committed, working, pending). Inputs are processed token by token through the recurrent path against `working`; at each chunk boundary it computes `ChunkSignals`, the `policy` turns them into a `Decision`, and the runner applies it (commit clones working into committed; rollback restores committed and reprocesses the chunk frozen; scale reprocesses with `beta_scale`; project edits the multi-layer S delta by a half-space projection against the coherence-canary gradient). `Calibration` is produced once per model from a benign stream in log-only mode and stores per-signal quantile thresholds at a target false-positive rate, the diagonal Fisher over state entries, CUSUM references, and canary baselines. `ArtifactStore` gains sessions with lineage, transaction logs, and the model-signature guard.

**Tech Stack:** torch, numpy, pytest. Pure PyTorch; all harness code is outside the training graph.

**Spec:** `docs/superpowers/specs/2026-09-21-plastic-design.md` sections 6, 9, 11.

## Global Constraints

- No string heuristics anywhere in the harness. Compression ratio may be computed for display only and must not influence any decision.
- `freeze=True` is the read-only mode (no write, no decay); `beta_scale` is the scale mode. The harness changes controls only at chunk boundaries.
- Projection invariant after projection: `⟨g, Δ⟩ ≤ ε`, `g = ∂score_coherence/∂S`, `Δ = S_working − S_committed`, flattened over layers and heads. If the removed fraction exceeds `project_max_removed`, fall back to rollback.
- No "legacy"/"v1", no schema versions; session load checks `model_signature`.
- Every task ends with `uv run pytest` green and a commit on `fuse`. Never push.

---

### Task 1: Statistics (`plastic/harness/stats.py`)

**Interfaces:**
- `robust_z(x: float, reference: Sequence[float]) -> float | None`: `(x − median)/(1.4826·MAD)`; `None` when fewer than 8 reference points; MAD floor 1e-9.
- `log_safe(x: float) -> float` = `log(max(x, 1e-12))` (norms are heavy-tailed; z-scores of norms are taken on the log).
- `class Cusum(k: float = 0.5, h: float = 4.5)`: `update(z: float) -> bool` two-sided (`s_hi = max(0, s_hi + z − k)`, `s_lo = max(0, s_lo − z − k)`, alarm when either exceeds `h`, then reset that side), `state() -> dict`, `Cusum.from_state(d)`.
- `quantile_threshold(values: Sequence[float], fpr: float, *, side: Literal["upper","lower"] = "upper") -> float`: empirical `1 − fpr` (or `fpr`) quantile with linear interpolation; raises on fewer than 20 values.
- `class SignalHistory(maxlen: int)`: append/values.

**Tests (`tests/test_harness_stats.py`):** z of the median is 0 and of an outlier large; `None` below 8 points; CUSUM stays silent on N(0,1) noise for 2000 draws with `h = 4.5` and alarms within 40 steps of a +1.0 shift; quantile of 1000 benign values at fpr 0.01 flags about 1% of a fresh benign sample and 100% of a shifted sample; Cusum state round-trip.

---

### Task 2: Canary suites (`plastic/harness/canary.py`)

**Interfaces:**
- `@dataclass CanarySuite`: `domain`, `coherence: list[list[int]]` (token ids) or `list[dict]` for physics (`{"mu": float, "actions": [[ax, ay], ...]}`), `poison: list[list[int]]` (text) or `list[dict]` (physics: actions with impossible dynamics: targets of a different mu than the one used to score), `to_json/from_json`, `default_text(tokenizer_or_ids, heldout_path, n_probe=3, probe_len=128, vocab_size, rng)` builds coherence probes from the first windows of the held-out file and poison probes from random tokens and single-token runs; `default_physics(n_probe=3, steps=64, mus=(0.05, 0.15, 0.25), rng)`.
- `score_suite(model, state: SessionState, suite, *, device) -> dict[str, float]` keys `coherence`, `poison`: mean NLL (text) or MSE against env-simulated targets (physics) of each probe run from a clone of `state` with `freeze=True`, `mode="recurrent"` for short probes; the clone is discarded; caller's state is unchanged (assert by tests).
- `canary_gradient(model, state, suite, *, device) -> list[Tensor]`: `∂ score_coherence / ∂ S[ℓ]` per layer, computed with `S` cloned and `requires_grad_(True)`, frozen forward, `torch.autograd.grad`; returned detached.

**Tests (`tests/test_harness_canary.py`):** scoring leaves the input state bit-identical; text suite from a small `.bin` has 3 coherence and ≥ 2 poison probes; gradient shapes equal `S` shapes and are finite and nonzero; physics suite scores finite; suite JSON round-trip.

---

### Task 3: Fisher and projection (`plastic/harness/fisher.py`, `plastic/harness/projection.py`)

**Interfaces:**
- `estimate_fisher_diag(model, chunks: Iterable[Tensor], *, device, n_chunks: int) -> list[Tensor]`: for each benign chunk, run from zero state through `chunk` mode with the incoming `S` of each layer cloned and `requires_grad_(True)` (a hook on `PlasticCore` is not needed: run the model layer by layer via `model.core.blocks` with explicit `LayerState`), take `(∂NLL/∂S)²` and average; returns per-layer `(H, d_h, d_h)` tensors (mean over batch). Physics analog with MSE.
- `fisher_norm(deltas: list[Tensor], fisher: list[Tensor]) -> float` = `Σ F ⊙ Δ²` summed over layers.
- `project_delta(deltas: list[Tensor], grads: list[Tensor], *, eps_dot: float, eps_cos: float) -> tuple[list[Tensor], ProjectionStats]` with `ProjectionStats(dot_before, dot_after, removed_ratio, applied: bool)`. Rule: `allowed = max(eps_dot, eps_cos·‖g‖·‖Δ‖)`; if `dot > allowed`: `Δ ← Δ − ((dot − allowed)/‖g‖²) g`.

**Tests (`tests/test_harness_projection.py`):** after projection `⟨g, Δ⟩ ≤ allowed + 1e-6` for random and adversarially aligned deltas; a delta already satisfying the constraint is returned unchanged with `applied=False`; removed ratio in [0, 1]; Fisher diag shapes and non-negativity; `fisher_norm` of zeros is 0.

---

### Task 4: Signals, policy, harness config (`plastic/harness/config.py`, `signals.py`, `policy.py`)

**Interfaces:**
- `HarnessConfig` (dataclass, JSON): `enable_rollback=True, enable_projection=True, enable_budget=True, enable_stats=True, log_only=False, canary_delta_max=0.5, poison_delta_min=-0.5, z_rollback=6.0, z_scale=3.0, scale_factor=0.25, budget_chunk=None|float, budget_session=None|float, project_eps_dot=0.0, project_eps_cos=0.02, project_max_removed=0.5, cusum_k=0.5, cusum_h=4.5, history_window=64, fisher_drift_max=None|float, learn_from_generation=False, target_fpr=0.01`.
- `@dataclass ChunkSignals`: `pos_start, pos_end, chunk_loss, surprise_mean, surprise_max, beta_mean, alpha_mean, write_norm_sum, delta_norm, delta_norm_per_layer, fisher_update, fisher_drift, canary_coherence_before, canary_coherence_after, canary_poison_before, canary_poison_after, canary_delta_coherence, canary_delta_poison, canary_alignment (cos), compression_ratio (display), z: dict[str, float|None], cusum_alarm: bool, budget_used, budget_remaining` and `to_dict()`.
- `compute_chunk_signals(...)` assembles from the runner's raw measurements plus the calibration reference windows and session history.
- `@dataclass Decision`: `kind: Literal["commit","rollback","scale","project","readonly"]`, `reasons: list[str]`, `scale: float = 1.0`.
- `decide(sig: ChunkSignals, cfg: HarnessConfig, thresholds: dict[str, float], *, read_only: bool) -> Decision` in this order: read-only session → `readonly`; `log_only` → `commit` (reasons list what would have fired); budget chunk cap → `scale` to `budget_chunk/delta_norm`; canary trigger (`canary_delta_coherence > canary_delta_max` or `canary_delta_poison < poison_delta_min`) → `rollback`; any z ≥ `z_rollback` among `chunk_loss, surprise_mean, delta_norm, fisher_update` (calibrated thresholds override z when present) → `rollback`; CUSUM alarm → `rollback` (+ freeze flag); projection wanted (`canary_alignment` positive and `enable_projection`) → `project`; any z ≥ `z_scale` → `scale` by `scale_factor`; else `commit`.

**Tests (`tests/test_harness_policy.py`):** a state-transition matrix: for each decision kind construct signals that trigger exactly it and assert kind and reason text; read-only dominates; log-only never blocks but records reasons; budget scale factor computed correctly.

---

### Task 5: Transaction runner (`plastic/harness/transaction.py`)

**Interfaces:**
- `class TransactionRunner(model, model_cfg, harness_cfg, calibration: Calibration | None, suite: CanarySuite | None, *, device)`; state: `committed: SessionState`, `working: SessionState`, `pending_inputs: list` (token ids or physics rows), `pending_outputs`, `history: dict[str, SignalHistory]`, `cusum: dict[str, Cusum]`, `budget_used: float`, `read_only: bool`, `anchor: SessionState` (session start, for drift), `transactions: list[dict]` (this call).
- `feed_tokens(ids: list[int], *, source: Literal["user","model"]) -> Tensor logits_last`: per token: recurrent forward against `working` (freeze when `source=="model"` and not `learn_from_generation`), append to pending; when pending length equals `L` call `_transact()`.
- `feed_physics(rows: Tensor (T, 7), targets: Tensor (T, 4)) -> Tensor preds`.
- `_transact() -> dict` (the transaction record): measure canaries before (from `committed`) and after (from `working`); compute deltas and signals; decide; apply:
  - commit: `committed = working.clone()`; append signal streams; update budget.
  - rollback: `working = committed.clone()`; reprocess `pending` with `freeze=True`; `committed = working.clone()`.
  - scale: `working = committed.clone()`; reprocess with `beta_scale=decision.scale`; `committed = working.clone()`.
  - project: compute `g` from committed, project Δ, set `working.S = committed.S + Δ_proj`; if removed ratio > max → rollback path; else `committed = working.clone()`.
  - readonly: as rollback (frozen reprocess).
  - Record everything with `decision`, `reasons`, signal dict, timings; clear pending; `pos` advances.
- `flush()`: transact a partial pending chunk (end of a turn) using the same policy.
- `state_dict()/load_state_dict()` for persistence (committed, working, pending, histories, cusum, budget, read_only, anchor).

**Tests (`tests/test_harness_transaction.py`)** on a tiny model (d=32, 2 layers, chunk 8) with a forced thresholds dict:
- commit path: after 8 tokens `committed == working` and one record with kind commit.
- rollback path (force `canary_delta_max = -1` so any chunk rolls back): committed S unchanged from before the chunk; `h` advanced; the record's kind is rollback; and the resulting state equals a fresh frozen pass over the same tokens from the pre-chunk state (consistency test).
- scale path: resulting S delta norm ≈ scale × unscaled delta norm (delta rule: writes scale linearly at first order; assert ratio in [0.6·s, 1.4·s] for s=0.25 or use exact per-token equivalence: reprocessing with `beta_scale=s` equals a direct forward with `beta_scale=s`).
- project path: after projection `⟨g, Δ⟩ ≤ allowed`; with `project_max_removed=0` it falls back to rollback.
- read-only: session flagged read-only never changes S.
- budget: `budget_session` small → after the cap, `read_only` becomes True and later chunks do not change S.
- streaming invariance: feeding 16 tokens as 16 calls equals feeding as 2 calls of 8 (state and records).
- persistence: `state_dict` round trip mid-chunk continues identically.

---

### Task 6: Calibration (`plastic/harness/calibrate.py`)

**Interfaces:**
- `@dataclass Calibration`: `model_signature`, `n_chunks`, `reference: dict[str, list[float]]` (per-signal benign values, capped at 512), `thresholds: dict[str, float]` (upper quantiles at `target_fpr` split by union bound across the signals used for rollback), `fisher: list[Tensor]` (saved as `fisher.pt` alongside), `canary_baseline: dict[str, float]`, `created_at`, `to_json/from_json`, `save(dir)/load(dir)`.
- `calibrate(model, model_cfg, store, model_id, *, data (TokenWindows or physics generator), harness_cfg, n_chunks=256, device) -> Calibration`: builds the default canary suite if absent (saved as `canary.json` in the model dir), runs the runner in `log_only` mode over benign sequences, collects signals, estimates Fisher, computes thresholds, writes `calibration.json` + `fisher.pt`.

**Tests (`tests/test_harness_calibrate.py`):** on the tiny fixture, calibration writes files, thresholds exist for the rollback signals, and replaying the calibration stream with the calibrated harness gates at most `2·target_fpr` of chunks.

---

### Task 7: Session store and runner (`plastic/store.py` additions, `plastic/session/runner.py`)

**Interfaces (store):**
- `sessions_dir`, `session_dir(id)`, `sessions_index`, `new_session_id(prefix)`, `create_session(session_id, *, model_id, domain, parent_session_id, harness_cfg, extra) -> meta` (writes `meta.json` with `model_signature`, `root_session_id`; `runner_state.pt` with zero state), `load_session_meta(id)`, `list_sessions() -> list[dict]` (lineage summaries with counts of commits/rollbacks), `save_runner_state(id, state_dict)`, `load_runner_state(id)`, `append_transaction(id, record)`, `read_transactions(id, limit)`, `append_trace(id, record)`, `fork_session(parent_id, child_id) -> meta` (copies runner state and meta lineage), `delete_session(id)`.
- Loading a session whose `model_signature` differs from `store.model_signature(model_id)` raises `ValueError`.

**Interfaces (runner):**
- `class Session`: `Session.open(store, session_id, *, device) -> Session` (loads model, calibration if present, canary suite, runner state); `chat(prompt: str, *, max_new_tokens=128, temperature=0.9, top_k=50) -> ChatResult(prompt, completion, transactions: list[dict], n_tokens_in, n_tokens_out)` (encodes prompt with bos, feeds as user tokens, samples new tokens through the recurrent path feeding each as `source="model"`, flushes, persists, appends trace); `physics_episode(*, steps: int, mu: float, seed: int, actions: Tensor | None) -> EpisodeResult(per_step: list[dict(t, base_mse, frozen_mse, adaptive_mse)], transactions)` computing the three-way comparison (base: zero state frozen; frozen: session state frozen; adaptive: session state learning) on the same trajectory; `reset()`; `fork(child_id)`; `summary() -> dict` (pos, budget, read_only, state norms, last decisions).

**Tests (`tests/test_session.py`):** create/list/fork lineage; signature mismatch refused after re-saving the model checkpoint; `chat` on a tiny trained-for-3-steps model returns a completion and records transactions and a trace; a second `chat` continues from persisted state (pos increases); `physics_episode` returns three curves of equal length and records transactions; fork's first chat starts from the parent's committed state (S equal before feeding).

---

### Task 8: CLI

`plastic calibrate <model_id> [--chunks N]`, `plastic session new --model <id> [--domain] [--harness-json]`, `plastic session list`, `plastic session fork <parent> <child>`, `plastic session reset <id>`, `plastic chat <session_id> "<prompt>"`, `plastic physics <session_id> --steps 256 --mu 0.12`. Tests in `tests/test_cli.py` for `session new/list/fork` and one `chat` on a tiny model.

## Self-review

- Spec coverage: 6.1 state and store (T5, T7), 6.2 transaction (T5), 6.3 signals (T4, T2, T3), 6.4 policy (T4, T5), 6.5 calibration (T6), 6.6 no heuristics (global), 4.2 physics three-way (T7), branching (T7), CLI (T8). Red team, adversarial gate, sleep are M4; API is M5.
- Type consistency: `SessionState`, `ChunkSignals`, `Decision`, `HarnessConfig`, `Calibration`, `CanarySuite`, `TransactionRunner`, `Session` used identically across tasks.
