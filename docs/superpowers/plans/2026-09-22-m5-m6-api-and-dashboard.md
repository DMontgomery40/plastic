# M5 API and M6 Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One FastAPI service over the artifact store (models, sessions, transactions, training jobs, red team, sleep) and one React dashboard that is a single product for both domains.

**Architecture:** `plastic/api/app.py` builds the app from an `ArtifactStore` root; routers are thin and call the Python APIs from M2 to M4 (`Session`, `calibrate_model`, `run_redteam`, `consolidate`, `train` via subprocess). A `SessionRegistry` keeps open `Session` objects (one model instance per model id) behind a per-session lock. The dashboard is React 18 + TypeScript + Vite + Tailwind + Recharts + Zustand with one store, one `fetchJson`, no mock data, and the design floor below.

**Tech Stack:** FastAPI 0.128, pydantic 2, uvicorn, httpx TestClient; React 18, Vite 5, Tailwind 3, Recharts 2, Zustand 4, vitest.

**Spec:** `docs/superpowers/specs/2026-09-21-plastic-design.md` section 10; design floor from `~/.claude/rules/design-legibility.md` (no emoji, no text under 11px, body 14px, antialiasing only under `min-resolution: 2dppx`, every color a defined token, no opacity for de-emphasis, no grain or dot-grid textures).

## Global Constraints

- The API never inspects text for safety; it only exposes what the harness produced.
- No "legacy"/"v1", no schema versions, no mock data paths in the dashboard: an empty state shows the command to run.
- API: every route has a TestClient test on a generated tiny artifact set (`tests/test_api.py`); `uv run pytest` green.
- Dashboard: `npm run build` (tsc + vite) green; vitest for the store and API client; every chart's data comes from the API types in `dashboard/src/api/types.ts`.
- Commit on `fuse`; never push.

---

## API contract (M5)

Base path `/api`. JSON everywhere. Errors: `{"detail": str}` with 404 for missing ids, 400 for invalid requests, 409 for conflicts (existing id, signature mismatch), 500 otherwise.

### Health and data
- `GET /api/health` → `{ok: true, artifacts_root: str, device: str, n_models: int, n_sessions: int}`
- `GET /api/data` → `[{name: str, dir: str, corpus: str, vocab_size: int, splits: {train: int, validation: int, test: int}}]` from `artifacts/data/*/meta.json`.

### Models
- `GET /api/models` → `ModelSummary[]` where `ModelSummary = {model_id, domain, status, params, created_at_unix, updated_at_unix, steps?, tokens?, eval?: EvalSummary, calibrated: bool, has_canary: bool, parent_model_id?: str, type?: str}` and `EvalSummary = {step, heldout_loss, heldout_loss_beta0, memory_value, mqar_accuracy?: {[pairs: string]: number}, beta_hist?: {edges: number[], counts: number[], beta_mean, beta_std, alpha_mean}}`.
- `GET /api/models/{model_id}` → `{record: ModelSummary, config: ModelConfig(dict), eval: EvalSummary|null, calibration: {n_chunks, thresholds: {[signal]: number}, canary_baseline: {[k]: number}, reference_sizes: {[signal]: number}, target_fpr, created_at_unix} | null, canary: {n_coherence: int, n_poison: int} | null, log: TrainLogRecord[] (last 200)}` where `TrainLogRecord = {step, loss?, grad_norm?, lr_scale?, tokens?, seconds?, tok_per_s?, adv_damage?, event?: "eval", heldout_loss?, memory_value?}`.
- `GET /api/models/{model_id}/log?limit=N` → `TrainLogRecord[]`.
- `POST /api/models/{model_id}/calibrate` body `{data_dir?: str, chunks?: int = 128, fisher_chunks?: int = 16, fpr?: number = 0.01}` → the `calibration` object above. Synchronous.

### Sessions
- `GET /api/sessions` → `SessionSummary[]` = `{session_id, model_id, domain, parent_session_id, root_session_id, forked_at_pos, created_at_unix, updated_at_unix, pos, n_transactions, commits, rollbacks, scales, projects, readonly, budget_used, read_only, read_only_reason}`.
- `POST /api/sessions` body `{model_id: str, session_id?: str, harness?: Partial<HarnessConfig>}` → `SessionSummary` (409 if the id exists).
- `GET /api/sessions/{id}` → `{meta: SessionSummary & {harness: HarnessConfig, model_signature}, summary: RunnerSummary, lineage: string[] (root first), transactions: TransactionRecord[] (last 100), trace: TraceRecord[] (last 50)}` where `RunnerSummary = {pos, pending, budget_used, budget_session, read_only, read_only_reason, n_transactions, cusum: {k,h,s_hi,s_lo,alarms}, state_norms: {s_norm: number[], h_norm: number[], s_norm_total, h_norm_total}, drift_from_anchor}` and `TransactionRecord = {index, t_unix, pos_start, pos_end, decision: {kind, reasons: string[], scale}, requested: {kind, reasons, scale}, signals: ChunkSignals, read_only, read_only_reason, seconds}` and `ChunkSignals` is the dict from `plastic/harness/signals.py::ChunkSignals.to_dict()` (fields: pos_start, pos_end, n_tokens, chunk_loss, surprise_mean, surprise_max, beta_mean, alpha_mean, write_norm_sum, delta_norm, delta_norm_per_layer, fisher_update, fisher_drift, canary_coherence_before/after, canary_poison_before/after, canary_delta_coherence, canary_delta_poison, canary_alignment, compression_ratio, z: {[signal]: number|null}, cusum_alarm, budget_used, budget_remaining, log_delta_norm, log_write_norm)`; `TraceRecord = {t_unix, kind: "chat"|"episode", prompt?, completion?, mu?, steps?, seed?, means?, pos_end, n_transactions}`.
- `GET /api/sessions/{id}/transactions?limit=100&offset=0` → `{total: int, items: TransactionRecord[]}`.
- `GET /api/sessions/{id}/state` → `{layers: [{s_norm_per_head: number[], h_norm: number, singular_values: number[][] (per head, top 8), drift_from_anchor: number}], pos}`.
- `POST /api/sessions/{id}/chat` body `{prompt: str, max_new_tokens?: 128, temperature?: 0.9, top_k?: 50, seed?: int}` → `{prompt, completion, transactions: TransactionRecord[], n_tokens_in, n_tokens_out, summary: RunnerSummary}` (400 for a physics session).
- `POST /api/sessions/{id}/physics` body `{steps?: 256, mu?: 0.12, seed?: 0, nonlinear?: false}` → `{mu, steps, per_step: [{t, base_mse, frozen_mse, adaptive_mse}], transactions: TransactionRecord[], means: {base_mse, frozen_mse, adaptive_mse}, summary: RunnerSummary}` (400 for a text session).
- `POST /api/sessions/{id}/fork` body `{child_session_id?: str}` → child `SessionSummary`.
- `POST /api/sessions/{id}/reset` → `SessionSummary`; `POST /api/sessions/{id}/resume` → `SessionSummary`.
- `DELETE /api/sessions/{id}` → `{deleted: true}`.

### Training jobs (local subprocesses)
- `GET /api/train/jobs` → `[{model_id, pid, status: "running"|"finished", exit_code, started_at_unix}]`.
- `POST /api/train` body `{domain: "text"|"physics", data_dir?: str, model_id?: str, steps: int, batch_size: int, seq_len: int, d_model?: int, layers?: int, heads?: int, chunk?: int, adversarial?: bool, device?: str, eval_every?: int, save_every?: int}` → `{model_id, pid}`; spawns `python -m plastic.cli train ...` with the process's interpreter.
- `GET /api/train/{model_id}` → `{model_id, status, phase?, pid?, exit_code?, latest: TrainLogRecord|null, eval: EvalSummary|null, error?: str}`.
- `POST /api/train/{model_id}/cancel` → `{model_id, status}`.

### Red team and sleep
- `GET /api/redteam` → `[{run_id, model_id, created_at_unix, families: {[family]: {n, damage_mean, damage_max, provisional_damage_max, gated_fraction, constraint_violated_fraction, over_threshold_fraction}}, threshold_coherence}]` from `artifacts/redteam/*/summary.json`.
- `POST /api/redteam` body `{model_id, data_dir?, prefixes?: 4, prefix_len?: 128, suffix_len?: 64, steps?: 30, families?: string[], record?: false}` → the summary above (synchronous).
- `GET /api/redteam/{run_id}` → `{summary, results: AttackResult[]}` (`AttackResult` = `plastic/redteam/attack.py::AttackResult.to_dict()`).
- `POST /api/sleep` body `{model_id, sessions?: string[], core_data_dir?, steps?: 200, lr?: 1e-4, core_ratio?: 0.8, seq_len?: 256}` → the manifest from `consolidate` (with `accepted`, `model_id` when accepted).

### Serving
- `plastic serve --artifacts-root artifacts --host 127.0.0.1 --port 13579 [--device cpu|mps]`; `start.sh` starts the API then the dashboard dev server with `VITE_API_URL`.

## Dashboard (M6)

One Zustand store (`models`, `sessions`, `currentSessionId`, `jobs`, `redteamRuns`, `activeTab`, `loading`, `error`), one `fetchJson` in `dashboard/src/api/client.ts`, types in `dashboard/src/api/types.ts` mirroring the contract. Tabs:

1. **Sessions**: lineage tree (root → forks, both domains) and a table (id, domain, model, pos, transactions, commits, rollbacks, budget, read-only); create (model picker + harness overrides), fork, reset, delete.
2. **Session**: header with the runner summary; transaction timeline (one mark per chunk: commit/rollback/scale/project/readonly with reasons on hover and click-to-select); signal panels for the selected chunk and over time: chunk loss, surprise, β mean, update norm (log) with the calibrated thresholds drawn as reference lines when present; canary suite before/after (coherence and poison) per chunk; CUSUM state and budget meter; per-layer state (β histogram from the model eval, ‖S‖ per head, top singular values, drift from anchor).
3. **Chat** (text sessions): prompt box, completion, and the transactions of that turn inline with their decisions and reasons; controls for max tokens, temperature, top-k, seed.
4. **Physics** (physics sessions): mu and steps controls; three-way error curve (base, frozen, adaptive) on one chart; mean table; the episode's transactions.
5. **Train**: models table (domain, status, params, held-out loss, memory value, MQAR accuracies, calibrated flag); start a local job form; live loss chart polled every 2 s while running; the three checkpoint numbers; a calibrate button.
6. **Red team**: run a campaign form; per-family results table with validated vs provisional damage and gated fraction; a scatter of damage vs payload NLL per attack; the recorded-payload count.
7. **Architecture**: a diagram of one block built from the live model config (SSM branch, memory branch, MLP, chunk size, heads), the harness stack (signals → policy → decisions), and a short explanation with links to the research docs.

Design floor: tokens for surface/text/accents defined in `tailwind.config.js` and all used; fonts ≥ 11 px (Recharts tick fontSize ≥ 11 too); body 14 px; `-webkit-font-smoothing: antialiased` only inside `@media (min-resolution: 2dppx)`; no emoji anywhere; saturated color only for status (commit green, rollback red, scale amber, project blue, readonly gray); no `opacity-*` on text; no textures.

Reuse from the old `dashboard/src` where sound: `WeightHeatmap`-style diverging color util, Recharts tooltip styling, the transaction timeline concept. Delete everything else (old tabs, mock data, nano types). Keep `dashboard/package.json` deps; add `vitest` and `@testing-library/react` as dev deps.

## Tasks

### M5
1. `plastic/api/app.py` (`create_app(artifacts_root, device)`), `plastic/api/registry.py` (`SessionRegistry` with per-session `threading.Lock`, model cache), `plastic/api/routers/{health,models,sessions,train,redteam,sleep}.py`, `plastic/api/jobs.py` (subprocess job manager), `plastic/api/schemas.py` (pydantic models for request bodies), `plastic serve` in `cli.py`, `start.sh`.
2. `tests/test_api.py`: a module fixture that trains a tiny text and a tiny physics model into a temp artifacts root (3 steps each, tiny config, chunk 8), writes canary suites, then exercises every route with `TestClient`, including chat, physics episode, fork, reset, calibrate, redteam (2 prefixes, 2 steps, families pgd+random), sleep (loose tolerance), and error codes.

### M6
3. Scaffold the new `dashboard/src` (delete old components/tabs/store/types/mockData), `api/client.ts`, `api/types.ts`, `store/index.ts`, `App.tsx` with tab routing and keyboard shortcuts (1 to 7 select tabs), layout `Header.tsx` (session picker, API status).
4. Tabs 1 to 7 as above with shared components in `components/{charts,panels}`.
5. `npm run build`, vitest for `store` and `client` (fetch mocked), and a smoke run against the live API with `plastic serve` (screenshot at devicePixelRatio 1 if a browser is available).

## Self-review

- Spec 10 coverage: API routes (M5 tasks 1, 2), dashboard tabs 1 to 7 (M6 tasks 3, 4, 5), design floor (global constraints).
- Type consistency: every dashboard type mirrors a Python `to_dict()`; names in this contract are the source of truth for both agents.
