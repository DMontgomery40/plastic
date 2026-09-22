// Types for the plastic API (docs/superpowers/plans/2026-09-22-m5-m6-api-and-dashboard.md).
// Every shape here mirrors a Python `to_dict()`; the plan's contract is the source of truth.
//
// Nullability note: several Python fields default to `float("nan")` or `None`
// (AttackResult canary/poison fields, redteam `provisional_damage_max`,
// fisher/canary signals before a calibration exists). They are typed
// `number | null` and must be guarded before formatting.

export type Domain = 'text' | 'physics';
export type DecisionKind = 'commit' | 'rollback' | 'scale' | 'project' | 'readonly';
export type Rule = 'delta' | 'chunk';
export type MemoryKind = 'linear' | 'mlp';
export type MemoryInput = 'ssm_out' | 'block_in';

// ---------------------------------------------------------------- health, data

export interface Health {
  ok: boolean;
  artifacts_root: string;
  device: string;
  n_models: number;
  n_sessions: number;
}

export interface DataDir {
  name: string;
  dir: string;
  corpus: string;
  vocab_size: number;
  splits: { train: number; validation: number; test: number };
}

// ---------------------------------------------------------------------- models

export interface BetaHist {
  edges: number[];
  counts: number[];
  beta_mean: number;
  beta_std: number;
  alpha_mean: number;
}

// A checkpoint evaluated before the first eval ran reports nulls for the three
// numbers; observed live on artifacts/models/lm_smoke_mps.
export interface EvalSummary {
  step: number;
  heldout_loss: number | null;
  heldout_loss_beta0: number | null;
  memory_value: number | null;
  mqar_accuracy?: Record<string, number> | null;
  beta_hist?: BetaHist | null;
  heldout_tokens?: number | null;
  heldout_elements?: number | null;
  evaluated_at_unix?: number | null;
}

export interface ModelSummary {
  model_id: string;
  domain: Domain;
  status: string;
  params: number;
  created_at_unix: number;
  updated_at_unix: number;
  steps?: number | null;
  tokens?: number | null;
  eval?: EvalSummary | null;
  calibrated: boolean;
  has_canary: boolean;
  /** Lineage: the model a sleep-consolidated child was distilled from. */
  parent_model_id?: string | null;
  /** e.g. "sleep_consolidation". Absent for a normally trained model. */
  type?: string | null;
  /**
   * The sleep consolidation manifest of a child model, with its accept/reject
   * outcome and canary deltas. Absent for a model that was never consolidated,
   * and for an older child whose record predates the field.
   */
  sleep?: SleepManifest | null;
  device?: string | null;
  error?: string | null;
}

export interface ModelConfig {
  domain: Domain;
  d_model: number;
  n_heads: number;
  n_layers: number;
  chunk: number;
  scan_chunk: number;
  conv_kernel: number;
  vocab_size: number;
  tie_embeddings: boolean;
  rule: Rule;
  memory: MemoryKind;
  memory_input: MemoryInput;
  mlp_mult: number;
  obs_dim: number;
  act_dim: number;
  ssm_c: number;
  beta_bias_init: number;
  alpha_bias_init: number;
  lam_init: number;
  chunk_momentum: number;
  chunk_orthogonalize: boolean;
  chunk_lr: number;
  mem_hidden_mult: number;
}

export interface CalibrationSummary {
  n_chunks: number;
  /**
   * Per-signal threshold. A null value means the bound is UNBOUNDED: the
   * quantile saturated or the signal has no finite limit. Never render a null
   * threshold as a number, and never as zero.
   */
  thresholds: Record<string, number | null>;
  /**
   * The false-positive rate the calibration sample can actually support per
   * signal. This, not `target_fpr`, is the honest number: `target_fpr` is what
   * was requested, and a small sample cannot deliver it.
   */
  achievable_fpr: Record<string, number | null>;
  canary_baseline: Record<string, number | null>;
  reference_sizes: Record<string, number>;
  /** What was asked for. A request, not a guarantee. */
  target_fpr: number;
  created_at_unix: number;
  model_signature?: string | null;
}

export interface CanaryCounts {
  n_coherence: number;
  n_poison: number;
}

export interface TrainLogRecord {
  step: number;
  loss?: number | null;
  grad_norm?: number | null;
  lr_scale?: number | null;
  tokens?: number | null;
  seconds?: number | null;
  tok_per_s?: number | null;
  adv_damage?: number | null;
  event?: 'eval' | null;
  heldout_loss?: number | null;
  memory_value?: number | null;
}

export interface ModelDetail {
  record: ModelSummary;
  config: ModelConfig;
  eval: EvalSummary | null;
  calibration: CalibrationSummary | null;
  canary: CanaryCounts | null;
  log: TrainLogRecord[];
}

// -------------------------------------------------------------------- harness

export interface HarnessConfig {
  enable_rollback: boolean;
  enable_projection: boolean;
  enable_budget: boolean;
  enable_stats: boolean;
  log_only: boolean;
  canary_delta_max: number;
  poison_delta_min: number;
  z_rollback: number;
  z_scale: number;
  scale_factor: number;
  budget_chunk: number | null;
  budget_session: number | null;
  fisher_drift_max: number | null;
  project_eps_dot: number;
  project_eps_cos: number;
  project_max_removed: number;
  cusum_k: number;
  cusum_h: number;
  freeze_on_alarm: boolean;
  history_window: number;
  target_fpr: number;
  learn_from_generation: boolean;
}

// The decision signals that can carry a calibrated threshold
// (plastic/harness/calibrate.py::ROLLBACK_DECISION_SIGNALS). No other signal
// ever gets a reference line: beta_mean in particular never has one.
export const THRESHOLD_SIGNALS = [
  'chunk_loss',
  'surprise_mean',
  'log_delta_norm',
  'log_write_norm',
  'fisher_update',
  'canary_delta_coherence',
] as const;

export type ThresholdSignal = (typeof THRESHOLD_SIGNALS)[number];

export interface ChunkSignals {
  pos_start: number;
  pos_end: number;
  n_tokens: number;
  chunk_loss: number;
  surprise_mean: number;
  surprise_max: number;
  beta_mean: number;
  alpha_mean: number;
  write_norm_sum: number;
  delta_norm: number;
  delta_norm_per_layer: number[];
  fisher_update: number | null;
  fisher_drift: number | null;
  canary_coherence_before: number | null;
  canary_coherence_after: number | null;
  canary_poison_before: number | null;
  canary_poison_after: number | null;
  canary_delta_coherence: number | null;
  canary_delta_poison: number | null;
  canary_alignment: number | null;
  compression_ratio: number | null;
  z: Record<string, number | null>;
  cusum_alarm: boolean;
  budget_used: number;
  budget_remaining: number | null;
  log_delta_norm: number;
  log_write_norm: number;
}

/**
 * What was actually committed, measured on the committed state AFTER the
 * decision. Distinct from `ChunkSignals`, which is the PROPOSED update measured
 * before it: a rollback reports `delta_norm` 0 here while `signals.delta_norm`
 * is large, and a scale reports the fraction it actually kept.
 *
 * The canary fields are absent when the model has no canary suite. Render them
 * as unavailable, never as zero.
 */
export interface AcceptedMetrics {
  delta_norm: number;
  budget_charge: number;
  budget_used: number;
  budget_remaining: number | null;
  canary_coherence_after?: number | null;
  canary_poison_after?: number | null;
  canary_delta_coherence?: number | null;
  canary_delta_poison?: number | null;
}

export interface Decision {
  kind: DecisionKind;
  reasons: string[];
  scale: number;
}

export interface TransactionRecord {
  index: number;
  t_unix: number;
  pos_start: number;
  pos_end: number;
  /** What was applied. */
  decision: Decision;
  /** What the policy asked for. Differs from `decision` when a later check overrode it. */
  requested: Decision;
  /** PROPOSED: measured on the provisional state, before the decision. */
  signals: ChunkSignals;
  /** ACCEPTED: measured on the committed state, after the decision. */
  accepted: AcceptedMetrics;
  // token counts by source in this chunk: `user` = prompt tokens, `model` = generated tokens
  // (INCLUDING the turn-closure tokens, not only sampled output). The prompt flushes before
  // generation, so a chunk is all-user or all-model, never mixed.
  sources?: { user: number; model: number };
  // whether this chunk was permitted to learn at all. read-only / ineligible is NOT a rollback and
  // NOT evidence of a damaging update -- it is a chunk the policy did not let write.
  eligible?: boolean;
  read_only: boolean;
  read_only_reason: string | null;
  seconds: number;
}

export interface CusumState {
  k: number;
  h: number;
  s_hi: number;
  s_lo: number;
  alarms: number;
}

// Plastic reports per-layer S/h norms; a pretrained backend (Qwen) reports per-memory-unit recurrent
// norms instead. All optional so a summary carries whichever shape its backend produced.
export interface StateNorms {
  s_norm?: number[];
  h_norm?: number[];
  s_norm_total?: number;
  h_norm_total?: number;
  recurrent_norm?: number[];
  recurrent_norm_total?: number;
}

// The session's ACTUAL calibration state as reported by Session.summary(), distinct from whether the
// model has a saved calibration artifact: only 'installed' means the thresholds gate this session.
// A rejected artifact exists but was refused (built for a different checkpoint, or unsigned).
export type CalibrationStatus =
  | 'installed'
  | 'absent'
  | 'rejected_unsigned'
  | 'rejected_signature_mismatch';

// TransactionRunner.summary() returns the ten required keys; Session.summary()
// adds session_id / model_id / domain, plus the backend/calibration/signals fields below.
export interface RunnerSummary {
  pos: number;
  pending: number;
  budget_used: number;
  budget_session: number | null;
  read_only: boolean;
  read_only_reason: string | null;
  n_transactions: number;
  cusum: CusumState;
  state_norms: StateNorms;
  drift_from_anchor: number;
  session_id?: string;
  model_id?: string;
  domain?: Domain;
  // Session.summary() (never the bare runner summary) adds these: which backend drives the session,
  // whether a persisted calibration was actually installed/rejected/absent ON THIS SESSION, and the
  // decision signals this backend can produce. Optional, so a plain RunnerSummary need not carry them.
  backend?: string;
  calibration?: CalibrationStatus;
  signals_available?: string[];
  // EFFECTIVE generation-write policy: model-source (generated) tokens are write-eligible when the
  // backend writes that source (Qwen's recurrent state does) OR learn_from_generation overrides it.
  // The read_only latch suppresses all writes regardless. Write-eligible is not retained learning.
  writes_generation?: boolean;
}

// -------------------------------------------------------------------- sessions

export interface SessionSummary {
  session_id: string;
  model_id: string;
  domain: Domain;
  parent_session_id: string | null;
  root_session_id: string | null;
  forked_at_pos: number | null;
  created_at_unix: number;
  updated_at_unix: number;
  pos: number;
  n_transactions: number;
  commits: number;
  rollbacks: number;
  scales: number;
  projects: number;
  readonly: number;
  budget_used: number;
  read_only: boolean;
  read_only_reason: string | null;
}

export interface SessionMeta extends SessionSummary {
  harness: HarnessConfig;
  model_signature: string;
}

export interface TraceRecord {
  t_unix: number;
  kind: 'chat' | 'episode';
  prompt?: string | null;
  completion?: string | null;
  mu?: number | null;
  steps?: number | null;
  seed?: number | null;
  means?: Record<string, number> | null;
  pos_end: number;
  n_transactions: number;
}

export interface SessionDetail {
  meta: SessionMeta;
  summary: RunnerSummary;
  // the calibration the session ACTUALLY loaded and verified at open (null if none is installed),
  // not the model's current saved artifact -- active policy lines/rates must come from this so a
  // same-model artifact replaced by a separate process is never shown as what the runner uses.
  calibration: CalibrationSummary | null;
  lineage: string[];
  transactions: TransactionRecord[];
  trace: TraceRecord[];
}

export interface TransactionPage {
  total: number;
  items: TransactionRecord[];
}

export interface LayerState {
  s_norm_per_head: number[];
  h_norm: number;
  singular_values: number[][];
  drift_from_anchor: number;
}

// A pretrained backend (Qwen) has no per-head S/h shape; its state is per-memory-unit recurrent
// norms and drift. Norms/drift are nullable: a failed or missing measurement is unavailable, never
// rendered as a measured zero.
export interface RecurrentUnit {
  index: number;
  recurrent_norm: number | null;
  drift_from_anchor: number | null;
}

export interface SessionState {
  // 'plastic' carries `layers`; 'recurrent' carries `units` + `recurrent_norm_total`. Both optional so
  // existing plastic-shape access stays valid; `kind` (absent on older payloads) selects the renderer.
  kind?: 'plastic' | 'recurrent';
  pos: number;
  layers?: LayerState[];
  units?: RecurrentUnit[];
  recurrent_norm_total?: number | null;
  backend?: string;
}

export interface ChatResult {
  prompt: string;
  completion: string;
  transactions: TransactionRecord[];
  n_tokens_in: number;
  n_tokens_out: number;
  summary: RunnerSummary;
}

export interface PhysicsStep {
  t: number;
  base_mse: number;
  frozen_mse: number;
  adaptive_mse: number;
}

export interface EpisodeResult {
  mu: number;
  steps: number;
  per_step: PhysicsStep[];
  transactions: TransactionRecord[];
  means: { base_mse: number; frozen_mse: number; adaptive_mse: number };
  summary: RunnerSummary;
}

// --------------------------------------------------------------- training jobs

export type JobStatus = 'running' | 'finished';

export interface TrainJob {
  model_id: string;
  pid: number;
  status: JobStatus;
  exit_code: number | null;
  started_at_unix: number;
}

export interface TrainStatus {
  model_id: string;
  status: string;
  phase?: string | null;
  pid?: number | null;
  exit_code?: number | null;
  latest: TrainLogRecord | null;
  eval: EvalSummary | null;
  error?: string | null;
}

export interface TrainRequest {
  domain: Domain;
  data_dir?: string;
  model_id?: string;
  steps: number;
  batch_size: number;
  seq_len: number;
  d_model?: number;
  layers?: number;
  heads?: number;
  chunk?: number;
  adversarial?: boolean;
  device?: string;
  eval_every?: number;
  save_every?: number;
}

export interface TrainStarted {
  model_id: string;
  pid: number;
}

// ------------------------------------------------------------ red team, sleep

export interface RedteamFamilyStats {
  n: number;
  /** Attacks whose payload met the plausibility constraint. */
  n_valid: number;
  damage_mean: number;
  damage_max: number;
  /** Same trajectory with the harness disabled: what the attack achieves undefended. */
  unprotected_damage_mean: number | null;
  /** Payload read but not learned: the activation-only change. */
  frozen_damage_mean: number | null;
  /** Restricted to constraint-satisfying attacks. Null when none qualified, never 0. */
  valid_damage_mean: number | null;
  valid_damage_max: number | null;
  /** Peak intermediate proposal, not a final endpoint. */
  provisional_damage_max: number | null;
  gated_fraction: number;
  constraint_violated_fraction: number;
  over_threshold_fraction: number | null;
  valid_over_threshold_fraction: number | null;
}

export interface RedteamSummary {
  run_id: string;
  model_id: string;
  created_at_unix: number;
  families: Record<string, RedteamFamilyStats>;
  threshold_coherence: number | null;
  recorded_payloads?: number | null;
}

export interface AttackResult {
  family: string;
  prefix_ids: number[];
  payload_ids: number[];
  damage_continuous: number | null;
  damage_validated: number;
  nll_payload: number;
  nll_prefix: number;
  nll_max: number;
  constraint_violated: boolean;
  decisions: string[];
  signals: Array<Record<string, unknown>>;
  canary_before: number | null;
  /** Peak intermediate proposal along the guarded run, not an endpoint. */
  canary_after_provisional: number | null;
  canary_after_accepted: number | null;
  /** Endpoint controls from the same post-prefix state. */
  canary_after_unprotected: number | null;
  canary_after_frozen: number | null;
  damage_unprotected: number | null;
  damage_frozen: number | null;
  /** Payload NLL along the guarded trajectory; can differ from `nll_payload`. */
  nll_payload_guarded: number | null;
  poison_before: number | null;
  poison_after_accepted: number | null;
  seconds: number;
}

export interface RedteamDetail {
  summary: RedteamSummary;
  results: AttackResult[];
}

export interface RedteamRequest {
  model_id: string;
  data_dir?: string;
  prefixes?: number;
  prefix_len?: number;
  suffix_len?: number;
  steps?: number;
  families?: string[];
  record?: boolean;
}

export interface SleepManifest {
  base_model_id: string;
  sessions: string[];
  memories: number;
  memory_tokens: number;
  steps: number;
  lr: number;
  core_ratio: number;
  loss_first: number | null;
  loss_last: number | null;
  canary_before: Record<string, number>;
  canary_after: Record<string, number>;
  delta_coherence: number;
  delta_poison: number;
  tolerance: Record<string, number>;
  accepted: boolean;
  seconds: number;
  created_at_unix: number;
  model_id?: string | null;
}

// -------------------------------------------------------------- request bodies

export interface CreateSessionRequest {
  model_id: string;
  session_id?: string;
  harness?: Partial<HarnessConfig>;
}

export interface CalibrateRequest {
  data_dir?: string;
  chunks?: number;
  fisher_chunks?: number;
  fpr?: number;
}

export interface ChatRequest {
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  seed?: number;
}

export interface PhysicsRequest {
  steps?: number;
  mu?: number;
  seed?: number;
  nonlinear?: boolean;
}
