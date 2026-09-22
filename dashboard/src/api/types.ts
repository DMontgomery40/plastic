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
  parent_model_id?: string | null;
  type?: string | null;
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
  thresholds: Record<string, number>;
  canary_baseline: Record<string, number>;
  reference_sizes: Record<string, number>;
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
  decision: Decision;
  requested: Decision;
  signals: ChunkSignals;
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

export interface StateNorms {
  s_norm: number[];
  h_norm: number[];
  s_norm_total: number;
  h_norm_total: number;
}

// TransactionRunner.summary() returns the ten required keys; Session.summary()
// adds session_id / model_id / domain. Both shapes satisfy this interface.
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

export interface SessionState {
  layers: LayerState[];
  pos: number;
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
  damage_mean: number;
  damage_max: number;
  provisional_damage_max: number | null;
  gated_fraction: number;
  constraint_violated_fraction: number;
  over_threshold_fraction: number | null;
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
  canary_after_provisional: number | null;
  canary_after_accepted: number | null;
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
