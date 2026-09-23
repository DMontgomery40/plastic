// Types for the playground API. Every shape mirrors a Python `to_dict()` in plastic/api; fields that
// Python may emit as None or a non-finite float are `number | null` and are guarded before rendering.
// A missing field means the backend never produces that quantity: the UI omits the panel, never shows 0.

export type DecisionKind = 'commit' | 'rollback' | 'scale' | 'project' | 'readonly';

export interface Capabilities {
  create_session: boolean;
  fork: boolean;
  reset: boolean;
  delete: boolean;
  resume: boolean;
  calibrate: boolean;
}

export interface Health {
  ok: boolean;
  artifacts_root: string;
  device: string;
  n_models: number;
  n_sessions: number;
  capabilities: Capabilities;
  public: boolean;
}

export interface ModelSummary {
  model_id: string;
  backend?: string; // 'ttt' | 'qwen' | 'plastic'; absent on old records means plastic
  domain: 'text' | 'physics';
  status: string;
  params: number;
  calibrated: boolean;
  has_canary: boolean;
  created_at_unix: number;
  updated_at_unix: number;
  eval?: { heldout_loss?: number | null; memory_value?: number | null } | null;
  /** Set on a chat-tuned checkpoint's record; absent on a base model. */
  chat_tuned?: boolean | null;
}

export interface Decision {
  kind: DecisionKind;
  reasons: string[];
  scale: number;
}

/** PROPOSED: measured on the provisional state before the decision. */
export interface ChunkSignals {
  pos_start: number;
  pos_end: number;
  n_tokens: number;
  chunk_loss: number | null;
  surprise_mean?: number | null;
  surprise_max?: number | null;
  beta_mean?: number | null;
  alpha_mean?: number | null;
  write_norm_sum?: number | null;
  delta_norm: number;
  fisher_update?: number | null;
  fisher_drift?: number | null;
  canary_delta_coherence?: number | null;
  canary_delta_poison?: number | null;
  canary_alignment?: number | null;
  z: Record<string, number | null>;
  cusum_alarm: boolean;
  budget_used: number;
  budget_remaining: number | null;
  log_delta_norm: number | null;
  log_write_norm?: number | null;
}

/** ACCEPTED: measured on the committed state after the decision. */
export interface AcceptedMetrics {
  delta_norm: number;
  budget_charge: number;
  budget_used: number;
  budget_remaining: number | null;
  canary_delta_coherence?: number | null;
  canary_delta_poison?: number | null;
}

export interface TransactionRecord {
  index: number;
  t_unix: number;
  pos_start: number;
  pos_end: number;
  /** What was applied. */
  decision: Decision;
  /** What the policy asked for; in observational mode this carries the would-have decision. */
  requested: Decision;
  signals: ChunkSignals;
  accepted: AcceptedMetrics;
  sources?: { user: number; model: number };
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

export type CalibrationStatus = 'installed' | 'absent' | 'rejected_unsigned' | 'rejected_signature_mismatch';

export interface RunnerSummary {
  pos: number;
  pending: number;
  budget_used: number;
  budget_session: number | null;
  read_only: boolean;
  read_only_reason: string | null;
  n_transactions: number;
  cusum: CusumState;
  state_norms: Record<string, number | number[] | undefined>;
  drift_from_anchor: number;
  session_id?: string;
  model_id?: string;
  backend?: string;
  calibration?: CalibrationStatus;
  signals_available?: string[];
  writes_generation?: boolean;
}

export interface HarnessConfig {
  log_only: boolean;
  enable_rollback: boolean;
  enable_stats: boolean;
  enable_projection: boolean;
  enable_budget: boolean;
  freeze_on_alarm: boolean;
  alarm_cooldown?: number;
  learn_from_generation: boolean;
  budget_session: number | null;
  target_fpr: number;
  [key: string]: unknown;
}

export interface SessionSummary {
  session_id: string;
  model_id: string;
  domain: 'text' | 'physics';
  parent_session_id: string | null;
  created_at_unix: number;
  updated_at_unix: number;
  pos: number;
  n_transactions: number;
  commits: number;
  rollbacks: number;
  scales: number;
  projects: number;
  readonly: number;
  read_only: boolean;
  read_only_reason: string | null;
}

export interface SessionMeta extends SessionSummary {
  harness: HarnessConfig;
}

export interface TraceRecord {
  t_unix: number;
  kind: 'chat' | 'episode';
  prompt?: string | null;
  completion?: string | null;
  pos_end: number;
  n_transactions: number;
}

export interface CalibrationSummary {
  n_chunks: number;
  thresholds: Record<string, number | null>;
  achievable_fpr: Record<string, number | null>;
  target_fpr: number;
  created_at_unix: number;
}

export interface SessionDetail {
  meta: SessionMeta;
  summary: RunnerSummary;
  calibration: CalibrationSummary | null;
  lineage: string[];
  transactions: TransactionRecord[];
  trace: TraceRecord[];
}

export interface TransactionPage {
  total: number;
  items: TransactionRecord[];
}

export interface MemoryUnit {
  index: number;
  recurrent_norm: number | null;
  drift_from_anchor: number | null;
}

export interface PlasticLayer {
  s_norm_per_head: number[];
  h_norm: number;
  drift_from_anchor: number;
}

export interface SessionState {
  kind?: 'plastic' | 'recurrent' | 'fast_weight';
  pos: number;
  backend?: string;
  layers?: PlasticLayer[];
  units?: MemoryUnit[];
  recurrent_norm_total?: number | null;
}

export interface ChatResult {
  prompt: string;
  completion: string;
  transactions: TransactionRecord[];
  n_tokens_in: number;
  n_tokens_out: number;
  summary: RunnerSummary;
}

export interface Sampling {
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  seed: number | null;
}

/** One chat exchange with the chunks it produced. */
export interface Turn {
  index: number;
  prompt: string;
  completion: string;
  t_unix: number;
  pos_end: number;
  chunks: TransactionRecord[];
}
