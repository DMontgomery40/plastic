// Shapes of the exported observatory JSON (scripts/export_sleep_observatory.py, schema plastic.sleep-observatory/1).
// Every nullable field is nullable because the source may not have recorded it: null means absent, never zero.

export type GroupName = 'taught' | 'boundary' | 'rolled' | 'poison' | 'general';
export type ArmName = 'floor' | 'ceiling' | 'anchor' | 'replay' | 'distill' | 'dream' | 'ungated';
export type ArmStatus = 'accepted' | 'rejected' | 'accepted_unmeasured';

export interface GroupCounts {
  n: number;
  recalled: number;
  n_paraphrase: number;
  recalled_paraphrase: number;
}

export interface CountsView {
  by_group: Partial<Record<GroupName, GroupCounts>> | null;
  source: 'saved' | 'saved, confirmed by recount' | 'recounted' | 'counted from replies' | 'absent';
  saved: Partial<Record<GroupName, GroupCounts>> | null;
}

export interface RecallTotals {
  n_probes: number | null;
  recalled: number | null;
  n_paraphrase: number | null;
  recalled_paraphrase: number | null;
  mean_answer_logprob: number | null;
  max_cluster_share: number | null;
  distinct_ratio: number | null;
}

export interface GateCheck {
  name: string;
  label: string;
  value: number | null;
  limit: number | null;
  /** null when the check was not in force at execution */
  passed: boolean | null;
  in_force: boolean;
  value_source?: string | null;
}

export interface Gate {
  measured: boolean | null;
  passed: boolean | null;
  checks: GateCheck[];
}

export interface Nll {
  mean?: number | null;
  median?: number | null;
  tokens?: number | null;
}

export interface HarvestSession {
  session_id: string;
  log_only: boolean | null;
  turns: number | null;
  accepted_turns: number | null;
  has_committed_state: boolean | null;
}

export interface Harvest {
  sessions: HarvestSession[];
  turns_by_reason: Record<string, number> | null;
  accepted_tokens: number | null;
  excluded_tokens: number | null;
  accepted_turns_flagged: number | null;
  flagged_excluded: number | null;
  provenance: string | null;
  flagged_policy: string | null;
  source_sessions: string[] | null;
  selected_turns: { value: number | null; source: string };
}

export interface ProbeRow {
  group: GroupName | null;
  group_attribution: string | null;
  variant: 'verbatim' | 'paraphrase';
  question: string;
  expected: string;
  reply: string;
  reply_truncated: boolean;
  hit: boolean;
  exact: boolean;
  answer_logprob: number | null;
}

export interface DreamKept {
  prompt: string | null;
  turn: string | null;
  text: string;
  session_id: string | null;
  reply_len: number | null;
  teacher_logprob: number | null;
  student_logprob: number | null;
  fastweight_logprob: number | null;
  gain: number | null;
  fastweight_gain: number | null;
  token_gain: (number | null)[] | null;
  token_fw_gain: (number | null)[] | null;
}

export interface Dreams {
  generated: number | null;
  kept_count: number;
  filtered: Record<string, number | null>;
  rejected_reasons: Record<string, number>;
  rejected_samples: { reason: string; text: string }[];
  kept: DreamKept[];
}

export interface PerTensor {
  quantity: string;
  tensors: string[];
  /** [layer][tensor] */
  values: (number | null)[][];
  total?: number | null;
}

export interface GradientNorms {
  quantity: string;
  steps: number[];
  total: (number | null)[];
  per_layer: (number | null)[][];
  other: (number | null)[] | null;
}

export interface Lineage {
  parent: string | null;
  child: string | null;
  outcome: 'committed' | 'pulled back' | 'unknown';
  where: string | null;
}

export interface SleepArm {
  arm: ArmName;
  kind: 'sleep';
  method: string;
  target: string | null;
  status: ArmStatus | null;
  status_conflict: boolean;
  reason: string | null;
  lineage: Lineage;
  gate: Gate | null;
  heldout_nll: { before: Nll | null; after: Nll | null };
  recall: { before: CountsView; after: CountsView; totals_before: RecallTotals | null; totals_after: RecallTotals | null };
  harvest: Harvest | null;
  batch: Record<string, unknown> | null;
  packed: Record<string, number> | null;
  losses: number[] | null;
  loss_terms: string | null;
  dreams: Dreams | null;
  anchor_relative_update: PerTensor | null;
  w0_relative_update: PerTensor | null;
  gradient_norms: GradientNorms | null;
  config: Record<string, unknown> | null;
  probes: ProbeRow[];
  report: { file: string; sha256: string; run_id: string | null; head_at_arm_start: string | null; code_at_import?: string | null; seconds: number | null } | null;
}

export interface ControlArm {
  arm: ArmName;
  kind: 'control';
  method: null;
  status: null;
  recall: { after: CountsView };
  mean_answer_logprob: number | null;
  max_cluster_share: { value: number | null; source: string };
  probes: ProbeRow[];
}

export type Arm = SleepArm | ControlArm;

export interface Checkpoint {
  name: string | null;
  step: number | null;
  label: string;
  digest_prefix: string | null;
  published: boolean;
}

export interface Protocol {
  facts: number | null;
  poison: boolean | null;
  seed: number | null;
  steps: number | null;
  target: string | null;
  lr: number | null;
  replay_ratio: number | null;
  batch_size: number | null;
  session_loss: string | null;
  prompt_loss_weight: number | null;
  augment: string | null;
  teach_temperature: number | null;
  dream_temperature: number | null;
  replay_revision: string | null;
  device: string | null;
  ceiling_mode: string | null;
  flagged_policy: string | null;
  wall_seconds: number | null;
}

export interface Run {
  id: string;
  question: string | null;
  summary: string | null;
  started_at_unix: number | null;
  checkpoint: Checkpoint;
  code: { recorded_at_launch: string | null; archive_readme: string | null };
  protocol: Protocol;
  arms: Arm[];
  notes: { recount: boolean };
  sources: { file: string; sha256: string }[];
  session_trajectory: string | null;
}

export interface IndexArm {
  arm: ArmName;
  status: ArmStatus | null;
  method: string | null;
  taught: GroupCounts | null;
  totals: RecallTotals | null;
  nll_before: number | null;
  nll_after: number | null;
}

export interface IndexRun {
  id: string;
  summary: string | null;
  started_at_unix: number | null;
  checkpoint: Checkpoint;
  code_recorded: string | null;
  facts: number | null;
  flagged_policy: string | null;
  steps: number | null;
  target: string | null;
  session_trajectory: string | null;
  arms: IndexArm[];
}

export interface ObservatoryIndex {
  schema: string;
  exporter_version: string;
  archive_digest: string;
  archive: string;
  chat_eval: Record<string, unknown> | null;
  runs: IndexRun[];
  sessions: string[];
}

export interface Chunk {
  i: number;
  pos: [number, number];
  n: number | null;
  source: 'user' | 'model';
  chunk_loss: number | null;
  surprise_mean: number | null;
  surprise_max: number | null;
  write_norm_sum: number | null;
  proposed: number | null;
  accepted: number | null;
  per_layer: (number | null)[] | null;
  requested: string | null;
  applied: string | null;
  scale: number | null;
  flags: string[];
  applied_reasons: string[];
  read_only: boolean;
}

export interface TrajectorySession {
  session_id: string;
  log_only: boolean;
  enable_rollback: boolean | null;
  z_rollback: number | null;
  z_scale: number | null;
  calibrated: boolean;
  counts: Record<string, number | null>;
  turns: { i: number; prompt: string; completion: string; tx: [number, number] }[];
  chunks: Chunk[];
  source: { transactions: string | null; transactions_sha256: string; trace_sha256: string | null };
}

export interface Trajectory {
  run_id: string;
  schema: string;
  provenance: { kind: string; controls_sha256: string; checks: string[] };
  per_layer: { quantity: string; layers: number; tensors: string[]; order: string };
  sessions: TrajectorySession[];
}
