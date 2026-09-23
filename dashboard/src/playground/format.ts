// Formatting and labeling. Every label here is a STATE, stated once, never an explanation.

import type { DecisionKind, HarnessConfig, RunnerSummary, TraceRecord, TransactionRecord, Turn } from './types';

export const DECISION_COLOR: Record<DecisionKind, string> = {
  commit: '#3fd17a',
  rollback: '#ff6b6b',
  scale: '#f0b429',
  project: '#58a6ff',
  readonly: '#94a3b4',
};

export const DECISION_LABEL: Record<DecisionKind, string> = {
  commit: 'Commit',
  rollback: 'Rollback',
  scale: 'Scale',
  project: 'Project',
  readonly: 'Read-only',
};

/** A second channel beside hue, so decisions are distinguishable without color. */
export const DECISION_GLYPH: Record<DecisionKind, string> = {
  commit: '●',
  rollback: '✕',
  scale: '◐',
  project: '◇',
  readonly: '○',
};

export function isFinite_(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

export function fmt(v: unknown, digits = 3): string {
  if (!isFinite_(v)) return 'n/a';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(Math.max(0, digits - 2));
  if (a >= 1) return v.toFixed(digits - 1);
  return v.toPrecision(digits);
}

export function fmtInt(v: unknown): string {
  return isFinite_(v) ? Math.round(v).toLocaleString() : 'n/a';
}

export function ago(unix: number | null | undefined, now = Date.now()): string {
  if (!isFinite_(unix)) return 'n/a';
  const s = Math.max(0, Math.round(now / 1000 - unix));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

/** Observational: no gate acts, decisions record what would have happened. Guarded: the harness intervenes. */
export function modeLabel(harness: HarnessConfig | undefined): 'observational' | 'guarded' {
  return harness?.log_only ? 'observational' : 'guarded';
}

/** Where the harness thresholds come from for THIS session. */
export function thresholdSource(summary: RunnerSummary | undefined): string {
  switch (summary?.calibration) {
    case 'installed':
      return 'calibrated thresholds';
    case 'rejected_signature_mismatch':
      return 'session-relative thresholds (saved calibration is for another checkpoint)';
    case 'rejected_unsigned':
      return 'session-relative thresholds (saved calibration is unsigned)';
    default:
      return 'session-relative thresholds';
  }
}

export function backendLabel(backend: string | undefined): string {
  switch (backend) {
    case 'ttt':
      return 'TTT-MLP fast weights';
    case 'qwen':
      return 'Gated DeltaNet (Qwen3.5)';
    case 'plastic':
    case undefined:
      return 'Plastic delta memory';
    default:
      return backend;
  }
}

/** An intervention is an applied decision other than commit; read-only is an observation, not an intervention. */
export function isIntervention(kind: DecisionKind): boolean {
  return kind === 'rollback' || kind === 'scale' || kind === 'project';
}

/**
 * The decision the policy would have applied. In observational (log-only) mode the runner commits and records
 * the policy's choice as `would_rollback:` / `would_scale:` / `would_project:` reasons on the requested decision;
 * in guarded mode `requested.kind` itself may differ from the applied decision when a later check overrode it.
 */
export function wouldKind(tx: TransactionRecord): DecisionKind | null {
  if (tx.requested.kind !== tx.decision.kind && isIntervention(tx.requested.kind)) return tx.requested.kind;
  for (const kind of ['rollback', 'project', 'scale'] as const) {
    if (tx.requested.reasons.some((r) => r.startsWith(`would_${kind}:`))) return kind;
  }
  return null;
}

export function wouldIntervene(tx: TransactionRecord): boolean {
  return wouldKind(tx) !== null;
}

/** The chunk's own size for the learning strip: its tokens' write norm where the backend reports one, else the proposed change. */
export function chunkSize(tx: TransactionRecord): number {
  const w = tx.signals.write_norm_sum;
  return isFinite_(w) ? w : tx.signals.delta_norm;
}

export function chunkSource(tx: TransactionRecord): 'prompt' | 'model' | 'mixed' | 'unknown' {
  const s = tx.sources;
  if (!s) return 'unknown';
  if (s.user > 0 && s.model > 0) return 'mixed';
  if (s.user > 0) return 'prompt';
  if (s.model > 0) return 'model';
  return 'unknown';
}

/** Group a session's transactions under its chat turns by position. Chunks before the first turn (none, normally) are dropped. */
export function turnsFromTrace(trace: TraceRecord[], transactions: TransactionRecord[]): Turn[] {
  const chats = trace.filter((t) => t.kind === 'chat').slice().sort((a, b) => a.pos_end - b.pos_end);
  const sorted = transactions.slice().sort((a, b) => a.pos_start - b.pos_start);
  const out: Turn[] = [];
  let start = 0;
  chats.forEach((t, i) => {
    const chunks = sorted.filter((tx) => tx.pos_start >= start && tx.pos_end <= t.pos_end);
    out.push({ index: i, prompt: t.prompt ?? '', completion: t.completion ?? '', t_unix: t.t_unix, pos_end: t.pos_end, chunks });
    start = t.pos_end;
  });
  return out;
}

export interface TurnTotalsData {
  chunks: number;
  committed: number;
  intervened: number;
  wouldIntervene: number;
  readOnly: number;
  proposed: number;
  accepted: number;
  promptTokens: number;
  modelTokens: number;
}

export function turnTotals(chunks: TransactionRecord[]): TurnTotalsData {
  const t: TurnTotalsData = { chunks: chunks.length, committed: 0, intervened: 0, wouldIntervene: 0, readOnly: 0, proposed: 0, accepted: 0, promptTokens: 0, modelTokens: 0 };
  for (const tx of chunks) {
    if (tx.decision.kind === 'commit') t.committed += 1;
    if (isIntervention(tx.decision.kind)) t.intervened += 1;
    if (tx.decision.kind === 'readonly') t.readOnly += 1;
    if (wouldIntervene(tx)) t.wouldIntervene += 1;
    if (isFinite_(tx.signals.delta_norm)) t.proposed += tx.signals.delta_norm;
    if (isFinite_(tx.accepted?.delta_norm)) t.accepted += tx.accepted.delta_norm;
    t.promptTokens += tx.sources?.user ?? 0;
    t.modelTokens += tx.sources?.model ?? 0;
  }
  return t;
}

/** Which per-chunk fields the backend actually produced (any finite value in the records). */
export function presentFields(transactions: TransactionRecord[], fields: string[]): string[] {
  return fields.filter((f) => transactions.some((tx) => isFinite_((tx.signals as unknown as Record<string, unknown>)[f])));
}

/** The signals named in the policy's reasons (applied or would-have), without the mode marker itself. */
export function firedSignals(tx: TransactionRecord): string[] {
  const reasons = tx.requested.reasons.length ? tx.requested.reasons : tx.decision.reasons;
  return reasons
    .filter((r) => r !== 'log_only' && r !== 'learning_ineligible' && r !== 'session_read_only')
    .map((r) => r.replace(/^would_(rollback|scale|project):/, '').split('(')[0]);
}
