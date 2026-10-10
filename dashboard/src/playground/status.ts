// What the user must know at a glance, derived from data the API already returns (no backend change):
//   1. is the model's memory safe to keep writing right now?   -> light + one sentence
//   2. was the last chunk committed or refused, and why?       -> lastChunk + explainReason
//   3. how close is it to the alarm and to the budget?         -> gauges
//   4. what changed since the last chunk?                      -> lastTurn / budget charge
//
// Every function is pure so the wording is unit-tested. Nothing here invents a number: a quantity the
// backend did not report stays null and the UI says "not reported", never 0.

import { DECISION_LABEL, fmt, isFinite_, isIntervention, wouldKind } from './format';
import { explainLock, explainReason } from './reasons';
import type { RunnerSummary, SessionDetail, TransactionRecord } from './types';

export type Light = 'empty' | 'ok' | 'watch' | 'locked' | 'unguarded';

export interface Gauge {
  label: string;
  /** 0..1+, null when there is nothing to compare against (no cap). */
  fraction: number | null;
  /** The reading as words and numbers, e.g. "2.1 of 5.0". */
  reading: string;
  /** Plain consequence of reaching the limit. */
  atLimit: string;
  level: 'ok' | 'watch' | 'over' | 'none';
}

export interface LockInfo {
  reason: string | null;
  title: string;
  sentence: string;
  kind: ReturnType<typeof explainLock>['kind'];
  /** Index of the chunk whose decision latched the lock, null when the loaded window starts already locked. */
  sinceChunk: number | null;
  /** True when the loaded chunks all precede the lock's start (the real start is earlier than shown). */
  startedBeforeWindow: boolean;
  /** Chunks read without learning since the lock. */
  refusedSince: number;
  /** Tokens read without learning since the lock. */
  tokensSince: number;
}

export interface SessionStatus {
  light: Light;
  label: string;
  sentence: string;
  lock: LockInfo | null;
  cusum: Gauge;
  budget: Gauge;
  /** Chunks among the last `window` that the harness applied an intervention to. */
  recent: { window: number; intervened: number; refused: number; wouldIntervene: number };
}

export const LIGHT_LABEL: Record<Light, string> = {
  empty: 'Ready',
  ok: 'Guarded',
  watch: 'Watch',
  locked: 'Locked',
  unguarded: 'Not guarded',
};

/** A second channel beside hue for the light. */
export const LIGHT_GLYPH: Record<Light, string> = { empty: '○', ok: '●', watch: '▲', locked: '■', unguarded: '◇' };

const WATCH_FRACTION = 0.6;

export function lockInfo(summary: RunnerSummary, transactions: TransactionRecord[]): LockInfo | null {
  if (!summary.read_only) return null;
  const sorted = transactions.slice().sort((a, b) => a.index - b.index);
  let i = sorted.length - 1;
  while (i >= 0 && sorted[i].read_only) i -= 1;
  const first = sorted[i + 1];
  const run = sorted.slice(i + 1);
  const reason = summary.read_only_reason ?? first?.read_only_reason ?? null;
  const e = explainLock(reason);
  const refused = run.filter((t) => t.decision.kind === 'readonly');
  return {
    reason,
    title: e.title,
    sentence: e.sentence,
    kind: e.kind,
    sinceChunk: first ? first.index : null,
    startedBeforeWindow: i < 0 && sorted.length > 0 && sorted[0].index > 0,
    refusedSince: refused.length,
    tokensSince: refused.reduce((a, t) => a + (isFinite_(t.signals?.n_tokens) ? t.signals.n_tokens : 0), 0),
  };
}

export function cusumGauge(summary: RunnerSummary, locked: boolean): Gauge {
  const c = summary.cusum;
  if (!c || !isFinite_(c.h) || c.h <= 0) return { label: 'Update-size alarm', fraction: null, reading: 'not reported', atLimit: 'Learning freezes when the alarm fires.', level: 'none' };
  const top = Math.max(isFinite_(c.s_hi) ? c.s_hi : 0, isFinite_(c.s_lo) ? c.s_lo : 0);
  const fraction = top / c.h;
  const alarms = c.alarms ?? 0;
  // the statistic resets to zero when the alarm fires and is not updated while locked, so a locked session reads 0:
  // say so rather than show a calm gauge
  const reading = locked ? `reset after alarm · ${alarms} alarm${alarms === 1 ? '' : 's'} so far` : `${fmt(top, 2)} of ${fmt(c.h, 2)}${alarms > 0 ? ` · ${alarms} alarm${alarms === 1 ? '' : 's'} so far` : ''}`;
  return {
    label: 'Update-size alarm',
    fraction: locked ? 1 : fraction,
    reading,
    atLimit: 'Learning freezes when the alarm fires.',
    level: locked || fraction >= 1 ? 'over' : fraction >= WATCH_FRACTION ? 'watch' : 'ok',
  };
}

export function budgetGauge(summary: RunnerSummary): Gauge {
  const cap = summary.budget_session;
  const used = summary.budget_used;
  if (cap === null || cap === undefined || !isFinite_(cap) || cap <= 0) {
    return { label: 'Session budget', fraction: null, reading: isFinite_(used) ? `${fmt(used)} used · no cap set` : 'not reported', atLimit: 'No session cap: only the per-chunk limits apply.', level: 'none' };
  }
  const fraction = (isFinite_(used) ? used : 0) / cap;
  return {
    label: 'Session budget',
    fraction,
    reading: `${fmt(used)} of ${fmt(cap)} used`,
    atLimit: 'At the cap the session locks; resuming does not restore it.',
    level: fraction >= 0.999 ? 'over' : fraction >= 0.8 ? 'watch' : 'ok',
  };
}

export function recentCounts(transactions: TransactionRecord[], window = 16) {
  const last = transactions.slice().sort((a, b) => a.index - b.index).slice(-window);
  return {
    window: last.length,
    intervened: last.filter((t) => isIntervention(t.decision.kind)).length,
    refused: last.filter((t) => t.decision.kind === 'rollback' || t.decision.kind === 'readonly').length,
    wouldIntervene: last.filter((t) => wouldKind(t) !== null).length,
  };
}

function firstReason(tx: TransactionRecord): string {
  const reasons = tx.decision.reasons.filter((r) => r !== 'log_only');
  const e = explainReason(reasons[0] ?? '');
  return e.sentence;
}

/** The status the whole page leads with. `detail.meta.harness.log_only` means nothing is ever blocked. */
export function sessionStatus(detail: SessionDetail | null, transactions: TransactionRecord[]): SessionStatus | null {
  if (!detail) return null;
  const summary = detail.summary;
  const sorted = transactions.slice().sort((a, b) => a.index - b.index);
  const last = sorted[sorted.length - 1] ?? null;
  const lock = lockInfo(summary, sorted);
  const cusum = cusumGauge(summary, lock !== null);
  const budget = budgetGauge(summary);
  const recent = recentCounts(sorted);
  const base = { lock, cusum, budget, recent };

  if (lock) {
    const since = lock.sinceChunk !== null ? `since chunk ${lock.sinceChunk}` : '';
    const counter = lock.refusedSince > 0 ? ` ${lock.refusedSince} chunk${lock.refusedSince === 1 ? '' : 's'} since ${lock.refusedSince === 1 ? 'has' : 'have'} been read without learning.` : '';
    return { light: 'locked', label: LIGHT_LABEL.locked, sentence: `Learning is frozen ${since}. ${lock.sentence}${counter}`.replace(/\s+/g, ' ').trim(), ...base };
  }
  if (detail.meta.harness.log_only) {
    const n = recent.wouldIntervene;
    return {
      light: 'unguarded',
      label: LIGHT_LABEL.unguarded,
      sentence: `Observational mode: every update is committed. ${n > 0 ? `The guard would have intervened on ${n} of the last ${recent.window} chunks.` : 'The guard has not flagged recent chunks.'}`,
      ...base,
    };
  }
  if (!last) return { light: 'empty', label: LIGHT_LABEL.empty, sentence: 'Guard is on. No chunks yet; send a message to start learning.', ...base };

  if (last.decision.kind === 'rollback') {
    return { light: 'watch', label: LIGHT_LABEL.watch, sentence: `Last chunk was rolled back. ${firstReason(last)}`, ...base };
  }
  if (last.decision.kind === 'scale' || last.decision.kind === 'project') {
    return { light: 'watch', label: LIGHT_LABEL.watch, sentence: `Last chunk was ${last.decision.kind === 'scale' ? 'committed at reduced strength' : 'committed with part of the update removed'}. ${firstReason(last)}`, ...base };
  }
  if (cusum.level === 'watch') {
    return { light: 'watch', label: LIGHT_LABEL.watch, sentence: `Update sizes are trending up: the alarm reading is at ${Math.round((cusum.fraction ?? 0) * 100)}% of its limit. At the limit, learning freezes.`, ...base };
  }
  if (budget.level === 'watch') {
    return { light: 'watch', label: LIGHT_LABEL.watch, sentence: `${Math.round((budget.fraction ?? 0) * 100)}% of the session budget is used. At the cap the session locks.`, ...base };
  }
  const flagged = recent.intervened;
  return {
    light: 'ok',
    label: LIGHT_LABEL.ok,
    sentence: `Last chunk committed in full. ${flagged > 0 ? `${flagged} of the last ${recent.window} chunks were intervened on.` : 'No interventions in recent chunks.'}`,
    ...base,
  };
}

// ---- proposed versus accepted -------------------------------------------------------------------

export type NormKind = 'full' | 'partial' | 'none' | 'larger' | 'unknown';

export interface NormOutcome {
  proposed: number | null;
  accepted: number | null;
  /** accepted / proposed; can exceed 1 (see scale below). */
  ratio: number | null;
  kind: NormKind;
  /** The scale the policy asked for; the accepted share can differ from it. */
  asked: number | null;
  sentence: string;
}

/**
 * Proposed = what the chunk wanted to write (measured before the decision). Accepted = what the committed state
 * actually moved by (measured after). A scale decision asks for a fraction but memory decay is applied unchanged,
 * so the kept share differs from the request (measured median 0.358 for a 0.25 request, and 4x LARGER in one
 * synthetic case): always show the measured accepted share, never the requested scale alone.
 */
export function normOutcome(tx: TransactionRecord): NormOutcome {
  const proposed = isFinite_(tx.signals?.delta_norm) ? tx.signals.delta_norm : null;
  const accepted = isFinite_(tx.accepted?.delta_norm) ? tx.accepted.delta_norm : null;
  const asked = tx.decision.kind === 'scale' && isFinite_(tx.decision.scale) ? tx.decision.scale : null;
  if (proposed === null || accepted === null) {
    return { proposed, accepted, ratio: null, kind: 'unknown', asked, sentence: 'The size of the change was not reported.' };
  }
  const ratio = proposed > 0 ? accepted / proposed : null;
  if (tx.decision.kind === 'readonly') return { proposed, accepted, ratio, kind: 'none', asked, sentence: 'Nothing was written: the session was read-only.' };
  if (ratio === null) return { proposed, accepted, ratio, kind: accepted > 0 ? 'larger' : 'none', asked, sentence: accepted > 0 ? `The chunk proposed no change but the state moved by ${fmt(accepted)}.` : 'The chunk proposed no change.' };
  const pct = (r: number) => `${Math.round(r * 100)}%`;
  if (ratio < 0.005) return { proposed, accepted, ratio, kind: 'none', asked, sentence: `None of the proposed ${fmt(proposed)} was kept.` };
  if (ratio > 1.02) {
    return { proposed, accepted, ratio, kind: 'larger', asked, sentence: `The state moved by ${fmt(accepted)}, more than the ${fmt(proposed)} proposed (${pct(ratio)})${asked !== null ? `, although ${pct(asked)} was requested` : ''}.` };
  }
  if (ratio >= 0.98) return { proposed, accepted, ratio, kind: 'full', asked, sentence: `Kept the full proposed change of ${fmt(proposed)}.` };
  return { proposed, accepted, ratio, kind: 'partial', asked, sentence: `Kept ${pct(ratio)} of the proposed change (${fmt(accepted)} of ${fmt(proposed)})${asked !== null ? `; the policy asked for ${pct(asked)}` : ''}.` };
}

// ---- what a decision did and did not do ---------------------------------------------------------

/** One sentence on what the applied decision means for the model's memory. */
export function verdictSentence(tx: TransactionRecord): string {
  const would = wouldKind(tx);
  switch (tx.decision.kind) {
    case 'commit':
      return would
        ? `Committed because the guard is observational. Guarded, it would have ${would === 'rollback' ? 'rolled this chunk back' : would === 'scale' ? 'scaled it down' : 'projected it'}.`
        : 'Committed in full. No check fired.';
    case 'rollback':
      return 'Refused. The fast-weight change from this chunk was discarded.';
    case 'scale':
      return 'Committed at reduced strength.';
    case 'project':
      return 'Committed with the part of the update that pointed at the canaries removed.';
    case 'readonly':
      return 'Read without learning: the session is locked.';
    default:
      return DECISION_LABEL[tx.decision.kind];
  }
}

/** What a rollback does NOT undo. Shown with every applied rollback; null for other decisions. */
export function rollbackLimits(tx: TransactionRecord): string | null {
  if (tx.decision.kind !== 'rollback') return null;
  return 'Fast weights were restored exactly. Text the model already generated stays. The model\'s short-term activation state still reflects this chunk, and that effect fades over roughly the next 64 tokens (measured on a small model).';
}
