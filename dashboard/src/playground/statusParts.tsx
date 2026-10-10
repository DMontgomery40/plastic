// Small shared pieces of the status header and the 'why' drawer. Colour is always paired with a glyph or a word.

import { DECISION_COLOR, DECISION_GLYPH, DECISION_LABEL, fmt, isFinite_, wouldKind } from './format';
import type { Gauge, Light, NormOutcome } from './status';
import { LIGHT_GLYPH } from './status';
import type { TransactionRecord } from './types';

export const LIGHT_CLASS: Record<Light, { text: string; border: string; fill: string }> = {
  empty: { text: 'text-ink-secondary', border: 'border-edge-strong', fill: 'bg-statusFill-readonly' },
  ok: { text: 'text-status-commit', border: 'border-status-commit', fill: 'bg-statusFill-commit' },
  watch: { text: 'text-status-scale', border: 'border-status-scale', fill: 'bg-statusFill-scale' },
  locked: { text: 'text-status-rollback', border: 'border-status-rollback', fill: 'bg-statusFill-rollback' },
  unguarded: { text: 'text-status-project', border: 'border-status-project', fill: 'bg-statusFill-project' },
};

export function LightBadge({ light, label }: { light: Light; label: string }) {
  const c = LIGHT_CLASS[light];
  return (
    <span className={`inline-flex items-center gap-2 rounded border-2 bg-surface-overlay px-3 py-1 text-lg font-bold ${c.border} ${c.text}`}>
      <span aria-hidden>{LIGHT_GLYPH[light]}</span>
      {label}
    </span>
  );
}

const LEVEL_FILL: Record<Gauge['level'], string> = { ok: 'bg-status-commit', watch: 'bg-status-scale', over: 'bg-status-rollback', none: 'bg-edge-strong' };
const LEVEL_WORD: Record<Gauge['level'], string> = { ok: 'OK', watch: 'Rising', over: 'At limit', none: '' };

/** A horizontal gauge. The reading and a level word carry the meaning; the bar and its hue only repeat it. */
export function GaugeBar({ gauge }: { gauge: Gauge }) {
  const pct = gauge.fraction === null ? null : Math.max(0, Math.min(1, gauge.fraction));
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-label font-semibold uppercase tracking-wide text-ink-muted">{gauge.label}</span>
        {LEVEL_WORD[gauge.level] ? <span className="text-xs font-semibold text-ink-secondary">{LEVEL_WORD[gauge.level]}</span> : null}
      </div>
      <div
        role={pct === null ? undefined : 'meter'}
        aria-label={gauge.label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct === null ? undefined : Math.round(pct * 100)}
        aria-valuetext={gauge.reading}
        className="mt-1 h-2.5 w-full overflow-hidden rounded-sm bg-surface-inset"
      >
        {pct !== null ? <div className={`h-full ${LEVEL_FILL[gauge.level]}`} style={{ width: `${Math.max(2, Math.round(pct * 100))}%` }} /> : null}
      </div>
      <p className="mt-1 break-words font-mono text-xs text-ink-primary">{gauge.reading}</p>
    </div>
  );
}

/** Applied decision as glyph + word in its colour. In observational mode, adds what the guard would have done. */
export function DecisionTag({ tx }: { tx: TransactionRecord }) {
  const k = tx.decision.kind;
  const would = wouldKind(tx);
  return (
    <span className="inline-flex flex-wrap items-baseline gap-x-2">
      <span className="font-semibold" style={{ color: DECISION_COLOR[k] }}>
        <span aria-hidden>{DECISION_GLYPH[k]} </span>
        {DECISION_LABEL[k]}
      </span>
      {would ? (
        <span className="text-xs text-ink-secondary">
          guarded: would <span style={{ color: DECISION_COLOR[would] }}>{DECISION_LABEL[would].toLowerCase()}</span>
        </span>
      ) : null}
    </span>
  );
}

/**
 * Proposed vs accepted change, drawn on one scale. The accepted bar is what the committed state actually moved by,
 * so it can be shorter than, equal to, or (for a scale decision) longer than the share the policy asked for.
 */
export function NormCompare({ outcome }: { outcome: NormOutcome }) {
  const { proposed, accepted } = outcome;
  const max = Math.max(isFinite_(proposed) ? proposed : 0, isFinite_(accepted) ? accepted : 0, 1e-12);
  const bar = (v: number | null, cls: string, label: string) => (
    <div className="flex items-center gap-3">
      <span className="w-20 shrink-0 text-xs text-ink-secondary">{label}</span>
      <div className="h-3 min-w-0 flex-1 rounded-sm bg-surface-inset">
        {v !== null ? <div className={`h-full rounded-sm ${cls}`} style={{ width: `${v === 0 ? 0 : Math.max(2, Math.round((v / max) * 100))}%` }} /> : null}
      </div>
      <span className="w-16 shrink-0 text-right font-mono text-xs text-ink-primary">{v === null ? 'n/a' : fmt(v)}</span>
    </div>
  );
  return (
    <div>
      <div className="space-y-1.5">
        {bar(proposed, 'bg-series-c', 'Proposed')}
        {bar(accepted, 'bg-status-commit', 'Accepted')}
      </div>
      <p className="mt-2 text-sm text-ink-primary">{outcome.sentence}</p>
    </div>
  );
}
