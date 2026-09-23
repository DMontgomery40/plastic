// Small building blocks shared by the three observatory views. Legibility floor: nothing below 11px, solid
// backgrounds, text in ink tokens (status hue only on marks and short state words), no opacity on text.

import { useEffect, useState, type ReactNode } from 'react';
import { OBS } from '../components/charts/theme';
import { checkState, num, OUTCOME_LABEL, type CheckState, type Outcome } from './format';
import type { GateCheck, Harvest } from './types';

export function useAsync<T>(load: (() => Promise<T>) | null, deps: unknown[]): { data: T | null; error: string | null } {
  const [state, setState] = useState<{ data: T | null; error: string | null }>({ data: null, error: null });
  useEffect(() => {
    if (!load) {
      setState({ data: null, error: null });
      return;
    }
    let live = true;
    setState({ data: null, error: null });
    load().then(
      (data) => live && setState({ data, error: null }),
      (e: unknown) => live && setState({ data: null, error: e instanceof Error ? e.message : String(e) })
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

const OUTCOME_TONE: Record<Outcome, string> = {
  committed: 'text-status-commit border-status-commit',
  'pulled back': 'text-status-rollback border-status-rollback',
  control: 'text-ink-secondary border-edge-strong',
  unknown: 'text-ink-secondary border-edge-strong',
};

/** Outcome glyph: a check for committed, a cross for pulled back, a dash for controls. Shape carries the state too. */
export function OutcomeGlyph({ outcome, size = 14 }: { outcome: Outcome; size?: number }) {
  const color = outcome === 'committed' ? OBS.pass : outcome === 'pulled back' ? OBS.fail : OBS.inkMuted;
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" className="shrink-0">
      {outcome === 'committed' ? (
        <path d="M3 8.5l3.2 3.2L13 5" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      ) : outcome === 'pulled back' ? (
        <path d="M4 4l8 8M12 4l-8 8" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
      ) : (
        <path d="M4 8h8" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
      )}
    </svg>
  );
}

export function OutcomeBadge({ outcome, label }: { outcome: Outcome; label?: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded border bg-surface-overlay px-2 py-0.5 text-xs font-semibold ${OUTCOME_TONE[outcome]}`}>
      <OutcomeGlyph outcome={outcome} size={12} />
      {label ?? OUTCOME_LABEL[outcome]}
    </span>
  );
}

export function Chip({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span title={title} className="inline-flex items-center rounded border border-edge-strong bg-surface-overlay px-2 py-0.5 font-mono text-xs text-ink-secondary">
      {children}
    </span>
  );
}

/** Marks content that is not a measurement. Appears on every illustrative element, not only on the toggle. */
export function IllustrativeTag() {
  return (
    <span className="inline-flex items-center rounded border border-illustrative bg-illustrative-soft px-2 py-0.5 text-xs font-semibold text-illustrative">
      Illustrative, not measured
    </span>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return <div className="text-label font-semibold uppercase tracking-wide text-ink-muted">{children}</div>;
}

export function Big({ children, tone = 'default' }: { children: ReactNode; tone?: 'default' | 'pass' | 'fail' | 'accent' | 'illustrative' }) {
  const cls = { default: 'text-ink-primary', pass: 'text-status-commit', fail: 'text-status-rollback', accent: 'text-accent', illustrative: 'text-illustrative' }[tone];
  return <div className={`font-mono text-xl font-semibold ${cls}`}>{children}</div>;
}

const CHECK_TONE: Record<CheckState, { dot: string; text: string }> = {
  passed: { dot: OBS.pass, text: 'text-status-commit' },
  failed: { dot: OBS.fail, text: 'text-status-rollback' },
  'not in force': { dot: OBS.notInForce, text: 'text-ink-secondary' },
  'not measured': { dot: OBS.notInForce, text: 'text-ink-secondary' },
};

/**
 * One damage-gate check as a value-against-limit track. The track spans zero, the limit and the value; the dashed line
 * is the limit, the dot is the measured value. A check that was not in force shows its value (if any) without a verdict.
 */
export function GateCheckRow({ check }: { check: GateCheck }) {
  const state = checkState(check);
  const tone = CHECK_TONE[state];
  const v = check.value;
  const limit = check.limit;
  const lo = Math.min(0, v ?? 0, limit ?? 0);
  const hi = Math.max(0, v ?? 0, limit ?? 0);
  const pad = (hi - lo || 1) * 0.15;
  const x = (t: number) => `${((t - lo + pad) / (hi - lo + 2 * pad)) * 100}%`;
  return (
    <li className="grid grid-cols-[minmax(9rem,13rem)_1fr_auto] items-center gap-3">
      <div className="min-w-0">
        <div className="text-sm font-semibold text-ink-primary">{check.label}</div>
        {check.value_source && check.value_source !== 'recorded measurement' ? <div className="text-micro text-ink-muted">{check.value_source}</div> : null}
      </div>
      <div className="relative h-11" role="img" aria-label={`${check.label}: value ${num(v)}, limit ${num(limit)}, ${state}`}>
        <div className="absolute inset-x-0 top-3.5 h-px bg-edge-strong" />
        {lo < 0 ? (
          <>
            <div className="absolute top-1.5 h-4 w-px bg-edge-strong" style={{ left: x(0) }} />
            <span className="absolute top-6 -translate-x-1/2 font-mono text-micro text-ink-muted" style={{ left: x(0) }}>0</span>
          </>
        ) : null}
        {limit !== null ? (
          <>
            <div className="absolute top-0 h-7 border-l-2 border-dashed border-ink-muted" style={{ left: x(limit) }} />
            <span className="absolute top-7 -translate-x-1/2 whitespace-nowrap text-micro text-ink-muted" style={{ left: x(limit) }}>limit</span>
          </>
        ) : null}
        {v !== null ? (
          <div className="absolute top-3.5 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2" style={{ left: x(v), backgroundColor: tone.dot, borderColor: OBS.surface }} />
        ) : null}
      </div>
      <div className="text-right font-mono text-sm">
        <span className="text-ink-primary">{num(v)}</span>
        <span className="text-ink-muted"> / {num(limit)}</span>
        <div className={`text-xs font-semibold ${tone.text}`}>{state}</div>
      </div>
    </li>
  );
}

/** What the arm's text selection kept, as a proportional bar with counts. Absent selections are said to be absent. */
export function TurnsBar({ harvest }: { harvest: Harvest }) {
  const tbr = harvest.turns_by_reason ?? {};
  const accepted = tbr.accepted ?? 0;
  const rejected = Object.entries(tbr).filter(([k]) => k !== 'accepted').reduce((a, [, v]) => a + (v ?? 0), 0);
  const all = String(harvest.provenance ?? '').startsWith('all');
  const sel = harvest.selected_turns.value;
  const flaggedOut = harvest.flagged_excluded ?? 0;
  const segments =
    sel === null
      ? [
          { n: accepted, label: 'accepted online', color: OBS.line },
          { n: rejected, label: 'rolled back', color: OBS.fail },
        ]
      : all
        ? [{ n: sel, label: `selected (${accepted} accepted, ${rejected} rolled back)`, color: OBS.line }]
        : [
            { n: sel, label: 'selected', color: OBS.line },
            { n: flaggedOut, label: 'flagged, excluded', color: OBS.flag },
            { n: rejected, label: 'rolled back, excluded', color: OBS.fail },
          ];
  const total = segments.reduce((a, s) => a + s.n, 0) || 1;
  return (
    <div>
      <div className="flex h-3 w-full gap-0.5 overflow-hidden rounded-sm bg-surface-inset" role="img" aria-label={segments.map((s) => `${s.n} ${s.label}`).join(', ')}>
        {segments.filter((s) => s.n > 0).map((s) => (
          <div key={s.label} style={{ width: `${(s.n / total) * 100}%`, backgroundColor: s.color }} />
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-secondary">
        {segments.map((s) => (
          <li key={s.label} className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: s.color }} />
            <span className="font-mono font-semibold text-ink-primary">{s.n}</span> {s.label}
          </li>
        ))}
        <li className="text-ink-muted">
          {sel === null ? 'selection not recorded for this run' : harvest.selected_turns.source === 'recorded' ? 'selection recorded' : 'selection derived from recorded counts'}
        </li>
      </ul>
    </div>
  );
}
