import type { CalibrationSummary, SessionSummary } from '../../api/types';
import { UNAVAILABLE, fmtInt, fmtPercent, isNum } from '../../utils/formatting';
import { Empty, KeyValue } from './index';

/**
 * Three quantities that are easy to confuse and must never be merged:
 *
 *   1. target_fpr        what the calibration ASKED for. A request, not a guarantee.
 *   2. achievable_fpr    what the calibration SAMPLE can support, per signal.
 *                        This is the honest number; a small sample cannot reach the target.
 *   3. intervention rate what a session actually DID: the fraction of chunks that were
 *                        not committed. It is measured against live traffic, not a
 *                        labelled benign stream, so it is NOT a false-positive rate.
 *
 * They are rendered in three separately titled blocks so no reader can read one
 * as evidence for another.
 */

export interface InterventionCounts {
  n_transactions: number;
  commits: number;
  rollbacks: number;
  scales: number;
  projects: number;
  readonly: number;
}

export function interventionRate(c: InterventionCounts): number | null {
  const total = c.n_transactions;
  if (!isNum(total) || total <= 0) return null;
  const nonCommit = (c.rollbacks ?? 0) + (c.scales ?? 0) + (c.projects ?? 0) + (c.readonly ?? 0);
  return nonCommit / total;
}

export function countsFromSession(s: SessionSummary): InterventionCounts {
  return {
    n_transactions: s.n_transactions ?? 0,
    commits: s.commits ?? 0,
    rollbacks: s.rollbacks ?? 0,
    scales: s.scales ?? 0,
    projects: s.projects ?? 0,
    readonly: s.readonly ?? 0,
  };
}

/** The calibration's requested target and its per-signal achievable rate. */
export function CalibratedRates({ calibration }: { calibration: CalibrationSummary | null }) {
  if (!calibration) {
    return (
      <Empty
        title="This model is not calibrated."
        detail="Without a calibration there is no target rate and no achievable rate; the policy falls back to robust z-scores over the session's own history."
        command="uv run plastic calibrate <model_id> --data artifacts/data/wikitext"
      />
    );
  }

  const achievable = Object.entries(calibration.achievable_fpr ?? {});

  return (
    <div className="space-y-4">
      <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
        <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">1. Requested target rate</p>
        <p className="mt-1 font-mono text-lg text-ink-primary">{fmtPercent(calibration.target_fpr, 2)}</p>
        <p className="mt-1 text-micro text-ink-secondary">
          What the calibration was asked for, over {fmtInt(calibration.n_chunks)} benign chunks. A request, not a guarantee.
        </p>
      </div>

      <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
        <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">2. Achievable rate, per signal</p>
        {achievable.length === 0 ? (
          <p className="mt-1 text-sm text-ink-secondary">
            No achievable rate was recorded for any signal, so the target above is unsupported by evidence.
          </p>
        ) : (
          <div className="mt-2">
            <KeyValue
              rows={achievable.map(([signal, rate]) => ({
                label: signal,
                value: isNum(rate) ? fmtPercent(rate, 3) : UNAVAILABLE,
              }))}
            />
          </div>
        )}
        <p className="mt-2 text-micro text-ink-secondary">
          What the calibration sample can actually support. This is the honest number when it differs from the target.
        </p>
      </div>
    </div>
  );
}

/** What a session actually did. Deliberately not called a false-positive rate. */
export function ObservedIntervention({ counts, label }: { counts: InterventionCounts; label: string }) {
  const rate = interventionRate(counts);
  const nonCommit = counts.rollbacks + counts.scales + counts.projects + counts.readonly;
  return (
    <div className="rounded border border-edge-strong bg-surface-overlay px-3 py-2.5">
      <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">3. Observed intervention rate</p>
      <p className="mt-1 font-mono text-lg text-ink-primary">{rate === null ? UNAVAILABLE : fmtPercent(rate, 2)}</p>
      <p className="mt-1 text-micro text-ink-secondary">
        {label}: {fmtInt(nonCommit)} of {fmtInt(counts.n_transactions)} chunks were not committed
        {counts.n_transactions > 0
          ? ` (${fmtInt(counts.rollbacks)} rollback, ${fmtInt(counts.scales)} scale, ${fmtInt(counts.projects)} project, ${fmtInt(counts.readonly)} read-only)`
          : ''}
        .
      </p>
      <p className="mt-1.5 text-micro text-ink-muted">
        This is not a false-positive rate. It is measured on whatever this session was fed, not on a labelled benign
        stream, so it says nothing about how many of these interventions were warranted.
      </p>
    </div>
  );
}

/** Compact variant for a table cell or a tile hint. */
export function rateSummaryText(counts: InterventionCounts): string {
  const rate = interventionRate(counts);
  return rate === null ? UNAVAILABLE : `${fmtPercent(rate, 1)} intervened`;
}
