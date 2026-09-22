import type { CalibrationSummary, SessionSummary } from '../../api/types';
import { UNAVAILABLE, fmtInt, fmtPercent, isNum } from '../../utils/formatting';
import { Empty, KeyValue } from './index';

/**
 * Three quantities that are easy to confuse and must never be merged:
 *
 *   1. target_fpr        what the calibration ASKED for. A request, not a guarantee.
 *   2. achievable_fpr    what the calibration SAMPLE can support, per signal.
 *                        This is the honest number; a small sample cannot reach the target.
 *   3. rejection rate    what a session actually DID: of the chunks that proposed a
 *                        write (eligible updates, i.e. NOT read-only observations),
 *                        the fraction the harness rejected or altered (rollback,
 *                        scale, project). A read-only chunk proposed no write and is
 *                        committed as its frozen observation, so it is never an
 *                        intervention. Measured against live traffic, not a labelled
 *                        benign stream, so it is NOT a false-positive rate.
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

/**
 * Interventions are the chunks where the harness rejected or altered a proposed
 * write: rollback, scale, or project. A read-only chunk proposed NO write and is
 * committed as its frozen observation, so readonly is never an intervention.
 */
export function interventionCount(c: InterventionCounts): number {
  return (c.rollbacks ?? 0) + (c.scales ?? 0) + (c.projects ?? 0);
}

/** Chunks that were eligible for a write: every chunk minus the read-only observations. */
export function eligibleUpdateCount(c: InterventionCounts): number {
  return Math.max(0, (c.n_transactions ?? 0) - (c.readonly ?? 0));
}

/** Fraction of ALL chunks that were an actual intervention. Readonly never counts. */
export function interventionRate(c: InterventionCounts): number | null {
  const total = c.n_transactions;
  if (!isNum(total) || total <= 0) return null;
  return interventionCount(c) / total;
}

/**
 * Of the chunks that proposed a write, the fraction the harness rejected or
 * altered. Null when there were no eligible updates (a pure-observation session
 * has no rejection rate to report): render as unavailable, never as 0%.
 */
export function eligibleRejectionRate(c: InterventionCounts): number | null {
  const eligible = eligibleUpdateCount(c);
  if (eligible <= 0) return null;
  return interventionCount(c) / eligible;
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
  const rate = eligibleRejectionRate(counts);
  const interventions = interventionCount(counts);
  const eligible = eligibleUpdateCount(counts);
  const readonly = counts.readonly ?? 0;
  return (
    <div className="rounded border border-edge-strong bg-surface-overlay px-3 py-2.5">
      <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">3. Observed rejection rate, eligible updates only</p>
      <p className="mt-1 font-mono text-lg text-ink-primary">{rate === null ? UNAVAILABLE : fmtPercent(rate, 2)}</p>
      <p className="mt-1 text-micro text-ink-secondary">
        {label}: {fmtInt(interventions)} of {fmtInt(eligible)} eligible update{eligible === 1 ? '' : 's'} rejected or altered
        {eligible > 0
          ? ` (${fmtInt(counts.rollbacks)} rollback, ${fmtInt(counts.scales)} scale, ${fmtInt(counts.projects)} project)`
          : ''}
        .
      </p>
      {readonly > 0 ? (
        <p className="mt-1 text-micro text-ink-secondary">
          {fmtInt(readonly)} read-only observation{readonly === 1 ? '' : 's'} (no write proposed) committed as frozen
          state; these are not interventions and are excluded from the rate above.
        </p>
      ) : null}
      <p className="mt-1.5 text-micro text-ink-muted">
        This is not a false-positive rate. It is measured on whatever this session was fed, not on a labelled benign
        stream, so it says nothing about how many of these interventions were warranted.
      </p>
    </div>
  );
}

/**
 * Compact variant for a table cell or a tile hint: the eligible-update rejection
 * rate. "unavailable" when no chunk proposed a write, so a pure-observation
 * session is never labelled as intervened.
 */
export function rateSummaryText(counts: InterventionCounts): string {
  const rate = eligibleRejectionRate(counts);
  return rate === null ? UNAVAILABLE : `${fmtPercent(rate, 1)} rejected`;
}
