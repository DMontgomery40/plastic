// Formatting helpers. Every number that reaches the screen goes through one of
// these, so a null or a NaN from the API renders as a dash instead of "NaN".

import type { CalibrationStatus, DecisionKind } from '../api/types';

/**
 * The single token for "there is no value here". Never render a missing metric
 * as 0, as an empty cell, or as anything a reader could mistake for a measured
 * result: a null threshold, an absent canary score and a rate nobody computed
 * all say the same honest thing.
 */
export const UNAVAILABLE = 'unavailable';

/** A null threshold means the bound is unlimited, which is not the same as unknown. */
export const UNBOUNDED = 'unbounded';

/**
 * A valid-only red team aggregate is null when no attack in that family met the
 * plausibility constraint. The campaign ran and nothing qualified, which is a
 * different statement from a number that was never reported, and very different
 * from zero damage.
 */
export const NO_VALID_PAYLOADS = 'no valid payloads';

/** True for a real, finite number. The API can send null or NaN for several signals. */
export function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/**
 * How a session's calibration status should read, and — crucially — whether it is ACTIVE. Only an
 * installed calibration gates the session: a rejected artifact (different model / unsigned) or an
 * absent one must never draw active policy threshold lines or claim active rates. `active` is the
 * single gate a caller uses to suppress those. A missing status (a loaded summary without the field)
 * reads as unknown, never as "not calibrated" — the caller distinguishes a failed/loading fetch,
 * which is not a status at all, by whether the summary itself is present.
 */
export interface CalibrationDisplay {
  active: boolean;
  label: string;
  detail: string;
}

export function calibrationDisplay(status: CalibrationStatus | string | null | undefined): CalibrationDisplay {
  switch (status) {
    case 'installed':
      return { active: true, label: 'Installed', detail: 'Calibrated thresholds are active for this session.' };
    case 'absent':
      return {
        active: false,
        label: 'Not calibrated',
        detail: 'No calibration is installed, so the policy falls back to robust z-scores and no thresholds are drawn.',
      };
    case 'rejected_signature_mismatch':
      return {
        active: false,
        label: 'Rejected: different model',
        detail: 'A saved calibration exists but was built for a different checkpoint, so it is not installed on this session.',
      };
    case 'rejected_unsigned':
      return {
        active: false,
        label: 'Rejected: unsigned',
        detail: 'A saved calibration exists but is unsigned, so it is not installed on this session.',
      };
    default:
      return { active: false, label: 'Unknown', detail: 'Calibration status is unavailable for this session.' };
  }
}

/**
 * The single rendering decision for a session's calibration, folding the session's status together
 * with whether the model detail (which carries the threshold artifact) has actually loaded. This is
 * one source of truth so the status tile, the signals-panel subtitle and the rates panel can never
 * contradict each other (an installed session must not simultaneously read "installed" and "not
 * calibrated" while its model detail is still loading; a rejected artifact must not read "not
 * calibrated"). Only the 'active' mode draws threshold lines / active rates; the fallback-to-robust-z
 * statement is asserted only when the session status actually establishes it (absent), or as a
 * distinct, honest reason for rejected / unknown — never for installed-but-details-pending.
 */
export type CalibrationRenderMode = 'active' | 'installed_pending' | 'rejected' | 'absent' | 'unknown';

export interface CalibrationView {
  mode: CalibrationRenderMode;
  drawThresholds: boolean;
  signalsSubtitle: string;
  inactive: { title: string; detail: string } | null; // for the rates panel when it is not active
}

export function calibrationView(
  status: CalibrationStatus | string | null | undefined,
  modelLoaded: boolean,
  hasThresholds: boolean,
): CalibrationView {
  const d = calibrationDisplay(status);
  const fallback = ' The policy falls back to robust z-scores over the session’s own history.';
  if (status === 'installed') {
    if (modelLoaded && hasThresholds) {
      return {
        mode: 'active',
        drawThresholds: true,
        signalsSubtitle: 'Dashed lines are the calibrated thresholds installed on this session.',
        inactive: null,
      };
    }
    return {
      mode: 'installed_pending',
      drawThresholds: false,
      signalsSubtitle: 'A calibration is installed on this session; its threshold details are loading or unavailable.',
      inactive: {
        title: 'Calibration installed; threshold details are loading or unavailable.',
        detail: 'The installed thresholds have not loaded, so target and achievable rates are not shown yet. This is not the same as an uncalibrated session.',
      },
    };
  }
  if (status === 'absent') {
    return { mode: 'absent', drawThresholds: false, signalsSubtitle: d.detail + fallback,
      inactive: { title: 'This session is not calibrated.', detail: d.detail } };
  }
  if (status === 'rejected_signature_mismatch' || status === 'rejected_unsigned') {
    return { mode: 'rejected', drawThresholds: false, signalsSubtitle: d.detail + fallback,
      inactive: { title: `${d.label}; not active on this session.`, detail: d.detail } };
  }
  return { mode: 'unknown', drawThresholds: false, signalsSubtitle: d.detail + fallback,
    inactive: { title: 'Calibration status is unavailable for this session.', detail: d.detail } };
}

export function fmt(v: unknown, decimals = 4): string {
  if (!isNum(v)) return UNAVAILABLE;
  if (v !== 0 && Math.abs(v) < 10 ** -decimals) return v.toExponential(2);
  if (Math.abs(v) >= 1e6) return v.toExponential(2);
  return v.toFixed(decimals);
}

export function fmtSigned(v: unknown, decimals = 4): string {
  if (!isNum(v)) return UNAVAILABLE;
  return `${v >= 0 ? '+' : ''}${fmt(v, decimals)}`;
}

export function fmtInt(v: unknown): string {
  if (!isNum(v)) return UNAVAILABLE;
  return Math.round(v).toLocaleString();
}

export function fmtPercent(v: unknown, decimals = 1): string {
  if (!isNum(v)) return UNAVAILABLE;
  return `${(v * 100).toFixed(decimals)}%`;
}

export function fmtParams(v: unknown): string {
  if (!isNum(v)) return UNAVAILABLE;
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return String(Math.round(v));
}

export function fmtTimestamp(unixSeconds: unknown): string {
  if (!isNum(unixSeconds) || unixSeconds <= 0) return UNAVAILABLE;
  return new Date(unixSeconds * 1000).toLocaleString();
}

export function fmtRelative(unixSeconds: unknown): string {
  if (!isNum(unixSeconds) || unixSeconds <= 0) return UNAVAILABLE;
  const diff = Math.max(0, Math.floor(Date.now() / 1000) - unixSeconds);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export function fmtDuration(seconds: unknown): string {
  if (!isNum(seconds)) return UNAVAILABLE;
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

export function truncate(text: string, length = 18): string {
  return text.length <= length ? text : `${text.slice(0, length - 1)}…`;
}

// ------------------------------------------------------------------- decisions

export const DECISION_COLOR: Record<DecisionKind, string> = {
  commit: '#3fd17a',
  rollback: '#ff6b6b',
  scale: '#f0b429',
  project: '#58a6ff',
  readonly: '#94a3b4',
};

export const DECISION_FILL: Record<DecisionKind, string> = {
  commit: '#1f7a45',
  rollback: '#9c2b2b',
  scale: '#8a6410',
  project: '#1f4f8f',
  readonly: '#48545f',
};

export const DECISION_LABEL: Record<DecisionKind, string> = {
  commit: 'Commit',
  rollback: 'Rollback',
  scale: 'Scale',
  project: 'Project',
  readonly: 'Read-only',
};

export function decisionColor(kind: string): string {
  return DECISION_COLOR[kind as DecisionKind] ?? DECISION_COLOR.readonly;
}

export function decisionFill(kind: string): string {
  return DECISION_FILL[kind as DecisionKind] ?? DECISION_FILL.readonly;
}

export function decisionLabel(kind: string): string {
  return DECISION_LABEL[kind as DecisionKind] ?? kind;
}

/**
 * Diverging blue/red ramp for signed values (adapted from the old weight
 * heatmap). Both ends stay dark enough that white-ish ink on top keeps its
 * contrast, so the scale is legible at devicePixelRatio 1.
 */
export function divergingColor(value: number, maxAbs: number): string {
  if (!isNum(value) || !isNum(maxAbs) || maxAbs <= 0) return '#1c242e';
  const t = Math.max(-1, Math.min(1, value / maxAbs));
  const mix = (from: number, to: number, a: number) => Math.round(from + (to - from) * a);
  if (t >= 0) {
    // neutral -> blue
    return `rgb(${mix(28, 40, t)}, ${mix(36, 108, t)}, ${mix(46, 200, t)})`;
  }
  // neutral -> red
  const a = -t;
  return `rgb(${mix(28, 190, a)}, ${mix(36, 56, a)}, ${mix(46, 56, a)})`;
}

/**
 * A calibrated threshold. A finite number is the bound; null means the signal
 * is unbounded; an absent key means no calibration covers it at all.
 */
export function fmtThreshold(v: number | null | undefined, decimals = 4): string {
  if (v === undefined) return UNAVAILABLE;
  if (v === null) return UNBOUNDED;
  return fmt(v, decimals);
}

/** Collapse a server message to one readable line: no markup, no stack, bounded length. */
export function cleanErrorMessage(raw: string, max = 240): string {
  const text = raw
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]*>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (!text) return 'the API returned an error with no readable message';
  const firstLine = text.split(/(?:Traceback|  File ")/)[0].trim() || text;
  return firstLine.length <= max ? firstLine : `${firstLine.slice(0, max - 1)}…`;
}

/** A valid-only aggregate: null means no payload qualified, not zero damage. */
export function fmtValidOnly(v: number | null | undefined, decimals = 4): string {
  return isNum(v) ? fmt(v, decimals) : NO_VALID_PAYLOADS;
}

/** A valid-only fraction, with the same null semantics. */
export function fmtValidOnlyPercent(v: number | null | undefined, decimals = 0): string {
  return isNum(v) ? fmtPercent(v, decimals) : NO_VALID_PAYLOADS;
}
