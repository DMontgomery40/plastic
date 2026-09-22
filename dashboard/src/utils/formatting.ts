// Formatting helpers. Every number that reaches the screen goes through one of
// these, so a null or a NaN from the API renders as a dash instead of "NaN".

import type { DecisionKind } from '../api/types';

/**
 * The single token for "there is no value here". Never render a missing metric
 * as 0, as an empty cell, or as anything a reader could mistake for a measured
 * result: a null threshold, an absent canary score and a rate nobody computed
 * all say the same honest thing.
 */
export const UNAVAILABLE = 'unavailable';

/** A null threshold means the bound is unlimited, which is not the same as unknown. */
export const UNBOUNDED = 'unbounded';

/** True for a real, finite number. The API can send null or NaN for several signals. */
export function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
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
