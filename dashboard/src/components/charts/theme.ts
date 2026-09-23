// Chart tokens. Recharts is styled per-chart from here, never from global CSS,
// so chart ink can never drift away from the Tailwind tokens.
//
// tickFontSize is 11 and is the smallest type anywhere in the app.

import type { CSSProperties } from 'react';

export const CHART = {
  grid: '#2c3642',
  axis: '#3d4a59',
  tick: '#94a3b4',
  tickFontSize: 11,
  axisLabel: '#b8c4d0',
  axisLabelFontSize: 11.5,
  cursor: '#3d4a59',
  reference: '#c792ea',
} as const;

/**
 * Sleep observatory: one sequential ramp (a single hue, dark to light) for every magnitude, so no chart needs a
 * categorical palette; status hues stay reserved for decisions and gate outcomes; illustrative content has its own hue.
 */
export const OBS = {
  ramp: ['#1c242e', '#1f4f8f', '#58a6ff', '#e9eff5'],
  line: '#58a6ff',
  lineMuted: '#94a3b4',
  illustrative: '#c792ea',
  illustrativeSoft: '#1b1626',
  pass: '#3fd17a',
  fail: '#ff6b6b',
  notInForce: '#94a3b4',
  flag: '#f0b429',
  surface: '#151b23',
  inset: '#090d12',
  edge: '#2c3642',
  edgeStrong: '#3d4a59',
  ink: '#e9eff5',
  inkSecondary: '#b8c4d0',
  inkMuted: '#94a3b4',
} as const;

export const SERIES_COLORS = ['#58a6ff', '#3fd17a', '#f0b429', '#c792ea', '#56d4dd'] as const;

export const tooltipContentStyle: CSSProperties = {
  backgroundColor: '#151b23',
  border: '1px solid #3d4a59',
  borderRadius: '6px',
  padding: '8px 10px',
  fontSize: '12px',
  lineHeight: '17px',
  color: '#e9eff5',
  boxShadow: '0 6px 18px rgba(0, 0, 0, 0.55)',
};

export const tooltipItemStyle: CSSProperties = {
  color: '#e9eff5',
  fontSize: '12px',
  padding: '1px 0',
};

export const tooltipLabelStyle: CSSProperties = {
  color: '#b8c4d0',
  fontSize: '11.5px',
  fontWeight: 600,
  marginBottom: '4px',
};

export const legendStyle: CSSProperties = {
  fontSize: '12px',
  color: '#b8c4d0',
  paddingBottom: '4px',
};

export const tickStyle = { fill: CHART.tick, fontSize: CHART.tickFontSize } as const;

export interface SeriesSpec {
  key: string;
  label: string;
  color: string;
  dashed?: boolean;
}
