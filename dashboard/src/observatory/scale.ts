// The observatory's one sequential scale: log magnitude mapped onto OBS.ramp (a single hue, dark to light).
import { OBS } from '../components/charts/theme';

function channels(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
}

const STOPS = OBS.ramp.map(channels);

/** t in [0, 1] to a color on the ramp. */
export function rampColor(t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x));
  const f = x - i;
  const c = STOPS[i].map((v, k) => Math.round(v + (STOPS[i + 1][k] - v) * f));
  return '#' + c.map((v) => v.toString(16).padStart(2, '0')).join('');
}

export const RAMP_CSS = `linear-gradient(to right, ${OBS.ramp.join(', ')})`;

/** A log scale over the positive finite values, clipped at the 2nd and 98th percentiles so one outlier cannot wash out the rest. */
export function logScale(values: (number | null | undefined)[]): { lo: number; hi: number; t: (v: number | null | undefined) => number | null } {
  const pos = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0).sort((a, b) => a - b);
  if (!pos.length) return { lo: 0, hi: 0, t: () => null };
  const q = (p: number) => pos[Math.min(pos.length - 1, Math.max(0, Math.round(p * (pos.length - 1))))];
  const lo = q(0.02);
  const hi = Math.max(q(0.98), lo * 1.0001);
  const a = Math.log10(lo);
  const b = Math.log10(hi);
  return {
    lo,
    hi,
    t: (v) => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? Math.min(1, Math.max(0, (Math.log10(v) - a) / (b - a))) : null),
  };
}
