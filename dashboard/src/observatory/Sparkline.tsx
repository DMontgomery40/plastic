import { OBS } from '../components/charts/theme';
import { num } from './format';

/** A single-series line over steps. Axis values are HTML text (crisp at any size); the SVG draws only the line. */
export function Sparkline({ values, label, unit, color = OBS.line, height = 72 }: { values: (number | null)[]; label: string; unit?: string; color?: string; height?: number }) {
  const finite = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  if (finite.length < 2) return <p className="text-sm text-ink-muted">{label}: not recorded</p>;
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const span = hi - lo || 1;
  const pts = values
    .map((v, i) => (typeof v === 'number' && Number.isFinite(v) ? `${(i / (values.length - 1)) * 100},${100 - ((v - lo) / span) * 92 - 4}` : null))
    .filter(Boolean)
    .join(' ');
  return (
    <figure className="min-w-0">
      <figcaption className="flex items-baseline justify-between gap-2 text-xs text-ink-secondary">
        <span>{label}</span>
        <span className="font-mono text-ink-muted">
          {num(finite[0])} → <span className="font-semibold text-ink-primary">{num(finite[finite.length - 1])}</span>
          {unit ? ` ${unit}` : ''}
        </span>
      </figcaption>
      <div className="mt-1 grid grid-cols-[auto_1fr] gap-2">
        <div className="flex flex-col justify-between py-0.5 text-right font-mono text-micro text-ink-muted" style={{ height }}>
          <span>{num(hi)}</span>
          <span>{num(lo)}</span>
        </div>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ height, width: '100%' }} role="img" aria-label={`${label}: ${values.length} steps, from ${num(finite[0])} to ${num(finite[finite.length - 1])}`}>
          <line x1="0" x2="100" y1="96" y2="96" stroke={OBS.edge} strokeWidth="1" vectorEffect="non-scaling-stroke" />
          <polyline points={pts} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
        </svg>
      </div>
      <div className="mt-0.5 flex justify-between pl-8 font-mono text-micro text-ink-muted">
        <span>step 1</span>
        <span>step {values.length}</span>
      </div>
    </figure>
  );
}
