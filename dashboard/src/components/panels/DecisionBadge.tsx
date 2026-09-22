import { decisionColor, decisionLabel } from '../../utils/formatting';

interface Props {
  kind: string;
  /** Shown after the label, e.g. the scale factor actually applied. */
  suffix?: string;
  size?: 'sm' | 'md';
}

/**
 * The decision of one chunk transaction. Colour is the status tier (bright
 * enough to read as text), on a solid overlay, never an alpha wash.
 */
export function DecisionBadge({ kind, suffix, size = 'md' }: Props) {
  const color = decisionColor(kind);
  const pad = size === 'sm' ? 'px-1.5 py-0.5 text-micro' : 'px-2 py-0.5 text-xs';
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border bg-surface-overlay font-semibold ${pad}`}
      style={{ color, borderColor: color }}
    >
      <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      {decisionLabel(kind)}
      {suffix ? <span className="font-mono font-normal text-ink-secondary">{suffix}</span> : null}
    </span>
  );
}

/** The decision colour legend, shown once next to the timeline. */
export function DecisionLegend() {
  const kinds = ['commit', 'rollback', 'scale', 'project', 'readonly'];
  return (
    <div className="flex flex-wrap items-center gap-3">
      {kinds.map((kind) => (
        <span key={kind} className="flex items-center gap-1.5 text-micro text-ink-secondary">
          <span
            aria-hidden
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ backgroundColor: decisionColor(kind) }}
          />
          {decisionLabel(kind)}
        </span>
      ))}
    </div>
  );
}
