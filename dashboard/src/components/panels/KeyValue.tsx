import type { ReactNode } from 'react';

export interface Row {
  label: string;
  value: ReactNode;
  /** A short note under the value: a threshold, a unit, a source. */
  note?: string;
}

interface Props {
  rows: Row[];
  columns?: 1 | 2;
  mono?: boolean;
}

/** Label / value table. Values are monospace by default so digits line up. */
export function KeyValue({ rows, columns = 1, mono = true }: Props) {
  return (
    <dl className={`grid grid-cols-1 gap-x-6 gap-y-2 ${columns === 2 ? 'sm:grid-cols-2' : ''}`}>
      {rows.map((row) => (
        <div key={row.label} className="flex items-baseline justify-between gap-4 border-b border-edge pb-1.5 last:border-b-0">
          <dt className="min-w-0 break-words text-xs text-ink-muted">{row.label}</dt>
          <dd className="min-w-0 break-words text-right">
            <span className={`text-sm text-ink-primary ${mono ? 'font-mono' : ''}`}>{row.value}</span>
            {row.note ? <span className="ml-2 text-micro text-ink-muted">{row.note}</span> : null}
          </dd>
        </div>
      ))}
    </dl>
  );
}
