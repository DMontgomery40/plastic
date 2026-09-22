import type { ReactNode } from 'react';

interface Props {
  title: string;
  /** What is missing and why, in one sentence. */
  detail?: string;
  /** The exact command that produces the missing data. Copied verbatim by the reader. */
  command?: string;
  children?: ReactNode;
}

/**
 * The only thing shown when data is absent. There is no mock data anywhere in
 * this dashboard: an empty state names the command that fills it.
 */
export function Empty({ title, detail, command, children }: Props) {
  return (
    <div className="rounded border border-dashed border-edge-strong bg-surface-overlay px-4 py-5">
      <p className="text-base font-semibold text-ink-primary">{title}</p>
      {detail ? <p className="mt-1 text-sm text-ink-secondary">{detail}</p> : null}
      {command ? (
        <pre className="mt-3 overflow-x-auto rounded border border-edge bg-surface-inset px-3 py-2 font-mono text-xs text-ink-secondary">
          {command}
        </pre>
      ) : null}
      {children ? <div className="mt-3">{children}</div> : null}
    </div>
  );
}
