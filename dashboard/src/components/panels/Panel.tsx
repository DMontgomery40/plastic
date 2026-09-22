import type { ReactNode } from 'react';

interface Props {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** A titled card. The single container used for every block on every tab. */
export function Panel({ title, subtitle, actions, children, className = '' }: Props) {
  return (
    <section className={`rounded-lg border border-edge bg-surface-raised ${className}`}>
      <header className="flex flex-wrap items-baseline justify-between gap-3 border-b border-edge px-4 py-3">
        <div>
          <h2 className="text-md font-semibold text-ink-primary">{title}</h2>
          {subtitle ? <p className="mt-0.5 text-xs text-ink-muted">{subtitle}</p> : null}
        </div>
        {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
      </header>
      <div className="px-4 py-4">{children}</div>
    </section>
  );
}

/** A large single number with its label; the loudest element on a tab. */
export function StatTile({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: 'default' | 'commit' | 'rollback' | 'scale' | 'project' | 'accent';
}) {
  const toneClass = {
    default: 'text-ink-primary',
    commit: 'text-status-commit',
    rollback: 'text-status-rollback',
    scale: 'text-status-scale',
    project: 'text-status-project',
    accent: 'text-accent',
  }[tone];
  return (
    <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
      <div className="text-label font-semibold uppercase tracking-wide text-ink-muted">{label}</div>
      <div className={`mt-1 font-mono text-xl font-semibold ${toneClass}`}>{value}</div>
      {hint ? <div className="mt-0.5 text-micro text-ink-muted">{hint}</div> : null}
    </div>
  );
}
