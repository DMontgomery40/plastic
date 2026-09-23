// Small controls with the legibility floor baked in: nothing under 11px, solid backgrounds, visible focus.

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react';

type Tone = 'default' | 'primary' | 'danger';

export function Button({ tone = 'default', className = '', ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { tone?: Tone }) {
  const tones: Record<Tone, string> = {
    default: 'border-edge-strong bg-surface-overlay text-ink-primary hover:border-accent',
    primary: 'border-accent bg-accent text-ink-inverse hover:bg-accent-hover',
    danger: 'border-status-rollback bg-surface-overlay text-status-rollback hover:bg-statusFill-rollback hover:text-ink-primary',
  };
  return (
    <button
      type="button"
      className={`rounded border px-3 py-1.5 text-sm font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:border-edge disabled:text-ink-muted ${tones[tone]} ${className}`}
      {...props}
    />
  );
}

export function Field({ label, htmlFor, children, hint }: { label: string; htmlFor: string; children: ReactNode; hint?: string }) {
  return (
    <div className="min-w-0">
      <label htmlFor={htmlFor} className="block text-label font-semibold uppercase tracking-wide text-ink-muted">
        {label}
      </label>
      <div className="mt-1">{children}</div>
      {hint ? <p className="mt-1 text-micro text-ink-muted">{hint}</p> : null}
    </div>
  );
}

const inputClass =
  'w-full rounded border border-edge-strong bg-surface-overlay px-2.5 py-1.5 font-mono text-sm text-ink-primary focus:border-accent focus:outline-none disabled:text-ink-muted';

export function NumberInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input type="number" className={inputClass} {...props} />;
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={inputClass} {...props} />;
}

export function ErrorBanner({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div role="alert" className="flex items-start justify-between gap-4 rounded border border-status-rollback bg-surface-raised px-4 py-3">
      <p className="text-sm text-ink-primary">{message}</p>
      <Button onClick={onDismiss} className="shrink-0">Dismiss</Button>
    </div>
  );
}

/** A one-line state label: what the session IS right now. Never an explanation. */
export function StateChip({ children, tone = 'default' }: { children: ReactNode; tone?: 'default' | 'accent' | 'warn' | 'bad' }) {
  const tones = {
    default: 'border-edge-strong text-ink-secondary',
    accent: 'border-accent text-accent',
    warn: 'border-status-scale text-status-scale',
    bad: 'border-status-rollback text-status-rollback',
  } as const;
  return <span className={`inline-flex items-center rounded border bg-surface-overlay px-2 py-0.5 text-xs font-semibold ${tones[tone]}`}>{children}</span>;
}
