import type { ChangeEvent, ReactNode } from 'react';

// Form controls. Resting opacity is full: nothing is de-emphasized with
// opacity, only with the muted ink tier and the overlay surface.

const controlBase =
  'w-full rounded border border-edge bg-surface-overlay px-2.5 py-1.5 text-sm text-ink-primary ' +
  'placeholder:text-ink-muted focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent ' +
  'disabled:border-edge disabled:bg-surface-inset disabled:text-ink-muted';

export function Label({ children, htmlFor }: { children: ReactNode; htmlFor?: string }) {
  return (
    <label htmlFor={htmlFor} className="mb-1 block text-label font-semibold uppercase tracking-wide text-ink-muted">
      {children}
    </label>
  );
}

export function Field({ label, htmlFor, hint, children }: { label: string; htmlFor?: string; hint?: string; children: ReactNode }) {
  return (
    <div>
      <Label htmlFor={htmlFor}>{label}</Label>
      {children}
      {hint ? <p className="mt-1 text-micro text-ink-muted">{hint}</p> : null}
    </div>
  );
}

export function TextInput({
  id,
  value,
  onChange,
  placeholder,
  disabled,
  mono = false,
}: {
  id?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  disabled?: boolean;
  mono?: boolean;
}) {
  return (
    <input
      id={id}
      type="text"
      className={`${controlBase} ${mono ? 'font-mono' : ''}`}
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(e.target.value)}
    />
  );
}

export function NumberInput({
  id,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  id?: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <input
      id={id}
      type="number"
      className={`${controlBase} font-mono`}
      value={Number.isFinite(value) ? value : ''}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(e: ChangeEvent<HTMLInputElement>) => {
        const next = Number(e.target.value);
        if (Number.isFinite(next)) onChange(next);
      }}
    />
  );
}

export function Select({
  id,
  value,
  onChange,
  options,
  disabled,
  ariaLabel,
}: {
  id?: string;
  value: string;
  onChange: (v: string) => void;
  options: Array<{ value: string; label: string }>;
  disabled?: boolean;
  /** Required when no visible <label> is associated with this control. */
  ariaLabel?: string;
}) {
  return (
    <select
      id={id}
      className={controlBase}
      aria-label={ariaLabel}
      value={value}
      disabled={disabled}
      onChange={(e: ChangeEvent<HTMLSelectElement>) => onChange(e.target.value)}
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

export function Slider({
  id,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  id?: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <input
        id={id}
        type="range"
        className="h-1.5 w-full cursor-pointer rounded bg-surface-overlay"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(Number(e.target.value))}
      />
      <span className="w-14 shrink-0 text-right font-mono text-sm text-ink-primary">{value}</span>
    </div>
  );
}

export function Checkbox({
  id,
  checked,
  onChange,
  label,
  disabled,
}: {
  id?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <label htmlFor={id} className="flex items-center gap-2 text-sm text-ink-secondary">
      <input
        id={id}
        type="checkbox"
        className="h-3.5 w-3.5 rounded border-edge-strong bg-surface-overlay accent-accent"
        checked={checked}
        disabled={disabled}
        onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}

export function Button({
  children,
  onClick,
  variant = 'default',
  disabled,
  type = 'button',
  size = 'md',
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'default' | 'primary' | 'danger';
  disabled?: boolean;
  type?: 'button' | 'submit';
  size?: 'sm' | 'md';
}) {
  const variants = {
    default: 'border-edge-strong bg-surface-overlay text-ink-primary hover:border-accent hover:text-accent',
    primary: 'border-accent bg-accent text-ink-inverse hover:bg-accent-hover hover:border-accent-hover',
    danger: 'border-status-rollback bg-surface-overlay text-status-rollback hover:bg-statusFill-rollback hover:text-ink-primary',
  }[variant];
  const pad = size === 'sm' ? 'px-2 py-1 text-xs' : 'px-3 py-1.5 text-sm';
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded border font-semibold transition-colors ${pad} ${variants} disabled:cursor-not-allowed disabled:border-edge disabled:bg-surface-inset disabled:text-ink-muted`}
    >
      {children}
    </button>
  );
}

export function ErrorBanner({ message, onDismiss }: { message: string; onDismiss?: () => void }) {
  return (
    <div className="flex items-start justify-between gap-4 rounded border border-status-rollback bg-surface-raised px-3 py-2">
      <p className="text-sm text-status-rollback">{message}</p>
      {onDismiss ? (
        <button type="button" onClick={onDismiss} className="text-xs font-semibold text-ink-secondary hover:text-ink-primary">
          Dismiss
        </button>
      ) : null}
    </div>
  );
}

/** A dense data table. Header cells are 11.5px; body cells are 13px. */
export function Table({ head, children }: { head: string[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-edge-strong text-left">
            {head.map((h) => (
              <th key={h} className="whitespace-nowrap px-2 py-1.5 text-label font-semibold uppercase tracking-wide text-ink-muted">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}
