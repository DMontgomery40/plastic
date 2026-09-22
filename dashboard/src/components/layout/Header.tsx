import { useMemo } from 'react';
import { useStore } from '../../store';
import type { Domain } from '../../api/types';
import { Select } from '../panels';

/** Tabs that only make sense for one domain; the session picker follows them. */
const TAB_DOMAIN: Partial<Record<string, Domain>> = {
  chat: 'text',
  physics: 'physics',
};

/** A closed loop with a live write point: the fast weights and the harness. */
function Mark() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden focusable="false">
      <rect x="1.5" y="1.5" width="19" height="19" rx="5" fill="none" stroke="#58a6ff" strokeWidth="1.6" />
      <path d="M5.5 14.5 C 8 5.5, 14 16.5, 16.5 7.5" fill="none" stroke="#3fd17a" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="16.5" cy="7.5" r="2.2" fill="#f0b429" />
    </svg>
  );
}

export function Header() {
  const health = useStore((s) => s.health);
  const error = useStore((s) => s.error);
  const sessions = useStore((s) => s.sessions);
  const currentSessionId = useStore((s) => s.currentSessionId);
  const activeTab = useStore((s) => s.activeTab);
  const setCurrentSession = useStore((s) => s.setCurrentSession);

  const wanted = TAB_DOMAIN[activeTab];
  const relevant = useMemo(
    () => (wanted ? sessions.filter((s) => s.domain === wanted) : sessions),
    [sessions, wanted],
  );

  const options = relevant.map((s) => ({
    value: s.session_id,
    label: `${s.session_id} · ${s.domain} · pos ${s.pos}`,
  }));

  const online = health?.ok === true;
  const statusColor = online ? '#3fd17a' : '#ff6b6b';

  return (
    <header className="sticky top-0 z-10 border-b border-edge bg-surface-raised">
      <div className="mx-auto flex max-w-[1700px] flex-wrap items-center gap-x-6 gap-y-2 px-5 py-3">
        <div className="flex items-center gap-2.5">
          <Mark />
          <div>
            <h1 className="text-lg font-bold leading-none tracking-tight text-ink-primary">plastic</h1>
            <p className="mt-1 text-micro text-ink-muted">Test-time training with a transactional safety harness</p>
          </div>
        </div>

        <div className="flex items-center gap-2 rounded border border-edge bg-surface-overlay px-2.5 py-1.5">
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: statusColor }} />
          <span className="text-xs font-semibold" style={{ color: statusColor }}>
            {online ? 'API online' : 'API unreachable'}
          </span>
          {health ? (
            <span className="font-mono text-micro text-ink-secondary">
              {health.device} · {health.n_models} models · {health.n_sessions} sessions
            </span>
          ) : (
            <span className="font-mono text-micro text-ink-secondary">{error ?? 'waiting for /api/health'}</span>
          )}
        </div>

        <div className="ml-auto flex min-w-[280px] items-center gap-2">
          <label htmlFor="session-picker" className="whitespace-nowrap text-label font-semibold uppercase tracking-wide text-ink-muted">
            Session
          </label>
          {options.length > 0 ? (
            <Select
              id="session-picker"
              value={currentSessionId && options.some((o) => o.value === currentSessionId) ? currentSessionId : options[0].value}
              onChange={setCurrentSession}
              options={options}
            />
          ) : (
            <span className="text-xs text-ink-secondary">
              {wanted ? `No ${wanted} session yet` : 'No sessions yet'}
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
