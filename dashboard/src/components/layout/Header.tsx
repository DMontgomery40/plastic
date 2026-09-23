import { useMemo } from 'react';
import { useStore } from '../../store';
import type { Domain } from '../../api/types';
import { Select } from '../panels';
import { isPublicDemo } from '../../publicMode';

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
  const healthLoading = useStore((s) => s.loading.health);
  const error = useStore((s) => s.error);
  const models = useStore((s) => s.models);
  const sessions = useStore((s) => s.sessions);
  const currentSessionId = useStore((s) => s.currentSessionId);
  const activeTab = useStore((s) => s.activeTab);
  const setCurrentSession = useStore((s) => s.setCurrentSession);
  const publicDemo = isPublicDemo();

  const wanted = publicDemo ? 'text' : TAB_DOMAIN[activeTab];
  const relevant = useMemo(
    () => (wanted ? sessions.filter((s) => s.domain === wanted) : sessions),
    [sessions, wanted],
  );

  const options = relevant.map((s) => ({
    value: s.session_id,
    label: publicDemo ? s.session_id : `${s.session_id} · ${s.domain} · pos ${s.pos}`,
  }));
  const active = currentSessionId !== null && options.some((o) => o.value === currentSessionId) ? currentSessionId : null;

  // Three states, not two: healthy, unreachable, and not-yet-known. Green is
  // only ever the API's own ok:true, never the absence of a loading flag.
  const state: 'online' | 'offline' | 'connecting' = health?.ok === true ? 'online' : healthLoading && !health ? 'connecting' : 'offline';
  const statusColor = { online: '#3fd17a', offline: '#ff6b6b', connecting: '#94a3b4' }[state];
  const statusText = publicDemo
    ? { online: 'Connected', offline: 'Unavailable', connecting: 'Connecting' }[state]
    : { online: 'API online', offline: 'API unreachable', connecting: 'Connecting to the API' }[state];

  return (
    <header className="sticky top-0 z-10 border-b border-edge bg-surface-raised">
      <div className="mx-auto flex max-w-[1700px] flex-wrap items-center gap-x-6 gap-y-2 px-5 py-3">
        <div className="flex items-center gap-2.5">
          <Mark />
          <div>
            <h1 className="text-lg font-bold leading-none tracking-tight text-ink-primary">plastic</h1>
            {!publicDemo ? <p className="mt-1 text-micro text-ink-muted">Test-time training with a transactional safety harness</p> : null}
          </div>
        </div>

        <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 rounded border border-edge bg-surface-overlay px-2.5 py-1.5">
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: statusColor }} />
          <span className="text-xs font-semibold" style={{ color: statusColor }}>
            {statusText}
          </span>
          {health && !publicDemo ? (
            // Counts come from the live store, not the health snapshot: a fork or a
            // new session updates them at once, where health.n_* only refreshes on
            // the next bootstrap.
            <span className="min-w-0 break-words font-mono text-micro text-ink-secondary">
              {health.device} · {models.length} models · {sessions.length} sessions
            </span>
          ) : !health && !publicDemo ? (
            <span className="font-mono text-micro text-ink-secondary">{state === 'connecting' ? 'waiting for /api/health' : (error ?? 'no response from /api/health')}</span>
          ) : null}
        </div>

        <div className="flex w-full min-w-0 items-center gap-2 sm:ml-auto sm:w-auto sm:min-w-[280px]">
          <label htmlFor="session-picker" className="whitespace-nowrap text-label font-semibold uppercase tracking-wide text-ink-muted">
            Session
          </label>
          {options.length > 0 ? (
            // When the active session is not in the filtered list the picker
            // must not name a different one: it shows an explicit placeholder
            // instead, so the header never claims a session the tabs are not on.
            <Select
              id="session-picker"
              value={active ?? ''}
              onChange={setCurrentSession}
              options={active ? options : [{ value: '', label: `Select a ${wanted ?? ''} session`.replace('  ', ' ') }, ...options]}
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
