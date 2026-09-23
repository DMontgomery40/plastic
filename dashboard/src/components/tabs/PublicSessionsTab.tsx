import { useStore } from '../../store';
import { fmtInt, fmtRelative } from '../../utils/formatting';
import { Button, Empty, Panel } from '../panels';

export function PublicSessionsTab() {
  const sessions = useStore((s) => s.sessions).filter((s) => s.domain === 'text');
  const currentSessionId = useStore((s) => s.currentSessionId);
  const setCurrentSession = useStore((s) => s.setCurrentSession);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const resetSession = useStore((s) => s.resetSession);
  const resumeSession = useStore((s) => s.resumeSession);
  const busy = useStore((s) => s.loading.mutation);

  const confirmReset = (sessionId: string) => {
    if (window.confirm(`Reset shared session ${sessionId}? Its conversation and measurements will be cleared for everyone.`)) {
      void resetSession(sessionId);
    }
  };

  if (sessions.length === 0) return <Empty title="No text session is available." detail="Try reconnecting in a moment." />;

  return (
    <Panel title="Text sessions">
      <ul className="space-y-2">
        {sessions.map((session) => (
          <li key={session.session_id} className="flex flex-wrap items-center justify-between gap-3 rounded border border-edge bg-surface-overlay px-3 py-2.5">
            <div className="min-w-0">
              <p className="break-all font-mono text-sm font-semibold text-ink-primary">
                {session.session_id}{session.session_id === currentSessionId ? <span className="ml-2 text-xs font-normal text-accent">Selected</span> : null}
              </p>
              <p className="mt-1 text-xs text-ink-secondary">
                {fmtInt(session.n_transactions)} chunks · position {fmtInt(session.pos)} · updated {fmtRelative(session.updated_at_unix)}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button size="sm" onClick={() => { setCurrentSession(session.session_id); setActiveTab('chat'); }}>Chat</Button>
              <Button size="sm" onClick={() => { setCurrentSession(session.session_id); setActiveTab('session'); }}>Measurements</Button>
              {session.read_only ? <Button size="sm" disabled={busy} onClick={() => void resumeSession(session.session_id)}>Resume</Button> : null}
              <Button size="sm" variant="danger" disabled={busy} onClick={() => confirmReset(session.session_id)}>Reset</Button>
            </div>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
