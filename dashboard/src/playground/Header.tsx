import { backendLabel, modeLabel } from './format';
import { TABS, TAB_LABELS, useStore } from './store';
import { Select, StateChip } from './ui';

export function Header() {
  const health = useStore((s) => s.health);
  const sessions = useStore((s) => s.sessions);
  const current = useStore((s) => s.currentSessionId);
  const detail = useStore((s) => s.detail);
  const models = useStore((s) => s.models);
  const selectSession = useStore((s) => s.selectSession);
  const stale = useStore((s) => s.stale);
  const tab = useStore((s) => s.tab);
  const setTab = useStore((s) => s.setTab);

  const text = sessions.filter((s) => s.domain === 'text');
  const model = models.find((m) => m.model_id === detail?.meta.model_id);
  const connected = health !== null && !stale;

  return (
    <header className="border-b border-edge bg-surface-raised">
      <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-5 gap-y-3 px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold tracking-tight text-ink-primary">plastic</span>
          <StateChip tone={connected ? 'accent' : 'bad'}>{connected ? 'connected' : health ? 'stale' : 'offline'}</StateChip>
          {health?.public ? <StateChip>shared public session</StateChip> : null}
        </div>
        <nav aria-label="Screens" className="flex items-center gap-1">
          {TABS.map((t, i) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              aria-current={tab === t ? 'page' : undefined}
              className={`rounded px-3 py-1.5 text-sm font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                tab === t ? 'bg-accent-soft text-accent' : 'text-ink-secondary hover:text-ink-primary'
              }`}
            >
              <span className="mr-1.5 font-mono text-micro text-ink-muted">{i + 1}</span>
              {TAB_LABELS[t]}
            </button>
          ))}
        </nav>
        <div className="ml-auto flex flex-wrap items-center gap-3">
          {detail ? (
            <>
              <StateChip>{backendLabel(detail.summary.backend ?? model?.backend)}</StateChip>
              <StateChip tone={modeLabel(detail.meta.harness) === 'guarded' ? 'warn' : 'default'}>{modeLabel(detail.meta.harness)}</StateChip>
              {model && model.chat_tuned === false ? <StateChip tone="warn">base model, not chat-tuned</StateChip> : null}
            </>
          ) : null}
          <label className="flex items-center gap-2 text-label font-semibold uppercase tracking-wide text-ink-muted">
            Session
            <span className="w-52">
              <Select aria-label="Session" value={current ?? ''} onChange={(e) => void selectSession(e.target.value || null)} disabled={text.length === 0}>
                {text.length === 0 ? <option value="">no text session</option> : null}
                {text.map((s) => (
                  <option key={s.session_id} value={s.session_id}>
                    {s.session_id}
                  </option>
                ))}
              </Select>
            </span>
          </label>
        </div>
      </div>
    </header>
  );
}
