import { backendLabel, modeLabel } from './format';
import { TABS, TAB_LABELS, useStore } from './store';
import type { ModelSummary, SessionSummary } from './types';
import { Select, StateChip } from './ui';

function shortParams(n: number | undefined): string {
  if (typeof n !== 'number' || !Number.isFinite(n) || n <= 0) return '';
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M` : `${Math.round(n / 1e3)}K`;
}

/** A model as a visitor should read it: its id, the learner, its size, and whether sleep produced it. */
export function modelLabel(m: ModelSummary): string {
  const size = shortParams(m.params);
  return [m.model_id, backendLabel(m.backend), size, m.parent_model_id ? `sleep child of ${m.parent_model_id}` : null].filter(Boolean).join(' · ');
}

/** Models a visitor can open: the server's catalog, text only, with at least one session to open it in. */
export function openableModels(models: ModelSummary[], sessions: SessionSummary[]): ModelSummary[] {
  return models.filter((m) => m.domain === 'text' && sessions.some((s) => s.domain === 'text' && s.model_id === m.model_id));
}

function Picker({ label, value, options, onChange }: { label: string; value: string; options: { value: string; text: string; title?: string }[]; onChange: (v: string) => void }) {
  const current = options.find((o) => o.value === value);
  return (
    <div className="flex items-center gap-2">
      <span className="text-label font-semibold uppercase tracking-wide text-ink-muted">{label}</span>
      {options.length > 1 ? (
        <span className="inline-block min-w-[15rem] max-w-[34rem]">
          <Select aria-label={label} value={value} onChange={(e) => onChange(e.target.value)}>
            {options.map((o) => (
              <option key={o.value} value={o.value} title={o.title}>
                {o.text}
              </option>
            ))}
          </Select>
        </span>
      ) : (
        // one choice is a fact, not a menu
        <span className="font-mono text-sm text-ink-primary" title={current?.title}>
          {current?.text ?? 'none'}
        </span>
      )}
    </div>
  );
}

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
  const openable = openableModels(models, sessions);
  const currentSession = text.find((s) => s.session_id === current);
  const modelId = currentSession?.model_id ?? detail?.meta.model_id ?? '';
  const model = models.find((m) => m.model_id === modelId);
  const modelSessions = text.filter((s) => s.model_id === modelId);
  const connected = health !== null && !stale;

  const chooseModel = (id: string) => {
    const first = text.find((s) => s.model_id === id);
    if (first) void selectSession(first.session_id);
  };

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
        <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-2">
          {text.length === 0 ? (
            <span className="text-sm text-ink-secondary">no text session</span>
          ) : (
            <>
              <Picker
                label="Model"
                value={modelId}
                options={openable.map((m) => ({ value: m.model_id, text: modelLabel(m), title: m.model_id }))}
                onChange={chooseModel}
              />
              {modelSessions.length > 1 ? (
                <Picker
                  label="Session"
                  value={current ?? ''}
                  options={modelSessions.map((s) => ({ value: s.session_id, text: s.session_id }))}
                  onChange={(v) => void selectSession(v || null)}
                />
              ) : null}
            </>
          )}
          {detail ? (
            <>
              <StateChip tone={modeLabel(detail.meta.harness) === 'guarded' ? 'warn' : 'default'}>{modeLabel(detail.meta.harness)}</StateChip>
              {model && model.chat_tuned === false ? <StateChip tone="warn">base model, not chat-tuned</StateChip> : null}
            </>
          ) : null}
        </div>
      </div>
    </header>
  );
}
