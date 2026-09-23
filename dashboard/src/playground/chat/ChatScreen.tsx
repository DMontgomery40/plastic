import { useEffect, useRef, useState } from 'react';
import { Panel } from '../../components/panels/Panel';
import { KeyValue } from '../../components/panels/KeyValue';
import { Empty } from '../../components/panels/Empty';
import { backendLabel, fmt, fmtInt, modeLabel, thresholdSource, turnTotals, turnsFromTrace } from '../format';
import { useStore } from '../store';
import type { Turn } from '../types';
import { Button, Field, NumberInput, StateChip } from '../ui';
import { LearningStrip, StripLegend } from './LearningStrip';

function TurnCard({ turn }: { turn: Turn }) {
  const t = turnTotals(turn.chunks);
  return (
    <article className="rounded border border-edge bg-surface-raised">
      <div className="border-b border-edge px-4 py-3">
        <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">You</p>
        <p className="mt-1 whitespace-pre-wrap text-base text-ink-primary">{turn.prompt}</p>
      </div>
      <div className="px-4 py-3">
        <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">Model</p>
        <p className="mt-1 whitespace-pre-wrap font-mono text-sm text-ink-primary">{turn.completion || <span className="text-ink-muted">(empty completion)</span>}</p>
      </div>
      <div className="border-t border-edge px-4 py-3">
        <LearningStrip chunks={turn.chunks} />
        <dl className="mt-2 flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-ink-secondary">
          <div><dt className="inline text-ink-muted">chunks </dt><dd className="inline">{t.chunks}</dd></div>
          <div><dt className="inline text-ink-muted">committed </dt><dd className="inline">{t.committed}</dd></div>
          <div><dt className="inline text-ink-muted">intervened </dt><dd className="inline">{t.intervened}</dd></div>
          {t.wouldIntervene > 0 ? <div><dt className="inline text-ink-muted">would intervene </dt><dd className="inline">{t.wouldIntervene}</dd></div> : null}
          {t.readOnly > 0 ? <div><dt className="inline text-ink-muted">read-only </dt><dd className="inline">{t.readOnly}</dd></div> : null}
          <div><dt className="inline text-ink-muted">proposed Δ </dt><dd className="inline">{fmt(t.proposed)}</dd></div>
          <div><dt className="inline text-ink-muted">accepted Δ </dt><dd className="inline">{fmt(t.accepted)}</dd></div>
          <div><dt className="inline text-ink-muted">tokens </dt><dd className="inline">{t.promptTokens} prompt · {t.modelTokens} generated</dd></div>
        </dl>
      </div>
    </article>
  );
}

function Composer() {
  const [text, setText] = useState('');
  const form = useRef<HTMLFormElement>(null);
  const isPublic = useStore((s) => s.health?.public ?? false);
  const busy = useStore((s) => s.busy.chat);
  const detail = useStore((s) => s.detail);
  const sendChat = useStore((s) => s.sendChat);
  const sampling = useStore((s) => s.sampling);
  const setSampling = useStore((s) => s.setSampling);
  const resume = useStore((s) => s.resumeSession);
  const caps = useStore((s) => s.capabilities)();
  const readOnly = detail?.summary.read_only ?? false;

  const submit = async () => {
    const prompt = text.trim();
    if (!prompt || busy || !detail || !form.current?.reportValidity()) return;
    if (await sendChat(prompt)) setText('');
  };

  return (
    <form ref={form} onSubmit={(e) => { e.preventDefault(); void submit(); }} className="rounded border border-edge bg-surface-raised px-4 py-3">
      <label htmlFor="prompt" className="text-label font-semibold uppercase tracking-wide text-ink-muted">
        Prompt
      </label>
      <textarea
        id="prompt"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            void submit();
          }
        }}
        rows={3}
        maxLength={isPublic ? 1024 : undefined}
        disabled={busy || !detail}
        placeholder={detail ? 'Type a message. Enter sends, Shift+Enter for a new line.' : 'Select a session first.'}
        className="mt-1 w-full resize-y rounded border border-edge-strong bg-surface-overlay px-3 py-2 text-base text-ink-primary focus:border-accent focus:outline-none disabled:text-ink-muted"
      />
      <div className="mt-2 flex flex-wrap items-end gap-3">
        <Button tone="primary" type="submit" disabled={busy || !detail || !text.trim()}>
          {busy ? 'Generating…' : 'Send'}
        </Button>
        {readOnly ? (
          <span className="flex items-center gap-2">
            <StateChip tone="warn">read-only: {detail?.summary.read_only_reason ?? 'latched'}</StateChip>
            {caps.resume && detail ? <Button onClick={() => void resume(detail.meta.session_id)}>Resume</Button> : null}
          </span>
        ) : null}
        <div className="ml-auto grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Field label="Max tokens" htmlFor="max-tokens">
            <NumberInput id="max-tokens" required min={0} max={isPublic ? 128 : 512} step={1} value={sampling.max_new_tokens} onChange={(e) => setSampling({ max_new_tokens: Number(e.target.value) })} />
          </Field>
          <Field label="Temperature" htmlFor="temperature">
            <NumberInput id="temperature" required min={0.01} max={2} step="any" value={sampling.temperature} onChange={(e) => setSampling({ temperature: Number(e.target.value) })} />
          </Field>
          <Field label="Top-k" htmlFor="top-k" hint="0 disables">
            <NumberInput id="top-k" required min={0} max={500} step={1} value={sampling.top_k} onChange={(e) => setSampling({ top_k: Number(e.target.value) })} />
          </Field>
          <Field label="Seed" htmlFor="seed" hint="blank = random">
            <NumberInput id="seed" min={0} max={4294967295} step={1} value={sampling.seed ?? ''} onChange={(e) => setSampling({ seed: e.target.value === '' ? null : Number(e.target.value) })} />
          </Field>
        </div>
      </div>
    </form>
  );
}

export function ChatScreen() {
  const detail = useStore((s) => s.detail);
  const transactions = useStore((s) => s.transactions);
  const models = useStore((s) => s.models);
  const busy = useStore((s) => s.busy);
  const bottom = useRef<HTMLDivElement>(null);
  const turns = detail ? turnsFromTrace(detail.trace, transactions) : [];

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' });
  }, [turns.length, busy.chat]);

  if (!detail) {
    return <Empty title="No text session selected." detail={busy.session ? 'Loading…' : 'Choose a session above, or create one under Sessions.'} />;
  }
  const model = models.find((m) => m.model_id === detail.meta.model_id);
  const summary = detail.summary;
  const present = summary.signals_available ?? [];

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
      <section className="min-w-0 space-y-3">
        {turns.length === 0 ? <Empty title="No turns yet." detail="Send a message to start." /> : turns.map((t) => <TurnCard key={t.index} turn={t} />)}
        {busy.chat ? <p role="status" className="text-sm text-ink-secondary">Generating…</p> : null}
        <div ref={bottom} />
        <Composer />
        <StripLegend />
      </section>
      <aside className="space-y-4">
        <Panel title="Session">
          <KeyValue
            rows={[
              { label: 'Model', value: detail.meta.model_id },
              { label: 'Learner', value: backendLabel(summary.backend ?? model?.backend) },
              { label: 'Mode', value: modeLabel(detail.meta.harness) },
              { label: 'Position', value: fmtInt(summary.pos) },
              { label: 'Chunks', value: fmtInt(summary.n_transactions) },
              { label: 'State', value: summary.read_only ? `read-only (${summary.read_only_reason ?? 'latched'})` : 'writable' },
            ]}
          />
        </Panel>
        <Panel title="Harness">
          <KeyValue
            rows={[
              { label: 'Thresholds', value: thresholdSource(summary) },
              { label: 'Generated tokens', value: summary.writes_generation ? 'write to memory' : 'read-only' },
              { label: 'CUSUM', value: `${fmt(summary.cusum?.s_hi, 2)} / ${fmt(summary.cusum?.s_lo, 2)}`, note: `h ${fmt(summary.cusum?.h, 2)} · ${summary.cusum?.alarms ?? 0} alarms` },
              { label: 'Budget used', value: fmt(summary.budget_used), note: summary.budget_session === null ? 'no session cap' : `cap ${fmt(summary.budget_session)}` },
              { label: 'Drift from anchor', value: fmt(summary.drift_from_anchor) },
              { label: 'Decision signals', value: present.length ? present.join(', ') : 'none' },
            ]}
            mono={false}
          />
        </Panel>
      </aside>
    </div>
  );
}
