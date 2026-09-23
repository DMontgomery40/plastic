import { useEffect, useState } from 'react';
import { Panel } from '../../components/panels/Panel';
import { ago, fmt, fmtInt } from '../format';
import { useStore } from '../store';
import type { ModelSummary, SleepMethod, SleepRun, SleepTarget } from '../types';
import { Button, Field, NumberInput, Select, StateChip } from '../ui';

const METHOD_LABEL: Record<SleepMethod, string> = {
  replay: 'Replay: fine-tune on accepted turns + SFT replay',
  distill: 'Distill: fast weights teach the reset model',
  anchor: 'Anchor: move W0 toward session fast weights',
};
const TARGET_LABEL: Record<SleepTarget, string> = { w0: 'initial fast weights (W0)', all: 'all parameters' };

function statusTone(status: SleepRun['status']): 'default' | 'accent' | 'warn' | 'bad' {
  switch (status) {
    case 'accepted':
      return 'accent';
    case 'running':
      return 'default';
    case 'accepted_unmeasured':
    case 'rejected':
      return 'warn';
    default:
      return 'bad';
  }
}

function RunCard({ run }: { run: SleepRun }) {
  const r = run.report;
  const before = r?.before ?? null;
  const after = r?.after ?? null;
  const gate = r?.gate ?? null;
  return (
    <li className="rounded border border-edge p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-semibold text-ink-primary">{run.run_id}</span>
        <StateChip tone={statusTone(run.status)}>{run.status === 'accepted_unmeasured' ? 'accepted, locality unmeasured' : run.status}</StateChip>
        {run.options ? <StateChip>{run.options.method} · {run.options.target}</StateChip> : null}
        {r?.model_id ? <StateChip tone="accent">child {r.model_id}</StateChip> : null}
        <span className="text-xs text-ink-secondary">{run.model_id} · started {ago(run.started_at_unix)}</span>
      </div>
      {r?.harvest ? (
        <p className="mt-1 text-xs text-ink-secondary">
          {fmtInt(r.harvest.accepted_tokens)} accepted tokens from {r.harvest.sessions.length} session{r.harvest.sessions.length === 1 ? '' : 's'}
          {r.harvest.excluded_tokens > 0 ? ` · ${fmtInt(r.harvest.excluded_tokens)} excluded (${Object.entries(r.harvest.turns_by_reason).filter(([k]) => k !== 'accepted').map(([k, v]) => `${v} ${k.replace('_', ' ')}`).join(', ')})` : ''}
        </p>
      ) : null}
      {before || after || gate ? (
        <dl className="mt-2 grid gap-x-6 gap-y-1 text-xs sm:grid-cols-3">
          <div>
            <dt className="text-ink-secondary">held-out NLL (mean / median)</dt>
            <dd className="font-mono text-ink-primary">
              {before?.heldout_nll ? `${fmt(before.heldout_nll.mean)} / ${fmt(before.heldout_nll.median)}` : 'n/a'}
              {' → '}
              {after?.heldout_nll ? `${fmt(after.heldout_nll.mean)} / ${fmt(after.heldout_nll.median)}` : 'n/a'}
            </dd>
          </div>
          <div>
            <dt className="text-ink-secondary">recall in a fresh session</dt>
            <dd className="font-mono text-ink-primary">
              {before?.recall ? `${before.recall.recalled}/${before.recall.n_probes}` : 'n/a'}
              {' → '}
              {after?.recall ? `${after.recall.recalled}/${after.recall.n_probes}` : 'n/a'}
              {after?.recall && after.recall.n_paraphrase > 0 ? ` (paraphrase ${before?.recall?.recalled_paraphrase ?? 'n/a'} → ${after.recall.recalled_paraphrase}/${after.recall.n_paraphrase})` : ''}
            </dd>
          </div>
          <div>
            <dt className="text-ink-secondary">locality gate</dt>
            <dd className="font-mono text-ink-primary">
              {gate ? (gate.checks.length ? gate.checks.map((c) => `${c.name} ${fmt(c.value)}${c.passed ? ' ok' : ' FAIL'}`).join(' · ') : 'not measured') : 'pending'}
            </dd>
          </div>
        </dl>
      ) : null}
      {r?.reason ? <p className="mt-1 text-xs text-ink-secondary">{r.reason}</p> : null}
      {run.status === 'running' && run.log_tail?.length ? (
        <pre className="mt-2 max-h-28 overflow-auto rounded bg-surface-sunken p-2 font-mono text-micro text-ink-secondary">{run.log_tail.slice(-6).join('\n')}</pre>
      ) : null}
      {run.status === 'failed' && run.stderr_tail?.length ? (
        <pre className="mt-2 max-h-28 overflow-auto rounded bg-surface-sunken p-2 font-mono text-micro text-ink-secondary">{run.stderr_tail.join('\n')}</pre>
      ) : null}
    </li>
  );
}

/**
 * Sleep: consolidate what the harness ACCEPTED in this model's sessions into a child model. The run is a
 * separate process; this panel shows its options, its state while it runs, and the before/after numbers
 * that say whether anything was retained (recall in a fresh session) and whether anything was harmed
 * (held-out NLL, canaries).
 */
export function SleepPanel({ models }: { models: ModelSummary[] }) {
  const caps = useStore((s) => s.capabilities)();
  const isPublic = useStore((s) => s.health?.public ?? false);
  const sessions = useStore((s) => s.sessions);
  const runs = useStore((s) => s.sleepRuns);
  const refreshSleep = useStore((s) => s.refreshSleep);
  const startSleep = useStore((s) => s.startSleep);
  const busy = useStore((s) => s.busy.mutation);
  // the shared demo consolidates into the root model only; locally any TTT model (including a child) can sleep
  const eligible = models.filter((m) => m.backend === 'ttt' && m.status === 'completed' && (!isPublic || !m.parent_model_id));
  const [modelId, setModelId] = useState(eligible[0]?.model_id ?? '');
  // the shared demo accepts anchor or replay on W0 with at most 10 steps (deploy/huggingface/app.py SLEEP_LIMITS)
  const maxSteps = isPublic ? 10 : 2000;
  const [method, setMethod] = useState<SleepMethod>(isPublic ? 'anchor' : 'replay');
  const [target, setTarget] = useState<SleepTarget>('w0');
  const [steps, setSteps] = useState(isPublic ? 5 : 40);
  const [selected, setSelected] = useState<string[]>([]);
  const [probesText, setProbesText] = useState('');
  const [probesError, setProbesError] = useState<string | null>(null);
  const model = eligible.find((m) => m.model_id === (modelId || eligible[0]?.model_id));
  const modelSessions = sessions.filter((s) => s.model_id === model?.model_id);
  const running = runs.some((r) => r.status === 'running');

  useEffect(() => {
    void refreshSleep();
  }, [refreshSleep]);
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => void refreshSleep(), 4000);
    return () => window.clearInterval(id);
  }, [running, refreshSleep]);

  if (!caps.sleep || eligible.length === 0) return null;

  const parseProbes = (): { question: string; answer: string; paraphrase?: string }[] | null => {
    const lines = probesText.split('\n').map((l) => l.trim()).filter(Boolean);
    const out: { question: string; answer: string; paraphrase?: string }[] = [];
    for (const line of lines) {
      const parts = line.split('|').map((p) => p.trim());
      if (parts.length < 2 || !parts[0] || !parts[1]) {
        setProbesError(`each probe line is "question | answer" or "question | answer | paraphrase": ${line}`);
        return null;
      }
      out.push({ question: parts[0], answer: parts[1], ...(parts[2] ? { paraphrase: parts[2] } : {}) });
    }
    setProbesError(null);
    return out;
  };

  const submit = () => {
    if (!model) return;
    const probes = parseProbes();
    if (probes === null) return;
    void startSleep(model.model_id, {
      method,
      target,
      steps: method === 'anchor' ? undefined : steps,
      sessions: selected.length ? selected : undefined,
      probes: probes.length ? probes : undefined,
    });
  };

  return (
    <Panel title="Sleep" subtitle="consolidate accepted session learning into a new model version">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="Model" htmlFor="sleep-model" hint={model?.parent_model_id ? `child of ${model.parent_model_id}` : undefined}>
          <Select id="sleep-model" value={model?.model_id ?? ''} onChange={(e) => { setModelId(e.target.value); setSelected([]); }}>
            {eligible.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.model_id}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Method" htmlFor="sleep-method" hint={METHOD_LABEL[method]}>
          <Select id="sleep-method" value={method} onChange={(e) => setMethod(e.target.value as SleepMethod)}>
            <option value="anchor">anchor</option>
            <option value="replay">replay</option>
            {isPublic ? null : <option value="distill">distill</option>}
          </Select>
        </Field>
        <Field label="Changes" htmlFor="sleep-target" hint={TARGET_LABEL[target]}>
          <Select id="sleep-target" value={target} onChange={(e) => setTarget(e.target.value as SleepTarget)}>
            <option value="w0">W0 only</option>
            {isPublic ? null : <option value="all">all parameters</option>}
          </Select>
        </Field>
        <Field label="Steps" htmlFor="sleep-steps" hint={method === 'anchor' ? 'not used by anchor' : isPublic ? `up to ${maxSteps} on the shared demo` : undefined}>
          <NumberInput id="sleep-steps" min={1} max={maxSteps} step={1} value={steps} disabled={method === 'anchor'} onChange={(e) => setSteps(Math.max(1, Math.min(maxSteps, Number(e.target.value) || 1)))} />
        </Field>
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <fieldset>
          <legend className="text-xs font-semibold uppercase tracking-wide text-ink-secondary">Source sessions</legend>
          <p className="mt-0.5 text-xs text-ink-secondary">{selected.length === 0 ? 'all sessions of this model; only accepted turns are used' : `${selected.length} selected; only accepted turns are used`}</p>
          <ul className="mt-1 max-h-32 space-y-1 overflow-auto">
            {modelSessions.map((s) => (
              <li key={s.session_id}>
                <label className="flex items-center gap-2 text-sm text-ink-primary">
                  <input
                    type="checkbox"
                    checked={selected.includes(s.session_id)}
                    onChange={(e) => setSelected(e.target.checked ? [...selected, s.session_id] : selected.filter((x) => x !== s.session_id))}
                  />
                  <span className="font-mono">{s.session_id}</span>
                  <span className="text-xs text-ink-secondary">{fmtInt(s.commits)} committed · {fmtInt(s.rollbacks)} rolled back</span>
                </label>
              </li>
            ))}
            {modelSessions.length === 0 ? <li className="text-xs text-ink-secondary">no sessions for this model yet</li> : null}
          </ul>
        </fieldset>
        <Field label="Recall probes (optional)" htmlFor="sleep-probes" hint={probesError ?? 'one per line: question | expected answer | paraphrase (optional); asked in a fresh session before and after'}>
          <textarea
            id="sleep-probes"
            rows={4}
            className="w-full rounded border border-edge bg-surface-sunken p-2 font-mono text-sm text-ink-primary"
            value={probesText}
            onChange={(e) => setProbesText(e.target.value)}
            placeholder={'What is my cat called? | Marlowe | Remind me of my cat\'s name.'}
          />
        </Field>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <Button tone="primary" disabled={busy || running || !model || modelSessions.length === 0} onClick={submit} title="Start a sleep run in a separate process">
          {running ? 'Sleeping…' : 'Sleep'}
        </Button>
        <span className="text-xs text-ink-secondary">{running ? 'a run is in progress; the card below updates every few seconds' : 'minutes on this machine; the model stays available while it runs'}</span>
      </div>
      {runs.length ? (
        <ul className="mt-4 space-y-2">
          {runs.slice(0, 6).map((r) => (
            <RunCard key={r.run_id} run={r} />
          ))}
        </ul>
      ) : null}
    </Panel>
  );
}
