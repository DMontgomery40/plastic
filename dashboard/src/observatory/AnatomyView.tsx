import { useState, type ReactNode } from 'react';
import { buildAnatomy, type AnatomyModel, type WakeSession } from './anatomy';
import { loadRun, loadTrajectory } from './data';
import { ARM_LABEL, isSleep, num, ratio, signed } from './format';
import { ILLUSTRATIVE } from './illustrative';
import { Big, GateCheckRow, IllustrativeTag, Label, OutcomeBadge, OutcomeGlyph, useAsync } from './parts';
import { defaultRunId } from './RunsView';
import { Sparkline } from './Sparkline';
import type { ObservatoryIndex, Run, SleepArm } from './types';
import { OBS } from '../components/charts/theme';

interface Props {
  index: ObservatoryIndex;
  runId: string | null;
  arm: string | null;
  onSelect: (run: string | null, arm: string | null) => void;
}

const METHOD_LEAD: Record<string, string> = {
  anchor: 'Moves the initial fast weights W0 part of the way toward the session’s final fast weights. No gradient steps.',
  replay: 'Fine-tunes the model on the selected turns, mixed with ordinary chat data so it keeps its general behavior.',
  distill: 'A frozen copy that holds the session’s state teaches the reset model to predict the way it does.',
  dream: 'The session-state teacher writes study items about each selected turn; the reset model learns to reproduce them without seeing the turn.',
};

function Stage({ n, title, lead, illustrative, children }: { n: number; title: string; lead: string; illustrative: boolean; children: ReactNode }) {
  return (
    <li className="grid grid-cols-[2.5rem_minmax(0,1fr)] gap-3">
      <div className="flex flex-col items-center">
        <span
          className={`flex h-9 w-9 items-center justify-center rounded-full border-2 font-mono text-sm font-bold ${
            illustrative ? 'border-illustrative text-illustrative' : 'border-accent text-accent'
          } bg-surface-raised`}
        >
          {n}
        </span>
        <span className="mt-1 w-0.5 flex-1 bg-edge-strong" aria-hidden="true" />
      </div>
      <section className={`mb-4 min-w-0 rounded-lg border px-4 py-3 ${illustrative ? 'border-illustrative bg-illustrative-soft' : 'border-edge bg-surface-raised'}`}>
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-lg font-semibold text-ink-primary">{title}</h3>
          {illustrative ? <IllustrativeTag /> : null}
        </div>
        <p className="mt-0.5 max-w-3xl text-sm text-ink-secondary">{lead}</p>
        <div className="mt-3">{children}</div>
      </section>
    </li>
  );
}

const CELL: Record<string, string> = {
  commit: 'bg-statusFill-commit',
  rollback: 'bg-statusFill-rollback',
  scale: 'bg-statusFill-scale',
  project: 'bg-statusFill-project',
  readonly: 'bg-statusFill-readonly',
};

function WakeStrip({ session }: { session: WakeSession }) {
  const chunks = session.turns.flat();
  const counts = chunks.reduce<Record<string, number>>((acc, c) => ({ ...acc, [c.applied]: (acc[c.applied] ?? 0) + 1 }), {});
  const flagged = chunks.filter((c) => c.flagged).length;
  const label = session.id === 'teach' ? 'Teaching session' : session.id === 'rolled' ? 'Rolled-back session' : session.id;
  return (
    <div>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm">
        <span className="font-semibold text-ink-primary">{label}</span>
        <span className="text-ink-secondary">
          {session.turns.length} turns · {chunks.length} chunks ·{' '}
          {Object.entries(counts)
            .map(([k, v]) => `${v} ${k === 'commit' ? 'committed' : k === 'rollback' ? 'rolled back' : k}`)
            .join(' · ')}
        </span>
        {session.forced ? <span className="text-xs text-ink-muted">rollbacks forced by the experiment</span> : null}
        {session.log_only ? <span className="text-xs text-ink-muted">observational: every change commits</span> : null}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-2 gap-y-2" role="img" aria-label={`${label}: ${chunks.length} chunks, ${flagged} flagged`}>
        {session.turns.map((t, i) => (
          <div key={i} className="flex gap-px rounded-sm border border-edge bg-surface-inset p-0.5" title={`turn ${i + 1}: ${t.length} chunks`}>
            {t.map((c, j) => (
              <div key={j} className="flex flex-col gap-px">
                <div className={`h-4 w-2 rounded-[1px] ${CELL[c.applied] ?? 'bg-statusFill-readonly'}`} />
                <div className={`h-1 w-2 rounded-[1px] ${c.flagged ? 'bg-status-scale' : 'bg-surface-inset'}`} />
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function WakeLegend({ observational }: { observational: boolean }) {
  return (
    <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-secondary">
      <li className="flex items-center gap-1.5"><span className="inline-block h-3 w-2 rounded-[1px] bg-statusFill-commit" /> chunk committed</li>
      <li className="flex items-center gap-1.5"><span className="inline-block h-3 w-2 rounded-[1px] bg-statusFill-rollback" /> chunk rolled back</li>
      <li className="flex items-center gap-1.5"><span className="inline-block h-1 w-2 rounded-[1px] bg-status-scale" /> intervention flagged{observational ? ' (advisory, not applied)' : ''}</li>
      <li className="text-ink-muted">one box per turn, one cell per 16-token chunk</li>
    </ul>
  );
}

function RecallCard({ label, before, after, variant = 'verbatim', tone }: { label: string; before?: Parameters<typeof ratio>[0]; after: Parameters<typeof ratio>[0]; variant?: 'verbatim' | 'paraphrase'; tone?: 'pass' | 'illustrative' }) {
  return (
    <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
      <Label>{label}</Label>
      <Big tone={tone}>
        {before !== undefined ? `${ratio(before, variant)} → ` : ''}
        {ratio(after, variant)}
      </Big>
    </div>
  );
}

function AnatomyStages({ m }: { m: AnatomyModel }) {
  const ill = m.mode === 'illustrative';
  const h = m.harvest;
  const c = m.consolidate;
  const observational = !!m.wake?.sessions.some((s) => s.log_only);
  return (
    <ol className="mt-2">
      <Stage n={1} title="Wake: learning during the conversation" illustrative={ill} lead="The model learns while it talks. Every 16-token chunk proposes a change to its fast weights; the harness commits the change or rolls it back.">
        {m.wake ? (
          <div className="space-y-4">
            {m.wake.sessions.map((s) => (
              <WakeStrip key={s.id} session={s} />
            ))}
            <WakeLegend observational={observational} />
          </div>
        ) : (
          <p className="text-sm text-ink-muted">Session log not archived for this run.</p>
        )}
      </Stage>

      <Stage n={2} title="Harvest: what may be consolidated" illustrative={ill} lead="Sleep reads back only turns whose every chunk was kept. A second rule decides whether kept turns that carried an intervention flag are used.">
        {h ? (
          <div className="space-y-3">
            <div className="grid gap-3 sm:grid-cols-4">
              <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
                <Label>Accepted online</Label>
                <Big>{h.accepted}</Big>
              </div>
              <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
                <Label>Rolled back</Label>
                <Big>{h.rolled_back}</Big>
                <div className="text-micro text-ink-muted">{h.all_turns ? 'included by this control' : 'excluded'}</div>
              </div>
              <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
                <Label>Flagged, kept online</Label>
                <Big>{h.flagged}</Big>
                <div className="text-micro text-ink-muted">rule: {h.policy ?? 'not recorded'}{h.policy === 'exclude' ? `, ${h.flagged_excluded} excluded` : ''}</div>
              </div>
              <div className="rounded border border-accent bg-surface-overlay px-3 py-2.5">
                <Label>Selected text turns</Label>
                <Big tone={ill ? 'illustrative' : 'accent'}>{h.selected ?? 'n/a'}</Big>
                <div className="text-micro text-ink-muted">{h.selected === null ? 'not recorded' : h.selected_source}</div>
              </div>
            </div>
            {h.state_sessions.length ? (
              <p className="text-sm text-ink-secondary">
                {m.method === 'anchor' ? 'Anchor' : m.method === 'distill' ? 'Distill' : 'Dream'} also loads the full committed state of: <span className="font-mono text-ink-primary">{h.state_sessions.join(', ')}</span>
              </p>
            ) : null}
          </div>
        ) : (
          <p className="text-sm text-ink-muted">Harvest not recorded.</p>
        )}
      </Stage>

      <Stage n={3} title={`Consolidate: ${m.method}${m.target === 'w0' ? ' on W0' : m.target === 'all' ? ' on all parameters' : ''}`} illustrative={ill} lead={METHOD_LEAD[m.method] ?? ''}>
        <div className="grid gap-4 md:grid-cols-2">
          {c.losses ? (
            <Sparkline values={c.losses} label={`training loss (${c.loss_terms ?? 'loss'})`} color={ill ? OBS.illustrative : OBS.line} />
          ) : (
            <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
              <Label>{m.method === 'anchor' ? 'Interpolation' : 'Steps'}</Label>
              <Big>{c.anchor_lambda !== null ? `λ ${c.anchor_lambda}` : 'n/a'}</Big>
              <div className="text-micro text-ink-muted">{m.method === 'anchor' ? 'W0 ← W0 + λ · (session − W0)' : 'no step log recorded'}</div>
            </div>
          )}
          <div className="grid content-start gap-3 sm:grid-cols-2">
            <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
              <Label>W0 relative change</Label>
              <Big tone={ill ? 'illustrative' : 'default'}>{num(c.w0_change_total, 4)}</Big>
              <div className="text-micro text-ink-muted">{c.w0_change_total !== null ? '‖ΔW0‖ / ‖W0‖, all tensors' : 'not recorded for this run'}</div>
            </div>
            {c.steps !== null && m.method !== 'anchor' ? (
              <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
                <Label>Budget</Label>
                <Big>{c.steps} steps</Big>
                <div className="text-micro text-ink-muted">lr {c.lr ?? 'n/a'} · batch {c.session_rows ?? '?'} session + {c.replay_rows ?? '?'} replay rows</div>
              </div>
            ) : null}
            {c.dreams ? (
              <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5 sm:col-span-2">
                <Label>Dreams</Label>
                <div className="font-mono text-base text-ink-primary">
                  {c.dreams.generated ?? 'n/a'} generated → <span className="font-semibold">{c.dreams.kept} kept</span>
                </div>
                <div className="text-micro text-ink-muted">{Object.entries(c.dreams.removed).map(([k, v]) => `${v} removed: ${k.replace('_', ' ')}`).join(' · ') || 'none removed'}</div>
                {c.dreams.example ? <p className="mt-1.5 text-sm text-ink-secondary">“{c.dreams.example}”</p> : null}
              </div>
            ) : null}
          </div>
        </div>
      </Stage>

      <Stage n={4} title="Gate: check the candidate child" illustrative={ill} lead="The child is measured against the parent on held-out chat and on the replies it gives to the probes. Any failed check rejects it.">
        <div className="space-y-3">
          <ul className="space-y-2">
            {m.gate.checks.map((ch) => (
              <GateCheckRow key={ch.name} check={ch} />
            ))}
          </ul>
          <p className="font-mono text-sm text-ink-secondary">
            held-out NLL {num(m.gate.nll_before)} → <span className="text-ink-primary">{num(m.gate.nll_after)}</span> nats/token ({signed(m.gate.nll_before !== null && m.gate.nll_after !== null ? m.gate.nll_after - m.gate.nll_before : null)})
          </p>
        </div>
      </Stage>

      <Stage
        n={5}
        title={m.outcome.outcome === 'committed' ? 'Commit: a child model' : m.outcome.outcome === 'pulled back' ? 'Pull back: no child' : 'Outcome'}
        illustrative={ill}
        lead={
          m.outcome.outcome === 'committed'
            ? 'The child is registered with a pointer to its parent. The parent never changes.'
            : m.outcome.outcome === 'pulled back'
              ? 'The candidate is discarded. The parent is unchanged, as if the sleep had not happened.'
              : 'Outcome not recorded.'
        }
      >
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded border border-edge-strong bg-surface-overlay px-3 py-1.5 font-mono text-sm text-ink-primary">{m.outcome.parent ?? 'parent'}</span>
            <span className="text-ink-muted" aria-hidden="true">────</span>
            {m.outcome.outcome === 'committed' ? (
              <span className="inline-flex items-center gap-2 rounded border-2 border-status-commit bg-surface-overlay px-3 py-1.5 font-mono text-sm text-ink-primary">
                <OutcomeGlyph outcome="committed" /> {m.outcome.child}
              </span>
            ) : m.outcome.outcome === 'pulled back' ? (
              <span className="inline-flex items-center gap-2 rounded border-2 border-dashed border-status-rollback bg-surface-overlay px-3 py-1.5 text-sm text-ink-primary">
                <OutcomeGlyph outcome="pulled back" /> no child
              </span>
            ) : (
              <span className="inline-flex items-center gap-2 rounded border border-edge-strong bg-surface-overlay px-3 py-1.5 text-sm text-ink-secondary">
                <OutcomeGlyph outcome="unknown" /> outcome unknown
              </span>
            )}
            <OutcomeBadge outcome={m.outcome.outcome === 'unknown' ? 'unknown' : m.outcome.outcome} />
          </div>
          <div>
            <Label>Fresh-session recall{m.outcome.outcome === 'pulled back' ? ' of the discarded candidate' : ''}</Label>
            <div className="mt-2 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <RecallCard label="Taught facts" before={m.recall.taught_before} after={m.recall.taught_after} tone={ill ? 'illustrative' : undefined} />
              <RecallCard label="Taught, unseen wording" before={m.recall.taught_before} after={m.recall.taught_after} variant="paraphrase" tone={ill ? 'illustrative' : undefined} />
              <RecallCard label="Rolled-back facts" after={m.recall.rolled_after} />
              <RecallCard label="General knowledge" before={m.recall.general_before} after={m.recall.general_after} />
            </div>
          </div>
        </div>
      </Stage>
    </ol>
  );
}

function sleepRuns(index: ObservatoryIndex) {
  return index.runs.filter((r) => r.arms.some((a) => a.method !== null)).slice().reverse();
}

export function AnatomyView({ index, runId, arm, onSelect }: Props) {
  const [mode, setMode] = useState<'measured' | 'illustrative'>('measured');
  const runs = sleepRuns(index);
  const id = runId && runs.some((r) => r.id === runId) ? runId : defaultRunId(index);
  const entry = index.runs.find((r) => r.id === id);
  const { data } = useAsync<{ run: Run; trajectory: Awaited<ReturnType<typeof loadTrajectory>> | null }>(
    id ? async () => ({ run: await loadRun(id), trajectory: entry?.session_trajectory ? await loadTrajectory(id) : null }) : null,
    [id]
  );
  const arms = (data?.run.arms.filter(isSleep) ?? []) as SleepArm[];
  const chosen = arms.find((a) => a.arm === arm) ?? arms.find((a) => a.lineage.outcome === 'pulled back') ?? arms[0] ?? null;
  const model = mode === 'illustrative' ? ILLUSTRATIVE : data && chosen ? buildAnatomy(data.run, chosen, data.trajectory) : null;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-3 rounded-lg border border-edge bg-surface-raised px-4 py-3">
        <div role="group" aria-label="What to show" className="flex rounded border border-edge-strong bg-surface-overlay p-0.5">
          <button
            type="button"
            aria-pressed={mode === 'measured'}
            onClick={() => setMode('measured')}
            className={`rounded px-3 py-1 text-sm font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${mode === 'measured' ? 'bg-accent-soft text-accent' : 'text-ink-secondary hover:text-ink-primary'}`}
          >
            Measured run
          </button>
          <button
            type="button"
            aria-pressed={mode === 'illustrative'}
            onClick={() => setMode('illustrative')}
            className={`rounded px-3 py-1 text-sm font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${mode === 'illustrative' ? 'bg-illustrative-soft text-illustrative' : 'text-ink-secondary hover:text-ink-primary'}`}
          >
            Illustrative: if consolidation worked
          </button>
        </div>
        {mode === 'measured' ? (
          <>
            <label className="flex min-w-0 flex-col gap-1">
              <Label>Run</Label>
              <select
                aria-label="Run"
                value={id ?? ''}
                onChange={(e) => onSelect(e.target.value, null)}
                className="max-w-[22rem] rounded border border-edge-strong bg-surface-overlay px-2.5 py-1.5 font-mono text-sm text-ink-primary focus:border-accent focus:outline-none"
              >
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id}
                  </option>
                ))}
              </select>
            </label>
            <div className="flex flex-col gap-1">
              <Label>Arm</Label>
              <div role="group" aria-label="Arm" className="flex flex-wrap gap-1">
                {arms.map((a) => (
                  <button
                    key={a.arm}
                    type="button"
                    aria-pressed={chosen?.arm === a.arm}
                    onClick={() => onSelect(id, a.arm)}
                    className={`inline-flex items-center gap-1.5 rounded border px-2.5 py-1 text-sm font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                      chosen?.arm === a.arm ? 'border-accent bg-accent-soft text-accent' : 'border-edge-strong bg-surface-overlay text-ink-secondary hover:text-ink-primary'
                    }`}
                  >
                    <OutcomeGlyph outcome={a.lineage.outcome} size={12} />
                    {ARM_LABEL[a.arm]}
                  </button>
                ))}
              </div>
            </div>
          </>
        ) : (
          <p className="max-w-xl text-sm text-illustrative">A prototype of the intended outcome with invented numbers. No run has produced this.</p>
        )}
      </div>
      {model ? <AnatomyStages m={model} /> : <p className="text-sm text-ink-secondary">Loading run…</p>}
    </div>
  );
}
