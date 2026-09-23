import { useMemo, useState } from 'react';
import { Panel } from '../components/panels/Panel';
import { loadRun } from './data';
import {
  ARM_LABEL,
  ARM_ROLE,
  GROUP_LABEL,
  GROUP_ORDER,
  dateLabel,
  hits,
  isSleep,
  methodLine,
  num,
  outcome,
  pct,
  ratio,
  signed,
} from './format';
import { Big, Chip, GateCheckRow, Label, OutcomeBadge, OutcomeGlyph, TurnsBar, useAsync } from './parts';
import type { Arm, GroupName, IndexRun, ObservatoryIndex, ProbeRow, Run, SleepArm } from './types';

const REPO = 'https://github.com/DMontgomery40/plastic/blob/main/';

interface Props {
  index: ObservatoryIndex;
  runId: string | null;
  arm: string | null;
  onSelect: (run: string | null, arm: string | null) => void;
}

/** The run shown when none is chosen: the most recent run that consolidated anything. */
export function defaultRunId(index: ObservatoryIndex): string | null {
  const withSleep = index.runs.filter((r) => r.arms.some((a) => a.status !== null));
  return (withSleep[withSleep.length - 1] ?? index.runs[index.runs.length - 1])?.id ?? null;
}

/** The arm shown when none is chosen: a pulled-back arm if the run has one, else its first consolidation arm. */
export function defaultArm(run: Run): string | null {
  const sleep = run.arms.filter(isSleep);
  return (sleep.find((a) => a.lineage.outcome === 'pulled back') ?? sleep[0] ?? run.arms[0])?.arm ?? null;
}

function groupRuns(runs: IndexRun[]): { key: string; label: string; runs: IndexRun[] }[] {
  const order: { key: string; label: string; test: (r: IndexRun) => boolean }[] = [
    { key: 'final', label: 'Final chat checkpoint', test: (r) => r.checkpoint.step === 250 },
    { key: 'step100', label: 'SFT step 100 (intermediate)', test: (r) => r.checkpoint.step === 100 },
    { key: 'step50', label: 'SFT step 50 (intermediate)', test: (r) => r.checkpoint.step === 50 },
    { key: 'base', label: 'Base model, not chat-tuned', test: (r) => r.checkpoint.step === null },
  ];
  return order.map((g) => ({ key: g.key, label: g.label, runs: runs.filter(g.test).slice().reverse() })).filter((g) => g.runs.length);
}

function RunList({ index, active, onSelect }: { index: ObservatoryIndex; active: string | null; onSelect: (id: string) => void }) {
  const groups = useMemo(() => groupRuns(index.runs), [index]);
  return (
    <nav aria-label="Archived runs" className="space-y-4">
      {groups.map((g) => (
        <section key={g.key}>
          <Label>{g.label}</Label>
          <ul className="mt-1.5 space-y-1">
            {g.runs.map((r) => {
              const sleepArms = r.arms.filter((a) => a.status !== null);
              const selected = r.id === active;
              return (
                <li key={r.id}>
                  <button
                    type="button"
                    onClick={() => onSelect(r.id)}
                    aria-current={selected ? 'true' : undefined}
                    className={`w-full rounded border px-3 py-2 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                      selected ? 'border-accent bg-accent-soft' : 'border-edge bg-surface-raised hover:border-edge-strong'
                    }`}
                  >
                    <div className={`truncate font-mono text-xs font-semibold ${selected ? 'text-accent' : 'text-ink-primary'}`}>{r.id}</div>
                    {r.summary ? <div className="mt-0.5 line-clamp-2 text-xs text-ink-secondary">{r.summary}</div> : null}
                    <div className="mt-1.5 flex flex-wrap gap-x-2.5 gap-y-1">
                      {sleepArms.length === 0 ? <span className="text-micro text-ink-muted">controls only</span> : null}
                      {sleepArms.map((a) => (
                        <span key={a.arm} className="inline-flex items-center gap-1 text-micro font-semibold text-ink-secondary" title={`${ARM_LABEL[a.arm]}: ${a.status}`}>
                          <OutcomeGlyph outcome={a.status === 'rejected' ? 'pulled back' : 'committed'} size={11} />
                          {a.arm}
                        </span>
                      ))}
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </nav>
  );
}

function RunHeader({ run }: { run: Run }) {
  const p = run.protocol;
  const readme = run.id === 'base_dryrun' ? run.sources[0]?.file.replace(/[^/]+$/, 'README.md') : `${run.sources[0]?.file.replace(/[^/]+$/, '')}`;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="font-mono text-lg font-semibold text-ink-primary">{run.id}</h2>
        <span className="text-xs text-ink-muted">{dateLabel(run.started_at_unix)}</span>
      </div>
      {run.summary ? <p className="text-base text-ink-secondary">{run.summary}</p> : null}
      <div className="flex flex-wrap gap-1.5">
        <Chip title={run.checkpoint.label}>{run.checkpoint.label}</Chip>
        {run.checkpoint.digest_prefix ? <Chip title="checkpoint digest (sha256 prefix)">{run.checkpoint.digest_prefix}</Chip> : null}
        <Chip title={run.code.archive_readme ?? undefined}>code {run.code.recorded_at_launch ?? 'unrecorded'}</Chip>
        {p.facts !== null ? <Chip>{p.facts} facts{p.poison ? ' + planted' : ''}</Chip> : null}
        {p.steps !== null ? <Chip>{p.target === 'w0' ? 'W0' : p.target} × {p.steps} steps</Chip> : null}
        {p.lr !== null ? <Chip>lr {p.lr}</Chip> : null}
        {p.flagged_policy ? <Chip title="rule for harness-accepted turns that carried an intervention flag">flagged turns: {p.flagged_policy}</Chip> : null}
        {p.ceiling_mode ? <Chip>ceiling: {p.ceiling_mode === 'single' ? 'one fact at a time' : 'all facts'}</Chip> : null}
        {p.augment && p.augment !== 'none' ? <Chip>teaching: {p.augment} set</Chip> : null}
      </div>
      <p className="text-xs text-ink-muted">
        <a className="text-accent hover:text-accent-hover" href={REPO + readme} target="_blank" rel="noreferrer">
          Archived files
        </a>
        {' · '}
        <a className="text-accent hover:text-accent-hover" href={REPO + 'docs/research/2026-09-23-sleep-consolidation.md'} target="_blank" rel="noreferrer">
          Method and readings
        </a>
      </p>
    </div>
  );
}

function RecallCell({ c, variant }: { c: Parameters<typeof ratio>[0]; variant?: 'verbatim' | 'paraphrase' }) {
  const text = ratio(c, variant);
  const lit = hits(c, variant) > 0;
  return <td className={`px-2 py-2 text-right font-mono ${text === 'n/a' ? 'text-ink-muted' : lit ? 'font-semibold text-ink-primary' : 'text-ink-secondary'}`}>{text}</td>;
}

function ArmMatrix({ run, active, onSelect }: { run: Run; active: string | null; onSelect: (arm: string) => void }) {
  const groups = GROUP_ORDER.filter((g) => run.arms.some((a) => a.recall.after.by_group?.[g]));
  const recounted = run.arms.some((a) => a.recall.after.source === 'recounted');
  const hasTotalsOnly = groups.length === 0;
  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] border-collapse text-sm">
          <caption className="sr-only">Recall after a reset, locality and outcome per arm</caption>
          <thead>
            <tr className="border-b border-edge text-left">
              <th scope="col" className="px-2 py-2 text-label font-semibold uppercase tracking-wide text-ink-muted">Arm</th>
              {hasTotalsOnly ? (
                <th scope="col" className="px-2 py-2 text-right text-label font-semibold uppercase tracking-wide text-ink-muted">Recall</th>
              ) : (
                groups.map((g) => (
                  <th key={g} scope="col" className="px-2 py-2 text-right text-label font-semibold uppercase tracking-wide text-ink-muted">
                    {GROUP_LABEL[g]}
                  </th>
                ))
              )}
              {groups.includes('taught') ? <th scope="col" className="px-2 py-2 text-right text-label font-semibold uppercase tracking-wide text-ink-muted">Taught, unseen wording</th> : null}
              <th scope="col" className="px-2 py-2 text-right text-label font-semibold uppercase tracking-wide text-ink-muted">Held-out NLL</th>
              <th scope="col" className="px-2 py-2 text-right text-label font-semibold uppercase tracking-wide text-ink-muted">Same-reply share</th>
              <th scope="col" className="px-2 py-2 text-label font-semibold uppercase tracking-wide text-ink-muted">Outcome</th>
            </tr>
          </thead>
          <tbody>
            {run.arms.map((a) => {
              const g = a.recall.after.by_group ?? {};
              const selected = a.arm === active;
              const share = isSleep(a) ? a.gate?.checks.find((c) => c.name === 'reply_cluster_share')?.value ?? a.recall.totals_after?.max_cluster_share ?? null : a.max_cluster_share.value;
              const nll = isSleep(a) ? a.heldout_nll : null;
              return (
                <tr
                  key={a.arm}
                  onClick={() => onSelect(a.arm)}
                  className={`cursor-pointer border-b border-edge ${selected ? 'bg-accent-soft' : 'hover:bg-surface-overlay'}`}
                >
                  <th scope="row" className="px-2 py-2 text-left">
                    <button type="button" onClick={() => onSelect(a.arm)} aria-pressed={selected} className="text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent">
                      <span className={`block font-semibold ${selected ? 'text-accent' : 'text-ink-primary'}`}>{ARM_LABEL[a.arm]}</span>
                      <span className="block text-micro font-normal text-ink-muted">{ARM_ROLE[a.arm]}</span>
                    </button>
                  </th>
                  {hasTotalsOnly ? (
                    <td className="px-2 py-2 text-right font-mono text-ink-secondary">
                      {isSleep(a) && a.recall.totals_after ? `${a.recall.totals_after.recalled}/${a.recall.totals_after.n_probes}` : 'n/a'}
                    </td>
                  ) : (
                    groups.map((grp) => <RecallCell key={grp} c={g[grp]} />)
                  )}
                  {groups.includes('taught') ? <RecallCell c={g.taught} variant="paraphrase" /> : null}
                  <td className="whitespace-nowrap px-2 py-2 text-right font-mono text-ink-secondary">
                    {nll?.before?.mean != null ? (
                      <>
                        {num(nll.before.mean)} <span className="text-ink-muted">→</span> <span className="text-ink-primary">{num(nll.after?.mean)}</span>
                      </>
                    ) : (
                      <span className="text-ink-muted">n/a</span>
                    )}
                  </td>
                  <td className="px-2 py-2 text-right font-mono text-ink-secondary">{pct(share)}</td>
                  <td className="px-2 py-2">
                    <OutcomeBadge outcome={outcome(a)} label={isSleep(a) ? undefined : a.arm === 'floor' ? 'Baseline' : 'In context'} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-ink-muted">
        Recall in a fresh session after consolidation; hits over probes.{recounted ? ' Counts recounted from saved replies.' : ''}
      </p>
    </div>
  );
}

function ProbeTable({ probes }: { probes: ProbeRow[] }) {
  const present = GROUP_ORDER.filter((g) => probes.some((p) => p.group === g));
  const [group, setGroup] = useState<GroupName | 'all'>(present.includes('taught') ? 'taught' : 'all');
  const [onlyHits, setOnlyHits] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const rows = probes.filter((p) => (group === 'all' || p.group === group) && (!onlyHits || p.hit));
  const shown = expanded ? rows : rows.slice(0, 12);
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <Label>Replies</Label>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Probe group">
          {(['all', ...present] as const).map((g) => (
            <button
              key={g}
              type="button"
              aria-pressed={group === g}
              onClick={() => setGroup(g)}
              className={`rounded border px-2 py-0.5 text-xs font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                group === g ? 'border-accent bg-accent-soft text-accent' : 'border-edge-strong bg-surface-overlay text-ink-secondary hover:text-ink-primary'
              }`}
            >
              {g === 'all' ? 'All' : GROUP_LABEL[g]}
            </button>
          ))}
        </div>
        <label className="ml-auto flex items-center gap-2 text-xs text-ink-secondary">
          <input type="checkbox" checked={onlyHits} onChange={(e) => setOnlyHits(e.target.checked)} />
          hits only
        </label>
      </div>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[720px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-edge text-left text-label font-semibold uppercase tracking-wide text-ink-muted">
              <th scope="col" className="px-2 py-1.5">Question</th>
              <th scope="col" className="px-2 py-1.5">Expected</th>
              <th scope="col" className="px-2 py-1.5">Reply</th>
              <th scope="col" className="px-2 py-1.5 text-right">Hit</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((p, i) => (
              <tr key={`${p.question}-${p.variant}-${i}`} className="border-b border-edge align-top">
                <td className="px-2 py-1.5 text-ink-secondary">
                  {p.question}
                  <span className="ml-1.5 text-micro text-ink-muted">{p.variant === 'paraphrase' ? 'unseen wording' : ''}</span>
                </td>
                <td className="px-2 py-1.5 font-mono text-ink-primary">{p.expected}</td>
                <td className="px-2 py-1.5 text-ink-primary">
                  {p.reply}
                  {p.reply_truncated ? '…' : ''}
                </td>
                <td className={`px-2 py-1.5 text-right text-xs font-semibold ${p.hit ? 'text-status-commit' : 'text-ink-muted'}`}>{p.hit ? 'hit' : 'miss'}</td>
              </tr>
            ))}
            {shown.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-2 py-3 text-sm text-ink-secondary">
                  No replies match.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      {rows.length > 12 ? (
        <button type="button" onClick={() => setExpanded(!expanded)} className="mt-2 text-sm font-semibold text-accent hover:text-accent-hover">
          {expanded ? 'Show fewer' : `Show all ${rows.length}`}
        </button>
      ) : null}
    </div>
  );
}

function Lineage({ run, arm }: { run: Run; arm: SleepArm }) {
  const out = arm.lineage.outcome;
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span className="rounded border border-edge-strong bg-surface-overlay px-2.5 py-1 font-mono text-ink-primary">{arm.lineage.parent ?? 'parent'}</span>
      <span className="text-ink-muted" aria-hidden="true">
        ──
      </span>
      <span className="rounded border border-edge bg-surface-inset px-2.5 py-1 text-ink-secondary">
        {arm.method} · {methodLine(arm)}
      </span>
      <span className="text-ink-muted" aria-hidden="true">
        ──
      </span>
      {out === 'committed' ? (
        <span className="rounded border border-status-commit bg-surface-overlay px-2.5 py-1 font-mono text-ink-primary">child {arm.lineage.child}</span>
      ) : (
        <span className="rounded border border-dashed border-status-rollback bg-surface-overlay px-2.5 py-1 text-ink-primary">no child</span>
      )}
      <span className="basis-full text-xs text-ink-muted">
        {out === 'committed' ? 'Registered in the run’s experiment store; not published.' : 'Pulled back: the child was discarded and the parent is unchanged.'}
        {run.checkpoint.digest_prefix ? ` Parent checkpoint ${run.checkpoint.digest_prefix}.` : ''}
      </span>
    </div>
  );
}

function SleepArmDetail({ run, arm }: { run: Run; arm: SleepArm }) {
  const before = arm.heldout_nll.before;
  const after = arm.heldout_nll.after;
  const tb = arm.recall.totals_before;
  const ta = arm.recall.totals_after;
  const taughtBefore = arm.recall.before.by_group?.taught;
  const taughtAfter = arm.recall.after.by_group?.taught;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="text-lg font-semibold text-ink-primary">{ARM_LABEL[arm.arm]}</h3>
        <OutcomeBadge outcome={outcome(arm)} />
        {arm.reason ? <span className="text-sm text-ink-secondary">{arm.reason}</span> : null}
      </div>
      <Lineage run={run} arm={arm} />
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
          <Label>Taught facts recalled</Label>
          <Big>
            {taughtAfter ? `${ratio(taughtBefore)} → ${ratio(taughtAfter)}` : tb && ta ? `${tb.recalled}/${tb.n_probes} → ${ta.recalled}/${ta.n_probes}` : 'n/a'}
          </Big>
          <div className="text-micro text-ink-muted">{taughtAfter ? 'fresh session, before → after' : 'all probes, fresh session, before → after'}</div>
        </div>
        <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
          <Label>Held-out NLL (nats/token)</Label>
          <Big>
            {num(before?.mean)} → {num(after?.mean)}
          </Big>
          <div className="text-micro text-ink-muted">change {signed(before?.mean != null && after?.mean != null ? after.mean - before.mean : null)} · {before?.tokens ?? 'n/a'} tokens</div>
        </div>
        <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
          <Label>Expected-answer log-prob</Label>
          <Big>
            {num(tb?.mean_answer_logprob, 2)} → {num(ta?.mean_answer_logprob, 2)}
          </Big>
          <div className="text-micro text-ink-muted">mean per token, verbatim probes</div>
        </div>
        <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
          <Label>Same-reply share</Label>
          <Big>
            {pct(tb?.max_cluster_share)} → {pct(ta?.max_cluster_share)}
          </Big>
          <div className="text-micro text-ink-muted">largest group of identical replies</div>
        </div>
      </div>
      {arm.gate ? (
        <div>
          <Label>Locality gate</Label>
          <ul className="mt-2 space-y-2">
            {arm.gate.checks.map((c) => (
              <GateCheckRow key={c.name} check={c} />
            ))}
          </ul>
        </div>
      ) : null}
      {arm.harvest ? (
        <div>
          <Label>Turns this arm trained on</Label>
          <div className="mt-2">
            <TurnsBar harvest={arm.harvest} />
          </div>
        </div>
      ) : null}
      {arm.dreams ? (
        <div>
          <Label>Dreams</Label>
          <p className="mt-1 text-sm text-ink-secondary">
            <span className="font-mono font-semibold text-ink-primary">{arm.dreams.generated ?? 'n/a'}</span> generated,{' '}
            <span className="font-mono font-semibold text-ink-primary">{arm.dreams.kept_count}</span> kept
            {Object.entries(arm.dreams.rejected_reasons).map(([k, v]) => `, ${v} removed as ${k}`).join('')}
          </p>
          {arm.dreams.kept.slice(0, 2).map((d, i) => (
            <blockquote key={i} className="mt-2 rounded border border-edge bg-surface-inset px-3 py-2 text-sm text-ink-primary">
              {d.text}
              <footer className="mt-1 font-mono text-xs text-ink-muted">
                gain {signed(d.gain, 2)} · no-quote session score {signed(d.fastweight_gain, 2)} nats/token
              </footer>
            </blockquote>
          ))}
        </div>
      ) : null}
      <ProbeTable probes={arm.probes} />
    </div>
  );
}

function ControlArmDetail({ arm }: { arm: Arm }) {
  if (isSleep(arm)) return null;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="text-lg font-semibold text-ink-primary">{ARM_LABEL[arm.arm]}</h3>
        <span className="text-sm text-ink-secondary">{ARM_ROLE[arm.arm]}</span>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
          <Label>Taught facts recalled</Label>
          <Big>{ratio(arm.recall.after.by_group?.taught)}</Big>
          <div className="text-micro text-ink-muted">unseen wording {ratio(arm.recall.after.by_group?.taught, 'paraphrase')}</div>
        </div>
        <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
          <Label>Expected-answer log-prob</Label>
          <Big>{num(arm.mean_answer_logprob, 2)}</Big>
          <div className="text-micro text-ink-muted">mean per token, verbatim probes</div>
        </div>
        <div className="rounded border border-edge bg-surface-overlay px-3 py-2.5">
          <Label>Same-reply share</Label>
          <Big>{pct(arm.max_cluster_share.value)}</Big>
          <div className="text-micro text-ink-muted">{arm.max_cluster_share.source}</div>
        </div>
      </div>
      <ProbeTable probes={arm.probes} />
    </div>
  );
}

export function RunsView({ index, runId, arm, onSelect }: Props) {
  const id = runId ?? defaultRunId(index);
  const { data: run, error } = useAsync(id ? () => loadRun(id) : null, [id]);
  const activeArm = run ? (arm && run.arms.some((a) => a.arm === arm) ? arm : defaultArm(run)) : null;
  const selected = run?.arms.find((a) => a.arm === activeArm) ?? null;
  const sleepArms = index.runs.flatMap((r) => r.arms.filter((a) => a.status !== null));
  const committed = sleepArms.filter((a) => a.status !== 'rejected').length;
  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-4">
        <div className="rounded-lg border border-edge bg-surface-raised px-4 py-3">
          <Label>Archived runs</Label>
          <Big>{index.runs.length}</Big>
        </div>
        <div className="rounded-lg border border-edge bg-surface-raised px-4 py-3">
          <Label>Consolidation attempts</Label>
          <Big>{sleepArms.length}</Big>
        </div>
        <div className="rounded-lg border border-edge bg-surface-raised px-4 py-3">
          <Label>Committed a child</Label>
          <Big tone="pass">{committed}</Big>
        </div>
        <div className="rounded-lg border border-edge bg-surface-raised px-4 py-3">
          <Label>Pulled back by the gate</Label>
          <Big tone="fail">{sleepArms.length - committed}</Big>
        </div>
      </div>
      <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
        <RunList index={index} active={id} onSelect={(r) => onSelect(r, null)} />
        <div className="min-w-0 space-y-4">
          {error ? <p className="text-sm text-status-rollback">This run could not be loaded: {error}</p> : null}
          {!run && !error ? <p className="text-sm text-ink-secondary">Loading run…</p> : null}
          {run ? (
            <>
              <Panel title="Run" subtitle="one protocol, several consolidation arms from the same parent and sessions">
                <RunHeader run={run} />
                <div className="mt-4">
                  <ArmMatrix run={run} active={activeArm} onSelect={(a) => onSelect(run.id, a)} />
                </div>
              </Panel>
              {selected ? (
                <Panel title="Arm" subtitle={isSleep(selected) ? 'what was consolidated, how it was checked, and what the child answered' : 'a control from the parent model'}>
                  {isSleep(selected) ? <SleepArmDetail run={run} arm={selected} /> : <ControlArmDetail arm={selected} />}
                </Panel>
              ) : null}
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
