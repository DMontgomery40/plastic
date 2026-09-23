import { useState, type ReactNode } from 'react';
import { Empty } from '../components/panels/Empty';
import { Panel } from '../components/panels/Panel';
import { Button } from '../playground/ui';
import { num } from './format';
import {
  ablationState,
  count,
  delta,
  loadLearning,
  MODE_LABEL,
  modeRole,
  rateCell,
  reportSets,
  speedLabel,
  VARIANT_LABEL,
  windowLabel,
  type AblationSet,
  type Before,
  type LearningIndex,
  type ModeResult,
  type ReportSet,
} from './learning';
import { useAsync } from './parts';

/** The one sentence that separates the two kinds of change this view reports. */
export const LEARNING_LEAD =
  'Within one episode the fast weights adapt and are then cleared; only a change to the slow weights that survives that reset counts as learning.';

const TH = 'px-3 py-2 text-label font-semibold uppercase tracking-wide text-ink-muted';

function Fact({ label, children, detail }: { label: string; children: ReactNode; detail?: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-label font-semibold uppercase tracking-wide text-ink-muted">{label}</dt>
      <dd className="mt-1 break-words text-base text-ink-primary">{children}</dd>
      {detail ? <dd className="mt-0.5 text-xs text-ink-secondary">{detail}</dd> : null}
    </div>
  );
}

function StateWord({ state }: { state: 'stale' | 'not yet archived' | 'not archived' }) {
  const tone = state === 'stale' ? 'border-status-scale text-status-scale' : 'border-edge-strong text-ink-secondary';
  return <span className={`inline-flex items-center whitespace-nowrap rounded border bg-surface-overlay px-2 py-0.5 text-xs font-semibold ${tone}`}>{state}</span>;
}

function Identity({ set }: { set: ReportSet }) {
  const cp = set.checkpoint;
  const stream = set.stream;
  return (
    <Panel title={cp.model_id ?? set.id} subtitle={set.tag ?? undefined}>
      <dl className="grid gap-x-6 gap-y-4 sm:grid-cols-2 xl:grid-cols-3">
        <Fact label="Checkpoint" detail={cp.step !== null ? `training step ${count(cp.step)}` : undefined}>
          <span className="font-mono">{cp.digest_prefix ?? 'n/a'}</span>
        </Fact>
        <Fact label="Execution commit">
          <span className="font-mono">{set.execution_commit ?? 'n/a'}</span>
        </Fact>
        <Fact label="Contract version" detail={set.current ? undefined : <StateWord state="stale" />}>
          <span className="font-mono">{set.contract_version ?? 'n/a'}</span>
        </Fact>
        <Fact label="Split" detail={set.split.train && set.split.heldout ? `${set.split.train.length} training pairings, ${set.split.heldout.length} held out` : undefined}>
          <span className="font-mono">{set.split.id ?? 'n/a'}</span>
        </Fact>
        <Fact label="Adaptation window">{windowLabel(set.adaptation_window)}</Fact>
        <Fact label="Experience stream">
          {stream ? `${count(stream.episodes)} episodes, ${count(stream.tokens)} steps, ${stream.policy ?? 'n/a'} policy` : 'n/a'}
        </Fact>
      </dl>
    </Panel>
  );
}

function BeforeTable({ before, noAdaptLabel, caption }: { before: Before; noAdaptLabel: string; caption: string }) {
  const rows = [
    ...before.policies.map((p) => ({ key: p.policy, label: `Held-out pairings, ${p.policy} policy`, pair: p })),
    { key: 'train', label: `Training distribution (${before.train.policies.join(', ') || 'n/a'})`, pair: before.train },
  ];
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-base">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-edge text-left">
            <th scope="col" className={TH}>Measurement</th>
            <th scope="col" className={`${TH} text-right`}>With adaptation</th>
            <th scope="col" className={`${TH} text-right`}>Without: {noAdaptLabel}</th>
            <th scope="col" className={`${TH} text-right`}>Elements</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className="border-b border-edge">
              <th scope="row" className="px-3 py-2 text-left font-normal text-ink-primary">{r.label}</th>
              <td className="px-3 py-2 text-right font-mono font-semibold text-ink-primary">{num(r.pair.adapt)}</td>
              <td className="px-3 py-2 text-right font-mono text-ink-secondary">{num(r.pair.no_adapt)}</td>
              <td className="px-3 py-2 text-right font-mono text-ink-secondary">{count(r.pair.elements)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BeforePanel({ set }: { set: ReportSet }) {
  const speed = set.before.speed;
  return (
    <Panel title="Before any stream: temporary adaptation" subtitle="MSE from a fresh state, one episode per row; lower is better">
      <BeforeTable before={set.before} noAdaptLabel={set.no_adapt_label} caption="Held-out MSE with and without within-episode adaptation" />
      <p className="mt-3 text-base text-ink-secondary">
        Adaptation speed, the adapting error as a fraction of the error without adaptation: {speedLabel(speed)}
        {speed?.episodes ? `, over ${speed.episodes} episodes` : ''}.
      </p>
    </Panel>
  );
}

interface MeasureRow {
  key: string;
  label: string;
  cell: (m: ModeResult) => ReactNode;
}

function revertCell(m: ModeResult): ReactNode {
  if (m.revert.ok === null) return <span className="text-ink-muted">n/a</span>;
  return m.revert.ok ? <span className="font-semibold text-status-commit">ok</span> : <span className="font-semibold text-status-rollback">not ok</span>;
}

function tokensMeasured(m: ModeResult): ReactNode {
  const without = m.tokens_measured_without_context;
  return (
    <>
      {count(m.tokens_measured)}
      {without !== null && without !== m.tokens_measured ? <span className="block text-xs text-ink-secondary">{count(without)} without the stream</span> : null}
    </>
  );
}

function AfterPanel({ set }: { set: ReportSet }) {
  const policies = set.before.policies.map((p) => p.policy);
  const rows: MeasureRow[] = [
    ...policies.map((p) => ({ key: `t-${p}`, label: `Transfer Δ, ${p} policy`, cell: (m: ModeResult) => delta(m.transfer.find((t) => t.policy === p)?.delta ?? null) })),
    { key: 'forget', label: 'Forgetting Δ, training distribution', cell: (m) => delta(m.forgetting_delta) },
    { key: 'harm-clean', label: 'Poison harm vs clean stream', cell: (m) => delta(m.poison_harm_vs_clean) },
    { key: 'harm-start', label: 'Poison harm vs start', cell: (m) => delta(m.poison_harm_vs_start) },
    { key: 'residual', label: 'Correction residual', cell: (m) => delta(m.correction_residual) },
    { key: 'revert', label: 'Revert to pre-stream parameters', cell: revertCell },
    { key: 'good', label: 'Accepted good', cell: (m) => rateCell(m.acceptance.accepted_good, m.acceptance.n_good) },
    { key: 'bad', label: 'Refused bad', cell: (m) => rateCell(m.acceptance.refused_bad, m.acceptance.n_bad) },
    { key: 'consumed', label: 'Tokens consumed', cell: (m) => count(m.tokens_consumed) },
    { key: 'measured', label: 'Tokens measured', cell: tokensMeasured },
    { key: 'params', label: 'Parameters', cell: (m) => count(m.parameters) },
  ];
  return (
    <Panel title="After the stream, from a fresh state: lasting change" subtitle="After minus before on identical inputs; negative is improvement">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-base">
          <caption className="sr-only">Lasting change per baseline mode after the experience stream</caption>
          <thead>
            <tr className="border-b border-edge text-left align-bottom">
              <th scope="col" className={TH}>Measure</th>
              {set.modes.map((m) => (
                <th key={m.mode} scope="col" className="px-3 py-2 text-right">
                  <span className="block text-base font-semibold text-ink-primary">{MODE_LABEL[m.mode] ?? m.mode}</span>
                  <span className="block text-xs font-normal text-ink-secondary">{modeRole(m.mode, set.learner)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className="border-b border-edge">
                <th scope="row" className="px-3 py-2 text-left font-normal text-ink-primary">{r.label}</th>
                {set.modes.map((m) => (
                  <td key={m.mode} data-mode={m.mode} className="px-3 py-2 text-right font-mono text-ink-primary">{r.cell(m)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {set.missing_modes.length ? <p className="mt-3 text-base text-ink-secondary">Not archived: {set.missing_modes.map((m) => MODE_LABEL[m] ?? m).join(', ')}.</p> : null}
    </Panel>
  );
}

const READING_GUIDE =
  'If decay only matches the full candidate, timescale adaptation explains the gain; if no meta-gradient matches it, meta-training is not needed; if the delta-rule baseline matches it, coupling learning to the dynamics does not help.';

function AblationTable({ set }: { set: AblationSet }) {
  const policies = set.variants[0]?.before.policies.map((p) => p.policy) ?? [];
  const rest = 6 + policies.length; // every column after the variant name
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[1080px] border-collapse text-base">
        <caption className="sr-only">Coordinate ablation: within-episode scores per variant, each against its own off-intervention</caption>
        <thead>
          <tr className="border-b border-edge text-left align-bottom">
            <th scope="col" className={TH}>Variant</th>
            <th scope="col" className={`${TH} text-right`}>Parameters</th>
            <th scope="col" className={`${TH} text-right`}>s/step</th>
            {policies.map((p) => (
              <th key={p} scope="col" className={`${TH} text-right`}>{p}: with / without</th>
            ))}
            <th scope="col" className={`${TH} text-right`}>Training dist.: with / without</th>
            <th scope="col" className={TH}>Adaptation speed</th>
            <th scope="col" className={`${TH} text-right`}>η per layer</th>
            <th scope="col" className={TH}>Without means</th>
          </tr>
        </thead>
        <tbody>
          {set.variants.map((v) => (
            <tr key={v.variant} className="border-b border-edge align-top">
              <th scope="row" className="px-3 py-2 text-left">
                <span className="block font-semibold text-ink-primary">{VARIANT_LABEL[v.variant] ?? v.variant}</span>
                <span className="block font-mono text-xs font-normal text-ink-secondary">{v.variant}</span>
                {v.current ? null : <StateWord state="stale" />}
              </th>
              <td className="px-3 py-2 text-right font-mono text-ink-secondary">{count(v.parameters)}</td>
              <td className="px-3 py-2 text-right font-mono text-ink-secondary">{num(v.s_per_step)}</td>
              {policies.map((p) => {
                const pair = v.before.policies.find((x) => x.policy === p);
                return (
                  <td key={p} className="whitespace-nowrap px-3 py-2 text-right font-mono">
                    <span className="font-semibold text-ink-primary">{num(pair?.adapt)}</span>
                    <span className="text-ink-secondary"> / {num(pair?.no_adapt)}</span>
                  </td>
                );
              })}
              <td className="whitespace-nowrap px-3 py-2 text-right font-mono">
                <span className="font-semibold text-ink-primary">{num(v.before.train.adapt)}</span>
                <span className="text-ink-secondary"> / {num(v.before.train.no_adapt)}</span>
              </td>
              <td className="px-3 py-2 text-ink-secondary">{speedLabel(v.before.speed)}</td>
              <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-ink-secondary">
                {v.eta_per_layer ? v.eta_per_layer.map((e) => num(e)).join(', ') : 'n/a'}
              </td>
              <td className="px-3 py-2 text-ink-secondary">{v.no_adapt_label ?? 'not recorded'}</td>
            </tr>
          ))}
          {set.missing.map((name) => (
            <tr key={name} className="border-b border-edge">
              <th scope="row" className="px-3 py-2 text-left">
                <span className="block font-semibold text-ink-primary">{VARIANT_LABEL[name] ?? name}</span>
                <span className="block font-mono text-xs font-normal text-ink-secondary">{name}</span>
              </th>
              <td colSpan={rest} className="px-3 py-2 text-ink-secondary">
                <StateWord state="not archived" />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AblationPanel({ index, setId, onSelect }: { index: LearningIndex; setId: string | null; onSelect: (id: string) => void }) {
  const sets = index.sets.filter((s): s is AblationSet => s.kind === 'ablation');
  const selected = sets.find((s) => s.id === setId) ?? sets[0];
  const st = ablationState(index, setId);
  return (
    <Panel
      title="Coordinate ablation"
      subtitle="Each variant trained from scratch, then scored before any stream: temporary adaptation only"
      actions={st.state === 'archived' ? null : <StateWord state={st.state === 'stale' ? 'stale' : 'not yet archived'} />}
    >
      {sets.length > 1 ? <SetPicker label="Ablation sets" sets={sets} active={selected.id} onSelect={onSelect} /> : null}
      {st.set === null ? (
        <p className="text-base text-ink-secondary">The ablation appears here once its results are archived beside this report.</p>
      ) : (
        <div className="space-y-3">
          <p className="max-w-4xl text-base text-ink-secondary">{READING_GUIDE}</p>
          <p className="text-base text-ink-secondary">
            Commit <span className="font-mono text-ink-primary">{st.set.execution_commit ?? 'n/a'}</span>, contract{' '}
            <span className="font-mono text-ink-primary">{st.set.contract_version ?? st.set.contract_versions.join(' and ')}</span>
            {st.state === 'stale' ? `; the current contract is ${index.current_contract_version}` : ''}.
          </p>
          <AblationTable set={st.set} />
        </div>
      )}
    </Panel>
  );
}

function SetPicker({ sets, active, onSelect, label = 'Report sets' }: { sets: { id: string }[]; active: string; onSelect: (id: string) => void; label?: string }) {
  return (
    <nav aria-label={label} className="flex flex-wrap gap-2">
      {sets.map((s) => (
        <button
          key={s.id}
          type="button"
          aria-current={s.id === active ? 'page' : undefined}
          onClick={() => onSelect(s.id)}
          className={`rounded border px-3 py-1.5 text-base font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
            s.id === active ? 'border-accent bg-accent-soft text-accent' : 'border-edge-strong bg-surface-raised text-ink-secondary hover:text-ink-primary'
          }`}
        >
          {s.id}
        </button>
      ))}
    </nav>
  );
}

export function LearningView({ setId, ablationId, onSelect, onSelectAblation }: {
  setId: string | null;
  ablationId: string | null;
  onSelect: (id: string) => void;
  onSelectAblation: (reportId: string, ablationId: string) => void;
}) {
  const [attempt, setAttempt] = useState(0);
  const { data, error } = useAsync(() => loadLearning(), [attempt]);
  if (error) {
    return (
      <Empty title="Learning results could not be loaded." detail={error}>
        <Button onClick={() => setAttempt((a) => a + 1)}>Retry</Button>
      </Empty>
    );
  }
  if (!data) return <p className="py-8 text-base text-ink-secondary">Loading learning results…</p>;
  const sets = reportSets(data);
  const set = sets.find((s) => s.id === setId) ?? sets[0];
  return (
    <div className="space-y-4">
      {sets.length > 1 ? <SetPicker sets={sets} active={set.id} onSelect={onSelect} /> : null}
      {set ? (
        <>
          <Identity set={set} />
          <BeforePanel set={set} />
          <AfterPanel set={set} />
        </>
      ) : (
        <Empty title="No learning-contract report is archived yet." />
      )}
      <AblationPanel index={data} setId={ablationId} onSelect={(id) => onSelectAblation(set?.id ?? id, id)} />
    </div>
  );
}
