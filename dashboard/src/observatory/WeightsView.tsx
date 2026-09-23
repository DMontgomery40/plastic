import { useMemo, useState } from 'react';
import { Panel } from '../components/panels/Panel';
import { OBS } from '../components/charts/theme';
import { loadRun, loadTrajectory } from './data';
import { ARM_LABEL, isSleep, num, pct, ratio, signed } from './format';
import { LayerMap, SessionPicker, useWidth } from './LayerMap';
import { Label, OutcomeGlyph, useAsync } from './parts';
import { defaultRunId } from './RunsView';
import { logScale, rampColor, RAMP_CSS } from './scale';
import { Sparkline } from './Sparkline';
import type { DreamKept, GradientNorms, ObservatoryIndex, PerTensor, Run, SleepArm, Trajectory } from './types';

interface Props {
  index: ObservatoryIndex;
  runId: string | null;
  arm: string | null;
  onSelect: (run: string | null, arm: string | null) => void;
}

/** 4 tensors × N layers, colored on a log scale (shared when several grids are compared). */
export function PerTensorGrid({ data, scale, title }: { data: PerTensor; scale: ReturnType<typeof logScale>; title?: string }) {
  const layers = data.values.length;
  return (
    <figure className="min-w-0">
      {title ? <figcaption className="mb-1 font-mono text-xs text-ink-secondary">{title}</figcaption> : null}
      <div className="grid grid-cols-[2rem_minmax(0,1fr)] items-center gap-x-2 gap-y-0.5">
        {data.tensors.map((t, k) => (
          <div key={t} className="contents">
            <span className="text-right font-mono text-xs text-ink-muted">{t}</span>
            <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${layers}, minmax(0, 1fr))` }}>
              {data.values.map((row, l) => {
                const v = row[k];
                const tt = scale.t(v);
                return <div key={l} className="h-4 rounded-[1px]" title={`layer ${l} ${t}: ${num(v, 4)}`} style={{ background: tt === null ? OBS.inset : rampColor(tt) }} />;
              })}
            </div>
          </div>
        ))}
        <span />
        <div className="flex justify-between font-mono text-micro text-ink-muted">
          <span>layer 0</span>
          <span>layer {layers - 1}</span>
        </div>
      </div>
    </figure>
  );
}

function ScaleLegend({ scale, label }: { scale: ReturnType<typeof logScale>; label: string }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-ink-secondary">
      <span className="font-mono">{num(scale.lo, 4)}</span>
      <span className="inline-block h-2.5 w-28 rounded-sm" style={{ background: RAMP_CSS }} />
      <span className="font-mono">{num(scale.hi, 4)}</span>
      <span className="text-ink-muted">{label}, log scale</span>
    </div>
  );
}

function GradGrid({ g }: { g: GradientNorms }) {
  const scale = logScale(g.per_layer.flat());
  const layers = g.per_layer.length;
  return (
    <div className="space-y-2">
      <Sparkline values={g.total} label="gradient norm, total (before clipping)" />
      <div className="grid gap-px" style={{ gridTemplateRows: `repeat(${layers}, 6px)`, gridTemplateColumns: `repeat(${g.steps.length}, minmax(0, 1fr))`, gridAutoFlow: 'column' }}>
        {g.steps.map((_, s) =>
          Array.from({ length: layers }, (_, i) => {
            const l = layers - 1 - i;
            const tt = scale.t(g.per_layer[l][s]);
            return <div key={`${s}-${l}`} title={`step ${g.steps[s]} layer ${l}: ${num(g.per_layer[l][s], 4)}`} style={{ background: tt === null ? OBS.inset : rampColor(tt) }} />;
          })
        )}
      </div>
      <ScaleLegend scale={scale} label="per-layer gradient norm; rows are layers, top is the last layer" />
    </div>
  );
}

function DreamGains({ dream, index }: { dream: DreamKept; index: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const gains = dream.token_gain ?? [];
  const max = Math.max(1e-6, ...gains.map((v) => Math.abs(v ?? 0)));
  const h = 64;
  const bw = gains.length ? width / gains.length : 0;
  return (
    <figure className="min-w-0">
      <figcaption className="flex flex-wrap items-baseline justify-between gap-2 text-xs text-ink-secondary">
        <span>
          Dream {index + 1}: gain per reply token, teacher minus reset student
        </span>
        <span className="font-mono text-ink-muted">
          mean {signed(dream.gain, 2)} nats/token · {gains.length} tokens
        </span>
      </figcaption>
      <div ref={ref} className="mt-1">
        {width > 0 && gains.length ? (
          <svg width={width} height={h} role="img" aria-label={`Per-token gains for dream ${index + 1}: ${gains.length} tokens, largest magnitude ${num(max, 2)}`} shapeRendering="crispEdges">
            <line x1={0} x2={width} y1={h / 2} y2={h / 2} stroke={OBS.edgeStrong} />
            {gains.map((v, i) => {
              const val = v ?? 0;
              const bh = (Math.abs(val) / max) * (h / 2 - 2);
              return <rect key={i} x={i * bw + 0.5} width={Math.max(1, bw - 1)} y={val >= 0 ? h / 2 - bh : h / 2} height={Math.max(val === 0 ? 0 : 1, bh)} fill={val >= 0 ? OBS.line : OBS.lineMuted} />;
            })}
          </svg>
        ) : null}
      </div>
      <p className="mt-1 line-clamp-2 text-sm text-ink-primary">“{dream.text}”</p>
    </figure>
  );
}

function WithinOneSleep({ index, runId, arm, onSelect }: Props) {
  const runs = index.runs.filter((r) => r.arms.some((a) => a.status !== null)).slice().reverse();
  const id = runId && runs.some((r) => r.id === runId) ? runId : defaultRunId(index);
  const entry = index.runs.find((r) => r.id === id);
  const { data } = useAsync<{ run: Run; trajectory: Trajectory | null }>(
    id ? async () => ({ run: await loadRun(id), trajectory: entry?.session_trajectory ? await loadTrajectory(id) : null }) : null,
    [id]
  );
  const [session, setSession] = useState('teach');
  const arms = (data?.run.arms.filter(isSleep) ?? []) as SleepArm[];
  const chosen = arms.find((a) => a.arm === arm) ?? arms.find((a) => a.lineage.outcome === 'pulled back') ?? arms[0] ?? null;
  const w0 = chosen?.w0_relative_update ?? chosen?.anchor_relative_update ?? null;
  const w0Scale = useMemo(() => logScale(w0?.values.flat() ?? []), [w0]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3 rounded-lg border border-edge bg-surface-raised px-4 py-3">
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
        {data?.trajectory ? <SessionPicker trajectory={data.trajectory} value={session} onChange={setSession} /> : null}
      </div>

      <Panel title="During the conversation" subtitle="proposed fast-weight change per layer, for every 16-token chunk of the session Sleep read">
        {data?.trajectory ? (
          <LayerMap trajectory={data.trajectory} session={session} />
        ) : (
          <p className="text-sm text-ink-muted">{data ? 'Session log not archived for this run.' : 'Loading…'}</p>
        )}
      </Panel>

      <Panel title="During sleep" subtitle="what one consolidation arm did to the model">
        <div role="group" aria-label="Arm" className="mb-4 flex flex-wrap gap-1">
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
              <OutcomeGlyph outcome={a.lineage.outcome === 'pulled back' ? 'pulled back' : 'committed'} size={12} />
              {ARM_LABEL[a.arm]}
            </button>
          ))}
        </div>
        {chosen ? (
          <div className="grid gap-6 xl:grid-cols-2">
            <div className="space-y-2">
              <Label>Training loss</Label>
              {chosen.losses ? (
                <Sparkline values={chosen.losses} label={chosen.loss_terms ?? 'loss'} />
              ) : (
                <p className="text-sm text-ink-secondary">{chosen.method === 'anchor' ? 'Anchor interpolates W0 directly; there are no gradient steps.' : 'Not recorded.'}</p>
              )}
            </div>
            <div className="space-y-2">
              <Label>Gradient norms by layer</Label>
              {chosen.gradient_norms ? <GradGrid g={chosen.gradient_norms} /> : <p className="text-sm text-ink-secondary">{chosen.method === 'anchor' ? 'No gradient steps.' : 'Not recorded for this run.'}</p>}
            </div>
            <div className="space-y-2">
              <Label>W0 change per tensor, child against parent</Label>
              {w0 ? (
                <>
                  <PerTensorGrid data={w0} scale={w0Scale} />
                  <ScaleLegend scale={w0Scale} label="‖ΔW0‖ / ‖W0‖" />
                  {w0.total != null ? <p className="font-mono text-xs text-ink-secondary">all tensors {num(w0.total, 4)}</p> : null}
                </>
              ) : (
                <p className="text-sm text-ink-secondary">Not recorded for this run.</p>
              )}
            </div>
            <div className="space-y-3">
              <Label>Dreams kept for training</Label>
              {chosen.dreams?.kept.length ? chosen.dreams.kept.slice(0, 3).map((d, i) => <DreamGains key={i} dream={d} index={i} />) : <p className="text-sm text-ink-secondary">{chosen.method === 'dream' ? 'No dream was kept.' : 'Not a Dream arm.'}</p>}
            </div>
          </div>
        ) : null}
      </Panel>
    </div>
  );
}

interface Row {
  run: Run;
  arm: SleepArm;
}

function Dot({ value, lo, hi, limit, tone }: { value: number | null; lo: number; hi: number; limit?: number | null; tone: string }) {
  const x = (v: number) => `${((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * 100}%`;
  return (
    <div className="relative h-5">
      <div className="absolute inset-x-0 top-1/2 h-px bg-edge" />
      {lo < 0 && hi > 0 ? <div className="absolute top-0.5 bottom-0.5 w-px bg-edge-strong" style={{ left: x(0) }} /> : null}
      {limit != null ? <div className="absolute top-0 bottom-0 border-l-2 border-dashed border-ink-muted" style={{ left: x(limit) }} /> : null}
      {value !== null ? <div className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2" style={{ left: x(value), background: tone, borderColor: OBS.surface }} /> : null}
    </div>
  );
}

function AcrossRuns({ index, onPick }: { index: ObservatoryIndex; onPick: (run: string, arm: string) => void }) {
  const { data: runs } = useAsync(() => Promise.all(index.runs.map((r) => loadRun(r.id))), [index]);
  const rows: Row[] = (runs ?? []).flatMap((run) => run.arms.filter(isSleep).map((arm) => ({ run, arm })));
  const anchors = rows.filter((r) => r.arm.anchor_relative_update || r.arm.w0_relative_update);
  const anchorScale = useMemo(() => logScale(anchors.flatMap((r) => (r.arm.w0_relative_update ?? r.arm.anchor_relative_update)!.values.flat())), [anchors]);
  if (!runs) return <p className="text-sm text-ink-secondary">Loading runs…</p>;
  const dNll = (a: SleepArm) => (a.heldout_nll.before?.mean != null && a.heldout_nll.after?.mean != null ? a.heldout_nll.after.mean - a.heldout_nll.before.mean : null);
  return (
    <div className="space-y-4">
      <Panel title="Every consolidation attempt" subtitle="held-out loss and reply collapse after sleep, with the gate’s outcome">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-edge text-left text-label font-semibold uppercase tracking-wide text-ink-muted">
                <th scope="col" className="px-2 py-1.5">Run · arm</th>
                <th scope="col" className="w-[26%] px-2 py-1.5">Held-out NLL change</th>
                <th scope="col" className="w-[26%] px-2 py-1.5">Same-reply share</th>
                <th scope="col" className="px-2 py-1.5 text-right">Taught recalled</th>
                <th scope="col" className="px-2 py-1.5">Outcome</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ run, arm }) => {
                const d = dNll(arm);
                const shareCheck = arm.gate?.checks.find((c) => c.name === 'reply_cluster_share');
                const share = shareCheck?.value ?? arm.recall.totals_after?.max_cluster_share ?? null;
                const out = arm.lineage.outcome;
                const shareTone = shareCheck && shareCheck.in_force ? (shareCheck.passed ? OBS.pass : OBS.fail) : OBS.notInForce;
                const taught = arm.recall.after.by_group?.taught;
                const nllLimit = arm.gate?.checks.find((c) => c.name === 'heldout_nll_mean_rise')?.limit ?? null;
                return (
                  <tr key={`${run.id}-${arm.arm}`} className="cursor-pointer border-b border-edge hover:bg-surface-overlay" onClick={() => onPick(run.id, arm.arm)}>
                    <td className="px-2 py-1.5">
                      <button type="button" onClick={() => onPick(run.id, arm.arm)} className="text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent">
                        <span className="block font-mono text-xs text-ink-primary">{run.id}</span>
                        <span className="text-xs text-ink-secondary">{ARM_LABEL[arm.arm]}</span>
                      </button>
                    </td>
                    <td className="px-2 py-1.5">
                      <div className="grid grid-cols-[1fr_4.5rem] items-center gap-2">
                        <Dot value={d} lo={-0.5} hi={0.1} limit={nllLimit !== null && nllLimit <= 0.1 ? nllLimit : null} tone={OBS.line} />
                        <span className="text-right font-mono text-xs text-ink-primary">{signed(d)}</span>
                      </div>
                    </td>
                    <td className="px-2 py-1.5">
                      <div className="grid grid-cols-[1fr_6rem] items-center gap-2">
                        <Dot value={share} lo={0} hi={1} limit={0.25} tone={shareTone} />
                        <span className="text-right font-mono text-xs text-ink-primary">
                          {pct(share)}
                          {shareCheck && !shareCheck.in_force ? <span className="block text-micro font-normal text-ink-muted">not in force</span> : null}
                        </span>
                      </div>
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-ink-primary">{taught ? ratio(taught) : arm.recall.totals_after ? `${arm.recall.totals_after.recalled}/${arm.recall.totals_after.n_probes} all` : 'n/a'}</td>
                    <td className="px-2 py-1.5">
                      <span className={`inline-flex items-center gap-1.5 whitespace-nowrap text-xs font-semibold ${out === 'committed' ? 'text-status-commit' : 'text-status-rollback'}`}>
                        <OutcomeGlyph outcome={out === 'committed' ? 'committed' : 'pulled back'} size={12} />
                        {out === 'committed' ? 'committed' : 'pulled back'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-ink-muted">
          NLL axis −0.5 to +0.1 nats/token with each run’s limit dashed; share axis 0 to 100% with the 25% limit dashed. Gray dots: the collapse check was not in force when the run executed.
        </p>
      </Panel>
      {anchors.length ? (
        <Panel title="W0 change by layer" subtitle="relative change of each initial fast-weight tensor, on one shared scale">
          <div className="space-y-4">
            {anchors.map(({ run, arm }) => (
              <PerTensorGrid key={`${run.id}-${arm.arm}`} data={(arm.w0_relative_update ?? arm.anchor_relative_update)!} scale={anchorScale} title={`${run.id} · ${ARM_LABEL[arm.arm]}`} />
            ))}
            <ScaleLegend scale={anchorScale} label="‖ΔW0‖ / ‖W0‖" />
          </div>
        </Panel>
      ) : null}
    </div>
  );
}

export function WeightsView(props: Props) {
  const [scope, setScope] = useState<'within' | 'across'>('within');
  return (
    <div className="space-y-3">
      <div role="group" aria-label="Scope" className="inline-flex rounded border border-edge-strong bg-surface-raised p-0.5">
        {(['within', 'across'] as const).map((s) => (
          <button
            key={s}
            type="button"
            aria-pressed={scope === s}
            onClick={() => setScope(s)}
            className={`rounded px-3 py-1 text-sm font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${scope === s ? 'bg-accent-soft text-accent' : 'text-ink-secondary hover:text-ink-primary'}`}
          >
            {s === 'within' ? 'Within one sleep' : 'Across runs'}
          </button>
        ))}
      </div>
      {scope === 'within' ? (
        <WithinOneSleep {...props} />
      ) : (
        <AcrossRuns
          index={props.index}
          onPick={(run, arm) => {
            setScope('within');
            props.onSelect(run, arm);
          }}
        />
      )}
    </div>
  );
}
