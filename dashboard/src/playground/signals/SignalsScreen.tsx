import { useState } from 'react';
import { LineChartPanel } from '../../components/charts/LineChartPanel';
import { Empty } from '../../components/panels/Empty';
import { Panel, StatTile } from '../../components/panels/Panel';
import { DECISION_COLOR, DECISION_GLYPH, DECISION_LABEL, chunkSource, firedSignals, fmt, fmtInt, isFinite_, presentFields, wouldIntervene } from '../format';
import { useStore } from '../store';
import type { SessionState, TransactionRecord } from '../types';

const SERIES_A = '#58a6ff';
const SERIES_B = '#3fd17a';
const SERIES_C = '#f0b429';
const SERIES_D = '#c792ea';

interface ChartSpec {
  field: string;
  title: string;
  color: string;
  log?: boolean;
}

// Each chart appears only when the backend produced the field (any finite value in the records).
const CHARTS: ChartSpec[] = [
  { field: 'chunk_loss', title: 'Chunk loss (nats/token)', color: SERIES_A },
  { field: 'surprise_mean', title: 'Inner surprise (reconstruction error)', color: SERIES_D },
  { field: 'beta_mean', title: 'Inner step size', color: SERIES_C, log: true },
  { field: 'write_norm_sum', title: 'Write norm (sum over the chunk)', color: SERIES_B },
  { field: 'canary_delta_coherence', title: 'Canary Δ coherence', color: SERIES_D },
  { field: 'canary_alignment', title: 'Canary gradient alignment', color: SERIES_C },
];

function rows(transactions: TransactionRecord[]) {
  return transactions.map((tx) => {
    const s = tx.signals as unknown as Record<string, unknown>;
    const row: Record<string, number | null | string> = { chunk: tx.index };
    for (const c of CHARTS) row[c.field] = isFinite_(s[c.field]) ? (s[c.field] as number) : null;
    row.proposed = isFinite_(tx.signals.delta_norm) ? tx.signals.delta_norm : null;
    row.accepted = isFinite_(tx.accepted?.delta_norm) ? tx.accepted.delta_norm : null;
    return row;
  });
}

function FastWeightPanel({ state }: { state: SessionState | null }) {
  if (!state) return null;
  if (state.kind === 'plastic' && state.layers) {
    return (
      <Panel title="Memory state" subtitle={`per layer · position ${fmtInt(state.pos)}`}>
        <ul className="grid gap-2 sm:grid-cols-2">
          {state.layers.map((l, i) => (
            <li key={i} className="rounded border border-edge bg-surface-overlay px-3 py-2 font-mono text-xs text-ink-secondary">
              <span className="text-ink-muted">layer {i}</span> · ‖S‖ {l.s_norm_per_head.map((v) => fmt(v, 2)).join(' ')} · ‖h‖ {fmt(l.h_norm, 2)} · drift {fmt(l.drift_from_anchor, 2)}
            </li>
          ))}
        </ul>
      </Panel>
    );
  }
  const units = state.units ?? [];
  if (units.length === 0) return null;
  const label = state.kind === 'fast_weight' ? 'Fast weights' : 'Recurrent memory';
  const per = state.kind === 'fast_weight' ? 4 : 1; // TTT reports W1, b1, W2, b2 per layer
  const layers = Math.ceil(units.length / per);
  const max = Math.max(1e-9, ...units.map((u) => u.recurrent_norm ?? 0));
  return (
    <Panel title={`${label} · ${units.length} units`} subtitle={`‖·‖ total ${fmt(state.recurrent_norm_total, 4)} · position ${fmtInt(state.pos)} · bar = norm, tick = drift from anchor`}>
      <ol role="list" aria-label={`${label} norms per layer`} className="flex h-28 items-end gap-px">
        {Array.from({ length: layers }, (_, l) => {
          const group = units.slice(l * per, (l + 1) * per);
          const norm = Math.sqrt(group.reduce((a, u) => a + (u.recurrent_norm ?? 0) ** 2, 0));
          const drift = group.some((u) => u.drift_from_anchor === null) ? null : Math.sqrt(group.reduce((a, u) => a + (u.drift_from_anchor ?? 0) ** 2, 0));
          const missing = group.some((u) => u.recurrent_norm === null);
          const frac = Math.min(1, norm / (max * Math.sqrt(per)));
          return (
            <li
              key={l}
              title={`layer ${l}: norm ${fmt(norm)} · drift ${drift === null ? 'n/a' : fmt(drift)}`}
              className="relative flex flex-1 items-end"
              aria-label={`layer ${l}: norm ${fmt(norm)}, drift ${drift === null ? 'unavailable' : fmt(drift)}`}
            >
              <div className="w-full rounded-sm" style={{ height: `${Math.max(3, Math.round(frac * 100))}%`, backgroundColor: missing ? '#48545f' : SERIES_A }} />
              {drift !== null && norm > 0 ? (
                <div aria-hidden className="absolute inset-x-0 border-t-2 border-status-scale" style={{ bottom: `${Math.min(100, Math.round((drift / norm) * 100))}%` }} />
              ) : null}
            </li>
          );
        })}
      </ol>
    </Panel>
  );
}

function ChunkTable({ transactions }: { transactions: TransactionRecord[] }) {
  const [open, setOpen] = useState<number | null>(null);
  const items = transactions.slice().sort((a, b) => b.index - a.index).slice(0, 60);
  const hasSurprise = presentFields(transactions, ['surprise_mean']).length > 0;
  const hasWrite = presentFields(transactions, ['write_norm_sum']).length > 0;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-left font-mono text-xs">
        <thead className="text-label uppercase tracking-wide text-ink-muted">
          <tr>
            <th className="py-1.5 pr-3">#</th>
            <th className="py-1.5 pr-3">decision</th>
            <th className="py-1.5 pr-3">src</th>
            <th className="py-1.5 pr-3">tok</th>
            <th className="py-1.5 pr-3">loss</th>
            {hasSurprise ? <th className="py-1.5 pr-3">surprise</th> : null}
            {hasWrite ? <th className="py-1.5 pr-3">write</th> : null}
            <th className="py-1.5 pr-3">proposed Δ</th>
            <th className="py-1.5 pr-3">accepted Δ</th>
            <th className="py-1.5 pr-3">fired</th>
          </tr>
        </thead>
        <tbody className="text-ink-primary">
          {items.map((tx) => {
            const would = wouldIntervene(tx);
            const fired = firedSignals(tx);
            const src = chunkSource(tx);
            return (
              <>
                <tr key={tx.index} className="cursor-pointer border-t border-edge hover:bg-surface-overlay" onClick={() => setOpen(open === tx.index ? null : tx.index)}>
                  <td className="py-1.5 pr-3 text-ink-muted">{tx.index}</td>
                  <td className="py-1.5 pr-3">
                    <span style={{ color: DECISION_COLOR[tx.decision.kind] }}>{DECISION_GLYPH[tx.decision.kind]} {DECISION_LABEL[tx.decision.kind]}</span>
                    {would ? <span className="ml-2 text-ink-secondary">would {DECISION_LABEL[tx.requested.kind].toLowerCase()}</span> : null}
                    {tx.signals.cusum_alarm ? <span className="ml-2 text-status-scale">CUSUM</span> : null}
                  </td>
                  <td className="py-1.5 pr-3 text-ink-secondary">{src === 'prompt' ? '▲ prompt' : src === 'model' ? '● model' : src}</td>
                  <td className="py-1.5 pr-3">{tx.signals.n_tokens}</td>
                  <td className="py-1.5 pr-3">{fmt(tx.signals.chunk_loss)}</td>
                  {hasSurprise ? <td className="py-1.5 pr-3">{fmt(tx.signals.surprise_mean)}</td> : null}
                  {hasWrite ? <td className="py-1.5 pr-3">{fmt(tx.signals.write_norm_sum)}</td> : null}
                  <td className="py-1.5 pr-3">{fmt(tx.signals.delta_norm)}</td>
                  <td className="py-1.5 pr-3">{fmt(tx.accepted?.delta_norm)}</td>
                  <td className="py-1.5 pr-3 text-ink-secondary">{fired.length ? fired.join(', ') : '—'}</td>
                </tr>
                {open === tx.index ? (
                  <tr key={`${tx.index}-detail`} className="border-t border-edge bg-surface-inset">
                    <td colSpan={10} className="px-3 py-2 text-ink-secondary">
                      <div className="grid gap-x-6 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
                        <div>z: {Object.entries(tx.signals.z ?? {}).filter(([, v]) => isFinite_(v)).map(([k, v]) => `${k} ${fmt(v, 2)}`).join(' · ') || 'none'}</div>
                        <div>reasons: {tx.decision.reasons.join('; ') || '—'}</div>
                        <div>requested: {tx.requested.reasons.join('; ') || '—'}</div>
                        <div>eligible: {tx.eligible === undefined ? 'n/a' : String(tx.eligible)} · read-only: {String(tx.read_only)}{tx.read_only_reason ? ` (${tx.read_only_reason})` : ''}</div>
                        <div>budget: used {fmt(tx.signals.budget_used)} · remaining {tx.signals.budget_remaining === null ? 'uncapped' : fmt(tx.signals.budget_remaining)}</div>
                        <div>positions {tx.pos_start}–{tx.pos_end} · {fmt(tx.seconds, 2)} s</div>
                        {isFinite_(tx.signals.canary_delta_coherence) ? <div>canary Δcoherence {fmt(tx.signals.canary_delta_coherence)} · Δpoison {fmt(tx.signals.canary_delta_poison)} · alignment {fmt(tx.signals.canary_alignment)}</div> : null}
                        {isFinite_(tx.signals.beta_mean) ? <div>step {fmt(tx.signals.beta_mean)} · alpha {fmt(tx.signals.alpha_mean)}</div> : null}
                      </div>
                    </td>
                  </tr>
                ) : null}
              </>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function SignalsScreen() {
  const detail = useStore((s) => s.detail);
  const transactions = useStore((s) => s.transactions);
  const state = useStore((s) => s.state);
  const stale = useStore((s) => s.stale);
  if (!detail) return <Empty title="No text session selected." detail="Choose a session above, or create one under Sessions." />;
  if (transactions.length === 0) return <Empty title="No chunks yet." detail="Send a message in Chat; every chunk it produces is measured here." />;
  const data = rows(transactions);
  const present = new Set(presentFields(transactions, CHARTS.map((c) => c.field)));
  const s = detail.summary;
  const meta = detail.meta;
  return (
    <div className="space-y-4">
      {stale ? <p role="status" className="text-xs text-status-scale">Showing the last data received; the connection failed since.</p> : null}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <StatTile label="Position" value={fmtInt(s.pos)} />
        <StatTile label="Chunks" value={fmtInt(s.n_transactions)} />
        <StatTile label="Committed" value={fmtInt(meta.commits)} tone="commit" />
        <StatTile label="Rolled back" value={fmtInt(meta.rollbacks)} tone={meta.rollbacks > 0 ? 'rollback' : 'default'} />
        <StatTile label="Scaled / projected / read-only" value={`${fmtInt(meta.scales)} / ${fmtInt(meta.projects)} / ${fmtInt(meta.readonly)}`} />
      </div>
      <Panel title="Fast-weight change" subtitle="proposed (measured before the decision) and accepted (measured after it), effective coordinates">
        <LineChartPanel
          data={data}
          xKey="chunk"
          series={[
            { key: 'proposed', label: 'Proposed Δ', color: SERIES_C },
            { key: 'accepted', label: 'Accepted Δ', color: SERIES_B },
          ]}
          xLabel="chunk"
          ariaLabel="Proposed and accepted fast-weight change per chunk"
        />
      </Panel>
      <div className="grid gap-4 lg:grid-cols-2">
        {CHARTS.filter((c) => present.has(c.field)).map((c) => (
          <Panel key={c.field} title={c.title}>
            <LineChartPanel data={data} xKey="chunk" series={[{ key: c.field, label: c.title, color: c.color }]} logScale={c.log} xLabel="chunk" showLegend={false} ariaLabel={`${c.title} per chunk`} />
          </Panel>
        ))}
      </div>
      <FastWeightPanel state={state} />
      <Panel title="Chunks" subtitle="newest first · click a row for z-scores, reasons, budget and canaries">
        <ChunkTable transactions={transactions} />
      </Panel>
    </div>
  );
}
