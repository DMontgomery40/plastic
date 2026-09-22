import { useEffect, useMemo } from 'react';
import { useStore } from '../../store';
import type { TransactionRecord } from '../../api/types';
import { fmt, fmtInt, fmtPercent, isNum } from '../../utils/formatting';
import { LineChartPanel, type ReferenceSpec } from '../charts';
import { DecisionBadge, Empty, KeyValue, Panel, StatTile } from '../panels';
import { TransactionTimeline } from '../session/TransactionTimeline';
import { BetaHistogramPanel, LayerStatePanel } from '../session/StatePanels';

interface SignalChart {
  key: 'chunk_loss' | 'surprise_mean' | 'beta_mean' | 'log_delta_norm';
  label: string;
  color: string;
  /** Calibrated threshold key, when this signal can have one at all. */
  thresholdKey: string | null;
  note: string;
}

// beta_mean deliberately has no threshold key: calibration only produces
// thresholds for ROLLBACK_DECISION_SIGNALS, and β is never one of them.
const SIGNAL_CHARTS: SignalChart[] = [
  { key: 'chunk_loss', label: 'Chunk loss', color: '#58a6ff', thresholdKey: 'chunk_loss', note: 'The out-of-distribution signal.' },
  { key: 'surprise_mean', label: 'Surprise, mean ‖e‖', color: '#c792ea', thresholdKey: 'surprise_mean', note: 'Inner-loop prediction error.' },
  { key: 'beta_mean', label: 'Write rate β, mean', color: '#3fd17a', thresholdKey: null, note: "The model's own gate. Never thresholded." },
  { key: 'log_delta_norm', label: 'log ‖Δ‖', color: '#f0b429', thresholdKey: 'log_delta_norm', note: 'Size of the committed state change.' },
];

function BudgetMeter({ used, total }: { used: number; total: number | null }) {
  if (!isNum(total) || total <= 0) {
    return (
      <KeyValue
        rows={[
          { label: 'Budget used', value: fmt(used, 4) },
          { label: 'Session budget', value: 'unbounded', note: 'no budget_session set' },
        ]}
      />
    );
  }
  const ratio = Math.max(0, Math.min(1, used / total));
  const color = ratio > 0.9 ? '#ff6b6b' : ratio > 0.6 ? '#f0b429' : '#3fd17a';
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-ink-muted">Session budget on ‖Δ‖</span>
        <span className="font-mono text-sm text-ink-primary">
          {fmt(used, 3)} / {fmt(total, 3)}
        </span>
      </div>
      <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded border border-edge bg-surface-inset">
        <div className="h-full" style={{ width: `${ratio * 100}%`, backgroundColor: color }} />
      </div>
      <p className="mt-1 text-micro text-ink-muted">{fmtPercent(ratio, 1)} spent. At 100% the session becomes read-only until reset.</p>
    </div>
  );
}

function CanaryPanel({ transactions }: { transactions: TransactionRecord[] }) {
  const rows = transactions.map((tx) => ({
    chunk: tx.index,
    coherence_before: tx.signals.canary_coherence_before,
    coherence_after: tx.signals.canary_coherence_after,
    poison_before: tx.signals.canary_poison_before,
    poison_after: tx.signals.canary_poison_after,
  }));
  const hasCanary = rows.some((r) => isNum(r.coherence_before) || isNum(r.poison_before));

  if (!hasCanary) {
    return (
      <Panel title="Canary suite" subtitle="Coherence must not get worse; poison must not get better.">
        <Empty
          title="No canary probes ran on these chunks."
          detail="The suite is written next to the checkpoint during training and is read by every session of that model."
          command="uv run plastic calibrate <model_id> --data artifacts/data/wikitext"
        />
      </Panel>
    );
  }

  return (
    <Panel title="Canary suite" subtitle="Read-only probes, scored before and after each chunk.">
      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">Coherence loss</p>
          <LineChartPanel
            data={rows}
            xKey="chunk"
            series={[
              { key: 'coherence_before', label: 'before', color: '#58a6ff', dashed: true },
              { key: 'coherence_after', label: 'after', color: '#3fd17a' },
            ]}
            height={180}
            xLabel="chunk"
          />
          <p className="mt-1 text-micro text-ink-muted">A rise after the chunk is the rollback trigger.</p>
        </div>
        <div>
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">Poison loss</p>
          <LineChartPanel
            data={rows}
            xKey="chunk"
            series={[
              { key: 'poison_before', label: 'before', color: '#58a6ff', dashed: true },
              { key: 'poison_after', label: 'after', color: '#ff6b6b' },
            ]}
            height={180}
            xLabel="chunk"
          />
          <p className="mt-1 text-micro text-ink-muted">A fall after the chunk means the model started liking the payload.</p>
        </div>
      </div>
    </Panel>
  );
}

function SelectedChunk({ tx, thresholds }: { tx: TransactionRecord; thresholds: Record<string, number> | null }) {
  const s = tx.signals;
  const z = s.z ?? {};
  const zRow = (name: string) => {
    const value = z[name];
    return { label: `z ${name}`, value: fmt(value, 2), note: thresholds?.[name] !== undefined ? `threshold ${fmt(thresholds[name], 3)}` : undefined };
  };
  return (
    <Panel
      title={`Chunk ${tx.index}`}
      subtitle={`Positions ${tx.pos_start} to ${tx.pos_end}, ${fmtInt(s.n_tokens)} tokens, ${fmt(tx.seconds, 3)} s`}
      actions={<DecisionBadge kind={tx.decision.kind} suffix={tx.decision.kind === 'scale' ? `β×${fmt(tx.decision.scale, 3)}` : undefined} />}
    >
      <div className="grid gap-4 lg:grid-cols-3">
        <KeyValue
          rows={[
            { label: 'Chunk loss', value: fmt(s.chunk_loss) },
            { label: 'Surprise mean', value: fmt(s.surprise_mean) },
            { label: 'Surprise max', value: fmt(s.surprise_max) },
            { label: 'β mean', value: fmt(s.beta_mean) },
            { label: 'α mean', value: fmt(s.alpha_mean) },
            { label: 'Compression ratio', value: fmt(s.compression_ratio, 3), note: 'display only' },
          ]}
        />
        <KeyValue
          rows={[
            { label: '‖Δ‖ Frobenius', value: fmt(s.delta_norm) },
            { label: 'log ‖Δ‖', value: fmt(s.log_delta_norm) },
            { label: 'Write norm sum', value: fmt(s.write_norm_sum) },
            { label: 'Fisher update', value: fmt(s.fisher_update) },
            { label: 'Fisher drift', value: fmt(s.fisher_drift) },
            { label: 'Canary alignment', value: fmt(s.canary_alignment, 3), note: 'cos(Δ, g_C)' },
          ]}
        />
        <KeyValue
          rows={[
            zRow('chunk_loss'),
            zRow('surprise_mean'),
            zRow('log_delta_norm'),
            zRow('log_write_norm'),
            zRow('fisher_update'),
            { label: 'CUSUM alarm', value: s.cusum_alarm ? 'yes' : 'no' },
          ]}
        />
      </div>
      {s.delta_norm_per_layer.length > 0 ? (
        <div className="mt-4">
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">‖Δ‖ per layer</p>
          <div className="flex flex-wrap gap-2">
            {s.delta_norm_per_layer.map((v, i) => (
              <span key={i} className="rounded border border-edge bg-surface-overlay px-2 py-1 font-mono text-xs text-ink-primary">
                L{i} {fmt(v, 4)}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </Panel>
  );
}

export function SessionTab() {
  const sessionDetail = useStore((s) => s.sessionDetail);
  const sessionState = useStore((s) => s.sessionState);
  const stateLoading = useStore((s) => s.loading.sessionState);
  const sessionLoading = useStore((s) => s.loading.session);
  const selected = useStore((s) => s.selectedTransaction);
  const setSelected = useStore((s) => s.setSelectedTransaction);
  const currentSessionId = useStore((s) => s.currentSessionId);
  const modelDetail = useStore((s) => s.modelDetail);
  const loadModel = useStore((s) => s.loadModel);
  const setActiveTab = useStore((s) => s.setActiveTab);

  const modelId = sessionDetail?.meta.model_id ?? null;

  useEffect(() => {
    if (modelId && modelDetail?.record.model_id !== modelId) void loadModel(modelId);
  }, [modelId, modelDetail?.record.model_id, loadModel]);

  const thresholds = modelDetail?.calibration?.thresholds ?? null;
  const transactions = sessionDetail?.transactions ?? [];

  const signalRows = useMemo(
    () =>
      transactions.map((tx) => ({
        chunk: tx.index,
        chunk_loss: tx.signals.chunk_loss,
        surprise_mean: tx.signals.surprise_mean,
        beta_mean: tx.signals.beta_mean,
        log_delta_norm: tx.signals.log_delta_norm,
      })),
    [transactions],
  );

  if (!currentSessionId) {
    return (
      <Empty
        title="No session selected."
        detail="Pick one in the header, or create one on the Sessions tab."
        command="uv run plastic session new --model <model_id>"
      />
    );
  }

  if (!sessionDetail) {
    return sessionLoading ? (
      <p className="text-sm text-ink-secondary">Loading session {currentSessionId}…</p>
    ) : (
      <Empty title={`Session ${currentSessionId} could not be read.`} detail="The API returned no detail for it." />
    );
  }

  const summary = sessionDetail.summary;
  const meta = sessionDetail.meta;
  const selectedTx = transactions.find((t) => t.index === selected) ?? null;
  const counts = transactions.reduce<Record<string, number>>((acc, tx) => {
    acc[tx.decision.kind] = (acc[tx.decision.kind] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <StatTile label="Position" value={fmtInt(summary.pos)} hint={`${fmtInt(summary.pending)} tokens pending`} />
        <StatTile label="Transactions" value={fmtInt(summary.n_transactions)} hint={`chunk size ${fmtInt(modelDetail?.config.chunk)}`} />
        <StatTile label="Commits" value={fmtInt(counts.commit ?? 0)} tone="commit" hint="of the last 100 chunks" />
        <StatTile label="Rollbacks" value={fmtInt(counts.rollback ?? 0)} tone="rollback" hint="of the last 100 chunks" />
        <StatTile label="Drift from anchor" value={fmt(summary.drift_from_anchor, 3)} hint="‖S − S_anchor‖" />
        <StatTile
          label="Mode"
          value={summary.read_only ? 'Read-only' : 'Learning'}
          tone={summary.read_only ? 'rollback' : 'commit'}
          hint={summary.read_only_reason ?? 'writes accepted'}
        />
      </div>

      <Panel
        title="Transaction timeline"
        subtitle="One mark per chunk, in order. Click to inspect, hover for the reasons the policy recorded."
        actions={
          <span className="font-mono text-micro text-ink-muted">
            {meta.session_id} · {meta.domain} · {meta.model_id}
          </span>
        }
      >
        {transactions.length === 0 ? (
          <Empty
            title="This session has no transactions yet."
            detail="Chunks are decided at every chunk boundary; feed the session first."
            command={
              meta.domain === 'text'
                ? 'uv run plastic chat <session_id> "a first prompt"'
                : 'uv run plastic physics <session_id> --steps 256 --mu 0.12'
            }
          >
            <button
              type="button"
              onClick={() => setActiveTab(meta.domain === 'text' ? 'chat' : 'physics')}
              className="text-sm font-semibold text-accent hover:text-accent-hover"
            >
              Open the {meta.domain === 'text' ? 'Chat' : 'Physics'} tab instead
            </button>
          </Empty>
        ) : (
          <TransactionTimeline transactions={transactions} selected={selected} onSelect={setSelected} />
        )}
      </Panel>

      {selectedTx ? <SelectedChunk tx={selectedTx} thresholds={thresholds} /> : null}

      {transactions.length > 0 ? (
        <Panel
          title="Signals over the session"
          subtitle={
            thresholds
              ? 'Dashed lines are the calibrated thresholds for this model.'
              : 'No calibration on this model, so the policy falls back to robust z-scores and no thresholds are drawn.'
          }
        >
          <div className="grid gap-5 lg:grid-cols-2">
            {SIGNAL_CHARTS.map((chart) => {
              const t = chart.thresholdKey ? thresholds?.[chart.thresholdKey] : undefined;
              const refs: ReferenceSpec[] = isNum(t) ? [{ value: t, label: `τ ${fmt(t, 3)}` }] : [];
              return (
                <div key={chart.key}>
                  <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">{chart.label}</p>
                  <LineChartPanel
                    data={signalRows}
                    xKey="chunk"
                    series={[{ key: chart.key, label: chart.label, color: chart.color }]}
                    height={180}
                    xLabel="chunk"
                    references={refs}
                    showLegend={false}
                    onPointClick={(i) => setSelected(signalRows[i]?.chunk ?? null)}
                  />
                  <p className="mt-1 text-micro text-ink-muted">{chart.note}</p>
                </div>
              );
            })}
          </div>
        </Panel>
      ) : null}

      <CanaryPanel transactions={transactions} />

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Change detection and budget" subtitle="Two-sided CUSUM on log ‖Δ‖, plus the session's write budget.">
          <div className="space-y-4">
            <KeyValue
              columns={2}
              rows={[
                { label: 'CUSUM s_hi', value: fmt(summary.cusum.s_hi, 3), note: `h ${fmt(summary.cusum.h, 2)}` },
                { label: 'CUSUM s_lo', value: fmt(summary.cusum.s_lo, 3), note: `k ${fmt(summary.cusum.k, 2)}` },
                { label: 'Alarms', value: fmtInt(summary.cusum.alarms) },
                { label: 'State ‖S‖ total', value: fmt(summary.state_norms.s_norm_total, 3) },
                { label: 'State ‖h‖ total', value: fmt(summary.state_norms.h_norm_total, 3) },
                { label: 'Lineage', value: sessionDetail.lineage.join(' → ') || meta.session_id },
              ]}
            />
            <BudgetMeter used={summary.budget_used} total={summary.budget_session} />
          </div>
        </Panel>

        <BetaHistogramPanel hist={modelDetail?.eval?.beta_hist ?? null} modelId={modelId} />
      </div>

      <LayerStatePanel state={sessionState} loading={stateLoading} />
    </div>
  );
}
