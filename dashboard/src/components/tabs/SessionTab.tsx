import { useEffect, useMemo } from 'react';
import { useStore } from '../../store';
import type { TransactionRecord } from '../../api/types';
import { UNAVAILABLE, calibrationDisplay, fmt, fmtInt, fmtPercent, fmtThreshold, isNum } from '../../utils/formatting';
import { LineChartPanel, type ReferenceSpec } from '../charts';
import { DecisionBadge, Empty, KeyValue, Panel, StatTile } from '../panels';
import { CalibratedRates, ObservedIntervention } from '../panels/RatePanel';
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
  { key: 'log_delta_norm', label: 'log ‖Δ‖', color: '#f0b429', thresholdKey: 'log_delta_norm', note: 'Size of the proposed state change, before the decision.' },
];

function BudgetMeter({ used, total }: { used: number; total: number | null }) {
  if (!isNum(total) || total <= 0) {
    return (
      <KeyValue
        rows={[
          { label: 'Budget used', value: fmt(used, 4) },
          { label: 'Session budget', value: total === null ? UNAVAILABLE : 'unbounded', note: 'no budget_session set' },
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

/**
 * Which empty/populated state the canary panel is in. Absence of measurements is
 * a fact about this session's traffic, not about the model: only an explicit
 * `has_canary === false` licenses the "no canary suite" claim. When the suite is
 * present (or its presence is not yet known) but nothing has been scored, the
 * honest statement is that no measurements are recorded yet.
 */
export type CanaryPanelState = 'populated' | 'no-suite' | 'no-measurements';

export function canaryPanelState(
  hasMeasurements: boolean,
  hasCanarySuite: boolean | null | undefined,
): CanaryPanelState {
  if (hasMeasurements) return 'populated';
  if (hasCanarySuite === false) return 'no-suite';
  return 'no-measurements';
}

function CanaryPanel({
  transactions,
  hasCanarySuite,
}: {
  transactions: TransactionRecord[];
  hasCanarySuite: boolean | null;
}) {
  // Three curves, not two. `signals.*_after` is the PROPOSED update's effect,
  // measured before the decision; `accepted.*_after` is what was actually
  // committed. On a rolled-back chunk they diverge, and that gap is the whole
  // point of the harness.
  const rows = transactions.map((tx) => ({
    chunk: tx.index,
    coherence_before: tx.signals.canary_coherence_before,
    coherence_proposed: tx.signals.canary_coherence_after,
    coherence_accepted: tx.accepted?.canary_coherence_after ?? null,
    poison_before: tx.signals.canary_poison_before,
    poison_proposed: tx.signals.canary_poison_after,
    poison_accepted: tx.accepted?.canary_poison_after ?? null,
  }));
  const hasMeasurements = rows.some((r) => isNum(r.coherence_before) || isNum(r.poison_before));
  const state = canaryPanelState(hasMeasurements, hasCanarySuite);

  if (state === 'no-suite') {
    return (
      <Panel title="Canary suite" subtitle="Coherence must not get worse; poison must not get better.">
        <Empty
          title="This model has no canary suite."
          detail="Without a calibration there are no canary probes, so every canary number on this session is unavailable, not zero. Calibrating the model builds the suite."
          command="uv run plastic calibrate <model_id> --data artifacts/data/wikitext"
        />
      </Panel>
    );
  }

  if (state === 'no-measurements') {
    return (
      <Panel title="Canary suite" subtitle="Coherence must not get worse; poison must not get better.">
        <Empty
          title="No canary measurements recorded yet."
          detail="No canary probes have been scored on this session's chunks so far. Feed the session and each chunk is scored before and after its update; the numbers are unavailable here only because none have been taken yet."
          command={'uv run plastic chat <session_id> "a first prompt"'}
        />
      </Panel>
    );
  }

  return (
    <Panel
      title="Canary suite, proposed against accepted"
      subtitle="Read-only probes scored before the chunk, after the proposed update, and after what was actually committed."
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div>
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">Coherence loss</p>
          <LineChartPanel
            data={rows}
            xKey="chunk"
            series={[
              { key: 'coherence_before', label: 'before', color: '#94a3b4', dashed: true },
              { key: 'coherence_proposed', label: 'after proposed', color: '#f0b429', dashed: true },
              { key: 'coherence_accepted', label: 'after accepted', color: '#3fd17a' },
            ]}
            height={190}
            xLabel="chunk"
            ariaLabel="Canary coherence loss per chunk, before the chunk, after the proposed update, and after the accepted update"
          />
          <p className="mt-1 text-micro text-ink-muted">
            A rise in the proposed curve is what triggers a rollback. The accepted curve is what the session lives with.
          </p>
        </div>
        <div>
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">Poison loss</p>
          <LineChartPanel
            data={rows}
            xKey="chunk"
            series={[
              { key: 'poison_before', label: 'before', color: '#94a3b4', dashed: true },
              { key: 'poison_proposed', label: 'after proposed', color: '#f0b429', dashed: true },
              { key: 'poison_accepted', label: 'after accepted', color: '#ff6b6b' },
            ]}
            height={190}
            xLabel="chunk"
            ariaLabel="Canary poison loss per chunk, before the chunk, after the proposed update, and after the accepted update"
          />
          <p className="mt-1 text-micro text-ink-muted">
            A fall means the model started liking the payload. A missing accepted curve means no canary suite.
          </p>
        </div>
      </div>
    </Panel>
  );
}

function EvidenceColumn({
  title,
  caption,
  tone,
  rows,
}: {
  title: string;
  caption: string;
  tone: 'proposed' | 'accepted';
  rows: Array<{ label: string; value: string; note?: string }>;
}) {
  const border = tone === 'proposed' ? 'border-status-scale' : 'border-status-commit';
  const ink = tone === 'proposed' ? 'text-status-scale' : 'text-status-commit';
  return (
    <div className={`rounded border-l-2 ${border} bg-surface-overlay px-3 py-2.5`}>
      <p className={`text-label font-semibold uppercase tracking-wide ${ink}`}>{title}</p>
      <p className="mb-2 mt-0.5 text-micro text-ink-secondary">{caption}</p>
      <KeyValue rows={rows} />
    </div>
  );
}

function SelectedChunk({ tx, thresholds }: { tx: TransactionRecord; thresholds: Record<string, number | null> | null }) {
  const s = tx.signals;
  const a = tx.accepted;
  const z = s.z ?? {};
  const overridden = tx.requested.kind !== tx.decision.kind;

  // A threshold key that is absent is unavailable; a key whose value is null is
  // an unbounded signal. Neither may render as a number.
  const zRow = (name: string) => ({
    label: `z ${name}`,
    value: fmt(z[name], 2),
    note: thresholds && name in thresholds ? `threshold ${fmtThreshold(thresholds[name], 3)}` : 'no calibrated threshold',
  });

  return (
    <Panel
      title={`Chunk ${tx.index}`}
      subtitle={`Positions ${tx.pos_start} to ${tx.pos_end}, ${fmtInt(s.n_tokens)} tokens, ${fmt(tx.seconds, 3)} s`}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          {overridden ? (
            <>
              <span className="text-micro text-ink-muted">requested</span>
              <DecisionBadge kind={tx.requested.kind} size="sm" />
              <span aria-hidden className="text-micro text-ink-muted">then</span>
            </>
          ) : null}
          <span className="text-micro text-ink-muted">applied</span>
          <DecisionBadge kind={tx.decision.kind} suffix={tx.decision.kind === 'scale' ? `beta x${fmt(tx.decision.scale, 3)}` : undefined} />
        </div>
      }
    >
      {overridden ? (
        <div className="mb-4 rounded border border-status-scale bg-surface-overlay px-3 py-2.5">
          <p className="text-sm font-semibold text-status-scale">
            The policy asked for {tx.requested.kind}; the runner applied {tx.decision.kind}.
          </p>
          <div className="mt-2 grid gap-3 sm:grid-cols-2">
            <div>
              <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">Requested, and why</p>
              <ReasonList reasons={tx.requested.reasons} />
            </div>
            <div>
              <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">Applied, and why</p>
              <ReasonList reasons={tx.decision.reasons} />
            </div>
          </div>
        </div>
      ) : (
        <div className="mb-4">
          <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">
            Reasons recorded ({tx.decision.kind}, as requested)
          </p>
          <ReasonList reasons={tx.decision.reasons} />
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <EvidenceColumn
          title="Proposed"
          caption="Measured on the provisional state, before the decision. This is what the chunk tried to do."
          tone="proposed"
          rows={[
            { label: 'Update norm ‖Δ‖', value: fmt(s.delta_norm) },
            { label: 'log ‖Δ‖', value: fmt(s.log_delta_norm) },
            { label: 'Write norm sum', value: fmt(s.write_norm_sum) },
            { label: 'Canary coherence after', value: fmt(s.canary_coherence_after) },
            { label: 'Canary Δ coherence', value: fmt(s.canary_delta_coherence) },
            { label: 'Canary Δ poison', value: fmt(s.canary_delta_poison) },
          ]}
        />
        <EvidenceColumn
          title="Accepted"
          caption="Measured on the committed state, after the decision. This is what the session actually learned."
          tone="accepted"
          rows={[
            { label: 'Update norm ‖Δ‖', value: fmt(a?.delta_norm), note: tx.decision.kind === 'rollback' ? 'nothing was learned' : undefined },
            { label: 'Budget charged', value: fmt(a?.budget_charge) },
            { label: 'Budget remaining', value: a?.budget_remaining === null ? 'unbounded' : fmt(a?.budget_remaining) },
            { label: 'Canary coherence after', value: fmt(a?.canary_coherence_after) },
            { label: 'Canary Δ coherence', value: fmt(a?.canary_delta_coherence) },
            { label: 'Canary Δ poison', value: fmt(a?.canary_delta_poison) },
          ]}
        />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <p className="mb-1.5 text-label font-semibold uppercase tracking-wide text-ink-muted">
            Chunk measurements (proposed)
          </p>
          <KeyValue
            rows={[
              { label: 'Chunk loss', value: fmt(s.chunk_loss) },
              { label: 'Surprise mean', value: fmt(s.surprise_mean) },
              { label: 'Surprise max', value: fmt(s.surprise_max) },
              { label: 'β mean', value: fmt(s.beta_mean) },
              { label: 'α mean', value: fmt(s.alpha_mean) },
              { label: 'Fisher update', value: fmt(s.fisher_update) },
              { label: 'Fisher drift', value: fmt(s.fisher_drift) },
              { label: 'Canary alignment', value: fmt(s.canary_alignment, 3), note: 'cos(Δ, g_C)' },
              { label: 'Compression ratio', value: fmt(s.compression_ratio, 3), note: 'display only, never gates' },
            ]}
          />
        </div>
        <div>
          <p className="mb-1.5 text-label font-semibold uppercase tracking-wide text-ink-muted">
            Robust z against the calibrated thresholds
          </p>
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
      </div>

      {s.delta_norm_per_layer.length > 0 ? (
        <div className="mt-4">
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">
            Proposed ‖Δ‖ per layer
          </p>
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

function ReasonList({ reasons }: { reasons: string[] }) {
  if (reasons.length === 0) {
    return <p className="mt-1 text-micro text-ink-secondary">No reason recorded: every signal stayed inside its threshold.</p>;
  }
  return (
    <ul className="mt-1 flex flex-wrap gap-1.5">
      {reasons.map((reason, i) => (
        <li key={`${reason}-${i}`} className="rounded border border-edge bg-surface-inset px-1.5 py-0.5 font-mono text-micro text-ink-secondary">
          {reason}
        </li>
      ))}
    </ul>
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

  // modelDetail is one shared slot; another tab may have left a different
  // model in it. Nothing from it is drawn until it is this session's model.
  const model = modelDetail?.record.model_id === modelId ? modelDetail : null;
  // Gate the ACTIVE calibration display on the session's own installed state, not the model's saved
  // artifact: a rejected (different model / unsigned) or absent calibration must not draw active
  // policy threshold lines or claim active rates, even though the model still holds a saved artifact.
  const calState = calibrationDisplay(sessionDetail?.summary?.calibration);
  const thresholds = calState.active ? (model?.calibration?.thresholds ?? null) : null;
  const transactions = sessionDetail?.transactions ?? [];

  const signalRows = useMemo(
    () =>
      transactions.map((tx) => ({
        chunk: tx.index,
        chunk_loss: tx.signals.chunk_loss,
        surprise_mean: tx.signals.surprise_mean,
        beta_mean: tx.signals.beta_mean,
        log_delta_norm: tx.signals.log_delta_norm,
        delta_proposed: tx.signals.delta_norm,
        delta_accepted: tx.accepted?.delta_norm ?? null,
      })),
    [transactions],
  );

  // How many chunks were overridden after the policy asked for something else.
  const overriddenCount = transactions.filter((tx) => tx.requested.kind !== tx.decision.kind).length;

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
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <StatTile label="Position" value={fmtInt(summary.pos)} hint={`${fmtInt(summary.pending)} tokens pending`} />
        <StatTile label="Transactions" value={fmtInt(summary.n_transactions)} hint={`chunk size ${fmtInt(model?.config.chunk)}`} />
        <StatTile label="Commits" value={fmtInt(counts.commit ?? 0)} tone="commit" hint="of the last 100 chunks" />
        <StatTile label="Rollbacks" value={fmtInt(counts.rollback ?? 0)} tone="rollback" hint="of the last 100 chunks" />
        <StatTile
          label="Overridden"
          value={fmtInt(overriddenCount)}
          tone={overriddenCount > 0 ? 'scale' : 'default'}
          hint="policy asked, runner applied something else"
        />
        <StatTile
          label="Mode"
          value={summary.read_only ? 'Read-only' : 'Learning'}
          tone={summary.read_only ? 'rollback' : 'commit'}
          hint={summary.read_only_reason ?? 'writes permitted (a write can still be rolled back)'}
        />
        <StatTile
          label="Backend"
          value={summary.backend ?? 'plastic'}
          hint={
            summary.signals_available && summary.signals_available.length > 0
              ? `decision signals: ${summary.signals_available.join(', ')}`
              : 'the fast-memory backend driving this session'
          }
        />
        <StatTile
          label="Calibration"
          value={calState.label}
          tone={calState.active ? 'commit' : calState.label.startsWith('Rejected') ? 'rollback' : 'default'}
          hint={calState.detail}
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
              ? 'Dashed lines are the calibrated thresholds installed on this session.'
              : `${calState.detail} The policy falls back to robust z-scores.`
          }
        >
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
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
                    ariaLabel={`${chart.label} per chunk over this session`}
                    onPointClick={(i) => setSelected(signalRows[i]?.chunk ?? null)}
                  />
                  <p className="mt-1 text-micro text-ink-muted">{chart.note}</p>
                </div>
              );
            })}
          </div>

          <div className="mt-5">
            <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">
              Update norm ‖Δ‖, proposed against accepted
            </p>
            <LineChartPanel
              data={signalRows}
              xKey="chunk"
              series={[
                { key: 'delta_proposed', label: 'proposed', color: '#f0b429', dashed: true },
                { key: 'delta_accepted', label: 'accepted', color: '#3fd17a' },
              ]}
              height={190}
              xLabel="chunk"
              ariaLabel="Proposed against accepted update norm per chunk"
            />
            <p className="mt-1 text-micro text-ink-muted">
              The gap is what the harness refused. A rolled-back chunk proposes a large update and accepts none; a
              scaled chunk accepts the fraction it kept. Neither value is derived from the other.
            </p>
          </div>
        </Panel>
      ) : null}

      <CanaryPanel transactions={transactions} hasCanarySuite={model?.record.has_canary ?? null} />

      <Panel
        title="Calibrated rates against what this session did"
        subtitle="Three distinct quantities. None of them is evidence for another."
      >
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <CalibratedRates calibration={calState.active ? (model?.calibration ?? null) : null} />
          <ObservedIntervention
            counts={{
              n_transactions: transactions.length,
              commits: counts.commit ?? 0,
              rollbacks: counts.rollback ?? 0,
              scales: counts.scale ?? 0,
              projects: counts.project ?? 0,
              readonly: counts.readonly ?? 0,
            }}
            label={`Session ${meta.session_id}, last ${transactions.length} chunks`}
          />
        </div>
      </Panel>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
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

        <BetaHistogramPanel hist={model?.eval?.beta_hist ?? null} modelId={modelId} />
      </div>

      <LayerStatePanel state={sessionState} loading={stateLoading} />
    </div>
  );
}
