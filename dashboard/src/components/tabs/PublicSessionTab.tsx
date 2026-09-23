import { useMemo } from 'react';
import { useStore } from '../../store';
import type { TransactionRecord } from '../../api/types';
import { fmt, fmtInt, isNum } from '../../utils/formatting';
import { LineChartPanel } from '../charts';
import { SERIES_COLORS } from '../charts/theme';
import { Button, DecisionBadge, Empty, Panel, StatTile } from '../panels';

const SIGNALS = [
  { key: 'chunk_loss', label: 'Chunk loss', color: SERIES_COLORS[0] },
  { key: 'surprise_mean', label: 'Memory surprise', color: SERIES_COLORS[3] },
  { key: 'beta_mean', label: 'Write rate β', color: SERIES_COLORS[1] },
  { key: 'log_delta_norm', label: 'Log state change', color: SERIES_COLORS[2] },
] as const;

export function measuredSignals(transactions: TransactionRecord[]) {
  return SIGNALS.filter(({ key }) => transactions.some((tx) => isNum(tx.signals[key])));
}

export function PublicSessionTab() {
  const currentSessionId = useStore((s) => s.currentSessionId);
  const sessionDetail = useStore((s) => s.sessionDetail);
  const loading = useStore((s) => s.loading.session);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const detail = sessionDetail?.meta.session_id === currentSessionId && sessionDetail.meta.domain === 'text' ? sessionDetail : null;
  const transactions = detail?.transactions ?? [];
  const charts = useMemo(() => measuredSignals(transactions), [transactions]);
  const rows = useMemo(() => transactions.map((tx) => ({
    chunk: tx.index,
    chunk_loss: tx.signals.chunk_loss,
    surprise_mean: tx.signals.surprise_mean,
    beta_mean: tx.signals.beta_mean,
    log_delta_norm: tx.signals.log_delta_norm,
    proposed: tx.signals.delta_norm,
    accepted: tx.accepted?.delta_norm ?? null,
  })), [transactions]);

  if (!currentSessionId) return <Empty title="No text session selected." detail="Choose a session to see its measurements." />;
  if (!detail) return loading ? <p className="text-sm text-ink-secondary">Loading measurements…</p> : <Empty title="Measurements unavailable." detail="Try reconnecting." />;

  const recent = transactions.slice(-20).reverse();

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Position" value={fmtInt(detail.summary.pos)} />
        <StatTile label="Chunks" value={fmtInt(detail.summary.n_transactions)} />
        <StatTile label="Commits" value={fmtInt(detail.meta.commits)} tone="commit" />
        <StatTile label="Rollbacks" value={fmtInt(detail.meta.rollbacks)} tone="rollback" />
      </div>

      <Panel title="Recent measurements" subtitle={`${detail.meta.session_id} · latest ${transactions.length} of ${detail.summary.n_transactions} chunks · ${detail.meta.harness.log_only ? 'Observation mode' : 'Guarded mode'}`} actions={<Button size="sm" onClick={() => setActiveTab('chat')}>Open Chat</Button>}>
        {transactions.length === 0 ? (
          <Empty title="No chunks measured yet." detail="Send a message in Chat to begin." />
        ) : (
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            {charts.map((signal) => (
              <div key={signal.key}>
                <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">{signal.label}</p>
                <LineChartPanel
                  data={rows}
                  xKey="chunk"
                  series={[{ key: signal.key, label: signal.label, color: signal.color }]}
                  height={180}
                  xLabel="chunk"
                  showLegend={false}
                  ariaLabel={`${signal.label} by chunk`}
                />
              </div>
            ))}
            {rows.some((row) => isNum(row.proposed) || isNum(row.accepted)) ? (
              <div className="lg:col-span-2">
                <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">State change · proposed and accepted</p>
                <LineChartPanel
                  data={rows}
                  xKey="chunk"
                  series={[
                    { key: 'proposed', label: 'Proposed', color: SERIES_COLORS[2], dashed: true },
                    { key: 'accepted', label: 'Accepted', color: SERIES_COLORS[1] },
                  ]}
                  height={190}
                  xLabel="chunk"
                  ariaLabel="Proposed and accepted state change by chunk"
                />
              </div>
            ) : null}
          </div>
        )}
      </Panel>

      {recent.length > 0 ? (
        <Panel title="Recent chunks" subtitle={`Latest ${recent.length} of ${detail.summary.n_transactions}`}>
          <ul className="space-y-1.5">
            {recent.map((tx) => (
              <li key={tx.index} className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded border border-edge bg-surface-overlay px-2.5 py-2 text-xs">
                <span className="font-mono text-ink-secondary">#{tx.index}</span>
                <DecisionBadge kind={tx.decision.kind} size="sm" />
                {isNum(tx.signals.chunk_loss) ? <span className="font-mono text-ink-secondary">loss {fmt(tx.signals.chunk_loss, 3)}</span> : null}
                {isNum(tx.signals.delta_norm) ? <span className="font-mono text-status-scale">proposed {fmt(tx.signals.delta_norm, 3)}</span> : null}
                {isNum(tx.accepted?.delta_norm) ? <span className="font-mono text-status-commit">accepted {fmt(tx.accepted.delta_norm, 3)}</span> : null}
              </li>
            ))}
          </ul>
        </Panel>
      ) : null}
    </div>
  );
}
