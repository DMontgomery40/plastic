import { useState } from 'react';
import type { BetaHist, SessionState } from '../../api/types';
import { fmt, fmtInt } from '../../utils/formatting';
import { BarChartPanel, LineChartPanel, SERIES_COLORS, type SeriesSpec } from '../charts';
import { Empty, KeyValue, Panel } from '../panels';

/**
 * Per-layer session state. Rendered defensively: the API may report an empty
 * layer list for a session that has never been advanced.
 */
export function LayerStatePanel({ state, loading }: { state: SessionState | null; loading: boolean }) {
  const [layer, setLayer] = useState(0);
  const layers = state?.layers ?? [];

  if (loading && !state) {
    return (
      <Panel title="Per-layer state" subtitle="Norms and spectrum of the fast weights S.">
        <p className="text-sm text-ink-secondary">Loading state…</p>
      </Panel>
    );
  }

  if (layers.length === 0) {
    return (
      <Panel title="Per-layer state" subtitle="Norms and spectrum of the fast weights S.">
        <Empty
          title="No state reported."
          detail="The session has no committed fast weights yet, so there is nothing to decompose."
          command="uv run plastic chat <session_id> &quot;a first prompt&quot;"
        />
      </Panel>
    );
  }

  const index = Math.min(layer, layers.length - 1);
  const current = layers[index];
  const heads = current.s_norm_per_head ?? [];
  const headBars = heads.map((v, i) => ({ head: `h${i}`, norm: v }));

  const singular = current.singular_values ?? [];
  const nValues = singular.reduce((max, row) => Math.max(max, row.length), 0);
  const singularRows = Array.from({ length: nValues }, (_, i) => {
    const row: Record<string, number | string> = { rank: i + 1 };
    singular.forEach((values, h) => {
      row[`h${h}`] = values[i] ?? Number.NaN;
    });
    return row;
  });
  const singularSeries: SeriesSpec[] = singular.map((_, h) => ({
    key: `h${h}`,
    label: `head ${h}`,
    color: SERIES_COLORS[h % SERIES_COLORS.length],
  }));

  return (
    <Panel
      title="Per-layer state"
      subtitle={`Position ${fmtInt(state?.pos)}. Layer ${index} of ${layers.length - 1}.`}
      actions={
        <div className="flex gap-1">
          {layers.map((_, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setLayer(i)}
              className={`rounded border px-2 py-0.5 font-mono text-xs font-semibold ${
                i === index ? 'border-accent text-accent' : 'border-edge text-ink-secondary hover:text-ink-primary'
              }`}
            >
              L{i}
            </button>
          ))}
        </div>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">
            Memory norm per head, layer {index}
          </p>
          {headBars.length > 0 ? (
            <BarChartPanel data={headBars} xKey="head" yKey="norm" label="‖S‖" color="#58a6ff" height={180} yLabel="‖S‖ Frobenius" />
          ) : (
            <p className="text-sm text-ink-secondary">No per-head norms reported for this layer.</p>
          )}
          <div className="mt-3">
            <KeyValue
              rows={[
                { label: 'Activation state ‖h‖', value: fmt(current.h_norm, 4) },
                { label: 'Drift from session anchor', value: fmt(current.drift_from_anchor, 4) },
                { label: 'Heads', value: fmtInt(heads.length) },
              ]}
            />
          </div>
        </div>
        <div>
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">
            Top singular values of S, layer {index}
          </p>
          {singularRows.length > 0 ? (
            <LineChartPanel
              data={singularRows}
              xKey="rank"
              series={singularSeries}
              height={180}
              dots
              xLabel="rank"
              yLabel="σ"
            />
          ) : (
            <p className="text-sm text-ink-secondary">No spectrum reported for this layer.</p>
          )}
        </div>
      </div>
    </Panel>
  );
}

/** The β histogram shipped with every checkpoint's eval: the learned write rate. */
export function BetaHistogramPanel({ hist, modelId }: { hist: BetaHist | null | undefined; modelId: string | null }) {
  if (!hist) {
    return (
      <Panel title="Learned write rate β" subtitle="From the model's evaluation, not from this session.">
        <Empty
          title="No β histogram on this checkpoint."
          detail="The histogram is written by the evaluation at the end of training."
          command={`uv run plastic train text --data artifacts/data/wikitext --steps 3000 --device mps`}
        />
      </Panel>
    );
  }
  const bars = hist.counts.map((count, i) => ({
    bin: ((hist.edges[i] + hist.edges[i + 1]) / 2).toFixed(3),
    count,
  }));
  return (
    <Panel title="Learned write rate β" subtitle={modelId ? `Model ${modelId}` : undefined}>
      <BarChartPanel data={bars} xKey="bin" yKey="count" label="tokens" color="#3fd17a" height={170} xLabel="β" yLabel="tokens" />
      <div className="mt-3">
        <KeyValue
          columns={2}
          rows={[
            { label: 'β mean', value: fmt(hist.beta_mean, 4) },
            { label: 'β std', value: fmt(hist.beta_std, 4) },
            { label: 'α mean', value: fmt(hist.alpha_mean, 4), note: 'retention' },
            { label: 'Bins', value: fmtInt(hist.counts.length) },
          ]}
        />
      </div>
    </Panel>
  );
}
