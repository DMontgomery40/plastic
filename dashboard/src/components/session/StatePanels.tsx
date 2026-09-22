import { useState } from 'react';
import type { BetaHist, SessionState } from '../../api/types';
import { UNAVAILABLE, fmt, fmtInt, isNum } from '../../utils/formatting';
import { BarChartPanel, LineChartPanel, SERIES_COLORS, type SeriesSpec } from '../charts';
import { Empty, KeyValue, Panel } from '../panels';

/**
 * Session state. A toy Plastic session has a per-layer S/h shape; a pretrained backend (Qwen) has a
 * per-memory-unit recurrent shape instead. This entry point dispatches on the state kind so each
 * backend renders its own honest fields -- never a fabricated S/h for the recurrent path.
 */
export function LayerStatePanel({ state, loading }: { state: SessionState | null; loading: boolean }) {
  if (state && (state.kind === 'recurrent' || (state.units !== undefined && state.layers === undefined))) {
    return <RecurrentStatePanel state={state} loading={loading} />;
  }
  return <PlasticStatePanel state={state} loading={loading} />;
}

/**
 * Per-memory-unit recurrent state for a pretrained backend (Qwen's gated-delta memory). Norms and
 * drift are nullable: a unit whose measurement failed or is missing is shown as unavailable, never
 * as a measured zero, and is excluded from the bars rather than plotted at 0.
 */
function RecurrentStatePanel({ state, loading }: { state: SessionState | null; loading: boolean }) {
  const units = state?.units ?? [];
  const subtitle = 'Per-unit norms of the gated-delta recurrent memory.';
  if (loading && !state) {
    return (
      <Panel title="Recurrent memory state" subtitle={subtitle}>
        <p className="text-sm text-ink-secondary">Loading state…</p>
      </Panel>
    );
  }
  if (units.length === 0) {
    return (
      <Panel title="Recurrent memory state" subtitle={subtitle}>
        <Empty
          title="No state reported."
          detail="The session has not advanced its recurrent memory yet, so there is nothing to show."
        />
      </Panel>
    );
  }
  const normBars = units.filter((u) => isNum(u.recurrent_norm)).map((u) => ({ unit: `u${u.index}`, norm: u.recurrent_norm as number }));
  const driftBars = units.filter((u) => isNum(u.drift_from_anchor)).map((u) => ({ unit: `u${u.index}`, drift: u.drift_from_anchor as number }));
  const normUnavailable = units.length - normBars.length;
  const driftUnavailable = units.length - driftBars.length;
  return (
    <Panel
      title="Recurrent memory state"
      subtitle={`Position ${fmtInt(state?.pos)}. ${fmtInt(units.length)} gated-delta memory units${state?.backend ? ` on the ${state.backend} backend` : ''}.`}
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div>
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">Memory norm per unit</p>
          {normBars.length > 0 ? (
            <BarChartPanel
              data={normBars}
              xKey="unit"
              yKey="norm"
              label="‖M‖"
              color="#58a6ff"
              height={200}
              yLabel="‖memory‖ Frobenius"
              ariaLabel="Frobenius norm of the recurrent memory per unit"
            />
          ) : (
            <p className="text-sm text-ink-secondary">No per-unit norms reported.</p>
          )}
          {normUnavailable > 0 ? (
            <p className="mt-1 text-sm text-ink-secondary">{fmtInt(normUnavailable)} unit(s) report no norm ({UNAVAILABLE}).</p>
          ) : null}
        </div>
        <div>
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">Drift from session anchor per unit</p>
          {driftBars.length > 0 ? (
            <BarChartPanel
              data={driftBars}
              xKey="unit"
              yKey="drift"
              label="Δ"
              color="#d29922"
              height={200}
              yLabel="‖drift‖"
              ariaLabel="Drift of each recurrent memory unit from the session anchor"
            />
          ) : (
            <p className="text-sm text-ink-secondary">No per-unit drift reported.</p>
          )}
          {driftUnavailable > 0 ? (
            <p className="mt-1 text-sm text-ink-secondary">{fmtInt(driftUnavailable)} unit(s) report no drift ({UNAVAILABLE}).</p>
          ) : null}
        </div>
      </div>
      <div className="mt-3">
        <KeyValue
          rows={[
            { label: 'Total recurrent norm', value: fmt(state?.recurrent_norm_total, 4) },
            { label: 'Memory units', value: fmtInt(units.length) },
          ]}
        />
      </div>
    </Panel>
  );
}

/**
 * Per-layer session state for the toy Plastic model. Rendered defensively: the API may report an
 * empty layer list for a session that has never been advanced.
 */
function PlasticStatePanel({ state, loading }: { state: SessionState | null; loading: boolean }) {
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
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div>
          <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">
            Memory norm per head, layer {index}
          </p>
          {headBars.length > 0 ? (
            <BarChartPanel
              data={headBars}
              xKey="head"
              yKey="norm"
              label="‖S‖"
              color="#58a6ff"
              height={180}
              yLabel="‖S‖ Frobenius"
              ariaLabel={`Frobenius norm of the memory state per head, layer ${index}`}
            />
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
              ariaLabel={`Top singular values of the memory state per head, layer ${index}`}
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
      <BarChartPanel
        data={bars}
        xKey="bin"
        yKey="count"
        label="tokens"
        color="#3fd17a"
        height={170}
        xLabel="β"
        yLabel="tokens"
        ariaLabel="Histogram of the learned write rate beta over evaluation tokens"
      />
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
