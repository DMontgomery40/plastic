import { useState } from 'react';
import { useStore } from '../../store';
import { UNAVAILABLE, fmt, fmtInt, fmtPercent } from '../../utils/formatting';
import { LineChartPanel } from '../charts';
import { Button, Checkbox, DecisionBadge, Empty, Field, NumberInput, Panel, Slider, StatTile, Table } from '../panels';

export function PhysicsTab() {
  const sessions = useStore((s) => s.sessions);
  const currentSessionId = useStore((s) => s.currentSessionId);
  const setCurrentSession = useStore((s) => s.setCurrentSession);
  const sessionDetail = useStore((s) => s.sessionDetail);
  const episode = useStore((s) => s.episodeResult);
  const runEpisode = useStore((s) => s.runEpisode);
  const busy = useStore((s) => s.loading.physics);

  const [mu, setMu] = useState(0.12);
  const [steps, setSteps] = useState(256);
  const [seed, setSeed] = useState(0);
  const [nonlinear, setNonlinear] = useState(false);
  const [logScale, setLogScale] = useState(false);

  const physicsSessions = sessions.filter((s) => s.domain === 'physics');
  const domain = sessionDetail?.meta.domain;

  if (physicsSessions.length === 0) {
    return (
      <Empty
        title="No physics session."
        detail="The hidden-mu comparison needs a session on a physics model."
        command={'uv run plastic train physics --steps 3000 --device mps\nuv run plastic session new --model <model_id>'}
      />
    );
  }

  // The API returns 400 for an episode against a text session.
  if (domain === 'text') {
    return (
      <Empty
        title={`Session ${currentSessionId} is a text session.`}
        detail="Episodes are only defined for physics sessions. Pick one in the header, or use one of these."
      >
        <div className="flex flex-wrap gap-2">
          {physicsSessions.map((s) => (
            <Button key={s.session_id} size="sm" onClick={() => setCurrentSession(s.session_id)}>
              {s.session_id}
            </Button>
          ))}
        </div>
      </Empty>
    );
  }

  const improvement =
    episode && episode.means.frozen_mse > 0
      ? (episode.means.frozen_mse - episode.means.adaptive_mse) / episode.means.frozen_mse
      : null;

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
      <div className="space-y-4">
        <Panel title="Episode" subtitle="A 2D point mass with hidden friction mu, resampled per episode.">
          <div className="space-y-3">
            <Field label="mu (hidden friction)" htmlFor="phys-mu" hint="The training range is 0.02 to 0.25.">
              <Slider id="phys-mu" value={mu} min={0.02} max={0.25} step={0.005} onChange={setMu} />
            </Field>
            <Field label="Steps" htmlFor="phys-steps">
              <NumberInput id="phys-steps" value={steps} min={8} max={4096} step={8} onChange={setSteps} />
            </Field>
            <Field label="Seed" htmlFor="phys-seed">
              <NumberInput id="phys-seed" value={seed} min={0} step={1} onChange={setSeed} />
            </Field>
            <Checkbox id="phys-nonlinear" label="Nonlinear dynamics" checked={nonlinear} onChange={setNonlinear} />
            <Button
              variant="primary"
              disabled={busy}
              onClick={() => void runEpisode({ mu, steps, seed, nonlinear })}
            >
              {busy ? 'Running…' : 'Run episode'}
            </Button>
            <p className="text-micro text-ink-muted">Session {currentSessionId}</p>
          </div>
        </Panel>

        {episode ? (
          <Panel title="Mean squared error" subtitle="Same trajectory, three ways of using the state.">
            <Table head={['Variant', 'Mean MSE', 'What it is']}>
              <tr className="border-b border-edge">
                <td className="px-2 py-1.5 font-semibold text-series-a">base</td>
                <td className="px-2 py-1.5 font-mono text-ink-primary">{fmt(episode.means.base_mse)}</td>
                <td className="px-2 py-1.5 text-xs text-ink-secondary">zero state, frozen</td>
              </tr>
              <tr className="border-b border-edge">
                <td className="px-2 py-1.5 font-semibold text-series-b">frozen</td>
                <td className="px-2 py-1.5 font-mono text-ink-primary">{fmt(episode.means.frozen_mse)}</td>
                <td className="px-2 py-1.5 text-xs text-ink-secondary">session state, no learning</td>
              </tr>
              <tr>
                <td className="px-2 py-1.5 font-semibold text-series-c">adaptive</td>
                <td className="px-2 py-1.5 font-mono text-ink-primary">{fmt(episode.means.adaptive_mse)}</td>
                <td className="px-2 py-1.5 text-xs text-ink-secondary">session state, learning through the harness</td>
              </tr>
            </Table>
          </Panel>
        ) : null}
      </div>

      <div className="space-y-4">
        {episode ? (
          <>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
              <StatTile label="mu" value={fmt(episode.mu, 3)} hint="hidden, must be identified" />
              <StatTile label="Steps" value={fmtInt(episode.steps)} />
              <StatTile label="Transactions" value={fmtInt(episode.transactions.length)} />
              <StatTile
                label="Adaptive vs frozen"
                value={improvement === null ? UNAVAILABLE : fmtPercent(improvement, 1)}
                tone={improvement !== null && improvement > 0 ? 'commit' : 'rollback'}
                hint="error removed by learning"
              />
            </div>

            <Panel
              title="Three-way error curve"
              subtitle="Per-step squared error on one trajectory."
              actions={
                <button
                  type="button"
                  onClick={() => setLogScale((v) => !v)}
                  className={`rounded border px-2 py-0.5 text-xs font-semibold ${
                    logScale ? 'border-accent text-accent' : 'border-edge text-ink-secondary hover:text-ink-primary'
                  }`}
                >
                  {logScale ? 'Log scale on' : 'Log scale off'}
                </button>
              }
            >
              <LineChartPanel
                data={episode.per_step}
                xKey="t"
                series={[
                  { key: 'base_mse', label: 'base', color: '#58a6ff' },
                  { key: 'frozen_mse', label: 'frozen', color: '#3fd17a' },
                  { key: 'adaptive_mse', label: 'adaptive', color: '#f0b429' },
                ]}
                height={320}
                logScale={logScale}
                xLabel="step"
                yLabel="squared error"
                ariaLabel="Per-step squared error for the base, frozen and adaptive variants on one trajectory"
              />
            </Panel>

            <Panel title="Transactions of this episode">
              {episode.transactions.length === 0 ? (
                <Empty title="No chunk boundary was crossed." detail="The episode was shorter than one chunk." />
              ) : (
                <Table head={['Chunk', 'Requested', 'Applied', 'Loss', 'Surprise', 'β', '‖Δ‖ proposed', '‖Δ‖ accepted', 'Reasons']}>
                  {episode.transactions.map((tx) => (
                    <tr key={tx.index} className="border-b border-edge">
                      <td className="px-2 py-1.5 font-mono text-ink-primary">{tx.index}</td>
                      <td className="px-2 py-1.5">
                        {tx.requested.kind === tx.decision.kind ? (
                          <span className="text-micro text-ink-muted">same</span>
                        ) : (
                          <DecisionBadge kind={tx.requested.kind} size="sm" />
                        )}
                      </td>
                      <td className="px-2 py-1.5">
                        <DecisionBadge kind={tx.decision.kind} size="sm" />
                      </td>
                      <td className="px-2 py-1.5 font-mono text-ink-primary">{fmt(tx.signals.chunk_loss)}</td>
                      <td className="px-2 py-1.5 font-mono text-ink-primary">{fmt(tx.signals.surprise_mean)}</td>
                      <td className="px-2 py-1.5 font-mono text-ink-primary">{fmt(tx.signals.beta_mean)}</td>
                      <td className="px-2 py-1.5 font-mono text-status-scale">{fmt(tx.signals.delta_norm)}</td>
                      <td className="px-2 py-1.5 font-mono text-status-commit">{fmt(tx.accepted?.delta_norm)}</td>
                      <td className="px-2 py-1.5 font-mono text-xs text-ink-secondary">{tx.decision.reasons.join(', ') || 'none recorded'}</td>
                    </tr>
                  ))}
                </Table>
              )}
            </Panel>
          </>
        ) : (
          <Empty
            title="No episode in this view yet."
            detail="Run one with the controls, or from the command line."
            command={`uv run plastic physics ${currentSessionId ?? '<session_id>'} --steps 256 --mu 0.12`}
          />
        )}
      </div>
    </div>
  );
}
