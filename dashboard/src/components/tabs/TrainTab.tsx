import { useEffect, useState } from 'react';
import { useStore } from '../../store';
import type { Domain, EvalSummary, TrainRequest } from '../../api/types';
import { fmt, fmtInt, fmtParams, fmtPercent, fmtRelative, isNum } from '../../utils/formatting';
import { LineChartPanel } from '../charts';
import { Button, Checkbox, Empty, Field, KeyValue, NumberInput, Panel, Select, StatTile, Table, TextInput } from '../panels';

function MqarChips({ ev }: { ev: EvalSummary | null | undefined }) {
  const mqar = ev?.mqar_accuracy;
  if (!mqar || Object.keys(mqar).length === 0) return <span className="text-ink-muted">—</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {Object.entries(mqar).map(([pairs, acc]) => (
        <span key={pairs} className="rounded border border-edge bg-surface-overlay px-1.5 py-0.5 font-mono text-micro text-ink-secondary">
          {pairs}: {fmtPercent(acc, 0)}
        </span>
      ))}
    </span>
  );
}

const DEFAULT_FORM: TrainRequest = {
  domain: 'text',
  data_dir: '',
  model_id: '',
  steps: 3000,
  batch_size: 16,
  seq_len: 1024,
  d_model: 256,
  layers: 4,
  heads: 4,
  chunk: 64,
  adversarial: false,
  device: 'mps',
  eval_every: 200,
  save_every: 200,
};

export function TrainTab() {
  const models = useStore((s) => s.models);
  const modelDetail = useStore((s) => s.modelDetail);
  const dataDirs = useStore((s) => s.dataDirs);
  const jobs = useStore((s) => s.jobs);
  const trainStatus = useStore((s) => s.trainStatus);
  const loadModel = useStore((s) => s.loadModel);
  const calibrate = useStore((s) => s.calibrate);
  const startTraining = useStore((s) => s.startTraining);
  const cancelTraining = useStore((s) => s.cancelTraining);
  const busy = useStore((s) => s.loading.mutation);
  const calibrating = useStore((s) => s.loading.calibrate);

  const [selected, setSelected] = useState<string | null>(null);
  const [form, setForm] = useState<TrainRequest>(DEFAULT_FORM);

  const selectedId = selected ?? models[0]?.model_id ?? null;
  const runningJob = selectedId ? jobs.find((j) => j.model_id === selectedId && j.status === 'running') : undefined;
  const liveStep = selectedId ? trainStatus[selectedId]?.latest?.step : undefined;

  useEffect(() => {
    if (selectedId && modelDetail?.record.model_id !== selectedId) void loadModel(selectedId);
  }, [selectedId, modelDetail?.record.model_id, loadModel]);

  // While a job runs, the poll loop moves `latest.step`; follow it so the loss
  // chart stays live.
  useEffect(() => {
    if (selectedId && runningJob) void loadModel(selectedId);
  }, [selectedId, runningJob, liveStep, loadModel]);

  const set = <K extends keyof TrainRequest>(key: K, value: TrainRequest[K]) => setForm((f) => ({ ...f, [key]: value }));

  const lossRows = (modelDetail?.log ?? [])
    .filter((r) => r.event !== 'eval' && isNum(r.loss))
    .map((r) => ({ step: r.step, loss: r.loss as number, grad_norm: isNum(r.grad_norm) ? r.grad_norm : Number.NaN }));
  const evalRows = (modelDetail?.log ?? [])
    .filter((r) => r.event === 'eval' && isNum(r.heldout_loss))
    .map((r) => ({ step: r.step, heldout_loss: r.heldout_loss as number, memory_value: isNum(r.memory_value) ? r.memory_value : Number.NaN }));

  const submit = async () => {
    const body: TrainRequest = { ...form };
    if (!body.data_dir) delete body.data_dir;
    if (!body.model_id) delete body.model_id;
    if (body.domain === 'physics') delete body.data_dir;
    await startTraining(body);
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-4">
        <StatTile label="Models" value={fmtInt(models.length)} hint={`${models.filter((m) => m.calibrated).length} calibrated`} />
        <StatTile label="Running jobs" value={fmtInt(jobs.filter((j) => j.status === 'running').length)} tone="accent" hint="polled every 2 s" />
        <StatTile label="Corpora" value={fmtInt(dataDirs.length)} hint="under artifacts/data" />
        <StatTile
          label="Best held-out loss"
          value={fmt(
            models.reduce<number | null>((best, m) => {
              const v = m.eval?.heldout_loss;
              return isNum(v) && (best === null || v < best) ? v : best;
            }, null),
            4,
          )}
          hint="lower is better"
        />
      </div>

      <Panel title="Models" subtitle="Every checkpoint ships three numbers: held-out loss, the value of the memory, and MQAR accuracy.">
        {models.length === 0 ? (
          <Empty
            title="No models yet."
            detail="Train one locally, or start a job with the form below."
            command={'uv run plastic train physics --steps 3000 --device mps'}
          />
        ) : (
          <Table
            head={['Model', 'Domain', 'Status', 'Params', 'Steps', 'Held-out loss', 'Memory value', 'MQAR', 'Calibrated', 'Updated', '']}
          >
            {models.map((m) => {
              const job = jobs.find((j) => j.model_id === m.model_id);
              return (
                <tr
                  key={m.model_id}
                  className={`border-b border-edge ${m.model_id === selectedId ? 'bg-accent-soft' : 'hover:bg-surface-overlay'}`}
                >
                  <td className="px-2 py-1.5">
                    <button type="button" onClick={() => setSelected(m.model_id)} className="font-mono text-sm text-accent hover:text-accent-hover">
                      {m.model_id}
                    </button>
                  </td>
                  <td className="px-2 py-1.5 text-ink-secondary">{m.domain}</td>
                  <td className="px-2 py-1.5">
                    <span
                      className={`font-semibold ${
                        m.status === 'completed'
                          ? 'text-status-commit'
                          : m.status === 'failed'
                            ? 'text-status-failed'
                            : 'text-status-running'
                      }`}
                    >
                      {m.status}
                    </span>
                  </td>
                  <td className="px-2 py-1.5 font-mono text-ink-primary">{fmtParams(m.params)}</td>
                  <td className="px-2 py-1.5 font-mono text-ink-primary">{fmtInt(m.steps)}</td>
                  <td className="px-2 py-1.5 font-mono text-ink-primary">{fmt(m.eval?.heldout_loss)}</td>
                  <td className="px-2 py-1.5 font-mono text-ink-primary">{fmt(m.eval?.memory_value)}</td>
                  <td className="px-2 py-1.5">
                    <MqarChips ev={m.eval} />
                  </td>
                  <td className="px-2 py-1.5 text-xs text-ink-secondary">{m.calibrated ? 'yes' : 'no'}</td>
                  <td className="px-2 py-1.5 text-xs text-ink-secondary">{fmtRelative(m.updated_at_unix)}</td>
                  <td className="px-2 py-1.5">
                    {job?.status === 'running' ? (
                      <Button size="sm" variant="danger" disabled={busy} onClick={() => void cancelTraining(m.model_id)}>
                        Cancel
                      </Button>
                    ) : (
                      <Button size="sm" disabled={calibrating} onClick={() => void calibrate(m.model_id, {})}>
                        {calibrating ? 'Calibrating…' : 'Calibrate'}
                      </Button>
                    )}
                  </td>
                </tr>
              );
            })}
          </Table>
        )}
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-4">
          <Panel
            title={selectedId ? `Training loss, ${selectedId}` : 'Training loss'}
            subtitle={runningJob ? 'Job is running; the chart follows the log every 2 seconds.' : 'From the model train log.'}
          >
            {lossRows.length === 0 ? (
              <Empty
                title="No loss records."
                detail="The log is written as the job steps; a model that never started has none."
                command={'uv run plastic train text --data artifacts/data/wikitext --steps 3000 --device mps'}
              />
            ) : (
              <LineChartPanel
                data={lossRows}
                xKey="step"
                series={[{ key: 'loss', label: 'train loss', color: '#58a6ff' }]}
                height={240}
                xLabel="step"
                yLabel="loss"
                showLegend={false}
              />
            )}
          </Panel>

          {evalRows.length > 0 ? (
            <Panel title="Evaluation over training" subtitle="Held-out loss and the value of the memory (β = 0 loss minus β loss).">
              <LineChartPanel
                data={evalRows}
                xKey="step"
                series={[
                  { key: 'heldout_loss', label: 'held-out loss', color: '#3fd17a' },
                  { key: 'memory_value', label: 'memory value', color: '#f0b429' },
                ]}
                height={220}
                xLabel="step"
              />
            </Panel>
          ) : null}

          <Panel title="Calibration" subtitle="Empirical thresholds at the target false-positive rate, plus the Fisher diagonal and the canary baselines.">
            {!modelDetail ? (
              <p className="text-sm text-ink-secondary">Select a model to see its calibration.</p>
            ) : modelDetail.calibration === null ? (
              <Empty
                title={`${modelDetail.record.model_id} is not calibrated.`}
                detail="Without it the harness falls back to robust z-scores over the session's own history and draws no thresholds."
                command={`uv run plastic calibrate ${modelDetail.record.model_id} --data artifacts/data/wikitext`}
              >
                <Button disabled={calibrating} onClick={() => void calibrate(modelDetail.record.model_id, {})}>
                  {calibrating ? 'Calibrating…' : 'Calibrate now'}
                </Button>
              </Empty>
            ) : (
              <div className="grid gap-4 lg:grid-cols-2">
                <div>
                  <p className="mb-2 text-label font-semibold uppercase tracking-wide text-ink-muted">Thresholds</p>
                  <KeyValue
                    rows={Object.entries(modelDetail.calibration.thresholds).map(([signal, value]) => ({
                      label: signal,
                      value: fmt(value, 4),
                    }))}
                  />
                </div>
                <div>
                  <p className="mb-2 text-label font-semibold uppercase tracking-wide text-ink-muted">Reference</p>
                  <KeyValue
                    rows={[
                      { label: 'Chunks observed', value: fmtInt(modelDetail.calibration.n_chunks) },
                      { label: 'Target FPR', value: fmtPercent(modelDetail.calibration.target_fpr, 2) },
                      { label: 'Calibrated', value: fmtRelative(modelDetail.calibration.created_at_unix) },
                      ...Object.entries(modelDetail.calibration.canary_baseline).map(([k, v]) => ({
                        label: `canary ${k}`,
                        value: fmt(v, 4),
                      })),
                      {
                        label: 'Canary probes',
                        value: modelDetail.canary ? `${modelDetail.canary.n_coherence} coherence, ${modelDetail.canary.n_poison} poison` : '—',
                      },
                    ]}
                  />
                </div>
              </div>
            )}
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title="Start a local job" subtitle="Runs plastic train as a subprocess of the API.">
            <div className="space-y-3">
              <Field label="Domain" htmlFor="tr-domain">
                <Select
                  id="tr-domain"
                  value={form.domain}
                  onChange={(v) => set('domain', v as Domain)}
                  options={[
                    { value: 'text', label: 'text' },
                    { value: 'physics', label: 'physics' },
                  ]}
                />
              </Field>
              {form.domain === 'text' ? (
                <Field label="Corpus" htmlFor="tr-data" hint={dataDirs.length === 0 ? 'None prepared yet.' : undefined}>
                  <Select
                    id="tr-data"
                    value={form.data_dir ?? ''}
                    onChange={(v) => set('data_dir', v)}
                    options={[
                      { value: '', label: 'API default' },
                      ...dataDirs.map((d) => ({ value: d.dir, label: `${d.name} · ${d.vocab_size} vocab` })),
                    ]}
                  />
                </Field>
              ) : null}
              <Field label="Model id" htmlFor="tr-model" hint="Left blank, the store assigns one.">
                <TextInput id="tr-model" value={form.model_id ?? ''} onChange={(v) => set('model_id', v)} mono />
              </Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Steps" htmlFor="tr-steps">
                  <NumberInput id="tr-steps" value={form.steps} min={1} step={100} onChange={(v) => set('steps', v)} />
                </Field>
                <Field label="Batch size" htmlFor="tr-batch">
                  <NumberInput id="tr-batch" value={form.batch_size} min={1} step={1} onChange={(v) => set('batch_size', v)} />
                </Field>
                <Field label="Sequence length" htmlFor="tr-seq">
                  <NumberInput id="tr-seq" value={form.seq_len} min={8} step={64} onChange={(v) => set('seq_len', v)} />
                </Field>
                <Field label="d_model" htmlFor="tr-dmodel">
                  <NumberInput id="tr-dmodel" value={form.d_model ?? 256} min={8} step={8} onChange={(v) => set('d_model', v)} />
                </Field>
                <Field label="Layers" htmlFor="tr-layers">
                  <NumberInput id="tr-layers" value={form.layers ?? 4} min={1} step={1} onChange={(v) => set('layers', v)} />
                </Field>
                <Field label="Heads" htmlFor="tr-heads">
                  <NumberInput id="tr-heads" value={form.heads ?? 4} min={1} step={1} onChange={(v) => set('heads', v)} />
                </Field>
                <Field label="Chunk" htmlFor="tr-chunk">
                  <NumberInput id="tr-chunk" value={form.chunk ?? 64} min={1} step={8} onChange={(v) => set('chunk', v)} />
                </Field>
                <Field label="Eval every" htmlFor="tr-eval">
                  <NumberInput id="tr-eval" value={form.eval_every ?? 200} min={0} step={50} onChange={(v) => set('eval_every', v)} />
                </Field>
              </div>
              <Field label="Device" htmlFor="tr-device">
                <Select
                  id="tr-device"
                  value={form.device ?? 'mps'}
                  onChange={(v) => set('device', v)}
                  options={[
                    { value: 'mps', label: 'mps' },
                    { value: 'cpu', label: 'cpu' },
                    { value: 'cuda', label: 'cuda' },
                    { value: 'auto', label: 'auto' },
                  ]}
                />
              </Field>
              <Checkbox
                id="tr-adv"
                label="Meta-train the write gate against the attacker"
                checked={Boolean(form.adversarial)}
                onChange={(v) => set('adversarial', v)}
              />
              <Button variant="primary" disabled={busy} onClick={() => void submit()}>
                {busy ? 'Starting…' : 'Start job'}
              </Button>
            </div>
          </Panel>

          <Panel title="Jobs" subtitle="Local subprocesses owned by the API.">
            {jobs.length === 0 ? (
              <p className="text-sm text-ink-secondary">No job has been started from this API process.</p>
            ) : (
              <ul className="space-y-2">
                {jobs.map((job) => {
                  const status = trainStatus[job.model_id];
                  return (
                    <li key={`${job.model_id}-${job.pid}`} className="rounded border border-edge bg-surface-overlay px-2.5 py-2">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-sm text-ink-primary">{job.model_id}</span>
                        <span className={`text-xs font-semibold ${job.status === 'running' ? 'text-status-running' : 'text-ink-secondary'}`}>
                          {job.status}
                        </span>
                      </div>
                      <p className="mt-1 font-mono text-micro text-ink-muted">
                        pid {job.pid} · started {fmtRelative(job.started_at_unix)}
                        {job.exit_code !== null ? ` · exit ${job.exit_code}` : ''}
                      </p>
                      {status?.latest ? (
                        <p className="mt-1 font-mono text-micro text-ink-secondary">
                          step {fmtInt(status.latest.step)} · loss {fmt(status.latest.loss)} · {fmt(status.latest.tok_per_s, 0)} tok/s
                        </p>
                      ) : null}
                      {status?.error ? <p className="mt-1 text-micro text-status-failed">{status.error}</p> : null}
                    </li>
                  );
                })}
              </ul>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
