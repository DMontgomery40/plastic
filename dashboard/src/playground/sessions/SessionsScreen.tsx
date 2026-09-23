import { useEffect, useState } from 'react';
import { Empty } from '../../components/panels/Empty';
import { Panel } from '../../components/panels/Panel';
import { ago, backendLabel, fmtInt } from '../format';
import { useStore } from '../store';
import { Button, Field, Select, StateChip } from '../ui';
import { SleepPanel } from './SleepPanel';

function CreateSession() {
  const models = useStore((s) => s.models).filter((m) => m.domain === 'text' && m.status === 'completed');
  const createSession = useStore((s) => s.createSession);
  const calibrate = useStore((s) => s.calibrate);
  const caps = useStore((s) => s.capabilities)();
  const busy = useStore((s) => s.busy.mutation);
  const calibrating = useStore((s) => s.calibrating);
  const [modelId, setModelId] = useState(models[0]?.model_id ?? '');
  const [guarded, setGuarded] = useState(false);
  const model = models.find((m) => m.model_id === (modelId || models[0]?.model_id));
  if (!caps.create_session) return null;
  if (models.length === 0) return <Empty title="No text model registered." detail="Register or train a text model with the CLI, then reload." command="uv run plastic --help" />;
  return (
    <Panel title="New session">
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto_auto] sm:items-end">
        <Field
          label="Model"
          htmlFor="new-model"
          hint={
            model
              ? `${backendLabel(model.backend)} · ${fmtInt(model.params)} params · ${
                  calibrating === model.model_id ? 'calibrating on real chats, a few minutes' : model.calibrated ? 'calibrated' : 'no calibration'
                }${model.parent_model_id ? ` · slept from ${model.parent_model_id}` : ''}`
              : undefined
          }
        >
          <Select id="new-model" value={model?.model_id ?? ''} onChange={(e) => setModelId(e.target.value)}>
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.model_id}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Mode" htmlFor="new-mode" hint={guarded && !model?.calibrated ? 'session-relative thresholds (no calibration)' : undefined}>
          <Select id="new-mode" value={guarded ? 'guarded' : 'observational'} onChange={(e) => setGuarded(e.target.value === 'guarded')}>
            <option value="observational">observational</option>
            <option value="guarded">guarded</option>
          </Select>
        </Field>
        <div className="flex gap-2">
          <Button tone="primary" disabled={busy || !model} onClick={() => model && void createSession(model.model_id, guarded)}>
            Create
          </Button>
          {caps.calibrate && model ? (
            <Button disabled={busy} onClick={() => void calibrate(model.model_id)} title="Fit thresholds on real chats with this model">
              {calibrating === model.model_id ? 'Calibrating…' : model.calibrated ? 'Recalibrate' : 'Calibrate'}
            </Button>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}

export function SessionsScreen() {
  const refreshSessions = useStore((s) => s.refreshSessions);
  const refreshModels = useStore((s) => s.refreshModels);
  // the catalog can change outside this screen (a chat turn, a CLI calibration, another client)
  useEffect(() => {
    void refreshSessions();
    void refreshModels();
  }, [refreshSessions, refreshModels]);
  const sessions = useStore((s) => s.sessions).filter((s) => s.domain === 'text');
  const models = useStore((s) => s.models);
  const current = useStore((s) => s.currentSessionId);
  const caps = useStore((s) => s.capabilities)();
  const busy = useStore((s) => s.busy.mutation);
  const selectSession = useStore((s) => s.selectSession);
  const setTab = useStore((s) => s.setTab);
  const reset = useStore((s) => s.resetSession);
  const fork = useStore((s) => s.forkSession);
  const del = useStore((s) => s.deleteSession);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  return (
    <div className="space-y-4">
      <CreateSession />
      <SleepPanel models={models} />
      <Panel title="Text sessions" subtitle={`${sessions.length} session${sessions.length === 1 ? '' : 's'}`}>
        {sessions.length === 0 ? (
          <Empty title="No text sessions." detail={caps.create_session ? 'Create one above.' : 'None available in this deployment.'} />
        ) : (
          <ul role="list" className="divide-y divide-edge">
            {sessions.map((s) => {
              const model = models.find((m) => m.model_id === s.model_id);
              const selected = s.session_id === current;
              return (
                <li key={s.session_id} className={`flex flex-wrap items-center gap-3 py-3 ${selected ? 'bg-accent-soft' : ''}`}>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-ink-primary">{s.session_id}</span>
                      {selected ? <StateChip tone="accent">selected</StateChip> : null}
                      {s.read_only ? <StateChip tone="warn">read-only</StateChip> : null}
                      {s.parent_session_id ? <StateChip>fork of {s.parent_session_id}</StateChip> : null}
                    </div>
                    <p className="mt-0.5 text-xs text-ink-secondary">
                      {s.model_id} · {backendLabel(model?.backend)} · position {fmtInt(s.pos)} · {fmtInt(s.n_transactions)} chunks · {fmtInt(s.commits)} committed · {fmtInt(s.rollbacks)} rolled back · updated {ago(s.updated_at_unix)}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button onClick={() => { void selectSession(s.session_id); setTab('chat'); }}>Chat</Button>
                    <Button onClick={() => { void selectSession(s.session_id); setTab('signals'); }}>Signals</Button>
                    {caps.fork ? <Button disabled={busy} onClick={() => void fork(s.session_id)}>Fork</Button> : null}
                    {caps.reset ? <Button disabled={busy} onClick={() => void reset(s.session_id)}>Reset</Button> : null}
                    {caps.delete ? (
                      confirmDelete === s.session_id ? (
                        <>
                          <Button tone="danger" disabled={busy} onClick={() => { setConfirmDelete(null); void del(s.session_id); }}>Confirm delete</Button>
                          <Button onClick={() => setConfirmDelete(null)}>Cancel</Button>
                        </>
                      ) : (
                        <Button tone="danger" disabled={busy} onClick={() => setConfirmDelete(s.session_id)}>Delete</Button>
                      )
                    ) : null}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </Panel>
    </div>
  );
}
