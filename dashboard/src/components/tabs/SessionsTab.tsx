import { useMemo, useState } from 'react';
import { useStore } from '../../store';
import type { HarnessConfig, SessionSummary } from '../../api/types';
import { fmt, fmtInt, fmtRelative } from '../../utils/formatting';
import {
  Button,
  Checkbox,
  Empty,
  Field,
  NumberInput,
  Panel,
  Select,
  Table,
  TextInput,
} from '../panels';

interface TreeNode {
  session: SessionSummary;
  children: TreeNode[];
}

/** Root-first forest over parent_session_id. Orphans become their own roots. */
export function buildLineageForest(sessions: SessionSummary[]): TreeNode[] {
  const byId = new Map(sessions.map((s) => [s.session_id, { session: s, children: [] as TreeNode[] }]));
  const roots: TreeNode[] = [];
  byId.forEach((node) => {
    const parentId = node.session.parent_session_id;
    const parent = parentId ? byId.get(parentId) : undefined;
    if (parent) parent.children.push(node);
    else roots.push(node);
  });
  const sortByCreated = (nodes: TreeNode[]) => {
    nodes.sort((a, b) => (a.session.created_at_unix ?? 0) - (b.session.created_at_unix ?? 0));
    nodes.forEach((n) => sortByCreated(n.children));
  };
  sortByCreated(roots);
  return roots;
}

function TreeRow({ node, depth, current, onSelect }: { node: TreeNode; depth: number; current: string | null; onSelect: (id: string) => void }) {
  const s = node.session;
  const active = s.session_id === current;
  return (
    <>
      <li>
        <button
          type="button"
          onClick={() => onSelect(s.session_id)}
          className={`flex w-full items-center gap-2 rounded border px-2 py-1.5 text-left ${
            active ? 'border-accent bg-accent-soft' : 'border-transparent hover:border-edge hover:bg-surface-overlay'
          }`}
          style={{ paddingLeft: `${8 + depth * 18}px` }}
        >
          {depth > 0 ? <span aria-hidden className="font-mono text-micro text-ink-muted">└</span> : null}
          <span className="font-mono text-sm text-ink-primary">{s.session_id}</span>
          <span className="rounded border border-edge bg-surface-overlay px-1.5 py-0.5 text-micro text-ink-secondary">{s.domain}</span>
          <span className="font-mono text-micro text-ink-muted">
            pos {fmtInt(s.pos)} · {fmtInt(s.n_transactions)} tx
          </span>
          {s.forked_at_pos !== null && s.forked_at_pos !== undefined ? (
            <span className="font-mono text-micro text-ink-muted">forked at {fmtInt(s.forked_at_pos)}</span>
          ) : null}
          {s.read_only ? <span className="text-micro font-semibold text-status-rollback">read-only</span> : null}
        </button>
      </li>
      {node.children.map((child) => (
        <TreeRow key={child.session.session_id} node={child} depth={depth + 1} current={current} onSelect={onSelect} />
      ))}
    </>
  );
}

const HARNESS_OVERRIDES: Array<{ key: keyof HarnessConfig; label: string; kind: 'number' | 'bool' }> = [
  { key: 'budget_chunk', label: 'Budget per chunk', kind: 'number' },
  { key: 'budget_session', label: 'Budget per session', kind: 'number' },
  { key: 'z_rollback', label: 'z rollback', kind: 'number' },
  { key: 'z_scale', label: 'z scale', kind: 'number' },
  { key: 'enable_rollback', label: 'Rollback enabled', kind: 'bool' },
  { key: 'enable_projection', label: 'Projection enabled', kind: 'bool' },
  { key: 'log_only', label: 'Log only (decide, never apply)', kind: 'bool' },
  { key: 'learn_from_generation', label: 'Learn from generated tokens', kind: 'bool' },
];

export function SessionsTab() {
  const models = useStore((s) => s.models);
  const sessions = useStore((s) => s.sessions);
  const currentSessionId = useStore((s) => s.currentSessionId);
  const setCurrentSession = useStore((s) => s.setCurrentSession);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const createSession = useStore((s) => s.createSession);
  const forkSession = useStore((s) => s.forkSession);
  const resetSession = useStore((s) => s.resetSession);
  const resumeSession = useStore((s) => s.resumeSession);
  const deleteSession = useStore((s) => s.deleteSession);
  const busy = useStore((s) => s.loading.mutation);

  const [modelId, setModelId] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [overrides, setOverrides] = useState<Partial<HarnessConfig>>({});

  const forest = useMemo(() => buildLineageForest(sessions), [sessions]);
  const usableModels = models.filter((m) => m.status === 'completed');
  const selectedModel = modelId || usableModels[0]?.model_id || '';

  const setOverride = (key: keyof HarnessConfig, value: number | boolean | null) => {
    setOverrides((prev) => {
      const next = { ...prev };
      if (value === null) delete next[key];
      else (next as Record<string, unknown>)[key] = value;
      return next;
    });
  };

  const submit = async () => {
    if (!selectedModel) return;
    const created = await createSession({
      model_id: selectedModel,
      session_id: sessionId.trim() || undefined,
      harness: Object.keys(overrides).length > 0 ? overrides : undefined,
    });
    if (created) setSessionId('');
  };

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-4">
        <Panel title="Lineage" subtitle="Forks share their parent's committed state at the fork position.">
          {sessions.length === 0 ? (
            <Empty
              title="No sessions yet."
              detail="A session pairs a trained model with its own fast weights, harness state, and transaction log."
              command="uv run plastic session new --model <model_id>"
            />
          ) : (
            <ul className="space-y-0.5">
              {forest.map((node) => (
                <TreeRow key={node.session.session_id} node={node} depth={0} current={currentSessionId} onSelect={setCurrentSession} />
              ))}
            </ul>
          )}
        </Panel>

        <Panel
          title="All sessions"
          subtitle="Counts come from the session's own transaction log."
          actions={
            currentSessionId ? (
              <Button size="sm" onClick={() => setActiveTab('session')}>
                Open session tab
              </Button>
            ) : null
          }
        >
          {sessions.length === 0 ? (
            <Empty title="Nothing to list." command="uv run plastic session new --model <model_id>" />
          ) : (
            <Table
              head={[
                'Session',
                'Domain',
                'Model',
                'Pos',
                'Tx',
                'Commit',
                'Rollback',
                'Scale',
                'Project',
                'Read-only',
                'Budget used',
                'Updated',
                '',
              ]}
            >
              {sessions.map((s) => (
                <tr
                  key={s.session_id}
                  className={`border-b border-edge ${
                    s.session_id === currentSessionId
                      ? 'border-l-2 border-l-accent bg-accent-soft'
                      : 'border-l-2 border-l-transparent hover:bg-surface-overlay'
                  }`}
                >
                  <td className="px-2 py-1.5">
                    <button
                      type="button"
                      onClick={() => setCurrentSession(s.session_id)}
                      className="font-mono text-sm text-accent hover:text-accent-hover"
                    >
                      {s.session_id}
                    </button>
                  </td>
                  <td className="px-2 py-1.5 text-ink-secondary">{s.domain}</td>
                  <td className="px-2 py-1.5 font-mono text-xs text-ink-secondary">{s.model_id}</td>
                  <td className="px-2 py-1.5 font-mono text-ink-primary">{fmtInt(s.pos)}</td>
                  <td className="px-2 py-1.5 font-mono text-ink-primary">{fmtInt(s.n_transactions)}</td>
                  <td className="px-2 py-1.5 font-mono text-status-commit">{fmtInt(s.commits)}</td>
                  <td className="px-2 py-1.5 font-mono text-status-rollback">{fmtInt(s.rollbacks)}</td>
                  <td className="px-2 py-1.5 font-mono text-status-scale">{fmtInt(s.scales)}</td>
                  <td className="px-2 py-1.5 font-mono text-status-project">{fmtInt(s.projects)}</td>
                  <td className="px-2 py-1.5 font-mono text-status-readonly">{fmtInt(s.readonly)}</td>
                  <td className="px-2 py-1.5 font-mono text-ink-primary">{fmt(s.budget_used, 3)}</td>
                  <td className="px-2 py-1.5 text-xs text-ink-secondary">{fmtRelative(s.updated_at_unix)}</td>
                  <td className="px-2 py-1.5">
                    <div className="flex gap-1.5">
                      <Button size="sm" disabled={busy} onClick={() => void forkSession(s.session_id)}>
                        Fork
                      </Button>
                      <Button size="sm" disabled={busy} onClick={() => void resetSession(s.session_id)}>
                        Reset
                      </Button>
                      {s.read_only ? (
                        <Button size="sm" disabled={busy} onClick={() => void resumeSession(s.session_id)}>
                          Resume
                        </Button>
                      ) : null}
                      <Button size="sm" variant="danger" disabled={busy} onClick={() => void deleteSession(s.session_id)}>
                        Delete
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </Table>
          )}
        </Panel>
      </div>

      <Panel title="New session" subtitle="Harness fields left blank keep the model's defaults.">
        {usableModels.length === 0 ? (
          <Empty
            title="No completed models."
            detail="A session needs a trained checkpoint to open."
            command={'uv run plastic train text --data artifacts/data/wikitext --steps 3000 --device mps'}
          />
        ) : (
          <div className="space-y-3">
            <Field label="Model" htmlFor="new-session-model">
              <Select
                id="new-session-model"
                value={selectedModel}
                onChange={setModelId}
                options={usableModels.map((m) => ({ value: m.model_id, label: `${m.model_id} · ${m.domain}` }))}
              />
            </Field>
            <Field label="Session id" htmlFor="new-session-id" hint="Left blank, the store assigns one.">
              <TextInput id="new-session-id" value={sessionId} onChange={setSessionId} placeholder="sess_1727000000" mono />
            </Field>

            <div className="space-y-2 rounded border border-edge bg-surface-overlay px-3 py-2.5">
              <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">Harness overrides</p>
              {HARNESS_OVERRIDES.map((item) =>
                item.kind === 'bool' ? (
                  <Checkbox
                    key={item.key}
                    id={`ov-${item.key}`}
                    label={item.label}
                    checked={Boolean(overrides[item.key])}
                    onChange={(v) => setOverride(item.key, v ? true : null)}
                  />
                ) : (
                  <div key={item.key} className="flex items-center gap-2">
                    <label htmlFor={`ov-${item.key}`} className="w-44 shrink-0 text-xs text-ink-secondary">
                      {item.label}
                    </label>
                    <NumberInput
                      id={`ov-${item.key}`}
                      value={Number(overrides[item.key] ?? Number.NaN)}
                      step={0.1}
                      onChange={(v) => setOverride(item.key, v)}
                    />
                  </div>
                ),
              )}
            </div>

            <Button variant="primary" disabled={busy || !selectedModel} onClick={() => void submit()}>
              {busy ? 'Working…' : 'Create session'}
            </Button>
          </div>
        )}
      </Panel>
    </div>
  );
}
