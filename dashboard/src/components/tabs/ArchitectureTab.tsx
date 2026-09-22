import { useEffect, useState } from 'react';
import { useStore } from '../../store';
import { fmtInt, fmtParams } from '../../utils/formatting';
import { BlockDiagram, HarnessDiagram } from '../architecture/BlockDiagram';
import { Empty, KeyValue, Panel, Select } from '../panels';

const RESEARCH_DOCS: Array<{ file: string; what: string }> = [
  { file: 'docs/research/2026-09-21-architecture-memo.md', what: 'The block equations, verified on CPU and MPS, with timing and parameter counts.' },
  { file: 'docs/research/2026-09-21-ttt-ssm-literature.md', what: 'TTT layers, Titans, LaCT, Gated DeltaNet, Mamba-3, and the 2026 delta-rule wave.' },
  { file: 'docs/research/2026-09-21-inference-time-learning-safety.md', what: 'Attacks on models that learn at inference, and the regex-free safety stack.' },
  { file: 'docs/research/2026-09-21-plastic-coordinate-recurrence.md', what: 'The coordinate recurrence behind the chunked log-space scan.' },
  { file: 'docs/research/2026-09-22-calibration-replay-audit.md', what: 'What the calibrated thresholds do and do not catch on replay.' },
  { file: 'docs/research/2026-09-22-copy-memory-content-audit.md', what: 'Copy and memory probes over the fast weights.' },
  { file: 'docs/research/2026-09-21-tooling-hf-jobs-torch.md', what: 'Hugging Face Jobs and torch on MPS.' },
];

export function ArchitectureTab() {
  const models = useStore((s) => s.models);
  const modelDetail = useStore((s) => s.modelDetail);
  const loadModel = useStore((s) => s.loadModel);
  const [selected, setSelected] = useState<string | null>(null);

  const usable = models.filter((m) => m.status === 'completed');
  // Only a model that is actually in the picker may be drawn, otherwise the
  // select shows one model and the diagram shows another.
  const loadedId = modelDetail?.record.model_id;
  const carriedOver = usable.some((m) => m.model_id === loadedId) ? loadedId : undefined;
  const selectedId = selected ?? carriedOver ?? usable[0]?.model_id ?? null;
  const drawing = modelDetail?.record.model_id === selectedId ? modelDetail : null;

  useEffect(() => {
    if (selectedId && modelDetail?.record.model_id !== selectedId) void loadModel(selectedId);
  }, [selectedId, modelDetail?.record.model_id, loadModel]);

  if (usable.length === 0) {
    return (
      <Empty
        title="No trained model to draw."
        detail="This tab renders the block from a live model configuration, never from a hardcoded default."
        command={'uv run plastic train physics --steps 3000 --device mps'}
      />
    );
  }

  const cfg = drawing?.config ?? null;
  const record = drawing?.record ?? null;

  return (
    <div className="space-y-4">
      <Panel
        title="One block"
        subtitle="Drawn from the selected model's own configuration."
        actions={
          <div className="min-w-[240px]">
            <Select
              value={selectedId ?? ''}
              onChange={setSelected}
              options={usable.map((m) => ({ value: m.model_id, label: `${m.model_id} · ${m.domain}` }))}
            />
          </div>
        }
      >
        {!cfg ? (
          <p className="text-sm text-ink-secondary">Loading the model configuration…</p>
        ) : (
          <div className="grid gap-5 xl:grid-cols-[auto_minmax(0,1fr)]">
            <BlockDiagram cfg={cfg} />
            <div className="space-y-4">
              <KeyValue
                columns={2}
                rows={[
                  { label: 'Domain', value: cfg.domain },
                  { label: 'Parameters', value: fmtParams(record?.params) },
                  { label: 'Width d_model', value: fmtInt(cfg.d_model) },
                  { label: 'Heads', value: fmtInt(cfg.n_heads), note: `head dim ${Math.floor(cfg.d_model / cfg.n_heads)}` },
                  { label: 'Layers', value: fmtInt(cfg.n_layers) },
                  { label: 'Transaction chunk L', value: fmtInt(cfg.chunk) },
                  { label: 'Scan chunk', value: fmtInt(cfg.scan_chunk), note: 'working set only' },
                  { label: 'Convolution kernel', value: fmtInt(cfg.conv_kernel) },
                  { label: 'Inner rule', value: cfg.rule },
                  { label: 'Memory', value: cfg.memory },
                  { label: 'Memory input', value: cfg.memory_input },
                  { label: 'Vocabulary', value: cfg.domain === 'text' ? fmtInt(cfg.vocab_size) : 'n/a' },
                ]}
              />
              <div className="space-y-2 text-sm text-ink-secondary">
                <p>
                  Each block runs two branches over the same input. The state-space branch carries a gated diagonal
                  recurrence whose decay is computed in log space, so only decay ratios in the unit interval are ever
                  formed. The memory branch is the test-time training part: a per-token gradient step on the
                  associative loss that binds each key to its value, with a learned write rate and a learned forget
                  rate.
                </p>
                <p>
                  The write rate is the model&apos;s own answer to &quot;should I learn from this token&quot;. It is
                  meta-trained end to end through the inner loop, which is what separates this from an adapter
                  fine-tuned on a model that was never trained to be adapted.
                </p>
              </div>
            </div>
          </div>
        )}
      </Panel>

      <Panel title="The harness" subtitle="Every chunk is a transaction: measure, decide, then commit or undo.">
        <HarnessDiagram />
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <div className="space-y-2 text-sm text-ink-secondary">
            <p>
              Inputs advance a working copy of the state token by token. At each chunk boundary the runner measures the
              chunk, asks the policy for a decision, and applies it. A commit promotes the working state; a rollback
              restores the committed state and reprocesses the chunk frozen, so the content is read but refused as
              training signal.
            </p>
            <p>
              A projection removes the part of the state change that aligns with the gradient of the coherence canary.
              If that removes more than the configured fraction of the change, the decision falls back to rollback,
              which recomputes a fully consistent state.
            </p>
          </div>
          <div className="space-y-2 text-sm text-ink-secondary">
            <p>
              Thresholds come from calibration: a benign stream is run with the harness in log-only mode and each
              decision signal gets an empirical quantile at the target false-positive rate, split across signals by a
              union bound. Without a calibration the policy falls back to robust z-scores over the session&apos;s own
              history.
            </p>
            <p>
              Magnitude alone is not enough. A change can stay under every norm cap and still push the model in a
              harmful direction, which is why the canary gradient alignment and the Fisher-weighted drift are measured
              alongside the norms.
            </p>
          </div>
        </div>
      </Panel>

      <Panel title="Where this is written down" subtitle="Companion documents in the repository.">
        <ul className="space-y-2">
          {RESEARCH_DOCS.map((doc) => (
            <li key={doc.file} className="border-b border-edge pb-2 last:border-b-0">
              <p className="font-mono text-sm text-ink-primary">{doc.file}</p>
              <p className="mt-0.5 text-xs text-ink-secondary">{doc.what}</p>
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}
