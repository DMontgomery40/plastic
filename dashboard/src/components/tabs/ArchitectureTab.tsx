import { useEffect, useState } from 'react';
import { useStore } from '../../store';
import type { ModelConfig } from '../../api/types';
import { fmtInt, fmtParams } from '../../utils/formatting';
import { BlockDiagram, HarnessDiagram } from '../architecture/BlockDiagram';
import { Empty, KeyValue, Panel, Select } from '../panels';

const RESEARCH_URL = 'https://github.com/DMontgomery40/plastic/blob/main/docs/research/current-status.md';

/** A native model can return config={}, and an incomplete Plastic job can too. */
export function hasPlasticConfig(config: Partial<ModelConfig> | null | undefined): config is ModelConfig {
  if (!config) return false;
  const positive = ['d_model', 'n_heads', 'n_layers', 'chunk', 'scan_chunk', 'conv_kernel', 'ssm_c', 'mlp_mult'];
  const nonnegative = ['vocab_size', 'obs_dim', 'act_dim'];
  const value = config as Record<string, unknown>;
  return (config.domain === 'text' || config.domain === 'physics')
    && positive.every((key) => typeof value[key] === 'number' && Number.isFinite(value[key]) && (value[key] as number) > 0)
    && nonnegative.every((key) => typeof value[key] === 'number' && Number.isFinite(value[key]) && (value[key] as number) >= 0)
    && typeof config.tie_embeddings === 'boolean'
    && (config.rule === 'delta' || config.rule === 'chunk')
    && (config.memory === 'linear' || config.memory === 'mlp')
    && (config.memory_input === 'ssm_out' || config.memory_input === 'block_in');
}

export function ArchitectureTab() {
  const models = useStore((s) => s.models);
  const modelDetail = useStore((s) => s.modelDetail);
  const loadModel = useStore((s) => s.loadModel);
  const [selected, setSelected] = useState<string | null>(null);

  const usable = models.filter((m) => m.status === 'completed');
  const loadedId = modelDetail?.record.model_id;
  const carriedOver = usable.some((m) => m.model_id === loadedId) ? loadedId : undefined;
  const selectedId = selected ?? carriedOver ?? usable[0]?.model_id ?? null;
  const drawing = modelDetail?.record.model_id === selectedId ? modelDetail : null;

  useEffect(() => {
    if (selectedId && modelDetail?.record.model_id !== selectedId) void loadModel(selectedId);
  }, [selectedId, modelDetail?.record.model_id, loadModel]);

  if (usable.length === 0) return <Empty title="No completed model available." />;

  const config = drawing?.config;
  const record = drawing?.record;
  const plastic = (record?.backend === undefined || record.backend === 'plastic') && hasPlasticConfig(config);

  return (
    <div className="space-y-4">
      <Panel
        title="Model architecture"
        actions={<div className="min-w-[240px]"><Select ariaLabel="Model" value={selectedId ?? ''} onChange={setSelected} options={usable.map((m) => ({ value: m.model_id, label: `${m.model_id} · ${m.domain}` }))} /></div>}
      >
        {!drawing ? (
          <p className="text-sm text-ink-secondary">Loading model…</p>
        ) : plastic ? (
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-[auto_minmax(0,1fr)]">
            <BlockDiagram cfg={config} />
            <KeyValue columns={2} rows={[
              { label: 'Model', value: drawing.record.model_id },
              { label: 'Backend', value: 'Plastic' },
              { label: 'Parameters', value: fmtParams(drawing.record.params) },
              { label: 'Width', value: fmtInt(config.d_model) },
              { label: 'Heads', value: fmtInt(config.n_heads) },
              { label: 'Layers', value: fmtInt(config.n_layers) },
              { label: 'Chunk', value: fmtInt(config.chunk) },
              { label: 'Inner rule', value: config.rule },
              { label: 'Memory', value: config.memory },
            ]} />
          </div>
        ) : (
          <KeyValue columns={2} rows={[
            { label: 'Model', value: drawing.record.model_id },
            { label: 'Backend', value: drawing.record.backend ?? 'Unavailable' },
            { label: 'Domain', value: drawing.record.domain },
            { label: 'Parameters', value: fmtParams(drawing.record.params) },
            ...(typeof config?.chunk === 'number' && Number.isFinite(config.chunk) ? [{ label: 'Chunk', value: fmtInt(config.chunk) }] : []),
          ]} />
        )}
        <p className="mt-4 text-sm"><a className="text-accent hover:text-accent-hover" href={RESEARCH_URL} target="_blank" rel="noreferrer">Research notes</a></p>
      </Panel>
      {plastic ? <Panel title="Transaction flow"><HarnessDiagram /></Panel> : null}
    </div>
  );
}
