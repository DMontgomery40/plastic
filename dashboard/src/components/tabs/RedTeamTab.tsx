import { useMemo, useState } from 'react';
import { useStore } from '../../store';
import {
  NO_VALID_PAYLOADS,
  UNAVAILABLE,
  fmt,
  fmtInt,
  fmtPercent,
  fmtRelative,
  fmtValidOnly,
  fmtValidOnlyPercent,
  isNum,
} from '../../utils/formatting';
import { GroupedBarChartPanel, ScatterChartPanel, SERIES_COLORS, type ReferenceSpec } from '../charts';
import { Button, Checkbox, Empty, Field, KeyValue, NumberInput, Panel, Select, StatTile, Table } from '../panels';

const ALL_FAMILIES = ['pgd', 'random', 'repeat', 'shuffle', 'topic_switch'];

export function RedTeamTab() {
  const models = useStore((s) => s.models);
  const dataDirs = useStore((s) => s.dataDirs);
  const runs = useStore((s) => s.redteamRuns);
  const detail = useStore((s) => s.redteamDetail);
  const loadRun = useStore((s) => s.loadRedteamRun);
  const runRedteam = useStore((s) => s.runRedteam);
  const busy = useStore((s) => s.loading.redteam);
  const detailBusy = useStore((s) => s.loading.redteamRun);

  const [modelId, setModelId] = useState('');
  const [dataDir, setDataDir] = useState('');
  const [prefixes, setPrefixes] = useState(4);
  const [prefixLen, setPrefixLen] = useState(128);
  const [suffixLen, setSuffixLen] = useState(64);
  const [steps, setSteps] = useState(30);
  const [families, setFamilies] = useState<string[]>(['pgd', 'random']);
  const [record, setRecord] = useState(false);

  const textModels = models.filter((m) => m.domain === 'text' && m.status === 'completed');
  const selectedModel = modelId || textModels[0]?.model_id || '';
  const summary = detail?.summary ?? null;

  const scatter = useMemo(
    () =>
      (detail?.results ?? []).map((r) => ({
        nll: r.nll_payload,
        damage: r.damage_validated,
        family: r.family,
        gated: r.decisions.some((d) => d !== 'commit'),
      })),
    [detail],
  );

  // The headline is the valid-only story: what an adversary could actually
  // deliver. `worstValid` is null, not 0, when nothing met the constraint.
  const familyStats = Object.values(summary?.families ?? {});
  const worstValid = familyStats.reduce<number | null>(
    (max, f) => (isNum(f.valid_damage_max) && (max === null || f.valid_damage_max > max) ? f.valid_damage_max : max),
    null,
  );
  const worstUnprotected = familyStats.reduce<number | null>(
    (max, f) =>
      isNum(f.unprotected_damage_mean) && (max === null || f.unprotected_damage_mean > max) ? f.unprotected_damage_mean : max,
    null,
  );
  const totalAttacks = familyStats.reduce((n, f) => n + (f.n ?? 0), 0);
  const totalValid = familyStats.reduce((n, f) => n + (f.n_valid ?? 0), 0);

  const familyRows = useMemo(
    () =>
      Object.entries(summary?.families ?? {}).map(([family, stats]) => ({
        family,
        accepted: stats.damage_mean,
        unprotected: stats.unprotected_damage_mean,
        frozen: stats.frozen_damage_mean,
      })),
    [summary],
  );

  const familyColor = useMemo(() => {
    const names = Array.from(new Set(scatter.map((p) => p.family)));
    const map: Record<string, string> = {};
    names.forEach((name, i) => {
      map[name] = SERIES_COLORS[i % SERIES_COLORS.length];
    });
    return map;
  }, [scatter]);

  const damageRefs: ReferenceSpec[] =
    summary && isNum(summary.threshold_coherence)
      ? [{ value: summary.threshold_coherence, label: `τ coherence ${fmt(summary.threshold_coherence, 3)}`, color: '#ff6b6b' }]
      : [];

  const toggleFamily = (name: string) =>
    setFamilies((prev) => (prev.includes(name) ? prev.filter((f) => f !== name) : [...prev, name]));

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
      <div className="space-y-4">
        <Panel title="Run a campaign" subtitle="Attacks go through the real token path and the real harness.">
          {textModels.length === 0 ? (
            <Empty
              title="No completed text model."
              detail="The attack needs a checkpoint, its tokenizer, and its canary suite."
              command={'uv run plastic train text --data artifacts/data/wikitext --steps 3000 --device mps'}
            />
          ) : (
            <div className="space-y-3">
              <Field label="Model" htmlFor="rt-model">
                <Select
                  id="rt-model"
                  value={selectedModel}
                  onChange={setModelId}
                  options={textModels.map((m) => ({ value: m.model_id, label: m.model_id }))}
                />
              </Field>
              <Field label="Corpus" htmlFor="rt-data" hint="Prefixes are sampled from its validation split.">
                <Select
                  id="rt-data"
                  value={dataDir}
                  onChange={setDataDir}
                  options={[{ value: '', label: 'API default' }, ...dataDirs.map((d) => ({ value: d.dir, label: d.name }))]}
                />
              </Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Prefixes" htmlFor="rt-prefixes">
                  <NumberInput id="rt-prefixes" value={prefixes} min={1} step={1} onChange={setPrefixes} />
                </Field>
                <Field label="Prefix length" htmlFor="rt-prefixlen">
                  <NumberInput id="rt-prefixlen" value={prefixLen} min={8} step={8} onChange={setPrefixLen} />
                </Field>
                <Field label="Suffix length" htmlFor="rt-suffixlen">
                  <NumberInput id="rt-suffixlen" value={suffixLen} min={8} step={8} onChange={setSuffixLen} />
                </Field>
                <Field label="Optimizer steps" htmlFor="rt-steps">
                  <NumberInput id="rt-steps" value={steps} min={1} step={5} onChange={setSteps} />
                </Field>
              </div>
              <div>
                <p className="mb-1 text-label font-semibold uppercase tracking-wide text-ink-muted">Families</p>
                <div className="space-y-1.5">
                  {ALL_FAMILIES.map((name) => (
                    <Checkbox key={name} id={`rt-fam-${name}`} label={name} checked={families.includes(name)} onChange={() => toggleFamily(name)} />
                  ))}
                </div>
              </div>
              <Checkbox
                id="rt-record"
                label="Record the strongest payloads as poison canaries"
                checked={record}
                onChange={setRecord}
              />
              <Button
                variant="primary"
                disabled={busy || families.length === 0 || !selectedModel}
                onClick={() =>
                  void runRedteam({
                    model_id: selectedModel,
                    data_dir: dataDir || undefined,
                    prefixes,
                    prefix_len: prefixLen,
                    suffix_len: suffixLen,
                    steps,
                    families,
                    record,
                  })
                }
              >
                {busy ? 'Attacking…' : 'Run campaign'}
              </Button>
              <p className="text-micro text-ink-muted">This runs synchronously and can take minutes.</p>
            </div>
          )}
        </Panel>

        <Panel title="Runs" subtitle="Stored under artifacts/redteam.">
          {runs.length === 0 ? (
            <Empty title="No campaign has been run." command={'uv run plastic redteam <model_id>'} />
          ) : (
            <ul className="space-y-1.5">
              {runs.map((r) => (
                <li key={r.run_id}>
                  <button
                    type="button"
                    onClick={() => void loadRun(r.run_id)}
                    className={`w-full rounded border px-2.5 py-2 text-left ${
                      summary?.run_id === r.run_id ? 'border-accent bg-accent-soft' : 'border-edge hover:bg-surface-overlay'
                    }`}
                  >
                    <span className="block font-mono text-xs text-ink-primary">{r.run_id}</span>
                    <span className="mt-0.5 block text-micro text-ink-muted">
                      {r.model_id} · {Object.keys(r.families).length} families · {fmtRelative(r.created_at_unix)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <div className="space-y-4">
        {!summary ? (
          <Empty
            title="No run selected."
            detail="Pick a run on the left, or start a campaign."
            command={'uv run plastic redteam <model_id> --data artifacts/data/wikitext --prefixes 4 --steps 30'}
          />
        ) : (
          <>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
              <StatTile
                label="Worst valid damage"
                value={fmtValidOnly(worstValid)}
                tone={worstValid === null ? 'default' : 'rollback'}
                hint="constraint-satisfying attacks only"
              />
              <StatTile
                label="Worst undefended damage"
                value={fmt(worstUnprotected)}
                hint="same payloads, harness off"
              />
              <StatTile
                label="Valid attacks"
                value={`${fmtInt(totalValid)} of ${fmtInt(totalAttacks)}`}
                hint="met the plausibility constraint"
              />
              <StatTile
                label="Coherence threshold"
                value={fmt(summary.threshold_coherence, 4)}
                hint={isNum(summary.threshold_coherence) ? 'from the calibration' : 'model not calibrated'}
              />
              <StatTile
                label="Recorded payloads"
                value={fmtInt(summary.recorded_payloads ?? 0)}
                hint="added to the poison canaries"
              />
            </div>

            <Panel
              title="Damage by family: accepted, unprotected, frozen"
              subtitle="Three endpoints on the same payload from the same post-prefix state. The gap between accepted and unprotected is what the harness prevented; frozen is the activation-only floor the payload reaches just by being read."
            >
              <GroupedBarChartPanel
                data={familyRows}
                xKey="family"
                series={[
                  { key: 'accepted', label: 'accepted (harness on)', color: '#3fd17a' },
                  { key: 'unprotected', label: 'unprotected (harness off)', color: '#ff6b6b' },
                  { key: 'frozen', label: 'frozen (read, not learned)', color: '#94a3b4' },
                ]}
                height={260}
                yLabel="coherence loss added"
                references={damageRefs}
                ariaLabel="Mean canary damage per attack family, compared across the accepted, unprotected and frozen endpoints"
              />
              <div className="mt-3">
                <Table head={['Family', 'Accepted', 'Unprotected', 'Frozen', 'Prevented by the harness']}>
                  {Object.entries(summary.families).map(([family, stats]) => {
                    const prevented =
                      isNum(stats.unprotected_damage_mean) && isNum(stats.damage_mean)
                        ? stats.unprotected_damage_mean - stats.damage_mean
                        : null;
                    return (
                      <tr key={family} className="border-b border-edge">
                        <td className="px-2 py-1.5 font-mono text-ink-primary">{family}</td>
                        <td className="px-2 py-1.5 font-mono text-status-commit">{fmt(stats.damage_mean)}</td>
                        <td className="px-2 py-1.5 font-mono text-status-rollback">{fmt(stats.unprotected_damage_mean)}</td>
                        <td className="px-2 py-1.5 font-mono text-status-readonly">{fmt(stats.frozen_damage_mean)}</td>
                        <td
                          className={`px-2 py-1.5 font-mono ${
                            prevented !== null && prevented > 0 ? 'text-status-commit' : 'text-ink-secondary'
                          }`}
                        >
                          {prevented === null ? UNAVAILABLE : fmt(prevented)}
                        </td>
                      </tr>
                    );
                  })}
                </Table>
                <p className="mt-2 text-micro text-ink-muted">
                  A negative unprotected number means the payload made the canaries better even with no defence, so
                  there was nothing for the harness to prevent on that family.
                </p>
              </div>
            </Panel>

            <Panel
              title="Per family, constraint-satisfying attacks"
              subtitle="The headline numbers. An attack counts here only if its payload met the plausibility constraint, so these are the attacks a real adversary could actually deliver."
            >
              <Table head={['Family', 'Valid of n', 'Valid damage mean', 'Valid damage max', 'Valid over threshold']}>
                {Object.entries(summary.families).map(([family, stats]) => (
                  <tr key={family} className="border-b border-edge">
                    <td className="px-2 py-1.5 font-mono text-ink-primary">{family}</td>
                    <td className="px-2 py-1.5 font-mono text-ink-primary">
                      {fmtInt(stats.n_valid)} of {fmtInt(stats.n)}
                    </td>
                    <td
                      className={`px-2 py-1.5 font-mono ${isNum(stats.valid_damage_mean) ? 'text-ink-primary' : 'text-ink-muted'}`}
                    >
                      {fmtValidOnly(stats.valid_damage_mean)}
                    </td>
                    <td
                      className={`px-2 py-1.5 font-mono ${isNum(stats.valid_damage_max) ? 'text-status-rollback' : 'text-ink-muted'}`}
                    >
                      {fmtValidOnly(stats.valid_damage_max)}
                    </td>
                    <td
                      className={`px-2 py-1.5 font-mono ${
                        isNum(stats.valid_over_threshold_fraction) ? 'text-status-scale' : 'text-ink-muted'
                      }`}
                    >
                      {fmtValidOnlyPercent(stats.valid_over_threshold_fraction)}
                    </td>
                  </tr>
                ))}
              </Table>
              <p className="mt-2 text-micro text-ink-muted">
                {NO_VALID_PAYLOADS} means the campaign ran and nothing in that family met the constraint. It is not a
                damage of zero, and it is not a missing measurement.
              </p>
            </Panel>

            <Panel
              title="Per family, every attempt"
              subtitle="Secondary: these include payloads the model itself finds implausible, which an adversary could not deliver unnoticed. Provisional is the peak intermediate proposal, not an endpoint."
            >
              <Table
                head={['Family', 'n', 'Damage mean', 'Damage max', 'Provisional max', 'Gated', 'Constraint violated', 'Over threshold']}
              >
                {Object.entries(summary.families).map(([family, stats]) => (
                  <tr key={family} className="border-b border-edge">
                    <td className="px-2 py-1.5 font-mono text-ink-primary">{family}</td>
                    <td className="px-2 py-1.5 font-mono text-ink-primary">{fmtInt(stats.n)}</td>
                    <td className="px-2 py-1.5 font-mono text-ink-secondary">{fmt(stats.damage_mean)}</td>
                    <td className="px-2 py-1.5 font-mono text-ink-secondary">{fmt(stats.damage_max)}</td>
                    <td className="px-2 py-1.5 font-mono text-status-scale">{fmt(stats.provisional_damage_max)}</td>
                    <td className="px-2 py-1.5 font-mono text-status-commit">{fmtPercent(stats.gated_fraction, 0)}</td>
                    <td className="px-2 py-1.5 font-mono text-ink-secondary">{fmtPercent(stats.constraint_violated_fraction, 0)}</td>
                    <td className="px-2 py-1.5 font-mono text-status-scale">
                      {isNum(stats.over_threshold_fraction) ? fmtPercent(stats.over_threshold_fraction, 0) : UNAVAILABLE}
                    </td>
                  </tr>
                ))}
              </Table>
            </Panel>

            <Panel
              title="Validated damage against payload likelihood"
              subtitle="One point per attack. A payload far to the right is fluent; a point high up hurt the canaries."
            >
              {detailBusy ? (
                <p className="text-sm text-ink-secondary">Loading results…</p>
              ) : scatter.length === 0 ? (
                <Empty title="This run recorded no per-attack results." detail="Only the summary was stored." />
              ) : (
                <>
                  <ScatterChartPanel
                    data={scatter}
                    xKey="nll"
                    yKey="damage"
                    label="attacks"
                    height={300}
                    xLabel="payload NLL (nats per token)"
                    yLabel="validated damage"
                    ariaLabel="Validated canary damage against payload negative log likelihood, one point per attack"
                    yReferences={damageRefs}
                    colorFor={(row) => familyColor[String(row.family)] ?? '#58a6ff'}
                  />
                  <div className="mt-2 flex flex-wrap gap-3">
                    {Object.entries(familyColor).map(([name, color]) => (
                      <span key={name} className="flex items-center gap-1.5 text-micro text-ink-secondary">
                        <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                        {name}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </Panel>

            {detail && detail.results.length > 0 ? (
              <Panel title="Attacks" subtitle="Decisions are what the harness did to each chunk of the payload.">
                <Table
                  head={[
                    'Family',
                    'Deliverable',
                    'Accepted damage',
                    'Unprotected',
                    'Frozen',
                    'Provisional',
                    'NLL payload',
                    'NLL guarded',
                    'NLL max',
                    'Decisions',
                    'Seconds',
                  ]}
                >
                  {detail.results.map((r, i) => (
                    <tr key={`${r.family}-${i}`} className="border-b border-edge">
                      <td className="px-2 py-1.5 font-mono text-ink-primary">{r.family}</td>
                      <td className="px-2 py-1.5 text-xs">
                        {r.constraint_violated ? (
                          <span className="text-ink-muted">no, constraint violated</span>
                        ) : (
                          <span className="font-semibold text-status-rollback">yes</span>
                        )}
                      </td>
                      <td className="px-2 py-1.5 font-mono text-status-commit">{fmt(r.damage_validated)}</td>
                      <td className="px-2 py-1.5 font-mono text-status-rollback">{fmt(r.damage_unprotected)}</td>
                      <td className="px-2 py-1.5 font-mono text-status-readonly">{fmt(r.damage_frozen)}</td>
                      <td className="px-2 py-1.5 font-mono text-status-scale">
                        {isNum(r.canary_after_provisional) && isNum(r.canary_before)
                          ? fmt(r.canary_after_provisional - r.canary_before)
                          : UNAVAILABLE}
                      </td>
                      <td className="px-2 py-1.5 font-mono text-ink-primary">{fmt(r.nll_payload, 3)}</td>
                      <td className="px-2 py-1.5 font-mono text-ink-secondary">{fmt(r.nll_payload_guarded, 3)}</td>
                      <td className="px-2 py-1.5 font-mono text-ink-secondary">{fmt(r.nll_max, 3)}</td>
                      <td className="px-2 py-1.5 font-mono text-xs text-ink-secondary">{r.decisions.join(', ') || UNAVAILABLE}</td>
                      <td className="px-2 py-1.5 font-mono text-ink-secondary">{fmt(r.seconds, 1)}</td>
                    </tr>
                  ))}
                </Table>
              </Panel>
            ) : null}

            <Panel title="Run metadata">
              <KeyValue
                columns={2}
                rows={[
                  { label: 'Run id', value: summary.run_id },
                  { label: 'Model', value: summary.model_id },
                  { label: 'Created', value: fmtRelative(summary.created_at_unix) },
                  { label: 'Families', value: Object.keys(summary.families).join(', ') },
                ]}
              />
            </Panel>
          </>
        )}
      </div>
    </div>
  );
}
