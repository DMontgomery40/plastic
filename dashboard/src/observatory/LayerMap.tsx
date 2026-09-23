import { useEffect, useMemo, useRef, useState } from 'react';
import { OBS } from '../components/charts/theme';
import { num } from './format';
import { Label } from './parts';
import { logScale, rampColor, RAMP_CSS } from './scale';
import type { Trajectory, TrajectorySession } from './types';

/** Width of an element in CSS pixels, so charts draw 1 unit = 1 px and text never scales. */
export function useWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setW(Math.floor(el.getBoundingClientRect().width));
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export type TensorPick = 'all' | 0 | 1 | 2 | 3;

/** [layer][chunk] magnitudes for one tensor, or all four combined in quadrature. Null stays null. */
export function layerGrid(session: TrajectorySession, layers: number, pick: TensorPick): (number | null)[][] {
  return Array.from({ length: layers }, (_, l) =>
    session.chunks.map((c) => {
      const v = c.per_layer;
      if (!v) return null;
      if (pick === 'all') {
        const parts = [0, 1, 2, 3].map((k) => v[l * 4 + k]);
        if (parts.some((x) => x === null || x === undefined)) return null;
        return Math.sqrt(parts.reduce((a: number, x) => a + (x as number) ** 2, 0));
      }
      const x = v[l * 4 + pick];
      return x === undefined ? null : x;
    })
  );
}

function turnOf(session: TrajectorySession, chunk: number): number | null {
  const t = session.turns.find((tr) => chunk >= tr.tx[0] && chunk < tr.tx[1]);
  return t ? t.i : null;
}

const LABEL_W = 64;
const ROW_FLAG = 13;
const ROW_DECISION = 13;

export function LayerMap({ trajectory, session: sessionId }: { trajectory: Trajectory; session: string }) {
  const session = trajectory.sessions.find((s) => s.session_id === sessionId) ?? trajectory.sessions[0];
  const layers = trajectory.per_layer.layers;
  const tensors = trajectory.per_layer.tensors;
  const [pick, setPick] = useState<TensorPick>('all');
  const [view, setView] = useState<'flat' | 'terrain'>('flat');
  const [tilt, setTilt] = useState(0.5);
  const [hover, setHover] = useState<{ chunk: number; layer: number } | null>(null);
  const [ref, width] = useWidth<HTMLDivElement>();
  const grid = useMemo(() => layerGrid(session, layers, pick), [session, layers, pick]);
  const scale = useMemo(() => logScale(grid.flat()), [grid]);
  const n = session.chunks.length;
  const plotW = Math.max(120, width - LABEL_W);
  const cw = plotW / n;
  const ch = 9;
  const mapH = layers * ch;
  const cell = hover ? session.chunks[hover.chunk] : null;
  const turn = hover ? turnOf(session, hover.chunk) : null;
  const turnText = turn !== null ? session.turns[turn] : null;

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - r.left - LABEL_W;
    const y = e.clientY - r.top - ROW_FLAG - ROW_DECISION - 6;
    const chunk = Math.floor(x / cw);
    const layer = layers - 1 - Math.floor(y / ch);
    setHover(chunk >= 0 && chunk < n && layer >= 0 && layer < layers ? { chunk, layer } : null);
  };

  const turnStarts = session.turns.map((t) => t.tx[0]).filter((x) => x > 0 && x < n);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <div role="group" aria-label="Tensor" className="flex flex-wrap gap-1">
          {(['all', 0, 1, 2, 3] as TensorPick[]).map((p) => (
            <button
              key={String(p)}
              type="button"
              aria-pressed={pick === p}
              onClick={() => setPick(p)}
              className={`rounded border px-2 py-0.5 font-mono text-xs font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                pick === p ? 'border-accent bg-accent-soft text-accent' : 'border-edge-strong bg-surface-overlay text-ink-secondary hover:text-ink-primary'
              }`}
            >
              {p === 'all' ? 'W1 b1 W2 b2 combined' : tensors[p]}
            </button>
          ))}
        </div>
        <div role="group" aria-label="View" className="ml-auto flex rounded border border-edge-strong bg-surface-overlay p-0.5">
          {(['flat', 'terrain'] as const).map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={view === v}
              onClick={() => setView(v)}
              className={`rounded px-2.5 py-0.5 text-xs font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${view === v ? 'bg-accent-soft text-accent' : 'text-ink-secondary hover:text-ink-primary'}`}
            >
              {v === 'flat' ? 'Map' : 'Terrain'}
            </button>
          ))}
        </div>
      </div>
      <div ref={ref} className="min-w-0">
        {width > 0 && view === 'flat' ? (
          <svg
            width={width}
            height={ROW_FLAG + ROW_DECISION + 6 + mapH + 22}
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
            role="img"
            aria-label={`Proposed fast-weight change per layer and chunk, ${session.session_id} session, ${layers} layers by ${n} chunks, log color scale from ${num(scale.lo)} to ${num(scale.hi)}`}
            shapeRendering="crispEdges"
          >
            <text x={LABEL_W - 6} y={ROW_FLAG - 3} textAnchor="end" fontSize={11} fill={OBS.inkMuted}>flag</text>
            <text x={LABEL_W - 6} y={ROW_FLAG + ROW_DECISION - 3} textAnchor="end" fontSize={11} fill={OBS.inkMuted}>kept</text>
            {session.chunks.map((c, i) => (
              <g key={i}>
                {c.flags.length ? <rect x={LABEL_W + i * cw} width={Math.max(1, cw - 0.5)} y={2} height={ROW_FLAG - 5} fill={OBS.flag} /> : null}
                <rect x={LABEL_W + i * cw} y={ROW_FLAG + 2} width={Math.max(1, cw - 0.5)} height={ROW_DECISION - 5} fill={c.applied === 'rollback' ? OBS.rollbackFill : c.applied === 'commit' ? OBS.commitFill : OBS.notInForce} />
              </g>
            ))}
            <g transform={`translate(0 ${ROW_FLAG + ROW_DECISION + 6})`}>
              {grid.map((row, l) =>
                row.map((v, i) => {
                  const t = scale.t(v);
                  return <rect key={`${l}-${i}`} x={LABEL_W + i * cw} y={(layers - 1 - l) * ch} width={Math.max(1, cw)} height={ch} fill={t === null ? OBS.inset : rampColor(t)} />;
                })
              )}
              {turnStarts.map((x) => (
                <line key={x} x1={LABEL_W + x * cw} x2={LABEL_W + x * cw} y1={0} y2={mapH} stroke={OBS.surface} strokeWidth={1} />
              ))}
              {hover ? <rect x={LABEL_W + hover.chunk * cw} y={(layers - 1 - hover.layer) * ch} width={Math.max(2, cw)} height={ch} fill="none" stroke={OBS.ink} strokeWidth={1.5} /> : null}
              {[0, Math.floor(layers / 2), layers - 1].map((l) => (
                <text key={l} x={LABEL_W - 6} y={(layers - 1 - l) * ch + ch - 1} textAnchor="end" fontSize={11} fill={OBS.inkMuted}>
                  layer {l}
                </text>
              ))}
              <text x={LABEL_W} y={mapH + 15} fontSize={11} fill={OBS.inkMuted}>chunk 0</text>
              <text x={LABEL_W + plotW} y={mapH + 15} textAnchor="end" fontSize={11} fill={OBS.inkMuted}>chunk {n - 1}</text>
              <text x={LABEL_W + plotW / 2} y={mapH + 15} textAnchor="middle" fontSize={11} fill={OBS.inkMuted}>
                {session.turns.length} turns, separated by thin gaps
              </text>
            </g>
          </svg>
        ) : null}
        {width > 0 && view === 'terrain' ? <Terrain grid={grid} scale={scale} width={width} tilt={tilt} /> : null}
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-ink-secondary">
        <span className="flex items-center gap-2">
          <span className="font-mono">{num(scale.lo)}</span>
          <span className="inline-block h-2.5 w-28 rounded-sm" style={{ background: RAMP_CSS }} />
          <span className="font-mono">{num(scale.hi)}</span>
          <span className="text-ink-muted">proposed change norm, log scale</span>
        </span>
        {view === 'flat' ? (
          <>
            <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: OBS.flag }} /> intervention flagged</span>
            <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: OBS.commitFill }} /> committed</span>
            <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: OBS.rollbackFill }} /> rolled back</span>
          </>
        ) : (
          <span className="text-ink-muted">height and color: the same log scale; layer 0 in front</span>
        )}
        {view === 'terrain' ? (
          <label className="ml-auto flex items-center gap-2">
            tilt
            <input type="range" min={0.15} max={0.9} step={0.05} value={tilt} onChange={(e) => setTilt(Number(e.target.value))} aria-label="Terrain tilt" />
          </label>
        ) : null}
      </div>
      {view === 'flat' ? (
      <div className="min-h-[3.25rem] rounded border border-edge bg-surface-inset px-3 py-2 text-sm" aria-live="polite">
        {hover && cell ? (
          <>
            <div className="font-mono text-ink-primary">
              turn {turn !== null ? turn + 1 : 'n/a'} · chunk {cell.i} ({cell.source} tokens) · layer {hover.layer} · {pick === 'all' ? 'combined' : tensors[pick]} {num(grid[hover.layer][hover.chunk])}
              <span className="text-ink-secondary"> · chunk total {num(cell.proposed)} proposed, {num(cell.accepted)} kept · surprise {num(cell.surprise_mean)} · loss {num(cell.chunk_loss)}</span>
            </div>
            <div className="truncate text-ink-secondary">
              {cell.flags.length ? <span className="text-status-scale">{cell.flags.join(', ')} · </span> : null}
              {turnText ? `“${turnText.prompt}”` : ''}
            </div>
          </>
        ) : (
          <span className="text-ink-muted">Point at a cell to read its layer, chunk, turn and signals.</span>
        )}
      </div>
      ) : null}
    </div>
  );
}

/** Secondary view: each layer as a ridge, stacked in depth. Labels stay upright; heights use the same log scale. */
function Terrain({ grid, scale, width, tilt }: { grid: (number | null)[][]; scale: ReturnType<typeof logScale>; width: number; tilt: number }) {
  const layers = grid.length;
  const n = grid[0]?.length ?? 0;
  const gap = 4 + tilt * 10;
  const skew = tilt * 3;
  const amp = 46;
  const plotW = width - LABEL_W - skew * layers;
  const height = amp + gap * layers + 30;
  const ridges = [];
  for (let l = layers - 1; l >= 0; l--) {
    const x0 = LABEL_W + l * skew;
    const y0 = amp + (layers - 1 - l) * gap;
    const pts = grid[l].map((v, i) => `${(x0 + (i / Math.max(1, n - 1)) * plotW).toFixed(1)},${(y0 - (scale.t(v) ?? 0) * amp).toFixed(1)}`);
    const d = `M${x0},${y0} L${pts.join(' L')} L${x0 + plotW},${y0} Z`;
    ridges.push(<path key={l} d={d} fill={OBS.surface} stroke={rampColor(0.35 + 0.65 * (l / Math.max(1, layers - 1)))} strokeWidth={1.25} strokeLinejoin="round" />);
  }
  return (
    <svg width={width} height={height} role="img" aria-label={`Terrain: ${layers} layers as ridges over ${n} chunks, heights on the log scale`}>
      {ridges}
      {[0, Math.floor(layers / 2), layers - 1].map((l) => (
        <text key={l} x={LABEL_W - 6} y={amp + (layers - 1 - l) * gap + 4} textAnchor="end" fontSize={11} fill={OBS.inkMuted}>
          layer {l}
        </text>
      ))}
      <text x={LABEL_W} y={height - 6} fontSize={11} fill={OBS.inkMuted}>chunk 0</text>
      <text x={LABEL_W + plotW} y={height - 6} textAnchor="end" fontSize={11} fill={OBS.inkMuted}>chunk {n - 1}</text>
    </svg>
  );
}

export function SessionPicker({ trajectory, value, onChange }: { trajectory: Trajectory; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex flex-col gap-1">
      <Label>Session</Label>
      <div role="group" aria-label="Session" className="flex gap-1">
        {trajectory.sessions.map((s) => (
          <button
            key={s.session_id}
            type="button"
            aria-pressed={value === s.session_id}
            onClick={() => onChange(s.session_id)}
            className={`rounded border px-2.5 py-1 text-sm font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              value === s.session_id ? 'border-accent bg-accent-soft text-accent' : 'border-edge-strong bg-surface-overlay text-ink-secondary hover:text-ink-primary'
            }`}
          >
            {s.session_id === 'teach' ? 'Teaching' : s.session_id === 'rolled' ? 'Rolled back' : s.session_id}
          </button>
        ))}
      </div>
    </div>
  );
}
