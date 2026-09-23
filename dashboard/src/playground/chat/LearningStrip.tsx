import { DECISION_COLOR, DECISION_GLYPH, DECISION_LABEL, chunkSize, chunkSource, fmt, isFinite_, wouldIntervene } from '../format';
import type { TransactionRecord } from '../types';

/**
 * One cell per chunk of a turn. Fill = the APPLIED decision; a dashed outline = the policy REQUESTED a different
 * decision (observational mode records what would have happened). Height = the chunk's own write (or proposed
 * change), width = its token count. A thin bar marks accepted change where it differs from proposed. Hue is
 * always paired with a glyph, so nothing depends on color alone.
 */
export function LearningStrip({ chunks, height = 44 }: { chunks: TransactionRecord[]; height?: number }) {
  if (chunks.length === 0) return <p className="text-xs text-ink-muted">No chunks recorded for this turn.</p>;
  const sizes = chunks.map(chunkSize);
  const max = Math.max(1e-9, ...sizes.filter(isFinite_));
  const tokens = chunks.map((c) => Math.max(1, c.signals.n_tokens));
  const totalTokens = tokens.reduce((a, b) => a + b, 0);
  return (
    <ol role="list" aria-label="Learning strip: one cell per chunk" className="flex w-full items-end gap-px" style={{ height }}>
      {chunks.map((tx, i) => {
        const applied = tx.decision.kind;
        const requested = tx.requested.kind;
        const frac = isFinite_(sizes[i]) ? Math.max(0.08, sizes[i] / max) : 0;
        const would = wouldIntervene(tx);
        const accepted = tx.accepted?.delta_norm;
        const proposed = tx.signals.delta_norm;
        const kept = isFinite_(accepted) && isFinite_(proposed) && proposed > 0 ? Math.min(1, accepted / proposed) : 1;
        const src = chunkSource(tx);
        const glyph = src === 'prompt' ? '▲' : src === 'model' ? '●' : src === 'mixed' ? '◆' : '·';
        const title = [
          `chunk ${tx.index} · ${DECISION_LABEL[applied]}${would ? ` (policy would ${DECISION_LABEL[requested].toLowerCase()})` : ''}`,
          `${tx.signals.n_tokens} tokens · ${src}`,
          `loss ${fmt(tx.signals.chunk_loss)}`,
          isFinite_(tx.signals.surprise_mean) ? `surprise ${fmt(tx.signals.surprise_mean)}` : null,
          isFinite_(tx.signals.beta_mean) ? `step ${fmt(tx.signals.beta_mean)}` : null,
          isFinite_(tx.signals.write_norm_sum) ? `write ${fmt(tx.signals.write_norm_sum)}` : null,
          `proposed Δ ${fmt(proposed)} · accepted Δ ${fmt(accepted)}`,
          tx.signals.cusum_alarm ? 'CUSUM alarm' : null,
        ]
          .filter(Boolean)
          .join('\n');
        return (
          <li
            key={tx.index}
            title={title}
            aria-label={title.replace(/\n/g, ', ')}
            className="relative flex min-w-[6px] items-end justify-center"
            style={{ width: `${(tokens[i] / totalTokens) * 100}%`, height: '100%' }}
          >
            <div
              className="relative w-full rounded-sm"
              style={{
                height: `${Math.round(frac * 100)}%`,
                backgroundColor: DECISION_COLOR[applied],
                outline: would ? `2px dashed ${DECISION_COLOR[requested]}` : undefined,
                outlineOffset: would ? '-2px' : undefined,
              }}
            >
              {kept < 0.999 ? (
                <div
                  aria-hidden
                  className="absolute inset-x-0 bottom-0 border-t-2 border-ink-primary"
                  style={{ height: `${Math.round(kept * 100)}%` }}
                />
              ) : null}
            </div>
            <span aria-hidden className="pointer-events-none absolute -top-0.5 text-micro leading-none text-ink-secondary">
              {glyph}
            </span>
            <span aria-hidden className="pointer-events-none absolute bottom-0 text-micro leading-none text-ink-inverse">
              {DECISION_GLYPH[applied]}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

export function StripLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-micro text-ink-secondary">
      {(Object.keys(DECISION_LABEL) as (keyof typeof DECISION_LABEL)[]).map((k) => (
        <span key={k} className="inline-flex items-center gap-1">
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: DECISION_COLOR[k] }} />
          {DECISION_GLYPH[k]} {DECISION_LABEL[k]}
        </span>
      ))}
      <span>▲ prompt tokens · ● generated tokens</span>
      <span>dashed outline = policy would have intervened</span>
      <span>bar = accepted share of proposed change</span>
    </div>
  );
}
