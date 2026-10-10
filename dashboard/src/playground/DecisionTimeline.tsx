import { useRef } from 'react';
import { DECISION_COLOR, DECISION_GLYPH, DECISION_LABEL, fmt, wouldKind } from './format';
import { explainReason } from './reasons';
import { normOutcome } from './status';
import type { TransactionRecord } from './types';

/**
 * One cell per chunk, oldest to newest. Glyph + fill = the APPLIED decision (never colour alone); a dashed
 * outline = the guard would have acted differently (observational mode); the thin bar under a cell is the share of
 * the proposed change that was actually kept. Cells are real buttons: arrow keys move, Enter or Space opens the
 * 'why' drawer for that chunk.
 */
export function DecisionTimeline({
  chunks,
  selected,
  onSelect,
  limit = 48,
  size = 'md',
}: {
  chunks: TransactionRecord[];
  selected: number | null;
  onSelect: (index: number) => void;
  limit?: number;
  size?: 'sm' | 'md';
}) {
  const sorted = chunks.slice().sort((a, b) => a.index - b.index);
  const shown = sorted.slice(-limit);
  const hidden = sorted.length - shown.length;
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  // roving tab stop: the selected chunk, else the newest
  const stop = shown.some((c) => c.index === selected) ? selected : shown[shown.length - 1]?.index;

  const move = (to: number) => {
    const t = Math.max(0, Math.min(shown.length - 1, to));
    refs.current[t]?.focus();
  };

  if (shown.length === 0) return <p className="text-sm text-ink-secondary">No chunks yet. Each message is split into chunks; every one gets a decision here.</p>;

  const cell = size === 'sm' ? 'h-7 w-6' : 'h-9 w-8';
  return (
    <div>
      <ol role="list" aria-label="Decision timeline, oldest to newest" className="flex flex-wrap gap-1">
        {hidden > 0 ? <li className="flex items-center pr-1 text-xs text-ink-secondary">+{hidden} earlier</li> : null}
        {shown.map((tx, i) => {
          const k = tx.decision.kind;
          const would = wouldKind(tx);
          const o = normOutcome(tx);
          const first = tx.decision.reasons.filter((r) => r !== 'log_only')[0];
          const label = `Chunk ${tx.index}: ${DECISION_LABEL[k]}${would ? `, guarded would ${DECISION_LABEL[would].toLowerCase()}` : ''}${first ? `. ${explainReason(first).title}` : ''}${o.ratio !== null && k !== 'readonly' ? `. Kept ${Math.round(o.ratio * 100)}% of proposed` : ''}`;
          const isSel = tx.index === selected;
          return (
            <li key={tx.index}>
              <button
                type="button"
                ref={(el) => { refs.current[i] = el; }}
                tabIndex={tx.index === stop ? 0 : -1}
                aria-label={label}
                aria-pressed={isSel}
                title={`${label}\n${fmt(tx.signals.n_tokens, 3)} tokens · positions ${tx.pos_start}–${tx.pos_end}`}
                onClick={() => onSelect(tx.index)}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowRight') { e.preventDefault(); move(i + 1); }
                  else if (e.key === 'ArrowLeft') { e.preventDefault(); move(i - 1); }
                  else if (e.key === 'Home') { e.preventDefault(); move(0); }
                  else if (e.key === 'End') { e.preventDefault(); move(shown.length - 1); }
                }}
                className={`relative flex ${cell} flex-col items-center justify-between rounded-sm border text-micro font-bold text-ink-inverse focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-1 focus-visible:ring-offset-surface-raised ${isSel ? 'border-ink-primary' : 'border-transparent'}`}
                style={{ backgroundColor: DECISION_COLOR[k], outline: would ? `2px dashed ${DECISION_COLOR[would]}` : undefined, outlineOffset: would ? '2px' : undefined }}
              >
                <span aria-hidden className="mt-0.5 leading-none">{DECISION_GLYPH[k]}</span>
                <span aria-hidden className="mb-px h-1 w-[80%] rounded-sm bg-ink-inverse" style={{ opacity: 1, width: `${o.ratio === null || k === 'readonly' ? 0 : Math.max(8, Math.min(100, Math.round(o.ratio * 100)))}%` }} />
              </button>
            </li>
          );
        })}
      </ol>
      <p className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-ink-secondary">
        {(Object.keys(DECISION_LABEL) as (keyof typeof DECISION_LABEL)[]).map((k) => (
          <span key={k}>
            <span aria-hidden style={{ color: DECISION_COLOR[k] }}>{DECISION_GLYPH[k]}</span> {DECISION_LABEL[k]}
          </span>
        ))}
        <span>Bar under a cell: share of the proposed change kept</span>
      </p>
    </div>
  );
}
