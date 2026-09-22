import { useState } from 'react';
import type { TransactionRecord } from '../../api/types';
import { decisionColor, decisionLabel, fmt } from '../../utils/formatting';
import { DecisionLegend } from '../panels';

interface Props {
  transactions: TransactionRecord[];
  selected: number | null;
  onSelect: (index: number) => void;
}

/**
 * One mark per chunk transaction, in order, coloured by the decision that was
 * actually applied. Clicking selects a chunk; hovering reveals the reasons the
 * policy recorded for it.
 */
export function TransactionTimeline({ transactions, selected, onSelect }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);
  const shown = hovered ?? selected;
  const detail = transactions.find((t) => t.index === shown) ?? null;

  return (
    <div>
      <div className="flex flex-wrap gap-[3px]" onMouseLeave={() => setHovered(null)}>
        {transactions.map((tx) => {
          const color = decisionColor(tx.decision.kind);
          const isSelected = selected === tx.index;
          return (
            <button
              key={tx.index}
              type="button"
              aria-label={`Chunk ${tx.index}, ${decisionLabel(tx.decision.kind)}`}
              aria-pressed={isSelected}
              onClick={() => onSelect(tx.index)}
              onMouseEnter={() => setHovered(tx.index)}
              onFocus={() => setHovered(tx.index)}
              className="h-7 w-3.5 rounded-sm border transition-transform hover:scale-y-110 focus:outline-none focus:ring-1 focus:ring-accent"
              style={{
                backgroundColor: color,
                borderColor: isSelected ? '#e9eff5' : color,
                borderWidth: isSelected ? 2 : 1,
              }}
            />
          );
        })}
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <DecisionLegend />
        <span className="text-micro text-ink-muted">
          {transactions.length} chunk{transactions.length === 1 ? '' : 's'}, oldest first
        </span>
      </div>

      {detail ? (
        <div className="mt-3 rounded border border-edge bg-surface-overlay px-3 py-2.5">
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <span className="font-mono text-sm font-semibold" style={{ color: decisionColor(detail.decision.kind) }}>
              chunk {detail.index} · {decisionLabel(detail.decision.kind)}
            </span>
            <span className="font-mono text-xs text-ink-secondary">
              pos {detail.pos_start}–{detail.pos_end}
            </span>
            <span className="font-mono text-xs text-ink-secondary">loss {fmt(detail.signals.chunk_loss, 4)}</span>
            <span className="font-mono text-xs text-ink-secondary">‖Δ‖ {fmt(detail.signals.delta_norm, 4)}</span>
            {detail.decision.kind === 'scale' ? (
              <span className="font-mono text-xs text-ink-secondary">β scale {fmt(detail.decision.scale, 3)}</span>
            ) : null}
          </div>
          {detail.decision.reasons.length > 0 ? (
            <ul className="mt-1.5 flex flex-wrap gap-1.5">
              {detail.decision.reasons.map((reason, i) => (
                <li key={`${reason}-${i}`} className="rounded border border-edge bg-surface-inset px-1.5 py-0.5 font-mono text-micro text-ink-secondary">
                  {reason}
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-1.5 text-micro text-ink-muted">No reason recorded: every signal stayed inside its threshold.</p>
          )}
          {detail.requested.kind !== detail.decision.kind ? (
            <p className="mt-1.5 text-micro text-ink-secondary">
              Policy asked for {decisionLabel(detail.requested.kind)}; the runner applied{' '}
              {decisionLabel(detail.decision.kind)}.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
