import { useEffect, useRef } from 'react';
import { DECISION_COLOR, DECISION_GLYPH, DECISION_LABEL, fmt, fmtInt, isFinite_, wouldKind } from './format';
import { explainReason } from './reasons';
import { normOutcome, rollbackLimits, verdictSentence } from './status';
import { DecisionTag, NormCompare } from './statusParts';
import { useStore } from './store';
import type { TransactionRecord } from './types';
import { Button } from './ui';

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-edge px-4 py-3">
      <h3 className="text-label font-semibold uppercase tracking-wide text-ink-muted">{title}</h3>
      <div className="mt-2">{children}</div>
    </section>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-0.5">
      <dt className="text-xs text-ink-secondary">{k}</dt>
      <dd className="text-right font-mono text-xs text-ink-primary">{v}</dd>
    </div>
  );
}

/** Reasons of the applied decision, plus (observational mode) the ones the guard would have acted on. */
function reasonsOf(tx: TransactionRecord) {
  const list = [...tx.decision.reasons, ...tx.requested.reasons.filter((r) => !tx.decision.reasons.includes(r))];
  return list.filter((r) => r !== 'log_only').map(explainReason);
}

/**
 * The body of the 'why' drawer, separate from the drawer chrome so it can be tested and reused.
 */
export function WhyBody({ tx }: { tx: TransactionRecord }) {
  const k = tx.decision.kind;
  const would = wouldKind(tx);
  const reasons = reasonsOf(tx);
  const norm = normOutcome(tx);
  const limits = rollbackLimits(tx);
  const s = tx.signals;
  const z = Object.entries(s.z ?? {}).filter(([, v]) => isFinite_(v));
  return (
    <>
      <div className="px-4 py-3">
        <p className="text-md font-semibold" style={{ color: DECISION_COLOR[k] }}>
          <span aria-hidden>{DECISION_GLYPH[k]} </span>
          {DECISION_LABEL[k]}
        </p>
        <p className="mt-1 text-base text-ink-primary">{verdictSentence(tx)}</p>
        {would && k === 'commit' ? <p className="mt-1 text-sm text-ink-secondary">Nothing was blocked: this session is observational.</p> : null}
      </div>

      <Section title={k === 'commit' && !would ? 'Checks' : would && k === 'commit' ? 'What the guard flagged' : 'Why'}>
        {reasons.length === 0 ? (
          <p className="text-sm text-ink-primary">No check fired.</p>
        ) : (
          <ul role="list" className="space-y-3">
            {reasons.map((r, i) => (
              <li key={`${r.raw}-${i}`}>
                <p className="text-sm font-semibold text-ink-primary">
                  {r.title}
                  {r.would ? <span className="ml-2 text-xs font-normal text-ink-secondary">would have fired</span> : null}
                </p>
                <p className="text-sm text-ink-secondary">{r.sentence}</p>
                <p className="mt-0.5 break-all font-mono text-xs text-ink-muted">{r.raw}</p>
              </li>
            ))}
          </ul>
        )}
        {s.cusum_alarm ? <p className="mt-3 text-sm font-semibold text-status-rollback">The update-size alarm fired on this chunk.</p> : null}
      </Section>

      <Section title="Memory change: proposed vs accepted">
        <NormCompare outcome={norm} />
        <dl className="mt-2">
          <Row k="Budget charged by this chunk" v={fmt(tx.accepted?.budget_charge)} />
          <Row k="Budget used after" v={`${fmt(tx.accepted?.budget_used)}${tx.accepted?.budget_remaining === null || tx.accepted?.budget_remaining === undefined ? ' · no cap' : ` · ${fmt(tx.accepted.budget_remaining)} left`}`} />
        </dl>
      </Section>

      {limits ? (
        <Section title="What rollback did not undo">
          <p className="text-sm text-ink-primary">{limits}</p>
        </Section>
      ) : null}

      {isFinite_(s.canary_delta_coherence) || isFinite_(s.canary_delta_poison) ? (
        <Section title="Canary check">
          <dl>
            <Row k="Benign text loss change (proposed)" v={fmt(s.canary_delta_coherence)} />
            <Row k="Benign text loss change (accepted)" v={fmt(tx.accepted?.canary_delta_coherence)} />
            <Row k="Poison text loss change (proposed)" v={fmt(s.canary_delta_poison)} />
            <Row k="Poison text loss change (accepted)" v={fmt(tx.accepted?.canary_delta_poison)} />
            {isFinite_(s.canary_alignment) ? <Row k="Overlap with canary-damage direction" v={fmt(s.canary_alignment)} /> : null}
          </dl>
        </Section>
      ) : null}

      <Section title="Chunk">
        <dl>
          <Row k="Positions" v={`${tx.pos_start}–${tx.pos_end} (${fmtInt(s.n_tokens)} tokens)`} />
          {tx.sources ? <Row k="Tokens" v={`${tx.sources.user} prompt · ${tx.sources.model} generated`} /> : null}
          <Row k="Loss" v={fmt(s.chunk_loss)} />
          {isFinite_(s.surprise_mean) ? <Row k="Inner surprise" v={fmt(s.surprise_mean)} /> : null}
          {isFinite_(s.beta_mean) ? <Row k="Inner step size" v={fmt(s.beta_mean)} /> : null}
          {isFinite_(s.write_norm_sum) ? <Row k="Write norm" v={fmt(s.write_norm_sum)} /> : null}
          {z.length ? <Row k="Robust z-scores" v={z.map(([name, v]) => `${name} ${fmt(v, 2)}`).join(' · ')} /> : null}
          <Row k="Decision time" v={`${fmt(tx.seconds, 2)} s`} />
        </dl>
      </Section>
    </>
  );
}

/** Right-hand drawer for the selected chunk. Esc closes; focus moves in on open and back to the chunk button on close. */
export function WhyDrawer() {
  const selected = useStore((s) => s.selectedChunk);
  const transactions = useStore((s) => s.transactions);
  const selectChunk = useStore((s) => s.selectChunk);
  const asideRef = useRef<HTMLElement>(null);
  const opener = useRef<Element | null>(null);
  const tx = selected === null ? null : transactions.find((t) => t.index === selected) ?? null;
  const open = tx !== null;

  useEffect(() => {
    if (!open) return;
    opener.current = document.activeElement;
    asideRef.current?.querySelector<HTMLElement>('[data-close]')?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') selectChunk(null);
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      if (opener.current instanceof HTMLElement && document.contains(opener.current)) opener.current.focus();
    };
  }, [open, selectChunk]);

  if (!tx) {
    // the chunk is no longer loaded (session switched or log trimmed): say so rather than render nothing
    return selected === null ? null : (
      <aside role="dialog" aria-label="Chunk detail" className="fixed inset-y-0 right-0 z-40 w-full max-w-[420px] border-l border-edge-strong bg-surface-raised p-4 shadow-xl">
        <p className="text-sm text-ink-primary">Chunk {selected} is not in the loaded history.</p>
        <Button className="mt-3" onClick={() => selectChunk(null)}>Close</Button>
      </aside>
    );
  }
  return (
    <aside ref={asideRef} role="dialog" aria-label={`Why chunk ${tx.index}`} className="fixed inset-y-0 right-0 z-40 flex w-full max-w-[420px] flex-col border-l border-edge-strong bg-surface-raised shadow-xl">
      <header className="flex items-center justify-between gap-3 px-4 py-3">
        <div className="min-w-0">
          <h2 className="text-md font-semibold text-ink-primary">Why chunk {tx.index}</h2>
          <p className="text-xs text-ink-secondary"><DecisionTag tx={tx} /></p>
        </div>
        <Button data-close onClick={() => selectChunk(null)} aria-label="Close detail">Close</Button>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto border-t border-edge">
        <WhyBody tx={tx} />
      </div>
    </aside>
  );
}
