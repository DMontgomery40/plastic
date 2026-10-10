import { fmt, fmtInt, turnTotals, turnsFromTrace } from './format';
import { DecisionTimeline } from './DecisionTimeline';
import { normOutcome, sessionStatus } from './status';
import { DecisionTag, GaugeBar, LIGHT_CLASS, LightBadge } from './statusParts';
import { useStore } from './store';
import { Button } from './ui';

/**
 * The first thing on every screen of a session: a traffic light with one sentence of reason, then the three things
 * to watch (last chunk, distance to the alarm, distance to the budget), then the decision timeline. If the session
 * is locked, a banner says why and how many chunks have been read without learning since.
 */
export function StatusHeader() {
  const detail = useStore((s) => s.detail);
  const transactions = useStore((s) => s.transactions);
  const stale = useStore((s) => s.stale);
  const busy = useStore((s) => s.busy);
  const selected = useStore((s) => s.selectedChunk);
  const selectChunk = useStore((s) => s.selectChunk);
  const resume = useStore((s) => s.resumeSession);
  const caps = useStore((s) => s.capabilities)();
  const status = sessionStatus(detail, transactions);
  if (!detail || !status) return null;

  const sorted = transactions.slice().sort((a, b) => a.index - b.index);
  const last = sorted[sorted.length - 1] ?? null;
  const norm = last ? normOutcome(last) : null;
  const turns = turnsFromTrace(detail.trace, transactions);
  const lastTurn = turns[turns.length - 1];
  const totals = lastTurn ? turnTotals(lastTurn.chunks) : null;
  const c = LIGHT_CLASS[status.light];
  const lock = status.lock;

  return (
    <section aria-label="Session status" className={`border-b-2 bg-surface-raised ${c.border}`}>
      <div className="mx-auto max-w-[1500px] px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <LightBadge light={status.light} label={status.label} />
          <p role="status" className="min-w-0 flex-1 basis-[22rem] text-md text-ink-primary">
            {status.sentence}
            {stale ? <span className="ml-2 text-sm font-semibold text-status-scale">Connection lost; this may be out of date.</span> : null}
          </p>
          {lock && caps.resume ? (
            <div className="flex flex-col items-start gap-1">
              <Button disabled={busy.mutation} onClick={() => void resume(detail.meta.session_id)}>Resume learning</Button>
              <span className="text-xs text-ink-secondary">Clears the lock. Spent budget stays spent.</span>
            </div>
          ) : null}
        </div>

        {lock ? (
          <div role="alert" className="mt-3 grid gap-x-8 gap-y-2 rounded border border-status-rollback bg-surface-overlay px-4 py-3 sm:grid-cols-3">
            <div>
              <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">Locked because</p>
              <p className="mt-0.5 text-sm font-semibold text-ink-primary">{lock.title}</p>
              {lock.reason ? <p className="font-mono text-xs text-ink-muted">{lock.reason}</p> : null}
            </div>
            <div>
              <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">Locked since</p>
              <p className="mt-0.5 font-mono text-sm text-ink-primary">
                {lock.sinceChunk === null ? 'unknown' : `${lock.startedBeforeWindow ? 'before ' : ''}chunk ${lock.sinceChunk}`}
              </p>
              {lock.sinceChunk !== null ? (
                <button type="button" className="text-xs font-semibold text-accent underline focus:outline-none focus-visible:ring-2 focus-visible:ring-accent" onClick={() => selectChunk(lock.sinceChunk)}>
                  Why it locked
                </button>
              ) : null}
            </div>
            <div>
              <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">Read without learning since</p>
              <p className="mt-0.5 font-mono text-sm text-ink-primary">
                {fmtInt(lock.refusedSince)} chunk{lock.refusedSince === 1 ? '' : 's'} · {fmtInt(lock.tokensSince)} tokens
              </p>
              <p className="text-xs text-ink-secondary">The model still answers; nothing it reads is stored.</p>
            </div>
          </div>
        ) : null}

        <div className="mt-3 grid gap-x-8 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="min-w-0">
            <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">Last chunk{last ? ` #${last.index}` : ''}</p>
            {last && norm ? (
              <>
                <p className="mt-1 text-sm"><DecisionTag tx={last} /></p>
                <p className="mt-1 font-mono text-xs text-ink-primary">
                  proposed {fmt(norm.proposed)} · accepted {fmt(norm.accepted)}
                  {norm.ratio !== null && last.decision.kind !== 'readonly' ? ` (${Math.round(norm.ratio * 100)}%)` : ''}
                </p>
                {totals && totals.chunks > 1 ? (
                  <p className="mt-0.5 text-xs text-ink-secondary">
                    This turn: {totals.chunks} chunks, {totals.committed} committed{totals.intervened > 0 ? `, ${totals.intervened} intervened` : ''}{totals.readOnly > 0 ? `, ${totals.readOnly} read-only` : ''}
                  </p>
                ) : null}
              </>
            ) : (
              <p className="mt-1 text-sm text-ink-secondary">None yet</p>
            )}
          </div>
          <GaugeBar gauge={status.cusum} />
          <GaugeBar gauge={status.budget} />
          <div className="min-w-0">
            <p className="text-label font-semibold uppercase tracking-wide text-ink-muted">Recent chunks</p>
            <p className="mt-1 font-mono text-sm text-ink-primary">
              {status.recent.window === 0 ? 'none yet' : `${status.recent.intervened} of ${status.recent.window} intervened`}
            </p>
            {status.recent.wouldIntervene > 0 && detail.meta.harness.log_only ? (
              <p className="text-xs text-ink-secondary">{status.recent.wouldIntervene} would have been</p>
            ) : (
              <p className="text-xs text-ink-secondary">Rollback, scale or project</p>
            )}
          </div>
        </div>

        {sorted.length > 0 ? (
          <div className="mt-3">
            <DecisionTimeline chunks={sorted} selected={selected} onSelect={selectChunk} limit={40} size="sm" />
          </div>
        ) : null}
      </div>
    </section>
  );
}
