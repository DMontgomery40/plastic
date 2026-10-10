import { describe, expect, it } from 'vitest';
import { explainLock, explainReason } from './reasons';
import { lockInfo, normOutcome, rollbackLimits, sessionStatus, verdictSentence, cusumGauge, budgetGauge } from './status';
import type { RunnerSummary, SessionDetail, TransactionRecord } from './types';

function tx(index: number, kind: TransactionRecord['decision']['kind'], over: Partial<TransactionRecord> = {}, reasons: string[] = [], proposed = 1, accepted = 1): TransactionRecord {
  const decision = { kind, reasons, scale: 1 };
  return {
    index, t_unix: 0, pos_start: index * 16, pos_end: index * 16 + 16, decision, requested: decision,
    signals: { pos_start: 0, pos_end: 16, n_tokens: 16, chunk_loss: 4, delta_norm: proposed, z: {}, cusum_alarm: false, budget_used: 0, budget_remaining: null, log_delta_norm: 0 },
    accepted: { delta_norm: accepted, budget_charge: accepted, budget_used: accepted, budget_remaining: null },
    read_only: false, read_only_reason: null, seconds: 0.1, ...over,
  } as TransactionRecord;
}

const summary = (over: Partial<RunnerSummary> = {}): RunnerSummary => ({
  pos: 0, pending: 0, budget_used: 0, budget_session: null, read_only: false, read_only_reason: null, n_transactions: 0,
  cusum: { k: 0.5, h: 5, s_hi: 0, s_lo: 0, alarms: 0 }, state_norms: {}, drift_from_anchor: 0, ...over,
});

function detail(s: RunnerSummary, logOnly = false): SessionDetail {
  return { meta: { session_id: 's', harness: { log_only: logOnly } }, summary: s, calibration: null, lineage: [], transactions: [], trace: [] } as unknown as SessionDetail;
}

describe('explainReason', () => {
  it('turns a canary reason into a sentence with the value and the limit', () => {
    const r = explainReason('canary_coherence(delta=+0.1234>0.0500)');
    expect(r.title).toBe('Canary coherence');
    expect(r.sentence).toContain('0.123');
    expect(r.sentence).toContain('0.05');
    expect(r.kind).toBe('canary');
    expect(r.raw).toBe('canary_coherence(delta=+0.1234>0.0500)');
  });
  it('reads the poison canary as the model getting better at what it must not learn', () => {
    expect(explainReason('canary_poison(delta=-0.6000<-0.5000)').sentence).toMatch(/fell by 0\.6/);
  });
  it('marks would_ reasons from observational mode and strips the prefix', () => {
    const r = explainReason('would_rollback:cusum_alarm');
    expect(r.would).toBe(true);
    expect(r.kind).toBe('cusum');
  });
  it('explains statistic thresholds, z-scores and budgets', () => {
    expect(explainReason('chunk_loss(5.2>4.9)').sentence).toContain('calibrated limit');
    expect(explainReason('log_delta_norm_z(6.31>=6.0)').sentence).toContain("session's own recent history");
    expect(explainReason('budget_chunk(delta_norm=0.4>0.15)').sentence).toContain('scaled down');
    expect(explainReason('budget_unsatisfiable(delta=0.2>cap=0.1)').sentence).toContain('memory decay');
  });
  it('marks a recheck after scaling', () => {
    expect(explainReason('final_candidate:canary_coherence(delta=+1>0.5)').sentence).toContain('after scaling or projection');
  });
  it('never throws and keeps unknown reasons readable', () => {
    const r = explainReason('something_new(1>2)');
    expect(r.kind).toBe('other');
    expect(r.raw).toBe('something_new(1>2)');
    expect(explainReason('').title).toBeDefined();
  });
  it('explains lock reasons', () => {
    expect(explainLock('cusum_alarm').title).toBe('Update-size alarm');
    expect(explainLock('budget_session(1.2>=1.2)').kind).toBe('budget');
    expect(explainLock(null).title).toBe('Locked');
  });
});

describe('normOutcome', () => {
  it('reports a full commit', () => expect(normOutcome(tx(0, 'commit')).kind).toBe('full'));
  it('reports a rollback as nothing kept', () => {
    const o = normOutcome(tx(0, 'rollback', {}, ['cusum_alarm'], 0.5, 0));
    expect(o.kind).toBe('none');
    expect(o.sentence).toContain('None of the proposed 0.5');
  });
  it('shows the measured share for a scale, not the requested one', () => {
    const t = tx(0, 'scale', {}, [], 1, 0.358);
    t.decision.scale = 0.25;
    const o = normOutcome(t);
    expect(o.kind).toBe('partial');
    expect(o.asked).toBe(0.25);
    expect(o.sentence).toContain('36%');
    expect(o.sentence).toContain('asked for 25%');
  });
  it('flags an accepted change larger than the proposal', () => {
    const t = tx(0, 'scale', {}, [], 0.00621, 0.0247);
    t.decision.scale = 0.5;
    const o = normOutcome(t);
    expect(o.kind).toBe('larger');
    expect(o.sentence).toContain('more than');
    expect(o.sentence).toContain('50% was requested');
  });
  it('treats a read-only chunk as nothing written and a missing norm as not reported', () => {
    expect(normOutcome(tx(0, 'readonly', {}, ['session_read_only'], 0, 0)).sentence).toContain('read-only');
    const t = tx(0, 'commit');
    (t.accepted as unknown as { delta_norm: unknown }).delta_norm = null;
    expect(normOutcome(t).kind).toBe('unknown');
  });
});

describe('lockInfo and sessionStatus', () => {
  const alarm = tx(14, 'rollback', { read_only: true, read_only_reason: 'cusum_alarm' }, ['cusum_alarm'], 0.9, 0);
  const locked = [tx(12, 'commit'), tx(13, 'commit'), alarm, tx(15, 'readonly', { read_only: true, read_only_reason: 'cusum_alarm' }, ['session_read_only'], 0, 0), tx(16, 'readonly', { read_only: true, read_only_reason: 'cusum_alarm' }, ['session_read_only'], 0, 0)];
  const lockedSummary = summary({ read_only: true, read_only_reason: 'cusum_alarm', cusum: { k: 0.5, h: 5, s_hi: 0, s_lo: 0, alarms: 1 } });

  it('finds where the lock began and counts chunks read without learning since', () => {
    const l = lockInfo(lockedSummary, locked)!;
    expect(l.sinceChunk).toBe(14);
    expect(l.refusedSince).toBe(2);
    expect(l.tokensSince).toBe(32);
    expect(l.startedBeforeWindow).toBe(false);
  });
  it('says the start is earlier when every loaded chunk is already locked', () => {
    const only = [tx(300, 'readonly', { read_only: true, read_only_reason: 'cusum_alarm' })];
    expect(lockInfo(lockedSummary, only)!.startedBeforeWindow).toBe(true);
  });
  it('leads with Locked and explains why, with the counter', () => {
    const s = sessionStatus(detail(lockedSummary), locked)!;
    expect(s.light).toBe('locked');
    expect(s.sentence).toContain('since chunk 14');
    expect(s.sentence).toContain('2 chunks have been read without learning since');
    expect(s.cusum.level).toBe('over');
    expect(s.cusum.reading).toContain('1 alarm');
  });
  it('is Ready with no chunks, OK when calm, Watch after a refusal', () => {
    expect(sessionStatus(detail(summary()), [])!.light).toBe('empty');
    expect(sessionStatus(detail(summary()), [tx(0, 'commit'), tx(1, 'commit')])!.light).toBe('ok');
    const w = sessionStatus(detail(summary()), [tx(0, 'commit'), tx(1, 'rollback', {}, ['canary_coherence(delta=+0.2>0.05)'], 1, 0)])!;
    expect(w.light).toBe('watch');
    expect(w.sentence).toContain('rolled back');
    expect(w.sentence).toContain('0.2');
  });
  it('goes to Watch as the alarm or the budget gets close, before anything is refused', () => {
    expect(sessionStatus(detail(summary({ cusum: { k: 0.5, h: 5, s_hi: 3.5, s_lo: 0, alarms: 0 } })), [tx(0, 'commit')])!.light).toBe('watch');
    expect(sessionStatus(detail(summary({ budget_session: 10, budget_used: 8.5 })), [tx(0, 'commit')])!.light).toBe('watch');
  });
  it('never calls an observational session guarded', () => {
    const chunks = [tx(0, 'commit', { requested: { kind: 'commit', reasons: ['log_only', 'would_rollback:cusum_alarm'], scale: 1 } }, ['log_only', 'would_rollback:cusum_alarm'])];
    const s = sessionStatus(detail(summary(), true), chunks)!;
    expect(s.light).toBe('unguarded');
    expect(s.sentence).toContain('would have intervened on 1 of the last 1');
  });
  it('returns null without a session', () => expect(sessionStatus(null, [])).toBeNull());
});

describe('gauges', () => {
  it('shows an uncapped budget as uncapped, never as 0%', () => {
    const g = budgetGauge(summary({ budget_used: 2.5 }));
    expect(g.fraction).toBeNull();
    expect(g.reading).toContain('no cap');
  });
  it('flags the budget at the cap', () => expect(budgetGauge(summary({ budget_session: 1, budget_used: 1 })).level).toBe('over'));
  it('reports a missing alarm limit as not reported', () => expect(cusumGauge(summary({ cusum: undefined as never }), false).reading).toBe('not reported'));
});

describe('what a decision did not do', () => {
  it('attaches the rollback limits only to a rollback', () => {
    expect(rollbackLimits(tx(0, 'rollback'))).toContain('64 tokens');
    expect(rollbackLimits(tx(0, 'rollback'))).toContain('already generated');
    expect(rollbackLimits(tx(0, 'commit'))).toBeNull();
    expect(rollbackLimits(tx(0, 'scale'))).toBeNull();
  });
  it('words observational commits as not blocked', () => {
    const t = tx(0, 'commit', { requested: { kind: 'commit', reasons: ['would_rollback:cusum_alarm'], scale: 1 } });
    expect(verdictSentence(t)).toContain('observational');
    expect(verdictSentence(t)).toContain('rolled this chunk back');
  });
});
