import { describe, expect, it } from 'vitest';
import { chunkSize, chunkSource, firedSignals, fmt, modeLabel, presentFields, thresholdSource, turnTotals, turnsFromTrace, wouldIntervene } from './format';
import type { HarnessConfig, RunnerSummary, TraceRecord, TransactionRecord } from './types';

function tx(index: number, pos_start: number, pos_end: number, opts: Partial<TransactionRecord> & { requested?: TransactionRecord['requested'] } = {}): TransactionRecord {
  const decision = opts.decision ?? { kind: 'commit', reasons: [], scale: 1 };
  return {
    index,
    t_unix: 0,
    pos_start,
    pos_end,
    decision,
    requested: opts.requested ?? decision,
    signals: {
      pos_start,
      pos_end,
      n_tokens: pos_end - pos_start,
      chunk_loss: 2.0,
      delta_norm: 10,
      z: {},
      cusum_alarm: false,
      budget_used: 0,
      budget_remaining: null,
      log_delta_norm: 2.3,
      ...(opts.signals ?? {}),
    },
    accepted: { delta_norm: 10, budget_charge: 10, budget_used: 10, budget_remaining: null, ...(opts.accepted ?? {}) },
    sources: opts.sources,
    eligible: true,
    read_only: false,
    read_only_reason: null,
    seconds: 0.1,
  };
}

describe('turns', () => {
  it('groups chunks under chat turns by position and keeps turn order', () => {
    const trace: TraceRecord[] = [
      { t_unix: 2, kind: 'chat', prompt: 'b', completion: 'B', pos_end: 40, n_transactions: 2 },
      { t_unix: 1, kind: 'chat', prompt: 'a', completion: 'A', pos_end: 20, n_transactions: 2 },
    ];
    const txs = [tx(0, 0, 16), tx(1, 16, 20), tx(2, 20, 36), tx(3, 36, 40)];
    const turns = turnsFromTrace(trace, txs);
    expect(turns.map((t) => t.prompt)).toEqual(['a', 'b']);
    expect(turns[0].chunks.map((c) => c.index)).toEqual([0, 1]);
    expect(turns[1].chunks.map((c) => c.index)).toEqual([2, 3]);
  });

  it('totals distinguish applied interventions, would-have interventions and read-only observations', () => {
    const chunks = [
      tx(0, 0, 8, { sources: { user: 8, model: 0 } }),
      tx(1, 8, 16, { requested: { kind: 'rollback', reasons: ['would_rollback:chunk_loss(9>8)'], scale: 1 }, sources: { user: 8, model: 0 } }),
      tx(2, 16, 24, { decision: { kind: 'rollback', reasons: ['chunk_loss(9>8)'], scale: 1 }, accepted: { delta_norm: 0, budget_charge: 0, budget_used: 10, budget_remaining: null }, sources: { user: 0, model: 8 } }),
      tx(3, 24, 32, { decision: { kind: 'readonly', reasons: ['learning_ineligible'], scale: 1 }, sources: { user: 0, model: 8 } }),
    ];
    const t = turnTotals(chunks);
    expect(t).toMatchObject({ chunks: 4, committed: 2, intervened: 1, wouldIntervene: 1, readOnly: 1, promptTokens: 16, modelTokens: 16 });
    expect(t.proposed).toBe(40);
    expect(t.accepted).toBe(30);
    expect(wouldIntervene(chunks[1])).toBe(true);
    expect(wouldIntervene(chunks[2])).toBe(false);
    expect(firedSignals(chunks[1])).toEqual(['chunk_loss']);
  });
});

describe('labels and gating', () => {
  it('sizes a cell by the chunk\'s own write norm when present, else by the proposed change', () => {
    expect(chunkSize(tx(0, 0, 8, { signals: { write_norm_sum: 3 } as never }))).toBe(3);
    expect(chunkSize(tx(0, 0, 8))).toBe(10);
  });

  it('gates fields on presence of any finite value, never on the decision-signal list', () => {
    const a = tx(0, 0, 8, { signals: { beta_mean: 0.01, surprise_mean: null } as never });
    const b = tx(1, 8, 16, { signals: { beta_mean: null, surprise_mean: null } as never });
    expect(presentFields([a, b], ['beta_mean', 'surprise_mean', 'write_norm_sum'])).toEqual(['beta_mean']);
  });

  it('names the mode and the threshold source as states', () => {
    expect(modeLabel({ log_only: true } as HarnessConfig)).toBe('observational');
    expect(modeLabel({ log_only: false } as HarnessConfig)).toBe('guarded');
    expect(thresholdSource({ calibration: 'installed' } as RunnerSummary)).toBe('calibrated thresholds');
    expect(thresholdSource({ calibration: 'absent' } as RunnerSummary)).toMatch(/session-relative/);
    expect(thresholdSource({ calibration: 'rejected_signature_mismatch' } as RunnerSummary)).toMatch(/another checkpoint/);
  });

  it('reads the chunk source from token counts', () => {
    expect(chunkSource(tx(0, 0, 8, { sources: { user: 8, model: 0 } }))).toBe('prompt');
    expect(chunkSource(tx(0, 0, 8, { sources: { user: 0, model: 3 } }))).toBe('model');
    expect(chunkSource(tx(0, 0, 8))).toBe('unknown');
  });

  it('never formats a non-finite value as a number', () => {
    expect(fmt(null)).toBe('n/a');
    expect(fmt(Number.NaN)).toBe('n/a');
    expect(fmt(0)).toBe('0');
    expect(fmt(12345.6)).toBe('12346');
  });
});
