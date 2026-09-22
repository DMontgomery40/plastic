import { describe, expect, it } from 'vitest';
import type { TransactionRecord } from '../api/types';

import { calibrationDisplay, calibrationView, generationLearningCopy, sourceAccounting } from './formatting';

describe('calibrationDisplay', () => {
  it('marks only an installed calibration active', () => {
    // ASTRA-081 #2: `active` is the single gate that lets policy threshold lines / active rates draw.
    expect(calibrationDisplay('installed').active).toBe(true);
    for (const status of ['absent', 'rejected_unsigned', 'rejected_signature_mismatch'] as const) {
      expect(calibrationDisplay(status).active).toBe(false);
    }
  });

  it('distinguishes the two rejection reasons from absent', () => {
    expect(calibrationDisplay('rejected_signature_mismatch').label).toBe('Rejected: different model');
    expect(calibrationDisplay('rejected_unsigned').label).toBe('Rejected: unsigned');
    expect(calibrationDisplay('absent').label).toBe('Not calibrated');
    // the three are different statements, never collapsed to one
    const labels = ['rejected_signature_mismatch', 'rejected_unsigned', 'absent'].map((s) => calibrationDisplay(s).label);
    expect(new Set(labels).size).toBe(3);
  });

  it('reads a missing or unknown status as Unknown and inactive, never as Not calibrated', () => {
    // a loaded summary lacking the field, or a value we do not recognize, is not the same as "absent"
    for (const status of [null, undefined, 'something_new']) {
      const d = calibrationDisplay(status);
      expect(d.active).toBe(false);
      expect(d.label).toBe('Unknown');
    }
    expect(calibrationDisplay(undefined).label).not.toBe(calibrationDisplay('absent').label);
  });

  it('always provides a human detail string', () => {
    for (const status of ['installed', 'absent', 'rejected_unsigned', 'rejected_signature_mismatch', null]) {
      expect(calibrationDisplay(status).detail.length).toBeGreaterThan(0);
    }
  });
});

describe('calibrationView', () => {
  it('only an installed session with loaded thresholds is active', () => {
    const active = calibrationView('installed', true, true);
    expect(active.mode).toBe('active');
    expect(active.drawThresholds).toBe(true);
    expect(active.inactive).toBeNull();
  });

  it('installed but details not loaded is installed_pending, never "not calibrated" or fallback', () => {
    for (const [loaded, has] of [[false, false], [true, false], [false, true]] as [boolean, boolean][]) {
      const v = calibrationView('installed', loaded, has);
      expect(v.mode).toBe('installed_pending');
      expect(v.drawThresholds).toBe(false);
      expect(v.inactive?.title).not.toMatch(/not calibrated/i);
      expect(v.signalsSubtitle).not.toMatch(/falls back/i); // the fallback is not asserted while installed
      expect(v.signalsSubtitle.toLowerCase()).toContain('installed');
    }
  });

  it('rejected states say why and are inactive, never "not calibrated"', () => {
    const mism = calibrationView('rejected_signature_mismatch', true, true);
    expect(mism.mode).toBe('rejected');
    expect(mism.drawThresholds).toBe(false); // even with a loaded artifact, a rejected session draws nothing
    expect(mism.inactive?.title).toMatch(/different model/i);
    expect(mism.inactive?.title).not.toMatch(/not calibrated/i);
    const uns = calibrationView('rejected_unsigned', true, true);
    expect(uns.inactive?.title).toMatch(/unsigned/i);
    expect(uns.inactive?.title).not.toMatch(/not calibrated/i);
  });

  it('absent is the only mode that reads "not calibrated" and asserts fallback', () => {
    const v = calibrationView('absent', true, false);
    expect(v.mode).toBe('absent');
    expect(v.inactive?.title).toMatch(/not calibrated/i);
    expect(v.signalsSubtitle).toMatch(/falls back/i);
  });

  it('an unknown/missing status is inactive but not "not calibrated"', () => {
    for (const s of [undefined, null, 'something_new']) {
      const v = calibrationView(s, true, true);
      expect(v.mode).toBe('unknown');
      expect(v.drawThresholds).toBe(false);
      expect(v.inactive?.title).not.toMatch(/not calibrated/i);
      // unknown does not establish the active policy, so it must NOT assert the robust-z fallback
      expect(v.signalsSubtitle).not.toMatch(/falls back/i);
    }
  });
});

describe('generationLearningCopy', () => {
  it('write-eligible generation is described as eligible/attempted, not guaranteed learning', () => {
    // ASTRA-101: the EFFECTIVE write policy -- Qwen generation is write-eligible even with
    // learn_from_generation off -- and eligibility is not retained learning.
    const on = generationLearningCopy(true, false);
    expect(on).toMatch(/generated tokens/i);
    expect(on).toMatch(/write-eligible/i);
    expect(on).toMatch(/closure/i);
    expect(on).toMatch(/rolled back|not guaranteed retained/i); // eligible != retained learning
    expect(on).not.toMatch(/read-only by default/i);
    expect(on).not.toMatch(/\blearned\b(?! )/i); // avoid claiming plain "learned"
  });

  it('generation that is not write-eligible reads read-only, not learned', () => {
    const off = generationLearningCopy(false, false);
    expect(off).toMatch(/read-only/i);
    expect(off).toMatch(/not write-eligible/i);
    expect(off).not.toMatch(/read-only by default/i);
  });

  it('a read-only session says no chunk writes', () => {
    const ro = generationLearningCopy(true, true);
    expect(ro).toMatch(/read-only/i);
    expect(ro).toMatch(/no chunk writes/i);
  });

  it('an unknown/loading effective policy is not asserted, and never affirms eligibility', () => {
    for (const v of [undefined, null]) {
      const copy = generationLearningCopy(v, false);
      expect(copy).toMatch(/not available yet/i);
      expect(copy).not.toMatch(/not write-eligible/i); // do not claim a policy we do not have
      expect(copy).not.toMatch(/write-eligible/i); // and do not affirm prompt eligibility either
    }
  });

  it('a KNOWN read-only latch wins over an unknown capability (ASTRA-104)', () => {
    // missing capability + read_only=True must honor the latch, not affirm write-eligibility
    const copy = generationLearningCopy(undefined, true);
    expect(copy).toMatch(/read-only/i);
    expect(copy).not.toMatch(/write-eligible/i);
  });
});

describe('sourceAccounting', () => {
  it('buckets chunks by source with eligible denominators and interventions', () => {
    const dec = (kind: TransactionRecord['decision']['kind']) => ({ kind, reasons: [], scale: 1 });
    const tx = (over: Partial<TransactionRecord>): TransactionRecord => ({
      index: 0, t_unix: 0, pos_start: 0, pos_end: 8, decision: dec('commit'),
      requested: dec('commit'), signals: {} as TransactionRecord['signals'],
      accepted: {} as TransactionRecord['accepted'], read_only: false, read_only_reason: null, seconds: 0,
      ...over,
    });
    const txns = [
      tx({ sources: { user: 8, model: 0 }, eligible: true, decision: dec('commit') }),   // user, eligible
      tx({ sources: { user: 0, model: 8 }, eligible: true, decision: dec('rollback') }), // model, eligible, intervention
      tx({ sources: { user: 0, model: 2 }, eligible: false, decision: dec('readonly') }),// model, NOT eligible, NOT intervention
    ];
    const acc = sourceAccounting(txns);
    expect(acc.user).toEqual({ chunks: 1, eligible: 1, interventions: 0, tokens: 8 });
    expect(acc.model).toEqual({ chunks: 2, eligible: 1, interventions: 1, tokens: 10 }); // 8 + 2 incl. closure
  });
});
