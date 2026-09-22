import { describe, expect, it } from 'vitest';

import { calibrationDisplay } from './formatting';

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
