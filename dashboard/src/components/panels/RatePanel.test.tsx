// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';

import { CalibratedRates } from './RatePanel';
import { calibrationView } from '../../utils/formatting';
import type { CalibrationSummary } from '../../api/types';

afterEach(cleanup);

// ASTRA-096 defect 1: the rendered rates panel must not say "not calibrated" for an installed session
// whose model detail is still loading, nor for a rejected artifact. Only a genuinely absent
// calibration reads "not calibrated". This is the rendered contradiction the helper-string tests miss.

describe('CalibratedRates rendering', () => {
  it('installed-but-loading renders an installed/loading message, not "not calibrated"', () => {
    const view = calibrationView('installed', false, false); // installed; model detail not loaded
    render(<CalibratedRates calibration={null} inactive={view.inactive} />);
    expect(screen.queryByText(/not calibrated/i)).toBeNull();
    expect(screen.getByText(/threshold details are loading or unavailable/i)).toBeTruthy();
  });

  it('a rejected calibration renders the rejection reason, not "not calibrated"', () => {
    const view = calibrationView('rejected_signature_mismatch', true, true);
    render(<CalibratedRates calibration={null} inactive={view.inactive} />);
    expect(screen.queryByText(/not calibrated/i)).toBeNull();
    expect(screen.getByText(/different model/i)).toBeTruthy();
  });

  it('an unknown status renders as unavailable, not "not calibrated"', () => {
    const view = calibrationView(undefined, true, true);
    render(<CalibratedRates calibration={null} inactive={view.inactive} />);
    expect(screen.queryByText(/not calibrated/i)).toBeNull();
  });

  it('a genuinely absent calibration still renders "not calibrated"', () => {
    render(<CalibratedRates calibration={null} inactive={null} />);
    expect(screen.getByText(/not calibrated/i)).toBeTruthy();
  });

  it('an active calibration renders the rates, not any empty state', () => {
    const cal = { target_fpr: 0.01, n_chunks: 100, achievable_fpr: { chunk_loss: 0.02 } } as unknown as CalibrationSummary;
    render(<CalibratedRates calibration={cal} inactive={null} />);
    expect(screen.queryByText(/not calibrated/i)).toBeNull();
    expect(screen.getByText(/Requested target rate/i)).toBeTruthy();
  });
});
