// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';

import { LayerStatePanel } from './StatePanels';
import type { SessionState } from '../../api/types';

// jsdom does not implement ResizeObserver, which recharts' ResponsiveContainer instantiates on mount.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = ResizeObserverStub;

afterEach(cleanup);

// ASTRA-081 #3: the native (Qwen) recurrent state must render its own honest per-unit view, never the
// toy per-layer S/h shape, and a null norm/drift must read as unavailable, never a plotted zero.

describe('LayerStatePanel dispatch', () => {
  it('renders the recurrent view for a recurrent state and counts unavailable units', () => {
    const state: SessionState = {
      kind: 'recurrent',
      pos: 24,
      backend: 'qwen',
      recurrent_norm_total: 3.5,
      units: [
        { index: 0, recurrent_norm: 1.0, drift_from_anchor: 0.5 },
        { index: 1, recurrent_norm: 2.0, drift_from_anchor: null }, // drift unavailable
        { index: 2, recurrent_norm: null, drift_from_anchor: 0.1 }, // norm unavailable
      ],
    };
    render(<LayerStatePanel state={state} loading={false} />);
    expect(screen.getByText(/Recurrent memory state/i)).toBeTruthy();
    expect(screen.queryByText(/Per-layer state/i)).toBeNull(); // never the toy plastic shape
    expect(screen.getByText(/1 unit\(s\) report no norm/i)).toBeTruthy();
    expect(screen.getByText(/1 unit\(s\) report no drift/i)).toBeTruthy();
  });

  it('renders the plastic view for a plastic state', () => {
    const state: SessionState = {
      kind: 'plastic',
      pos: 512,
      layers: [{ s_norm_per_head: [1, 2], h_norm: 0.4, singular_values: [[1, 0.5]], drift_from_anchor: 0.1 }],
    };
    render(<LayerStatePanel state={state} loading={false} />);
    expect(screen.getByText(/Per-layer state/i)).toBeTruthy();
    expect(screen.queryByText(/Recurrent memory state/i)).toBeNull();
  });

  it('shows an empty recurrent state honestly', () => {
    render(<LayerStatePanel state={{ kind: 'recurrent', pos: 0, units: [] }} loading={false} />);
    expect(screen.getByText(/No state reported/i)).toBeTruthy();
  });
});
