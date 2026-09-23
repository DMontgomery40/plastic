// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { clearObservatoryCache } from './data';
import { layerGrid } from './LayerMap';
import { ObservatoryScreen } from './ObservatoryScreen';
import { logScale } from './scale';
import { exportFetch, readExport } from './testData';
import type { Trajectory } from './types';

beforeEach(() => {
  clearObservatoryCache();
  vi.stubGlobal('fetch', vi.fn(exportFetch));
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('weights and changes', () => {
  it('combines the four tensors of a layer in quadrature and keeps missing values missing', () => {
    const t = readExport<Trajectory>('sessions/final_step250_seed0_exclude.json');
    const teach = t.sessions[0];
    const grid = layerGrid(teach, 24, 'all');
    expect(grid.length).toBe(24);
    expect(grid[0].length).toBe(165);
    const v = teach.chunks[0].per_layer!;
    expect(grid[0][0]).toBeCloseTo(Math.hypot(v[0]!, v[1]!, v[2]!, v[3]!), 6);
    expect(layerGrid(teach, 24, 2)[5][10]).toBe(teach.chunks[10].per_layer![5 * 4 + 2]);
    const missing = layerGrid({ ...teach, chunks: [{ ...teach.chunks[0], per_layer: null }] }, 24, 'all');
    expect(missing[3][0]).toBeNull();
  });

  it('maps magnitudes on a log scale clipped to the central range', () => {
    const s = logScale([null, 0, 0.001, 0.01, 0.1, 1, 10]);
    expect(s.t(null)).toBeNull();
    expect(s.t(0)).toBeNull(); // zero has no log position: drawn as "no value", not as the minimum
    expect(s.t(s.lo)).toBe(0);
    expect(s.t(s.hi)).toBe(1);
    expect(s.t(1000)).toBe(1);
  });

  it('lists every consolidation attempt across runs and marks checks that were not in force', async () => {
    window.history.replaceState(null, '', '/#sleep/weights');
    render(<ObservatoryScreen />);
    fireEvent.click(await screen.findByRole('button', { name: 'Across runs' }));
    const table = (await screen.findByText('Every consolidation attempt')).closest('section')!;
    const rows = within(table).getAllByRole('row').slice(1);
    expect(rows.length).toBe(27);
    expect(within(table).getAllByText('pulled back').length).toBe(4);
    expect(within(table).getAllByText('not in force').length).toBeGreaterThan(5);
  });

  it('shows the per-layer map of the teaching session and the anchor’s W0 change', async () => {
    window.history.replaceState(null, '', '/#sleep/weights/final_step250_seed0_exclude/anchor');
    render(<ObservatoryScreen />);
    expect(await screen.findByText('W0 change per tensor, child against parent')).toBeTruthy();
    expect(screen.getByText('Anchor interpolates W0 directly; there are no gradient steps.')).toBeTruthy();
    expect(await screen.findByRole('button', { name: 'Terrain' })).toBeTruthy();
  });
});
