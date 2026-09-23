// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { clearObservatoryCache } from './data';
import { ARM_LABEL } from './format';
import { layerGrid } from './LayerMap';
import { ObservatoryScreen } from './ObservatoryScreen';
import { logScale } from './scale';
import { exportFetch, readExport } from './testData';
import type { ArmStatus, Lineage, ObservatoryIndex, Run, SleepArm, Trajectory } from './types';

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
    // The index is the public inventory; the view loads separate per-run files.
    // Compare every listed arm, so dropping a new run, duplicating one, including
    // controls, or changing an outcome fails without pinning the archive's size.
    const index = readExport<ObservatoryIndex>('index.json');
    const attempts = index.runs.flatMap((run) => run.arms
      .filter((arm) => arm.method !== null)
      .map((arm) => ({ run: run.id, arm: arm.arm, status: arm.status })));
    expect(rows.length).toBe(attempts.length);
    for (const attempt of attempts) {
      const row = within(table).getByRole('button', {
        name: `${attempt.run} ${ARM_LABEL[attempt.arm]}`,
      }).closest('tr')!;
      const outcome = attempt.status === null ? 'unknown' : attempt.status === 'rejected' ? 'pulled back' : 'committed';
      expect(within(row).getByText(outcome)).toBeTruthy();
    }
    // Fixed historical cases preserve the meaning of earlier gate outcomes.
    for (const arm of ['Replay', 'Distill']) {
      const row = within(table).getByRole('button', { name: `sleep_controls_step100_w0_40 ${arm}` }).closest('tr')!;
      expect(within(row).getByText('distinct-reply ratio (earlier rule)')).toBeTruthy();
      expect(within(row).getByText('not in force')).toBeTruthy();
    }
    const dream = within(table).getByRole('button', { name: 'final_step250_seed0_exclude Dream' }).closest('tr')!;
    expect(within(dream).getByText('largest identical-reply share')).toBeTruthy();
  });

  it.each<{
    status: ArmStatus | null; outcome: Lineage['outcome']; committed: number; rejected: number;
  }>([
    { status: null, outcome: 'unknown', committed: 0, rejected: 0 },
    { status: 'accepted', outcome: 'committed', committed: 1, rejected: 0 },
    { status: 'accepted_unmeasured', outcome: 'committed', committed: 1, rejected: 0 },
    { status: 'rejected', outcome: 'pulled back', committed: 0, rejected: 1 },
  ])('preserves $status through the run summary, weights and anatomy', async ({ status, outcome, committed, rejected }) => {
    const index = readExport<ObservatoryIndex>('index.json');
    const run = readExport<Run>('runs/final_step250_seed0_exclude.json');
    const original = index.runs.find((r) => r.id === run.id)!;
    run.id = 'outcome_fixture';
    run.session_trajectory = null;
    run.arms = run.arms.filter((a) => a.arm === 'replay' || a.arm === 'floor');
    const replay = run.arms.find((a) => a.arm === 'replay') as SleepArm;
    replay.status = status;
    replay.reason = null;
    replay.gate = null; // outcome recording and recorded gate measurements are separate
    replay.lineage = { ...replay.lineage, outcome, child: committed ? 'fixture-child' : null };
    index.runs = [
      { ...original, id: run.id, session_trajectory: null, arms: original.arms
        .filter((a) => a.arm === 'replay' || a.arm === 'floor')
        .map((a) => a.arm === 'replay' ? { ...a, status } : a) },
      index.runs.find((r) => r.id === 'final_step250_seed0_ceiling_single')!,
    ];
    vi.stubGlobal('fetch', vi.fn((url: string | URL | Request) => {
      const path = String(url);
      const fixture = path.endsWith('/index.json') ? index : path.endsWith(`/runs/${run.id}.json`) ? run : null;
      return fixture ? Promise.resolve(new Response(JSON.stringify(fixture), { status: 200 })) : exportFetch(url);
    }));

    window.history.replaceState(null, '', '/#sleep/runs');
    render(<ObservatoryScreen />);
    await screen.findByRole('heading', { name: run.id });
    for (const [label, count] of [['Consolidation attempts', 1], ['Committed a child', committed], ['Rejected attempts', rejected]] as const) {
      expect(within(screen.getByText(label).parentElement!).getByText(String(count))).toBeTruthy();
    }
    const archive = screen.getByRole('navigation', { name: 'Archived runs' });
    const entry = within(archive).getByRole('button', { name: /outcome_fixture/ });
    expect(within(entry).queryByText('controls only')).toBeNull();
    if (outcome === 'unknown') {
      expect(screen.getAllByText('Unknown').length).toBeGreaterThan(0);
      expect(screen.queryByText('no child')).toBeNull();
      expect(screen.queryByText(/Pulled back: the child was discarded/)).toBeNull();
    }

    fireEvent.click(screen.getByRole('button', { name: 'Weights and changes' }));
    const select = await screen.findByRole('combobox', { name: 'Run' });
    expect(within(select).getAllByRole('option').map((o) => o.getAttribute('value'))).toEqual([run.id]);
    fireEvent.click(screen.getByRole('button', { name: 'Across runs' }));
    const table = (await screen.findByText('Every consolidation attempt')).closest('section')!;
    expect(within(table).getAllByRole('row')).toHaveLength(2); // one attempt; neither control belongs here
    const row = within(table).getByRole('button', { name: `${run.id} Replay` }).closest('tr')!;
    expect(within(row).getByText(outcome)).toBeTruthy();
    if (outcome === 'unknown') {
      expect(within(row).queryByText(/committed|pulled back/)).toBeNull();
    }
    fireEvent.click(within(row).getByRole('button'));
    expect((await screen.findByRole('combobox', { name: 'Run' }) as HTMLSelectElement).value).toBe(run.id);

    fireEvent.click(screen.getByRole('button', { name: 'Anatomy of a sleep' }));
    const anatomySelect = await screen.findByRole('combobox', { name: 'Run' });
    expect(within(anatomySelect).getAllByRole('option').map((o) => o.getAttribute('value'))).toEqual([run.id]);
    if (outcome === 'unknown') {
      await screen.findByText('Unknown');
      expect(screen.queryByText('no child')).toBeNull();
      expect(screen.queryByText(/The candidate is discarded/)).toBeNull();
    } else {
      await screen.findByText(committed ? 'Commit: a child model' : 'Pull back: no child');
    }
  });

  it('shows the per-layer map of the teaching session and the anchor’s W0 change', async () => {
    window.history.replaceState(null, '', '/#sleep/weights/final_step250_seed0_exclude/anchor');
    render(<ObservatoryScreen />);
    expect(await screen.findByText('W0 change per tensor, child against parent')).toBeTruthy();
    expect(screen.getByText('Anchor interpolates W0 directly; there are no gradient steps.')).toBeTruthy();
    expect(await screen.findByRole('button', { name: 'Terrain' })).toBeTruthy();
  });
});
