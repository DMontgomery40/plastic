// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { buildAnatomy } from './anatomy';
import { clearObservatoryCache } from './data';
import { ILLUSTRATIVE } from './illustrative';
import { ObservatoryScreen } from './ObservatoryScreen';
import { exportFetch, exportFiles, readExport } from './testData';
import type { Run, SleepArm, Trajectory } from './types';

beforeEach(() => {
  clearObservatoryCache();
  vi.stubGlobal('fetch', vi.fn(exportFetch));
  window.history.replaceState(null, '', '/#sleep/anatomy/final_step250_seed0_exclude/dream');
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const SEED0 = 'final_step250_seed0_exclude';

describe('anatomy of a sleep', () => {
  it('builds the measured walk-through from the exported run and its session log', () => {
    const run = readExport<Run>(`runs/${SEED0}.json`);
    const dream = run.arms.find((a) => a.arm === 'dream') as SleepArm;
    const m = buildAnatomy(run, dream, readExport<Trajectory>(`sessions/${SEED0}.json`));
    expect(m.mode).toBe('measured');
    const [teach, rolled] = m.wake!.sessions;
    expect(teach.turns.length).toBe(30);
    expect(teach.turns.flat().length).toBe(165);
    expect(teach.turns.flat().every((c) => c.applied === 'commit')).toBe(true);
    expect(rolled.forced).toBe(true);
    expect(m.harvest).toMatchObject({ accepted: 30, rolled_back: 2, flagged: 29, flagged_excluded: 29, selected: 1, state_sessions: ['teach'] });
    expect(m.outcome).toEqual({ outcome: 'pulled back', parent: 'parent', child: null });
    expect(m.gate.checks.find((c) => c.passed === false)?.value).toBe(0.307692);
  });

  it('labels every stage of the illustrative mode and none of the measured one', async () => {
    render(<ObservatoryScreen />);
    await screen.findByText('Pull back: no child');
    expect(screen.queryAllByText('Illustrative, not measured')).toHaveLength(0);
    expect(screen.getByText('Damage gate: check the candidate child')).toBeTruthy();
    expect(screen.getByText(/no retention or contamination check\. Any failed check rejects it\./)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Illustrative: if consolidation worked' }));
    await screen.findByText('Commit: a child model');
    expect(screen.getAllByText('Illustrative, not measured')).toHaveLength(5);
    expect(screen.getByText(/invented numbers/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Measured run' }));
    await screen.findByText('Pull back: no child');
    expect(screen.queryAllByText('Illustrative, not measured')).toHaveLength(0);
  });

  it('keeps the invented scenario out of the exported data', () => {
    const marker = ILLUSTRATIVE.consolidate.dreams!.example!;
    for (const file of exportFiles()) {
      expect(JSON.stringify(readExport(file))).not.toContain(marker);
    }
    expect(ILLUSTRATIVE.mode).toBe('illustrative');
  });
});
