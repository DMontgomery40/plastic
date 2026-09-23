import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { clearObservatoryCache, isRun, loadIndex, loadRun, loadTrajectory, ObservatoryDataError } from './data';
import { checkState, ratio } from './format';
import { formatRoute, parseRoute } from './ObservatoryScreen';
import { exportFetch, readExport } from './testData';
import type { ObservatoryIndex, Run } from './types';

beforeEach(() => {
  clearObservatoryCache();
  vi.stubGlobal('fetch', vi.fn(exportFetch));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe('observatory data', () => {
  it('loads the committed index and every run it lists', async () => {
    const index = await loadIndex();
    expect(index.runs.length).toBeGreaterThan(10);
    for (const r of index.runs) {
      const run = await loadRun(r.id);
      expect(isRun(run)).toBe(true);
      expect(run.id).toBe(r.id);
    }
    for (const id of index.sessions) {
      const t = await loadTrajectory(id);
      expect(t.per_layer.layers * t.per_layer.tensors.length).toBe(t.sessions[0].chunks[0].per_layer?.length);
    }
  });

  it('rejects unsafe ids and unexpected formats, and retries after a failure', async () => {
    await expect(loadRun('../index')).rejects.toBeInstanceOf(ObservatoryDataError);
    await expect(loadRun('no_such_run')).rejects.toThrow('HTTP 404');
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ schema: 'other' }))));
    clearObservatoryCache();
    await expect(loadIndex()).rejects.toThrow('unexpected format');
    vi.stubGlobal('fetch', vi.fn(exportFetch));
    clearObservatoryCache();
    await expect(loadIndex()).resolves.toMatchObject({ schema: 'plastic.sleep-observatory/1' });
  });

  it('keeps absent apart from zero', () => {
    expect(ratio(undefined)).toBe('n/a');
    expect(ratio({ n: 0, recalled: 0, n_paraphrase: 0, recalled_paraphrase: 0 })).toBe('n/a');
    expect(ratio({ n: 24, recalled: 0, n_paraphrase: 24, recalled_paraphrase: 2 })).toBe('0/24');
    expect(ratio({ n: 24, recalled: 0, n_paraphrase: 24, recalled_paraphrase: 2 }, 'paraphrase')).toBe('2/24');
    const index = readExport<ObservatoryIndex>('index.json');
    const seed0 = readExport<Run>('runs/final_step250_seed0_exclude.json');
    expect(index.runs.find((r) => r.id === seed0.id)?.session_trajectory).toBe('sessions/final_step250_seed0_exclude.json');
  });

  it('never gives a verdict for a check that was not in force', () => {
    const all40 = readExport<Run>('runs/sleep_controls_step100_all40.json');
    const replay = all40.arms.find((a) => a.arm === 'replay');
    expect(replay?.kind).toBe('sleep');
    if (replay?.kind !== 'sleep') return;
    const share = replay.gate?.checks.find((c) => c.name === 'reply_cluster_share');
    expect(share && checkState(share)).toBe('not in force');
    expect(replay.status).toBe('accepted');
  });

  it('round-trips shareable routes', () => {
    const r = { view: 'runs' as const, run: 'final_step250_seed0_exclude', arm: 'dream' };
    expect(parseRoute(formatRoute(r))).toEqual(r);
    expect(parseRoute('#sleep')).toEqual({ view: 'runs', run: null, arm: null });
    expect(parseRoute('#chat')).toEqual({ view: 'runs', run: null, arm: null });
    expect(formatRoute({ view: 'runs', run: null, arm: 'dream' })).toBe('#sleep/runs');
  });
});
