// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { clearObservatoryCache } from './data';
import { ObservatoryScreen } from './ObservatoryScreen';
import { exportFetch, readExport } from './testData';
import type { ObservatoryIndex, Run, SleepArm } from './types';

beforeEach(() => {
  clearObservatoryCache();
  vi.stubGlobal('fetch', vi.fn(exportFetch));
  window.history.replaceState(null, '', '/#sleep/runs');
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('Sleep observatory: runs', () => {
  it.each(['runs', 'anatomy'])('summarizes the saved seed-2 Dream removals in %s', async (view) => {
    window.history.replaceState(null, '', `/#sleep/${view}/final_step250_seed2_include/dream`);
    render(<ObservatoryScreen />);
    const dreams = (await screen.findByText('Dreams', { exact: true })).parentElement!;
    expect(dreams.textContent).toContain('180 generated');
    expect(dreams.textContent).toContain('24 kept');
    expect(dreams.textContent).toContain('144 duplicates');
    expect(dreams.textContent).toContain('12 over the cap');
    expect(dreams.textContent).not.toMatch(/over cap 24|\(gain |removed as/);
    expect(dreams.textContent).toContain('The Moon is larger than the Earth.');
  });

  it.each(['runs', 'anatomy'])('groups numeric removal reasons without losing counts in %s', async (view) => {
    const id = 'final_step250_seed2_include';
    const run = readExport<Run>(`runs/${id}.json`);
    const dream = run.arms.find((arm) => arm.arm === 'dream') as SleepArm;
    dream.dreams!.rejected_reasons = {
      duplicate: 1,
      degenerate: 2,
      'gain -0.120 < 0.2': 3,
      'gain nan < 0.8': 1,
      low_gain: 2,
      'over cap 8 (gain 0.312)': 2,
      'over cap 64 (gain 1.792)': 3,
      'too long for seq_len 64 (student 80, teacher 90 tokens)': 2,
      'too long for seq_len 128 (student 130, teacher 160 tokens)': 1,
      'new filter with diagnostic value 123.456': 2,
      'another internal detail': 1,
      too_long: 0,
    };
    vi.stubGlobal('fetch', vi.fn((url: string | URL | Request) =>
      String(url).endsWith(`/runs/${id}.json`)
        ? Promise.resolve(new Response(JSON.stringify(run), { status: 200 }))
        : exportFetch(url)
    ));
    window.history.replaceState(null, '', `/#sleep/${view}/${id}/dream`);
    render(<ObservatoryScreen />);
    const dreams = (await screen.findByText('Dreams', { exact: true })).parentElement!;
    for (const summary of ['1 duplicate', '2 too short or repetitive', '6 failed the gain check', '5 over the cap', '3 too long', '3 other removals']) {
      expect(dreams.textContent).toContain(summary);
    }
    expect(dreams.textContent).not.toMatch(/seq_len|gain nan|123\.456|internal detail|0 too long/);
  });

  it.each(['runs', 'anatomy'])('keeps an empty removal summary clean in %s', async (view) => {
    const id = 'final_step250_seed2_include';
    const run = readExport<Run>(`runs/${id}.json`);
    (run.arms.find((arm) => arm.arm === 'dream') as SleepArm).dreams!.rejected_reasons = {};
    vi.stubGlobal('fetch', vi.fn((url: string | URL | Request) =>
      String(url).endsWith(`/runs/${id}.json`)
        ? Promise.resolve(new Response(JSON.stringify(run), { status: 200 }))
        : exportFetch(url)
    ));
    window.history.replaceState(null, '', `/#sleep/${view}/${id}/dream`);
    render(<ObservatoryScreen />);
    const dreams = (await screen.findByText('Dreams', { exact: true })).parentElement!;
    expect(dreams.textContent).toContain('24 kept');
    expect(dreams.textContent).not.toContain('other removals');
    if (view === 'anatomy') expect(dreams.textContent).toContain('none removed');
  });

  it('preserves a historical pulled-back arm: no child, the failed check and its value', async () => {
    window.history.replaceState(null, '', '/#sleep/runs/final_step250_seed0_exclude/dream');
    render(<ObservatoryScreen />);
    await screen.findByRole('heading', { name: 'final_step250_seed0_exclude' });
    const armPanel = (await screen.findByText('locality gate failed')).closest('section')!;
    expect(within(armPanel).getByText('no child')).toBeTruthy();
    expect(within(armPanel).getByText(/Pulled back: the child was discarded/)).toBeTruthy();
    expect(within(armPanel).getByRole('img', { name: /largest identical-reply share: value 0\.308, limit 0\.250, failed/ })).toBeTruthy();
    expect(within(armPanel).getByText('selected')).toBeTruthy();
    expect(screen.queryByText(/Illustrative/)).toBeNull(); // synthetic numbers never appear in the archive view
  });

  it.each([
    { run: 'final_step250_seed1_include', arm: 'dream', reason: 'no dream carried session information above the gain threshold' },
    { run: 'final_step250_seed0_exclude', arm: 'dream', reason: 'locality gate failed' },
    { run: 'final_step250_seed1_include', arm: 'anchor', reason: null },
    { run: 'final_step250_seed1_include', arm: 'replay', reason: null },
  ])('keeps selection separate from training and rejection cause for $run / $arm', async ({ run, arm, reason }) => {
    window.history.replaceState(null, '', `/#sleep/runs/${run}/${arm}`);
    render(<ObservatoryScreen />);
    await screen.findByRole('heading', { name: run });
    const selection = await screen.findByText('Turns selected for this arm');
    expect(within(selection.parentElement!).getByText('selected')).toBeTruthy();
    expect(screen.queryByText('Turns this arm trained on')).toBeNull();
    const rejections = screen.getByText('Rejected attempts').parentElement!;
    expect(within(rejections).getByText('no child retained')).toBeTruthy();
    expect(screen.queryByText('a locality check failed')).toBeNull();
    if (reason) expect(screen.getByText(reason)).toBeTruthy();
    if (run === 'final_step250_seed1_include' && arm === 'dream') {
      const detail = selection.closest('section')!;
      expect(within(detail).queryByText('Locality gate')).toBeNull();
      expect(within(detail).getByText('No replies match.')).toBeTruthy();
      expect(within(detail).getAllByText('n/a').length).toBeGreaterThan(0);
    }
  });

  it.each([
    { ids: ['sleep_controls_step50', 'final_step250_seed0_exclude'], run: 'final_step250_seed0_exclude', arm: 'Dream' },
    { ids: ['final_step250_seed0_exclude', 'final_step250_seed0_include'], run: 'final_step250_seed0_include', arm: 'Dream' },
    { ids: ['final_step250_seed0_exclude', 'final_step250_seed0_ceiling_single'], run: 'final_step250_seed0_exclude', arm: 'Dream' },
    { ids: ['sleep_controls_step50', 'study_step100_w0'], run: 'study_step100_w0', arm: 'Replay' },
    { ids: ['final_step250_seed0_ceiling_single'], run: 'final_step250_seed0_ceiling_single', arm: 'Ceiling' },
  ])('opens $run / $arm for archive $ids', async ({ ids, run, arm }) => {
    // Keep these archive transitions stable when more real runs are published.
    // Choosing an older run, a newer control over a sleep run, or the wrong arm must fail.
    const index = readExport<ObservatoryIndex>('index.json');
    const runs = ids.map((id) => {
      const entry = index.runs.find((r) => r.id === id);
      if (!entry) throw new Error(`missing archived fixture ${id}`);
      return entry;
    });
    vi.stubGlobal('fetch', vi.fn((url: string | URL | Request) =>
      String(url).endsWith('/index.json')
        ? Promise.resolve(new Response(JSON.stringify({ ...index, runs }), { status: 200 }))
        : exportFetch(url)
    ));
    render(<ObservatoryScreen />);
    expect(await screen.findByRole('heading', { name: run })).toBeTruthy();
    expect(await screen.findByRole('heading', { name: arm })).toBeTruthy();
  });

  it('shows a later rule as not in force on a run that predates it, and keeps the recorded outcome', async () => {
    render(<ObservatoryScreen />);
    fireEvent.click(await screen.findByRole('button', { name: /sleep_controls_step100_all40/ }));
    await screen.findByRole('heading', { name: 'sleep_controls_step100_all40' });
    await waitFor(() => expect(window.location.hash).toBe('#sleep/runs/sleep_controls_step100_all40'));
    const img = await screen.findByRole('img', { name: /largest identical-reply share: value 0\.467, limit 0\.250, not in force/ });
    const armPanel = img.closest('section')!;
    expect(within(armPanel).getAllByText('Committed').length).toBeGreaterThan(0);
    expect(within(armPanel).getByText(/not published/)).toBeTruthy();
    expect(within(armPanel).getByText('selection not recorded for this run')).toBeTruthy();
  });

  it('filters replies by group and to hits', async () => {
    window.history.replaceState(null, '', '/#sleep/runs/final_step250_seed0_exclude/floor');
    render(<ObservatoryScreen />);
    await screen.findByRole('heading', { name: 'Floor' });
    fireEvent.click(screen.getByRole('button', { name: 'General' }));
    fireEvent.click(screen.getByLabelText('hits only'));
    const rows = screen.getAllByText('hit');
    expect(rows.length).toBe(6); // 3/7 verbatim + 3/7 unseen wording at the floor
  });
});
