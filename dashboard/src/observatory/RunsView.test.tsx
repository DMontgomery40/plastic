// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { clearObservatoryCache } from './data';
import { ObservatoryScreen } from './ObservatoryScreen';
import { exportFetch } from './testData';

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
  it('opens on the latest run with a pulled-back arm: no child, the failed check and its value', async () => {
    render(<ObservatoryScreen />);
    await screen.findByRole('heading', { name: 'final_step250_seed0_exclude' });
    const armPanel = (await screen.findByText('locality gate failed')).closest('section')!;
    expect(within(armPanel).getByText('no child')).toBeTruthy();
    expect(within(armPanel).getByText(/Pulled back: the child was discarded/)).toBeTruthy();
    expect(within(armPanel).getByRole('img', { name: /largest identical-reply share: value 0\.308, limit 0\.250, failed/ })).toBeTruthy();
    expect(within(armPanel).getByText('selected')).toBeTruthy();
    expect(screen.queryByText(/Illustrative/)).toBeNull(); // synthetic numbers never appear in the archive view
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
