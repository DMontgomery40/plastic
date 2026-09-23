// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { clearObservatoryCache } from './data';
import { ablationState, clearLearningCache, rateCell, speedLabel, windowLabel, type AblationSet, type LearningIndex, type ReportSet, type Variant } from './learning';
import { LEARNING_LEAD } from './LearningView';
import { formatRoute, ObservatoryScreen, parseRoute } from './ObservatoryScreen';
import { exportFetch, readLearningExport } from './testData';

const committed = () => readLearningExport<LearningIndex>();
const phys = (i: LearningIndex) => i.sets.find((s): s is ReportSet => s.kind === 'report' && s.id === 'phys_mps_3k')!;
const reportsOnly = (): LearningIndex => ({ ...committed(), sets: committed().sets.filter((s) => s.kind === 'report') });

/** Serve a given learning index (or an HTTP failure) and, optionally, fail the Sleep export. */
function stub(learning: LearningIndex | number, sleep: 'ok' | number = 'ok') {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string | URL | Request) => {
      if (String(url).includes('assets/learning/')) {
        return typeof learning === 'number' ? new Response('unavailable', { status: learning }) : new Response(JSON.stringify(learning), { status: 200 });
      }
      return sleep === 'ok' ? exportFetch(url) : new Response('unavailable', { status: sleep });
    })
  );
}

function variant(name: string, over: Partial<Variant> = {}): Variant {
  const current = committed().current_contract_version;
  return {
    variant: name, config: {}, parameters: 409095, size: { d_model: 128, n_heads: 4, n_layers: 3, chunk: 16 }, train_steps: 1500,
    s_per_step: 0.19, final_train_loss: 0.06, eta_per_layer: name === 'delta_baseline' ? null : [0.276, 0.339, 0.386],
    no_adapt_label: name === 'delta_baseline' ? 'writes disabled (beta_scale=0)' : 'fast parameters frozen (freeze=True)',
    contract_version: current, current: true, split_id: 'split', adaptation_window: { update_period: 16, boundaries_per_episode: 3, checked: true },
    before: phys(committed()).before, ...over,
  };
}

function withAblation(variants: Variant[], missing: string[], over: Partial<AblationSet> = {}): LearningIndex {
  const current = committed().current_contract_version;
  const set: AblationSet = { id: 'coordinate-ablation', kind: 'ablation', tag: null, execution_commit: 'abcdef012345', contract_version: current,
    contract_versions: [current], current: true, variants, missing, sources: [], ...over };
  return { ...reportsOnly(), sets: [...reportsOnly().sets, set] };
}

beforeEach(() => {
  clearObservatoryCache();
  clearLearningCache();
  vi.stubGlobal('fetch', vi.fn(exportFetch));
  window.history.replaceState(null, '', '/#sleep/learning');
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('Learning view', () => {
  it('shows the checkpoint, contract, split and adaptation window of the committed report set', async () => {
    const set = phys(committed());
    render(<ObservatoryScreen />);
    expect(screen.getByRole('heading', { level: 1, name: 'Learning' })).toBeTruthy();
    expect(screen.getAllByText(LEARNING_LEAD)).toHaveLength(1);
    const identity = (await screen.findByRole('heading', { name: 'phys_mps_3k' })).closest('section')!;
    for (const value of [set.checkpoint.digest_prefix!, set.execution_commit!, set.contract_version!, set.split.id!]) {
      expect(within(identity).getByText(value)).toBeTruthy();
    }
    expect(within(identity).getByText('fast update every step, 63 boundaries inside each episode')).toBeTruthy();
    expect(within(identity).getByText('10 training pairings, 5 held out')).toBeTruthy();
    expect(within(identity).queryByText('stale')).toBeNull();
  });

  it('labels the before-stream "without" column with the learner’s own off-intervention', async () => {
    render(<ObservatoryScreen />);
    const before = await screen.findByRole('table', { name: /Held-out MSE with and without/ });
    expect(within(before).getByRole('columnheader', { name: 'Without: writes disabled (beta_scale=0)' })).toBeTruthy();
    const cells = (label: RegExp) => within(within(before).getByRole('row', { name: label })).getAllByRole('cell').map((c) => c.textContent);
    expect(cells(/impulse policy/)).toEqual(['0.364', '0.530', '2,048']);
    expect(cells(/hold policy/)).toEqual(['0.068', '0.360', '2,048']);
    expect(cells(/Training distribution \(gaussian\)/)).toEqual(['0.249', '0.474', '2,048']);
    expect(screen.getByText(/mean ratio 0\.392; below one half at step 2, over 8 episodes/)).toBeTruthy();
  });

  it('keeps an empty acceptance side as n/a and a measured zero as zero, per mode', async () => {
    render(<ObservatoryScreen />);
    const after = await screen.findByRole('table', { name: /Lasting change per baseline mode/ });
    const row = (label: string) => within(within(after).getByRole('row', { name: new RegExp(label) })).getAllByRole('cell').map((c) => [c.dataset.mode, c.textContent]);
    expect(row('Accepted good')).toEqual([['frozen', 'n/a (n=0)'], ['continued', '1.00 (n=2)'], ['in_context', '1.00 (n=2)']]);
    expect(row('Refused bad')).toEqual([['frozen', 'n/a (n=0)'], ['continued', '0.00 (n=1)'], ['in_context', '0.00 (n=1)']]);
    expect(row('Transfer Δ, hold policy')[1]).toEqual(['continued', '+0.221']);
    expect(row('Poison harm vs clean stream')[1]).toEqual(['continued', '−0.011']);
    expect(row('Poison harm vs start')[1]).toEqual(['continued', '+0.019']);
    expect(row('Revert to pre-stream parameters').map(([, t]) => t)).toEqual(['ok', 'ok', 'ok']);
    expect(row('Tokens measured')[2]).toEqual(['in_context', '353,28025,600 without the stream']);
    expect(row('Parameters')[0]).toEqual(['frozen', '3,564,764']);
    expect(rateCell(0, 1)).toBe('0.00 (n=1)');
    expect(rateCell(0, 0)).toBe('n/a (n=0)');
    expect(rateCell(null, null)).toBe('n/a (n=0)');
  });

  it('says the coordinate ablation is not yet archived, distinct from failed and stale', async () => {
    stub(reportsOnly());
    render(<ObservatoryScreen />);
    const panel = (await screen.findByRole('heading', { name: 'Coordinate ablation' })).closest('section')!;
    expect(within(panel).getByText('not yet archived')).toBeTruthy();
    expect(within(panel).queryByText('stale')).toBeNull();
    expect(screen.queryByText(/could not be loaded/)).toBeNull();
    expect(ablationState(reportsOnly()).state).toBe('not archived');
  });

  it('shows each archived variant against its own off-intervention and lists missing variants', async () => {
    stub(withAblation([variant('full'), variant('delta_baseline')], ['no_fast', 'decay_only']));
    render(<ObservatoryScreen />);
    const table = await screen.findByRole('table', { name: /Coordinate ablation/ });
    const row = (name: string) => within(table).getByRole('row', { name: new RegExp(name) });
    expect(within(row('Full candidate')).getByText('fast parameters frozen (freeze=True)')).toBeTruthy();
    expect(within(row('Full candidate')).getByText('0.276, 0.339, 0.386')).toBeTruthy();
    expect(within(row('Delta-rule baseline')).getByText('writes disabled (beta_scale=0)')).toBeTruthy();
    expect(within(row('Delta-rule baseline')).getByText('n/a')).toBeTruthy(); // no learned step size, not zero
    expect(within(row('No fast updates')).getByText('not archived')).toBeTruthy();
    expect(within(row('Decay only')).getByText('not archived')).toBeTruthy();
    expect(within(table).queryByRole('columnheader', { name: /^Without: / })).toBeNull(); // no single shared off-intervention
    const panel = table.closest('section')!;
    expect(within(panel).queryByText('not yet archived')).toBeNull();
    expect(within(panel).queryByText('stale')).toBeNull();
  });

  it('marks an ablation measured under a superseded contract as stale', async () => {
    const old = variant('full', { contract_version: '2026-09-23.1', current: false });
    stub(withAblation([old], [], { contract_version: '2026-09-23.1', contract_versions: ['2026-09-23.1'], current: false }));
    render(<ObservatoryScreen />);
    const panel = (await screen.findByRole('heading', { name: 'Coordinate ablation' })).closest('section')!;
    expect(within(panel).getAllByText('stale').length).toBe(2); // the set and its variant
    expect(within(panel).getByText(/the current contract is 2026-09-23\.2/)).toBeTruthy();
    expect(within(panel).queryByText('not yet archived')).toBeNull();
  });

  it('reports a failed load with a retry, and renders without the Sleep export', async () => {
    stub(503, 500);
    render(<ObservatoryScreen />);
    expect(await screen.findByText('Learning results could not be loaded.')).toBeTruthy();
    expect(screen.queryByText('not yet archived')).toBeNull();
    stub(committed(), 500);
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(await screen.findByRole('heading', { name: 'phys_mps_3k' })).toBeTruthy();
    expect(screen.queryByText('Sleep runs could not be loaded.')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Runs' }));
    expect(await screen.findByText('Sleep runs could not be loaded.')).toBeTruthy();
  });

  it('labels an undeclared window and a speed that never halves, and routes to a set', () => {
    expect(windowLabel({ update_period: null, boundaries_per_episode: null, checked: false })).toBe('unchecked: the learner declared no update period');
    expect(speedLabel({ mean_ratio: 0.8, steps_to_half: null, reached_half: false, horizon: 64, episodes: 8 })).toBe('mean ratio 0.800; not below one half within 64 steps');
    expect(speedLabel(null)).toBe('n/a');
    const r = { view: 'learning' as const, run: 'phys_mps_3k', arm: null };
    expect(parseRoute(formatRoute(r))).toEqual(r);
  });
});
