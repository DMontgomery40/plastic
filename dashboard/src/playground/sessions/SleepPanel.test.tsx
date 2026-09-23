// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { SleepPanel } from './SleepPanel';
import { useStore } from '../store';
import type { ModelSummary, SessionSummary, SleepRun } from '../types';

const ttt: ModelSummary = {
  model_id: 'ttt_base', backend: 'ttt', domain: 'text', status: 'completed', params: 759_000_000,
  calibrated: false, has_canary: false, created_at_unix: 0, updated_at_unix: 0, chat_tuned: false,
};
const toy: ModelSummary = { ...ttt, model_id: 'lm_toy', backend: 'plastic' };
const session = {
  session_id: 'teach', model_id: 'ttt_base', domain: 'text', parent_session_id: null, created_at_unix: 0, updated_at_unix: 0,
  pos: 40, n_transactions: 3, commits: 3, rollbacks: 0, scales: 0, projects: 0, readonly: 0, read_only: false, read_only_reason: null,
} as unknown as SessionSummary;

const initial = useStore.getState();
let posts: { url: string; body: unknown }[];
let serverRuns: SleepRun[];

function run(status: SleepRun['status'], extra: Partial<SleepRun> = {}): SleepRun {
  return {
    run_id: 'sleep_1_replay_w0', model_id: 'ttt_base', status, exit_code: null, started_at_unix: 0,
    options: { method: 'replay', target: 'w0', steps: 40 }, sessions: null, n_probes: 1, report: null, log_tail: ['[sleep] step 5/40 loss 1.12'], ...extra,
  };
}

beforeEach(() => {
  posts = [];
  serverRuns = [];
  vi.stubGlobal('fetch', vi.fn(async (url: string, options: RequestInit = {}) => {
    if (options.method === 'POST') {
      posts.push({ url, body: JSON.parse(options.body as string) });
      const started = run('running');
      serverRuns = [started, ...serverRuns];
      return new Response(JSON.stringify(started));
    }
    if (url.endsWith('/api/sleep')) return new Response(JSON.stringify(serverRuns));
    if (url.endsWith('/models')) return new Response(JSON.stringify([ttt, toy]));
    return new Response(JSON.stringify([]));
  }));
  useStore.setState({ ...initial, models: [ttt, toy], sessions: [session],
    health: { ok: true, artifacts_root: '', device: 'mps', n_models: 2, n_sessions: 1, public: false,
      capabilities: { create_session: true, fork: true, reset: true, delete: true, resume: true, calibrate: true, sleep: true } } }, true);
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); useStore.setState(initial, true); });

describe('sleep panel', () => {
  it('is hidden when the server does not allow sleep or no TTT model exists', () => {
    useStore.setState({ health: { ...useStore.getState().health!, capabilities: { ...useStore.getState().capabilities(), sleep: false } } });
    const { container } = render(<SleepPanel models={[ttt]} />);
    expect(container.textContent).toBe('');
    cleanup();
    useStore.setState({ health: { ...useStore.getState().health!, capabilities: { ...useStore.getState().capabilities(), sleep: true } } });
    const second = render(<SleepPanel models={[toy]} />);
    expect(second.container.textContent).toBe('');
  });

  it('starts a run with the chosen options and parsed probes, then shows it running and polls to completion', async () => {
    render(<SleepPanel models={[ttt, toy]} />);
    await waitFor(() => expect(screen.getByText('Sleep', { selector: 'button' })).toBeTruthy());
    fireEvent.change(screen.getByLabelText('Method'), { target: { value: 'distill' } });
    fireEvent.change(screen.getByLabelText('Changes'), { target: { value: 'all' } });
    fireEvent.change(screen.getByLabelText('Steps'), { target: { value: '12' } });
    fireEvent.click(screen.getByLabelText(/teach/));
    fireEvent.change(screen.getByLabelText(/Recall probes/), { target: { value: 'What is my cat called? | Marlowe | Remind me of my cat\'s name.\nWhere do I live? | Denver' } });
    await act(async () => { fireEvent.click(screen.getByText('Sleep', { selector: 'button' })); });
    expect(posts).toHaveLength(1);
    expect(posts[0].url).toContain('/api/models/ttt_base/sleep');
    expect(posts[0].body).toEqual({
      method: 'distill', target: 'all', steps: 12, sessions: ['teach'],
      probes: [{ question: 'What is my cat called?', answer: 'Marlowe', paraphrase: "Remind me of my cat's name." }, { question: 'Where do I live?', answer: 'Denver' }],
    });
    expect(screen.getByText('Sleeping…')).toBeTruthy();
    expect((screen.getByText('Sleeping…') as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText('running')).toBeTruthy();
    // the server finishes: the next poll shows the outcome with before/after numbers and the child
    serverRuns = [run('accepted', {
      report: {
        run_id: 'sleep_1_replay_w0', parent_model_id: 'ttt_base', status: 'accepted', model_id: 'sleep_child',
        harvest: { sessions: [{ session_id: 'teach', log_only: true, turns: 3, accepted_turns: 3, has_committed_state: true }], turns_by_reason: { accepted: 3 }, accepted_tokens: 172, excluded_tokens: 0 },
        before: { heldout_nll: { mean: 2.085, median: 1.422, tokens: 887 }, canary: null, recall: { n_probes: 2, recalled: 0, recalled_exact: 0, n_paraphrase: 1, recalled_paraphrase: 0 } },
        after: { heldout_nll: { mean: 2.051, median: 1.379, tokens: 887 }, canary: null, recall: { n_probes: 2, recalled: 2, recalled_exact: 1, n_paraphrase: 1, recalled_paraphrase: 1 } },
        gate: { passed: true, checks: [{ name: 'heldout_nll_mean_rise', value: -0.034, limit: 0.05, passed: true }] },
      },
    })];
    await act(async () => { await useStore.getState().refreshSleep(); });
    expect(screen.getByText('accepted')).toBeTruthy();
    expect(screen.getByText('child sleep_child')).toBeTruthy();
    expect(screen.getByText(/0\/2 → 2\/2/)).toBeTruthy();
    expect(screen.getByText(/2\.08 \/ 1\.42 → 2\.05 \/ 1\.38/)).toBeTruthy();
    expect(screen.getByText(/heldout_nll_mean_rise -0\.0340 ok/)).toBeTruthy();
    expect(screen.getByText('Sleep', { selector: 'button' })).toBeTruthy();
  });

  it('offers only the root model on the shared demo, every TTT model locally', async () => {
    const child: ModelSummary = { ...ttt, model_id: 'sleep_child', parent_model_id: 'ttt_base', type: 'sleep' };
    render(<SleepPanel models={[ttt, child]} />);
    await waitFor(() => expect(screen.getByText('Sleep', { selector: 'button' })).toBeTruthy());
    expect(Array.from((screen.getByLabelText('Model') as HTMLSelectElement).options).map((o) => o.value)).toEqual(['ttt_base', 'sleep_child']);
    cleanup();
    useStore.setState({ health: { ...useStore.getState().health!, public: true } });
    render(<SleepPanel models={[ttt, child]} />);
    await waitFor(() => expect(screen.getByText('Sleep', { selector: 'button' })).toBeTruthy());
    expect(Array.from((screen.getByLabelText('Model') as HTMLSelectElement).options).map((o) => o.value)).toEqual(['ttt_base']);
  });

  it('rejects a malformed probe line before sending anything', async () => {
    render(<SleepPanel models={[ttt]} />);
    await waitFor(() => expect(screen.getByText('Sleep', { selector: 'button' })).toBeTruthy());
    fireEvent.change(screen.getByLabelText(/Recall probes/), { target: { value: 'no separator here' } });
    await act(async () => { fireEvent.click(screen.getByText('Sleep', { selector: 'button' })); });
    expect(posts).toHaveLength(0);
    expect(screen.getByText(/each probe line is/)).toBeTruthy();
  });

  it('shows a failed run with its stderr tail and a rejected run with its reason', async () => {
    serverRuns = [
      run('failed', { run_id: 'sleep_2_anchor_w0', options: { method: 'anchor', target: 'w0' }, exit_code: 1, stderr_tail: ['RuntimeError: out of memory'] }),
      run('rejected', { run_id: 'sleep_3_replay_all', report: { run_id: 'sleep_3_replay_all', parent_model_id: 'ttt_base', status: 'rejected', reason: 'locality gate failed',
        gate: { passed: false, checks: [{ name: 'heldout_nll_mean_rise', value: 0.31, limit: 0.05, passed: false }] } } }),
    ];
    render(<SleepPanel models={[ttt]} />);
    await waitFor(() => expect(screen.getByText('failed')).toBeTruthy());
    expect(screen.getByText('RuntimeError: out of memory')).toBeTruthy();
    expect(screen.getByText('rejected')).toBeTruthy();
    expect(screen.getByText('locality gate failed')).toBeTruthy();
    expect(screen.getByText(/0\.310 FAIL/)).toBeTruthy();
  });
});
