// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ChatScreen } from './ChatScreen';
import { useStore } from '../store';
import type { SessionDetail } from '../types';

const detail: SessionDetail = {
  meta: {
    session_id: 'demo_text', model_id: 'test', domain: 'text', parent_session_id: null,
    created_at_unix: 0, updated_at_unix: 0, pos: 0, n_transactions: 0,
    commits: 0, rollbacks: 0, scales: 0, projects: 0, readonly: 0,
    read_only: false, read_only_reason: null,
    harness: { log_only: true, enable_rollback: false, enable_stats: true,
      enable_projection: false, enable_budget: false, freeze_on_alarm: false,
      learn_from_generation: true, budget_session: null, target_fpr: 0.01 },
  },
  summary: { pos: 0, pending: 0, budget_used: 0, budget_session: null,
    read_only: false, read_only_reason: null, n_transactions: 0,
    cusum: { k: 0, h: 5, s_hi: 0, s_lo: 0, alarms: 0 }, state_norms: {}, drift_from_anchor: 0 },
  calibration: null, lineage: [], transactions: [], trace: [],
};
const initial = useStore.getState();
let responseStatus = 422;
let posts: unknown[];
beforeEach(() => {
  posts = [];
  responseStatus = 422;
  vi.stubGlobal('fetch', vi.fn(async (url: string, options: RequestInit = {}) => {
    if (options.method === 'POST') {
      posts.push(JSON.parse(options.body as string));
      if (responseStatus === 0) throw new TypeError('offline');
      return new Response(JSON.stringify(responseStatus === 200
        ? { prompt: 'Hello.', completion: 'Hello!', transactions: [], n_tokens_in: 1, n_tokens_out: 1, summary: detail.summary }
        : { detail: 'Request rejected' }), { status: responseStatus });
    }
    const data = url.endsWith('/state') ? { pos: 0, layers: [] }
      : url.includes('/transactions') ? { total: 0, items: [] }
      : url.endsWith('/sessions') ? [detail.meta] : detail;
    return new Response(JSON.stringify(data));
  }));
  Element.prototype.scrollIntoView = vi.fn();
  useStore.setState({ ...initial, detail, currentSessionId: 'demo_text',
    health: { ok: true, artifacts_root: '', device: 'cpu', n_models: 1, n_sessions: 1, public: true,
      capabilities: { create_session: false, fork: false, reset: true, delete: false, resume: true, calibrate: false, sleep: false } } }, true);
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); useStore.setState(initial, true); });

function typePrompt() {
  const prompt = screen.getByLabelText('Prompt') as HTMLTextAreaElement;
  fireEvent.change(prompt, { target: { value: 'Hello.' } });
  return prompt;
}

describe('turn text', () => {
  it('wraps a reply with no spaces inside its card', () => {
    const looped = 'an'.repeat(120);
    useStore.setState({ detail: { ...useStore.getState().detail!, trace: [{ t_unix: 0, kind: 'chat', prompt: 'x'.repeat(200), completion: looped, pos_end: 10, n_transactions: 0 }] } });
    render(<ChatScreen />);
    expect(screen.getByText(looped).className).toContain('[overflow-wrap:anywhere]');
    expect(screen.getByText('x'.repeat(200)).className).toContain('[overflow-wrap:anywhere]');
  });
});

describe('chat request validation and recovery', () => {
  it.each([['Max tokens', '129'], ['Max tokens', '1.5'], ['Temperature', '0'], ['Top-k', '-1'], ['Seed', '-1']])(
    'keeps invalid %s input out of the public request', async (label, value) => {
      render(<ChatScreen />);
      const prompt = typePrompt();
      fireEvent.change(screen.getByLabelText(label), { target: { value } });
      fireEvent.click(screen.getByRole('button', { name: 'Send' }));
      await act(async () => {});
      expect(posts).toEqual([]);
      expect(prompt.value).toBe('Hello.');
    },
  );
  it('uses the same validation for Enter', async () => {
    render(<ChatScreen />);
    const prompt = typePrompt();
    fireEvent.change(screen.getByLabelText('Max tokens'), { target: { value: '129' } });
    fireEvent.keyDown(prompt, { key: 'Enter' });
    await act(async () => {});
    expect(posts).toEqual([]);
    expect(prompt.value).toBe('Hello.');
  });
  it.each([409, 413, 422, 429, 503, 0])('retains the draft after a %s failure and permits a successful retry', async (status) => {
    responseStatus = status;
    render(<ChatScreen />);
    const prompt = typePrompt();
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));
    await waitFor(() => expect(useStore.getState().error).not.toBeNull());
    expect(prompt.value).toBe('Hello.');
    expect(useStore.getState().stale).toBe(status === 0 || status >= 500);
    responseStatus = 200;
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));
    await waitFor(() => expect(prompt.value).toBe(''));
    expect(posts).toHaveLength(2);
    expect(useStore.getState().lastChat?.completion).toBe('Hello!');
    expect(useStore.getState().stale).toBe(false);
  });
  it('keeps local sampling above the public cap available', async () => {
    responseStatus = 200;
    useStore.setState({ health: { ...useStore.getState().health!, public: false } });
    render(<ChatScreen />);
    const prompt = typePrompt();
    fireEvent.change(screen.getByLabelText('Max tokens'), { target: { value: '256' } });
    fireEvent.keyDown(prompt, { key: 'Enter' });
    await waitFor(() => expect(useStore.getState().lastChat?.completion).toBe('Hello!'));
    expect(posts).toHaveLength(1);
    expect((posts[0] as { max_new_tokens: number }).max_new_tokens).toBe(256);
  });
  it('accepts the public upper token boundary with fractional temperature', async () => {
    responseStatus = 200;
    render(<ChatScreen />);
    const prompt = typePrompt();
    fireEvent.change(screen.getByLabelText('Max tokens'), { target: { value: '128' } });
    fireEvent.change(screen.getByLabelText('Temperature'), { target: { value: '0.8' } });
    fireEvent.keyDown(prompt, { key: 'Enter' });
    await waitFor(() => expect(useStore.getState().lastChat?.completion).toBe('Hello!'));
    expect(posts).toEqual([{ prompt: 'Hello.', max_new_tokens: 128, temperature: 0.8, top_k: 40, seed: null }]);
  });
});
