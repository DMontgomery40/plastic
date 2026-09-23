// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { SessionsScreen } from './SessionsScreen';
import { useStore } from '../store';
import type { ModelSummary } from '../types';

const model: ModelSummary = {
  model_id: 'ttt_base', backend: 'ttt', domain: 'text', status: 'completed', params: 759_000_000,
  calibrated: false, has_canary: false, created_at_unix: 0, updated_at_unix: 0, chat_tuned: false,
};
const initial = useStore.getState();
let release: () => void = () => {};
let calibratedOnServer = false;

beforeEach(() => {
  calibratedOnServer = false;
  vi.stubGlobal('fetch', vi.fn(async (url: string, options: RequestInit = {}) => {
    if (options.method === 'POST' && url.endsWith('/calibrate')) {
      // the real-chat calibration generates every response: it holds the request open for minutes
      await new Promise<void>((resolve) => { release = resolve; });
      calibratedOnServer = true;
      return new Response(JSON.stringify({ n_chunks: 12, thresholds: {}, achievable_fpr: {}, canary_baseline: {}, reference_sizes: {}, target_fpr: 0.01, created_at_unix: 1 }));
    }
    if (url.endsWith('/models')) return new Response(JSON.stringify([{ ...model, calibrated: calibratedOnServer }]));
    return new Response(JSON.stringify([]));
  }));
  useStore.setState({ ...initial, models: [model], sessions: [],
    health: { ok: true, artifacts_root: '', device: 'cpu', n_models: 1, n_sessions: 0, public: false,
      capabilities: { create_session: true, fork: true, reset: true, delete: true, resume: true, calibrate: true } } }, true);
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); useStore.setState(initial, true); });

describe('sessions screen', () => {
  it('re-reads the model catalog on entry so a calibration done elsewhere shows up', async () => {
    calibratedOnServer = true;
    render(<SessionsScreen />);
    await waitFor(() => expect(screen.getByText(/· calibrated$/)).toBeTruthy());
    expect(screen.getByTitle('Fit thresholds on real chats with this model').textContent).toBe('Recalibrate');
  });
});

describe('calibrate action', () => {
  it('shows the running calibration as a state, disables mutations, then reports the model as calibrated', async () => {
    render(<SessionsScreen />);
    expect(screen.getByText(/no calibration/)).toBeTruthy();
    const button = screen.getByTitle('Fit thresholds on real chats with this model');
    expect(button.textContent).toBe('Calibrate');
    await act(async () => { fireEvent.click(button); });
    expect(useStore.getState().calibrating).toBe('ttt_base');
    expect(screen.getByTitle('Fit thresholds on real chats with this model').textContent).toBe('Calibrating…');
    expect(screen.getByText(/calibrating on real chats/)).toBeTruthy();
    expect((screen.getByText('Create') as HTMLButtonElement).disabled).toBe(true);
    await act(async () => { release(); });
    await waitFor(() => expect(useStore.getState().calibrating).toBeNull());
    expect(screen.getByTitle('Fit thresholds on real chats with this model').textContent).toBe('Recalibrate');
    expect(screen.getByText(/· calibrated$/)).toBeTruthy();
    expect((screen.getByText('Create') as HTMLButtonElement).disabled).toBe(false);
  });

  it('clears the calibrating state and surfaces the error when the request fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string, options: RequestInit = {}) => {
      if (options.method === 'POST') return new Response(JSON.stringify({ detail: 'model ttt_base has no checkpoint yet' }), { status: 400 });
      return new Response(JSON.stringify(url.endsWith('/models') ? [model] : []));
    }));
    render(<SessionsScreen />);
    await act(async () => { fireEvent.click(screen.getByTitle('Fit thresholds on real chats with this model')); });
    await waitFor(() => expect(useStore.getState().calibrating).toBeNull());
    expect(useStore.getState().error).toMatch(/no checkpoint yet/);
    expect(screen.getByTitle('Fit thresholds on real chats with this model').textContent).toBe('Calibrate');
  });
});
