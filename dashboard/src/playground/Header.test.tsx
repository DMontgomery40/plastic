// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { Header, modelLabel, openableModels } from './Header';
import { useStore } from './store';
import type { ModelSummary, SessionSummary } from './types';

const qwen: ModelSummary = {
  model_id: 'qwen3_5_0_8b_abliterated', backend: 'qwen', domain: 'text', status: 'completed', params: 752_393_024,
  calibrated: false, has_canary: false, created_at_unix: 0, updated_at_unix: 0, parent_model_id: null,
};
const core: ModelSummary = { ...qwen, model_id: 'lm_wikitext_l4', backend: 'plastic', params: 6_845_984, calibrated: true, has_canary: true, chat_tuned: false };
const physics: ModelSummary = { ...core, model_id: 'phys', domain: 'physics' };
const child: ModelSummary = { ...qwen, model_id: 'sleep_1', parent_model_id: 'qwen3_5_0_8b_abliterated' };

function session(id: string, model: string, domain: 'text' | 'physics' = 'text'): SessionSummary {
  return { session_id: id, model_id: model, domain, parent_session_id: null, created_at_unix: 0, updated_at_unix: 0, pos: 0, n_transactions: 0,
    commits: 0, rollbacks: 0, scales: 0, projects: 0, readonly: 0, read_only: false, read_only_reason: null } as unknown as SessionSummary;
}

const initial = useStore.getState();
let selected: (string | null)[];

beforeEach(() => {
  selected = [];
  useStore.setState({ ...initial, health: { ok: true, artifacts_root: '', device: 'cpu', n_models: 1, n_sessions: 1, public: true,
    capabilities: { create_session: false, fork: false, reset: true, delete: false, resume: true, calibrate: false, sleep: false } },
    selectSession: async (id: string | null) => { selected.push(id); useStore.setState({ currentSessionId: id }); } }, true);
});
afterEach(() => { cleanup(); useStore.setState(initial, true); vi.restoreAllMocks(); });

describe('header model and session pickers', () => {
  it('shows a single model and session as plain text, not a one-option menu', () => {
    useStore.setState({ models: [qwen], sessions: [session('demo_text', qwen.model_id)], currentSessionId: 'demo_text' });
    render(<Header />);
    expect(screen.queryByRole('combobox')).toBeNull();
    expect(screen.getByText('qwen3_5_0_8b_abliterated · Gated DeltaNet (Qwen3.5) · 752M')).toBeTruthy();
  });

  it('offers every openable model and opens the chosen model’s session', () => {
    useStore.setState({ models: [qwen, core, physics], sessions: [session('demo_text', qwen.model_id), session('demo_core', core.model_id), session('demo_physics', 'phys', 'physics')], currentSessionId: 'demo_text' });
    render(<Header />);
    const picker = screen.getByRole('combobox', { name: 'Model' }) as HTMLSelectElement;
    expect([...picker.options].map((o) => o.textContent)).toEqual(['qwen3_5_0_8b_abliterated · Gated DeltaNet (Qwen3.5) · 752M', 'lm_wikitext_l4 · Plastic delta memory · 6.8M']);
    fireEvent.change(picker, { target: { value: 'lm_wikitext_l4' } });
    expect(selected).toEqual(['demo_core']);
    expect(screen.queryByRole('combobox', { name: 'Session' })).toBeNull(); // one session per model
  });

  it('adds a session menu only when the chosen model has several sessions', () => {
    useStore.setState({ models: [qwen], sessions: [session('a', qwen.model_id), session('b', qwen.model_id)], currentSessionId: 'a' });
    render(<Header />);
    expect(screen.queryByRole('combobox', { name: 'Model' })).toBeNull();
    const s = screen.getByRole('combobox', { name: 'Session' }) as HTMLSelectElement;
    fireEvent.change(s, { target: { value: 'b' } });
    expect(selected).toEqual(['b']);
  });

  it('lists only models the server returned that have a text session, and names sleep children', () => {
    const list = openableModels([qwen, core, child], [session('demo_text', qwen.model_id), session('demo_sleep_1', 'sleep_1')]);
    expect(list.map((m) => m.model_id)).toEqual(['qwen3_5_0_8b_abliterated', 'sleep_1']);
    expect(modelLabel(child)).toBe('sleep_1 · Gated DeltaNet (Qwen3.5) · 752M · sleep child of qwen3_5_0_8b_abliterated');
  });
});
