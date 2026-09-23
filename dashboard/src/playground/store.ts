// One store, three concerns: what the server allows, which session is open, and what it produced.
// Components read slices and call actions; nothing else talks to the API.

import { create } from 'zustand';
import * as api from './client';
import { ApiError } from './client';
import type {
  Capabilities,
  ChatResult,
  Health,
  ModelSummary,
  Sampling,
  SessionDetail,
  SessionState,
  SessionSummary,
  SleepOptions,
  SleepRun,
  TransactionRecord,
} from './types';

export const TABS = ['chat', 'signals', 'sessions'] as const;
export type Tab = (typeof TABS)[number];
export const TAB_LABELS: Record<Tab, string> = { chat: 'Chat', signals: 'Signals', sessions: 'Sessions' };

const NO_CAPABILITIES: Capabilities = { create_session: false, fork: false, reset: false, delete: false, resume: false, calibrate: false, sleep: false };

export interface PlaygroundState {
  health: Health | null;
  healthChecked: boolean;
  models: ModelSummary[];
  sessions: SessionSummary[];
  currentSessionId: string | null;
  detail: SessionDetail | null;
  transactions: TransactionRecord[];
  state: SessionState | null;
  lastChat: ChatResult | null;
  /** True after a request failed since the last successful refresh: data shown may be stale. */
  stale: boolean;
  sampling: Sampling;
  tab: Tab;
  busy: { chat: boolean; session: boolean; mutation: boolean };
  /** model id whose real-chat calibration is running (it generates every response, so it takes minutes) */
  calibrating: string | null;
  /** sleep runs known to the server, newest first; a running one is polled by the Sessions screen */
  sleepRuns: SleepRun[];
  error: string | null;

  capabilities: () => Capabilities;
  setTab: (tab: Tab) => void;
  setSampling: (patch: Partial<Sampling>) => void;
  clearError: () => void;
  bootstrap: () => Promise<void>;
  refreshSessions: () => Promise<void>;
  /** Re-read the model catalog (calibration can also happen from the CLI or another client). */
  refreshModels: () => Promise<void>;
  selectSession: (sessionId: string | null) => Promise<void>;
  reloadCurrent: () => Promise<void>;
  sendChat: (prompt: string) => Promise<boolean>;
  createSession: (modelId: string, guarded: boolean) => Promise<string | null>;
  resetSession: (sessionId: string) => Promise<void>;
  resumeSession: (sessionId: string) => Promise<void>;
  forkSession: (sessionId: string) => Promise<string | null>;
  deleteSession: (sessionId: string) => Promise<void>;
  calibrate: (modelId: string) => Promise<void>;
  refreshSleep: () => Promise<void>;
  startSleep: (modelId: string, options: SleepOptions) => Promise<void>;
}

function message(err: unknown): string {
  if (err instanceof ApiError) return err.status === 0 ? `Connection failed: ${err.message}` : err.message;
  return err instanceof Error ? err.message : String(err);
}

export const useStore = create<PlaygroundState>((set, get) => ({
  health: null,
  healthChecked: false,
  models: [],
  sessions: [],
  currentSessionId: null,
  detail: null,
  transactions: [],
  state: null,
  lastChat: null,
  stale: false,
  sampling: { max_new_tokens: 96, temperature: 0.8, top_k: 40, seed: null },
  tab: 'chat',
  busy: { chat: false, session: false, mutation: false },
  calibrating: null,
  sleepRuns: [],
  error: null,

  capabilities: () => get().health?.capabilities ?? NO_CAPABILITIES,
  setTab: (tab) => set({ tab }),
  setSampling: (patch) => set({ sampling: { ...get().sampling, ...patch } }),
  clearError: () => set({ error: null }),

  bootstrap: async () => {
    try {
      const [health, models, sessions] = await Promise.all([api.getHealth(), api.getModels(), api.getSessions()]);
      set({ health, models, sessions, healthChecked: true, stale: false, error: null });
      const text = sessions.filter((s) => s.domain === 'text');
      const current = get().currentSessionId;
      const pick = current && text.some((s) => s.session_id === current) ? current : text[0]?.session_id ?? null;
      await get().selectSession(pick);
    } catch (err) {
      set({ healthChecked: true, stale: true, error: message(err) });
    }
  },

  refreshSessions: async () => {
    try {
      const sessions = await api.getSessions();
      set({ sessions, stale: false });
    } catch (err) {
      set({ stale: true, error: message(err) });
    }
  },

  refreshModels: async () => {
    try {
      set({ models: await api.getModels() });
    } catch (err) {
      set({ error: message(err) });
    }
  },

  selectSession: async (sessionId) => {
    set({ currentSessionId: sessionId, detail: null, transactions: [], state: null, lastChat: null });
    if (!sessionId) return;
    await get().reloadCurrent();
  },

  reloadCurrent: async () => {
    const id = get().currentSessionId;
    if (!id) return;
    set({ busy: { ...get().busy, session: true } });
    try {
      const [detail, page, state] = await Promise.all([api.getSession(id), api.getTransactions(id), api.getSessionState(id)]);
      if (get().currentSessionId !== id) return;
      set({ detail, transactions: page.items, state, stale: false, error: null });
    } catch (err) {
      set({ stale: true, error: message(err) });
    } finally {
      set({ busy: { ...get().busy, session: false } });
    }
  },

  sendChat: async (prompt) => {
    const id = get().currentSessionId;
    if (!id || !prompt.trim() || get().busy.chat) return false;
    set({ busy: { ...get().busy, chat: true }, error: null });
    try {
      const result = await api.chat(id, prompt, get().sampling);
      set({ lastChat: result });
      await Promise.all([get().reloadCurrent(), get().refreshSessions()]);
      return true;
    } catch (err) {
      set({ stale: get().stale || !(err instanceof ApiError) || err.status < 400 || err.status >= 500, error: message(err) });
      return false;
    } finally {
      set({ busy: { ...get().busy, chat: false } });
    }
  },

  createSession: async (modelId, guarded) => {
    set({ busy: { ...get().busy, mutation: true }, error: null });
    try {
      const created = await api.createSession({ model_id: modelId, harness: guarded ? { log_only: false } : { log_only: true, freeze_on_alarm: false } });
      await get().refreshSessions();
      await get().selectSession(created.session_id);
      set({ tab: 'chat' });
      return created.session_id;
    } catch (err) {
      set({ error: message(err) });
      return null;
    } finally {
      set({ busy: { ...get().busy, mutation: false } });
    }
  },

  resetSession: async (sessionId) => {
    set({ busy: { ...get().busy, mutation: true }, error: null });
    try {
      await api.resetSession(sessionId);
      await get().refreshSessions();
      if (get().currentSessionId === sessionId) await get().selectSession(sessionId);
    } catch (err) {
      set({ error: message(err) });
    } finally {
      set({ busy: { ...get().busy, mutation: false } });
    }
  },

  resumeSession: async (sessionId) => {
    set({ busy: { ...get().busy, mutation: true }, error: null });
    try {
      await api.resumeSession(sessionId);
      await get().refreshSessions();
      if (get().currentSessionId === sessionId) await get().reloadCurrent();
    } catch (err) {
      set({ error: message(err) });
    } finally {
      set({ busy: { ...get().busy, mutation: false } });
    }
  },

  forkSession: async (sessionId) => {
    set({ busy: { ...get().busy, mutation: true }, error: null });
    try {
      const child = await api.forkSession(sessionId);
      await get().refreshSessions();
      await get().selectSession(child.session_id);
      return child.session_id;
    } catch (err) {
      set({ error: message(err) });
      return null;
    } finally {
      set({ busy: { ...get().busy, mutation: false } });
    }
  },

  deleteSession: async (sessionId) => {
    set({ busy: { ...get().busy, mutation: true }, error: null });
    try {
      await api.deleteSession(sessionId);
      const sessions = await api.getSessions();
      set({ sessions });
      if (get().currentSessionId === sessionId) {
        const next = sessions.find((s) => s.domain === 'text')?.session_id ?? null;
        await get().selectSession(next);
      }
    } catch (err) {
      set({ error: message(err) });
    } finally {
      set({ busy: { ...get().busy, mutation: false } });
    }
  },

  refreshSleep: async () => {
    if (!get().capabilities().sleep) return;
    try {
      const runs = await api.listSleep();
      const wasRunning = get().sleepRuns.some((r) => r.status === 'running');
      set({ sleepRuns: runs });
      // a run just finished: the catalog may hold a new child model
      if (wasRunning && !runs.some((r) => r.status === 'running')) await get().refreshModels();
    } catch (err) {
      set({ error: message(err) });
    }
  },

  startSleep: async (modelId, options) => {
    set({ busy: { ...get().busy, mutation: true }, error: null });
    try {
      const run = await api.startSleep(modelId, options);
      set({ sleepRuns: [run, ...get().sleepRuns.filter((r) => r.run_id !== run.run_id)] });
    } catch (err) {
      set({ error: message(err) });
    } finally {
      set({ busy: { ...get().busy, mutation: false } });
    }
  },

  calibrate: async (modelId) => {
    set({ busy: { ...get().busy, mutation: true }, calibrating: modelId, error: null });
    try {
      await api.calibrateModel(modelId);
      const models = await api.getModels();
      set({ models });
      if (get().currentSessionId) await get().reloadCurrent();
    } catch (err) {
      set({ error: message(err) });
    } finally {
      set({ busy: { ...get().busy, mutation: false }, calibrating: null });
    }
  },
}));
