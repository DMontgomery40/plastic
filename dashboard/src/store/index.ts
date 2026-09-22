// One store for the whole dashboard. Components read slices from it and call
// its actions; nothing else talks to the API.

import { create } from 'zustand';
import * as api from '../api/client';
import { ApiError } from '../api/client';
import type {
  CalibrateRequest,
  ChatRequest,
  ChatResult,
  CreateSessionRequest,
  DataDir,
  EpisodeResult,
  Health,
  ModelDetail,
  ModelSummary,
  PhysicsRequest,
  RedteamDetail,
  RedteamRequest,
  RedteamSummary,
  SessionDetail,
  SessionState,
  SessionSummary,
  TrainJob,
  TrainRequest,
  TrainStatus,
} from '../api/types';

export const TAB_KEYS = ['sessions', 'session', 'chat', 'physics', 'train', 'redteam', 'architecture'] as const;
export type TabKey = (typeof TAB_KEYS)[number];

export const TAB_LABELS: Record<TabKey, string> = {
  sessions: 'Sessions',
  session: 'Session',
  chat: 'Chat',
  physics: 'Physics',
  train: 'Train',
  redteam: 'Red team',
  architecture: 'Architecture',
};

/** Which loading flag a given request owns. Kept explicit so the UI never guesses. */
export type LoadingKey =
  | 'health'
  | 'models'
  | 'model'
  | 'sessions'
  | 'session'
  | 'sessionState'
  | 'jobs'
  | 'redteam'
  | 'redteamRun'
  | 'data'
  | 'chat'
  | 'physics'
  | 'calibrate'
  | 'mutation';

export interface PlasticState {
  // data
  health: Health | null;
  models: ModelSummary[];
  modelDetail: ModelDetail | null;
  sessions: SessionSummary[];
  currentSessionId: string | null;
  sessionDetail: SessionDetail | null;
  sessionState: SessionState | null;
  selectedTransaction: number | null;
  chatResult: ChatResult | null;
  episodeResult: EpisodeResult | null;
  jobs: TrainJob[];
  trainStatus: Record<string, TrainStatus>;
  redteamRuns: RedteamSummary[];
  redteamDetail: RedteamDetail | null;
  dataDirs: DataDir[];

  // ui
  activeTab: TabKey;
  loading: Record<LoadingKey, boolean>;
  error: string | null;

  // actions
  setActiveTab: (tab: TabKey) => void;
  setCurrentSession: (sessionId: string | null) => void;
  setSelectedTransaction: (index: number | null) => void;
  clearError: () => void;

  refreshHealth: () => Promise<void>;
  refreshModels: () => Promise<void>;
  refreshSessions: () => Promise<void>;
  refreshDataDirs: () => Promise<void>;
  refreshJobs: () => Promise<void>;
  refreshRedteam: () => Promise<void>;
  bootstrap: () => Promise<void>;

  loadModel: (modelId: string) => Promise<void>;
  calibrate: (modelId: string, body?: CalibrateRequest) => Promise<void>;

  loadSession: (sessionId: string) => Promise<void>;
  loadSessionState: (sessionId: string) => Promise<void>;
  createSession: (body: CreateSessionRequest) => Promise<SessionSummary | null>;
  forkSession: (sessionId: string, childSessionId?: string) => Promise<SessionSummary | null>;
  resetSession: (sessionId: string) => Promise<void>;
  resumeSession: (sessionId: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;

  sendChat: (body: ChatRequest) => Promise<void>;
  runEpisode: (body: PhysicsRequest) => Promise<void>;

  startTraining: (body: TrainRequest) => Promise<void>;
  cancelTraining: (modelId: string) => Promise<void>;

  loadRedteamRun: (runId: string) => Promise<void>;
  runRedteam: (body: RedteamRequest) => Promise<void>;
}

const NO_LOADING: Record<LoadingKey, boolean> = {
  health: false,
  models: false,
  model: false,
  sessions: false,
  session: false,
  sessionState: false,
  jobs: false,
  redteam: false,
  redteamRun: false,
  data: false,
  chat: false,
  physics: false,
  calibrate: false,
  mutation: false,
};

export const initialState = {
  health: null,
  models: [],
  modelDetail: null,
  sessions: [],
  currentSessionId: null,
  sessionDetail: null,
  sessionState: null,
  selectedTransaction: null,
  chatResult: null,
  episodeResult: null,
  jobs: [],
  trainStatus: {},
  redteamRuns: [],
  redteamDetail: null,
  dataDirs: [],
  activeTab: 'sessions' as TabKey,
  loading: NO_LOADING,
  error: null,
};

export function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.status ? `${err.message} (HTTP ${err.status})` : err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}

export const useStore = create<PlasticState>()((set, get) => {
  /** Run a request with its loading flag set, recording any failure in `error`. */
  const withLoading = async <T,>(key: LoadingKey, run: () => Promise<T>): Promise<T | null> => {
    set((s) => ({ loading: { ...s.loading, [key]: true }, error: null }));
    try {
      return await run();
    } catch (err) {
      set({ error: describeError(err) });
      return null;
    } finally {
      set((s) => ({ loading: { ...s.loading, [key]: false } }));
    }
  };

  return {
    ...initialState,

    setActiveTab: (tab) => set({ activeTab: tab }),

    setCurrentSession: (sessionId) => {
      set({
        currentSessionId: sessionId,
        sessionDetail: null,
        sessionState: null,
        selectedTransaction: null,
        chatResult: null,
        episodeResult: null,
      });
      if (sessionId) void get().loadSession(sessionId);
    },

    setSelectedTransaction: (index) => set({ selectedTransaction: index }),

    clearError: () => set({ error: null }),

    refreshHealth: async () => {
      const health = await withLoading('health', api.getHealth);
      if (health) set({ health });
    },

    refreshModels: async () => {
      const models = await withLoading('models', api.getModels);
      if (models) set({ models });
    },

    refreshSessions: async () => {
      const sessions = await withLoading('sessions', api.getSessions);
      if (!sessions) return;
      const current = get().currentSessionId;
      const stillThere = current !== null && sessions.some((s) => s.session_id === current);
      set({ sessions, currentSessionId: stillThere ? current : (sessions[0]?.session_id ?? null) });
    },

    refreshDataDirs: async () => {
      const dataDirs = await withLoading('data', api.getDataDirs);
      if (dataDirs) set({ dataDirs });
    },

    refreshJobs: async () => {
      const jobs = await withLoading('jobs', api.getTrainJobs);
      if (!jobs) return;
      set({ jobs });
      const statuses = await Promise.all(
        jobs.map(async (job) => {
          try {
            return await api.getTrainStatus(job.model_id);
          } catch {
            return null;
          }
        }),
      );
      const next: Record<string, TrainStatus> = { ...get().trainStatus };
      statuses.forEach((status) => {
        if (status) next[status.model_id] = status;
      });
      set({ trainStatus: next });
    },

    refreshRedteam: async () => {
      const redteamRuns = await withLoading('redteam', api.getRedteamRuns);
      if (redteamRuns) set({ redteamRuns });
    },

    bootstrap: async () => {
      await Promise.all([
        get().refreshHealth(),
        get().refreshModels(),
        get().refreshSessions(),
        get().refreshDataDirs(),
        get().refreshJobs(),
        get().refreshRedteam(),
      ]);
      const sessionId = get().currentSessionId;
      if (sessionId) await get().loadSession(sessionId);
    },

    loadModel: async (modelId) => {
      const detail = await withLoading('model', () => api.getModel(modelId));
      if (detail) set({ modelDetail: detail });
    },

    calibrate: async (modelId, body = {}) => {
      const result = await withLoading('calibrate', () => api.calibrateModel(modelId, body));
      if (!result) return;
      await get().refreshModels();
      await get().loadModel(modelId);
    },

    loadSession: async (sessionId) => {
      const detail = await withLoading('session', () => api.getSession(sessionId));
      if (!detail) return;
      set({ sessionDetail: detail, currentSessionId: sessionId });
      await get().loadSessionState(sessionId);
    },

    loadSessionState: async (sessionId) => {
      const state = await withLoading('sessionState', () => api.getSessionState(sessionId));
      set({ sessionState: state });
    },

    createSession: async (body) => {
      const created = await withLoading('mutation', () => api.createSession(body));
      if (!created) return null;
      await get().refreshSessions();
      await get().loadSession(created.session_id);
      return created;
    },

    forkSession: async (sessionId, childSessionId) => {
      const child = await withLoading('mutation', () => api.forkSession(sessionId, childSessionId));
      if (!child) return null;
      await get().refreshSessions();
      await get().loadSession(child.session_id);
      return child;
    },

    resetSession: async (sessionId) => {
      const done = await withLoading('mutation', () => api.resetSession(sessionId));
      if (!done) return;
      await get().refreshSessions();
      await get().loadSession(sessionId);
    },

    resumeSession: async (sessionId) => {
      const done = await withLoading('mutation', () => api.resumeSession(sessionId));
      if (!done) return;
      await get().refreshSessions();
      await get().loadSession(sessionId);
    },

    deleteSession: async (sessionId) => {
      const done = await withLoading('mutation', () => api.deleteSession(sessionId));
      if (!done) return;
      if (get().currentSessionId === sessionId) {
        set({ currentSessionId: null, sessionDetail: null, sessionState: null, selectedTransaction: null });
      }
      await get().refreshSessions();
    },

    sendChat: async (body) => {
      const sessionId = get().currentSessionId;
      if (!sessionId) {
        set({ error: 'no session selected' });
        return;
      }
      const result = await withLoading('chat', () => api.chat(sessionId, body));
      if (!result) return;
      set({ chatResult: result });
      await get().loadSession(sessionId);
    },

    runEpisode: async (body) => {
      const sessionId = get().currentSessionId;
      if (!sessionId) {
        set({ error: 'no session selected' });
        return;
      }
      const result = await withLoading('physics', () => api.runPhysics(sessionId, body));
      if (!result) return;
      set({ episodeResult: result });
      await get().loadSession(sessionId);
    },

    startTraining: async (body) => {
      const started = await withLoading('mutation', () => api.startTraining(body));
      if (!started) return;
      await get().refreshJobs();
      await get().refreshModels();
    },

    cancelTraining: async (modelId) => {
      const done = await withLoading('mutation', () => api.cancelTraining(modelId));
      if (!done) return;
      await get().refreshJobs();
    },

    loadRedteamRun: async (runId) => {
      const detail = await withLoading('redteamRun', () => api.getRedteamRun(runId));
      if (detail) set({ redteamDetail: detail });
    },

    runRedteam: async (body) => {
      const summary = await withLoading('redteam', () => api.runRedteam(body));
      if (!summary) return;
      await get().refreshRedteam();
      await get().loadRedteamRun(summary.run_id);
    },
  };
});

// ------------------------------------------------------------------- polling
// Running training jobs are polled every 2 s. The interval only exists while at
// least one job is running, and is torn down as soon as none is.

const POLL_MS = 2000;
let pollTimer: ReturnType<typeof setInterval> | null = null;

export function hasRunningJob(jobs: TrainJob[]): boolean {
  return jobs.some((job) => job.status === 'running');
}

export function startJobPolling(): void {
  if (pollTimer !== null) return;
  pollTimer = setInterval(() => {
    const { jobs, refreshJobs } = useStore.getState();
    if (!hasRunningJob(jobs)) {
      stopJobPolling();
      return;
    }
    void refreshJobs();
  }, POLL_MS);
}

export function stopJobPolling(): void {
  if (pollTimer === null) return;
  clearInterval(pollTimer);
  pollTimer = null;
}

export function isPolling(): boolean {
  return pollTimer !== null;
}

/** Keep the poll loop in step with the job list. Called whenever jobs change. */
export function syncJobPolling(jobs: TrainJob[]): void {
  if (hasRunningJob(jobs)) startJobPolling();
  else stopJobPolling();
}
