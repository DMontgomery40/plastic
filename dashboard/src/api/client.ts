// One fetch helper, one typed function per route. No component calls fetch directly.
//
// The dev server proxies /api to VITE_API_URL (vite.config.ts), so the default
// base is the empty string and every URL is same-origin. Setting VITE_API_URL in
// a built bundle points the client straight at the API instead.

import type {
  CalibrateRequest,
  CalibrationSummary,
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
  SleepManifest,
  TrainJob,
  TrainLogRecord,
  TrainRequest,
  TrainStarted,
  TrainStatus,
  TransactionPage,
} from './types';

export const API_BASE: string = (import.meta.env?.VITE_API_URL as string | undefined) ?? '';

export class ApiError extends Error {
  readonly status: number;
  readonly url: string;

  constructor(message: string, status: number, url: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.url = url;
  }
}

type Query = Record<string, string | number | boolean | undefined | null>;

/** Build a request path: base + path + non-empty query parameters. */
export function buildUrl(path: string, query?: Query): string {
  const qs = Object.entries(query ?? {})
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join('&');
  return `${API_BASE}${path}${qs ? `?${qs}` : ''}`;
}

interface FetchOptions {
  method?: 'GET' | 'POST' | 'DELETE';
  body?: unknown;
  query?: Query;
  signal?: AbortSignal;
}

export async function fetchJson<T>(path: string, options: FetchOptions = {}): Promise<T> {
  const { method = 'GET', body, query, signal } = options;
  const url = buildUrl(path, query);
  const init: RequestInit = { method, signal };
  if (body !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(body);
  }

  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (err) {
    throw new ApiError(err instanceof Error ? err.message : 'network request failed', 0, url);
  }

  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      if (!response.ok) throw new ApiError(text.slice(0, 400), response.status, url);
      throw new ApiError('response was not JSON', response.status, url);
    }
  }

  if (!response.ok) {
    const detail =
      payload && typeof payload === 'object' && 'detail' in payload
        ? String((payload as { detail: unknown }).detail)
        : `request failed with status ${response.status}`;
    throw new ApiError(detail, response.status, url);
  }
  return payload as T;
}

// ---------------------------------------------------------------- health, data

export const getHealth = () => fetchJson<Health>('/api/health');
export const getDataDirs = () => fetchJson<DataDir[]>('/api/data');

// ---------------------------------------------------------------------- models

export const getModels = () => fetchJson<ModelSummary[]>('/api/models');

export const getModel = (modelId: string) => fetchJson<ModelDetail>(`/api/models/${encodeURIComponent(modelId)}`);

export const getModelLog = (modelId: string, limit = 200) =>
  fetchJson<TrainLogRecord[]>(`/api/models/${encodeURIComponent(modelId)}/log`, { query: { limit } });

export const calibrateModel = (modelId: string, body: CalibrateRequest = {}) =>
  fetchJson<CalibrationSummary>(`/api/models/${encodeURIComponent(modelId)}/calibrate`, { method: 'POST', body });

// -------------------------------------------------------------------- sessions

export const getSessions = () => fetchJson<SessionSummary[]>('/api/sessions');

export const createSession = (body: CreateSessionRequest) =>
  fetchJson<SessionSummary>('/api/sessions', { method: 'POST', body });

export const getSession = (sessionId: string) =>
  fetchJson<SessionDetail>(`/api/sessions/${encodeURIComponent(sessionId)}`);

export const getSessionTransactions = (sessionId: string, limit = 100, offset = 0) =>
  fetchJson<TransactionPage>(`/api/sessions/${encodeURIComponent(sessionId)}/transactions`, {
    query: { limit, offset },
  });

export const getSessionState = (sessionId: string) =>
  fetchJson<SessionState>(`/api/sessions/${encodeURIComponent(sessionId)}/state`);

export const chat = (sessionId: string, body: ChatRequest) =>
  fetchJson<ChatResult>(`/api/sessions/${encodeURIComponent(sessionId)}/chat`, { method: 'POST', body });

// The plan names this route /physics; spec section 10 calls it /episode.
// The plan is the stated source of truth for both agents.
export const runPhysics = (sessionId: string, body: PhysicsRequest) =>
  fetchJson<EpisodeResult>(`/api/sessions/${encodeURIComponent(sessionId)}/physics`, { method: 'POST', body });

export const forkSession = (sessionId: string, childSessionId?: string) =>
  fetchJson<SessionSummary>(`/api/sessions/${encodeURIComponent(sessionId)}/fork`, {
    method: 'POST',
    body: { child_session_id: childSessionId },
  });

export const resetSession = (sessionId: string) =>
  fetchJson<SessionSummary>(`/api/sessions/${encodeURIComponent(sessionId)}/reset`, { method: 'POST' });

export const resumeSession = (sessionId: string) =>
  fetchJson<SessionSummary>(`/api/sessions/${encodeURIComponent(sessionId)}/resume`, { method: 'POST' });

export const deleteSession = (sessionId: string) =>
  fetchJson<{ deleted: boolean }>(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });

// --------------------------------------------------------------- training jobs

export const getTrainJobs = () => fetchJson<TrainJob[]>('/api/train/jobs');

export const startTraining = (body: TrainRequest) => fetchJson<TrainStarted>('/api/train', { method: 'POST', body });

export const getTrainStatus = (modelId: string) =>
  fetchJson<TrainStatus>(`/api/train/${encodeURIComponent(modelId)}`);

export const cancelTraining = (modelId: string) =>
  fetchJson<{ model_id: string; status: string }>(`/api/train/${encodeURIComponent(modelId)}/cancel`, {
    method: 'POST',
  });

// ------------------------------------------------------------- red team, sleep

export const getRedteamRuns = () => fetchJson<RedteamSummary[]>('/api/redteam');

export const runRedteam = (body: RedteamRequest) => fetchJson<RedteamSummary>('/api/redteam', { method: 'POST', body });

export const getRedteamRun = (runId: string) =>
  fetchJson<RedteamDetail>(`/api/redteam/${encodeURIComponent(runId)}`);

export interface SleepRequest {
  model_id: string;
  sessions?: string[];
  core_data_dir?: string;
  steps?: number;
  lr?: number;
  core_ratio?: number;
  seq_len?: number;
}

export const runSleep = (body: SleepRequest) => fetchJson<SleepManifest>('/api/sleep', { method: 'POST', body });
