// The only module that talks HTTP. Errors are surfaced as ApiError with a clean message; an HTML error
// page or a stack trace never reaches the banner raw.

import type {
  CalibrationSummary,
  ChatResult,
  Health,
  ModelSummary,
  Sampling,
  SessionDetail,
  SessionState,
  SessionSummary,
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

export function buildUrl(path: string, query?: Query): string {
  const qs = Object.entries(query ?? {})
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join('&');
  return `${API_BASE}${path}${qs ? `?${qs}` : ''}`;
}

export function cleanErrorMessage(text: string): string {
  const stripped = text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
  return stripped.length > 240 ? `${stripped.slice(0, 237)}...` : stripped || 'request failed';
}

export async function fetchJson<T>(path: string, options: { method?: 'GET' | 'POST' | 'DELETE'; body?: unknown; query?: Query } = {}): Promise<T> {
  const { method = 'GET', body, query } = options;
  const url = buildUrl(path, query);
  const init: RequestInit = { method };
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
      if (!response.ok) throw new ApiError(cleanErrorMessage(text), response.status, url);
      throw new ApiError('response was not JSON', response.status, url);
    }
  }
  if (!response.ok) {
    const detail = payload && typeof payload === 'object' && 'detail' in payload
      ? cleanErrorMessage(String((payload as { detail: unknown }).detail))
      : `request failed with status ${response.status}`;
    throw new ApiError(detail, response.status, url);
  }
  return payload as T;
}

export const getHealth = () => fetchJson<Health>('/api/health');
export const getModels = () => fetchJson<ModelSummary[]>('/api/models');
export const calibrateModel = (modelId: string) =>
  fetchJson<CalibrationSummary>(`/api/models/${encodeURIComponent(modelId)}/calibrate`, { method: 'POST', body: {} });

export const getSessions = () => fetchJson<SessionSummary[]>('/api/sessions');
export const createSession = (body: { model_id: string; session_id?: string; harness?: Record<string, unknown> }) =>
  fetchJson<SessionSummary>('/api/sessions', { method: 'POST', body });
export const getSession = (sessionId: string) => fetchJson<SessionDetail>(`/api/sessions/${encodeURIComponent(sessionId)}`);
export const getTransactions = (sessionId: string, limit = 400, offset = 0) =>
  fetchJson<TransactionPage>(`/api/sessions/${encodeURIComponent(sessionId)}/transactions`, { query: { limit, offset } });
export const getSessionState = (sessionId: string) => fetchJson<SessionState>(`/api/sessions/${encodeURIComponent(sessionId)}/state`);
export const chat = (sessionId: string, prompt: string, sampling: Sampling) =>
  fetchJson<ChatResult>(`/api/sessions/${encodeURIComponent(sessionId)}/chat`, {
    method: 'POST',
    body: { prompt, max_new_tokens: sampling.max_new_tokens, temperature: sampling.temperature, top_k: sampling.top_k, seed: sampling.seed },
  });
export const forkSession = (sessionId: string) =>
  fetchJson<SessionSummary>(`/api/sessions/${encodeURIComponent(sessionId)}/fork`, { method: 'POST', body: {} });
export const resetSession = (sessionId: string) => fetchJson<unknown>(`/api/sessions/${encodeURIComponent(sessionId)}/reset`, { method: 'POST' });
export const resumeSession = (sessionId: string) => fetchJson<unknown>(`/api/sessions/${encodeURIComponent(sessionId)}/resume`, { method: 'POST' });
export const deleteSession = (sessionId: string) => fetchJson<unknown>(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
