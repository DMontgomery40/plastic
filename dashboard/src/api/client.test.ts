import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  ApiError,
  buildUrl,
  calibrateModel,
  chat,
  deleteSession,
  fetchJson,
  forkSession,
  getModelLog,
  getRedteamRun,
  getSessionState,
  getSessionTransactions,
  runPhysics,
  startTraining,
} from './client';

interface Call {
  url: string;
  init: RequestInit | undefined;
}

const calls: Call[] = [];

function mockOnce(body: unknown, status = 200) {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    return {
      ok: status >= 200 && status < 300,
      status,
      text: async () => (body === undefined ? '' : JSON.stringify(body)),
    } as Response;
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => {
  calls.length = 0;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('buildUrl', () => {
  it('returns the bare path when there is no query', () => {
    expect(buildUrl('/api/models')).toBe('/api/models');
  });

  it('appends only the parameters that carry a value', () => {
    expect(buildUrl('/api/x', { limit: 100, offset: 0, cursor: undefined, tag: null, name: '' })).toBe(
      '/api/x?limit=100&offset=0',
    );
  });

  it('encodes keys and values', () => {
    expect(buildUrl('/api/x', { 'a b': 'c/d' })).toBe('/api/x?a%20b=c%2Fd');
  });

  it('keeps false as a real value', () => {
    expect(buildUrl('/api/x', { record: false })).toBe('/api/x?record=false');
  });
});

describe('route URLs', () => {
  it('builds the transactions page URL with limit and offset', async () => {
    mockOnce({ total: 0, items: [] });
    await getSessionTransactions('s1', 50, 100);
    expect(calls[0].url).toBe('/api/sessions/s1/transactions?limit=50&offset=100');
  });

  it('builds the model log URL', async () => {
    mockOnce([]);
    await getModelLog('lm_1', 25);
    expect(calls[0].url).toBe('/api/models/lm_1/log?limit=25');
  });

  it('builds the session state URL', async () => {
    mockOnce({ layers: [], pos: 0 });
    await getSessionState('s1');
    expect(calls[0].url).toBe('/api/sessions/s1/state');
  });

  it('builds the red team detail URL', async () => {
    mockOnce({ summary: {}, results: [] });
    await getRedteamRun('rt_lm_1_123');
    expect(calls[0].url).toBe('/api/redteam/rt_lm_1_123');
  });

  it('percent-encodes ids that contain path characters', async () => {
    mockOnce({ layers: [], pos: 0 });
    await getSessionState('a/b c');
    expect(calls[0].url).toBe('/api/sessions/a%2Fb%20c/state');
  });

  it('uses the plan route name for physics episodes', async () => {
    mockOnce({ mu: 0.1, steps: 4, per_step: [], transactions: [], means: {}, summary: {} });
    await runPhysics('s2', { steps: 4 });
    expect(calls[0].url).toBe('/api/sessions/s2/physics');
    expect(calls[0].init?.method).toBe('POST');
  });
});

describe('request bodies', () => {
  it('posts JSON with a content type', async () => {
    mockOnce({ prompt: 'hi', completion: '', transactions: [], n_tokens_in: 1, n_tokens_out: 0, summary: {} });
    await chat('s1', { prompt: 'hi', max_new_tokens: 8 });
    expect(calls[0].init?.method).toBe('POST');
    expect(calls[0].init?.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(JSON.parse(String(calls[0].init?.body))).toEqual({ prompt: 'hi', max_new_tokens: 8 });
  });

  it('sends an empty object for a calibrate call with no options', async () => {
    mockOnce({ n_chunks: 1, thresholds: {}, canary_baseline: {}, reference_sizes: {}, target_fpr: 0.01, created_at_unix: 1 });
    await calibrateModel('lm_1');
    expect(JSON.parse(String(calls[0].init?.body))).toEqual({});
  });

  it('sends child_session_id on a fork', async () => {
    mockOnce({ session_id: 'child' });
    await forkSession('parent', 'child');
    expect(JSON.parse(String(calls[0].init?.body))).toEqual({ child_session_id: 'child' });
  });

  it('sends the training request through unchanged', async () => {
    mockOnce({ model_id: 'lm_2', pid: 42 });
    await startTraining({ domain: 'physics', steps: 10, batch_size: 2, seq_len: 64 });
    expect(JSON.parse(String(calls[0].init?.body))).toEqual({ domain: 'physics', steps: 10, batch_size: 2, seq_len: 64 });
  });

  it('sends no body on a delete', async () => {
    mockOnce({ deleted: true });
    await deleteSession('s1');
    expect(calls[0].init?.method).toBe('DELETE');
    expect(calls[0].init?.body).toBeUndefined();
  });
});

describe('errors', () => {
  it('raises ApiError carrying the API detail and status', async () => {
    mockOnce({ detail: 'session not found' }, 404);
    await expect(fetchJson('/api/sessions/nope')).rejects.toMatchObject({
      name: 'ApiError',
      message: 'session not found',
      status: 404,
    });
  });

  it('falls back to the status when there is no detail', async () => {
    mockOnce({}, 500);
    await expect(fetchJson('/api/health')).rejects.toMatchObject({ status: 500, message: 'request failed with status 500' });
  });

  it('reports a network failure as status 0', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      }),
    );
    const err = await fetchJson('/api/health').catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(0);
  });

  it('accepts an empty response body', async () => {
    mockOnce(undefined, 200);
    await expect(fetchJson('/api/health')).resolves.toBeNull();
  });
});
