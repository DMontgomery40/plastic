import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hasRunningJob, initialState, isPolling, startJobPolling, stopJobPolling, useStore } from './index';
import { buildLineageForest } from '../components/tabs/SessionsTab';
import type {
  ChunkSignals,
  Health,
  ModelDetail,
  ModelSummary,
  RunnerSummary,
  SessionDetail,
  SessionSummary,
  TrainJob,
  TransactionRecord,
} from '../api/types';

// ---------------------------------------------------------------- sample data
// Shapes copied from the contract in
// docs/superpowers/plans/2026-09-22-m5-m6-api-and-dashboard.md.

const HEALTH: Health = { ok: true, artifacts_root: 'artifacts', device: 'cpu', n_models: 2, n_sessions: 2 };

const MODEL: ModelSummary = {
  model_id: 'lm_1',
  domain: 'text',
  status: 'completed',
  params: 6_800_000,
  created_at_unix: 1_726_900_000,
  updated_at_unix: 1_726_900_500,
  steps: 3000,
  tokens: 3_000_000,
  calibrated: true,
  has_canary: true,
  eval: {
    step: 3000,
    heldout_loss: 3.21,
    heldout_loss_beta0: 3.55,
    memory_value: 0.34,
    mqar_accuracy: { '4': 0.91, '8': 0.72 },
    beta_hist: { edges: [0, 0.5, 1], counts: [10, 20], beta_mean: 0.42, beta_std: 0.11, alpha_mean: 0.98 },
  },
};

const PHYSICS_MODEL: ModelSummary = { ...MODEL, model_id: 'phys_1', domain: 'physics', calibrated: false };

const MODEL_DETAIL: ModelDetail = {
  record: MODEL,
  config: {
    domain: 'text',
    d_model: 256,
    n_heads: 4,
    n_layers: 4,
    chunk: 64,
    scan_chunk: 16,
    conv_kernel: 4,
    vocab_size: 8192,
    tie_embeddings: true,
    rule: 'delta',
    memory: 'linear',
    memory_input: 'ssm_out',
    mlp_mult: 4,
    obs_dim: 4,
    act_dim: 2,
    ssm_c: 8,
    beta_bias_init: 0,
    alpha_bias_init: -4,
    lam_init: 2.197,
    chunk_momentum: 0.9,
    chunk_orthogonalize: true,
    chunk_lr: 0.1,
    mem_hidden_mult: 2,
  },
  eval: MODEL.eval ?? null,
  calibration: {
    n_chunks: 256,
    thresholds: { chunk_loss: 4.8, surprise_mean: 1.2, log_delta_norm: -1.4, canary_delta_coherence: 0.05 },
    canary_baseline: { coherence: 3.4, poison: 8.1 },
    reference_sizes: { chunk_loss: 256 },
    target_fpr: 0.01,
    created_at_unix: 1_726_901_000,
  },
  canary: { n_coherence: 16, n_poison: 16 },
  log: [
    { step: 10, loss: 5.4, grad_norm: 0.8 },
    { step: 20, loss: 5.1, grad_norm: 0.7 },
    { step: 20, event: 'eval', heldout_loss: 5.0, memory_value: 0.2 },
  ],
};

const SESSION_A: SessionSummary = {
  session_id: 's1',
  model_id: 'lm_1',
  domain: 'text',
  parent_session_id: null,
  root_session_id: 's1',
  forked_at_pos: null,
  created_at_unix: 1_726_902_000,
  updated_at_unix: 1_726_902_500,
  pos: 512,
  n_transactions: 8,
  commits: 6,
  rollbacks: 1,
  scales: 1,
  projects: 0,
  readonly: 0,
  budget_used: 1.25,
  read_only: false,
  read_only_reason: null,
};

const SESSION_B: SessionSummary = {
  ...SESSION_A,
  session_id: 's1b',
  parent_session_id: 's1',
  root_session_id: 's1',
  forked_at_pos: 256,
  created_at_unix: 1_726_903_000,
};

const SIGNALS: ChunkSignals = {
  pos_start: 0,
  pos_end: 64,
  n_tokens: 64,
  chunk_loss: 3.9,
  surprise_mean: 0.8,
  surprise_max: 2.1,
  beta_mean: 0.41,
  alpha_mean: 0.97,
  write_norm_sum: 2.4,
  delta_norm: 0.31,
  delta_norm_per_layer: [0.1, 0.12, 0.2, 0.09],
  fisher_update: 0.004,
  fisher_drift: 0.02,
  canary_coherence_before: 3.4,
  canary_coherence_after: 3.39,
  canary_poison_before: 8.1,
  canary_poison_after: 8.15,
  canary_delta_coherence: -0.01,
  canary_delta_poison: 0.05,
  canary_alignment: -0.12,
  compression_ratio: 0.62,
  z: { chunk_loss: 0.4, surprise_mean: 0.2, log_delta_norm: 0.1, log_write_norm: null, fisher_update: null },
  cusum_alarm: false,
  budget_used: 1.25,
  budget_remaining: null,
  log_delta_norm: -1.17,
  log_write_norm: 0.87,
};

const TRANSACTION: TransactionRecord = {
  index: 0,
  t_unix: 1_726_902_400,
  pos_start: 0,
  pos_end: 64,
  decision: { kind: 'commit', reasons: [], scale: 1 },
  requested: { kind: 'commit', reasons: [], scale: 1 },
  signals: SIGNALS,
  read_only: false,
  read_only_reason: null,
  seconds: 0.08,
};

const RUNNER: RunnerSummary = {
  pos: 512,
  pending: 12,
  budget_used: 1.25,
  budget_session: 10,
  read_only: false,
  read_only_reason: null,
  n_transactions: 8,
  cusum: { k: 0.5, h: 5, s_hi: 0.2, s_lo: 0, alarms: 0 },
  state_norms: { s_norm: [1, 1.1], h_norm: [0.4, 0.5], s_norm_total: 1.49, h_norm_total: 0.64 },
  drift_from_anchor: 0.9,
};

const SESSION_DETAIL: SessionDetail = {
  meta: { ...SESSION_A, harness: {} as SessionDetail['meta']['harness'], model_signature: 'abc123' },
  summary: RUNNER,
  lineage: ['s1'],
  transactions: [TRANSACTION],
  trace: [],
};

const RUNNING_JOB: TrainJob = { model_id: 'lm_2', pid: 4242, status: 'running', exit_code: null, started_at_unix: 1_726_904_000 };

// ------------------------------------------------------------------ fetch mock

type Routes = Record<string, unknown>;

/** Serve one payload per path; unlisted paths 404 the way the API does. */
function mockRoutes(routes: Routes, failures: Record<string, number> = {}) {
  const seen: string[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const path = url.split('?')[0];
      seen.push(path);
      if (failures[path]) {
        const status = failures[path];
        return { ok: false, status, text: async () => JSON.stringify({ detail: `boom ${status}` }) } as Response;
      }
      if (!(path in routes)) {
        return { ok: false, status: 404, text: async () => JSON.stringify({ detail: `no route ${path}` }) } as Response;
      }
      return { ok: true, status: 200, text: async () => JSON.stringify(routes[path]) } as Response;
    }),
  );
  return seen;
}

const FULL_ROUTES: Routes = {
  '/api/health': HEALTH,
  '/api/models': [MODEL, PHYSICS_MODEL],
  '/api/models/lm_1': MODEL_DETAIL,
  '/api/sessions': [SESSION_A, SESSION_B],
  '/api/sessions/s1': SESSION_DETAIL,
  '/api/sessions/s1/state': { layers: [{ s_norm_per_head: [1, 2], h_norm: 0.4, singular_values: [[1, 0.5]], drift_from_anchor: 0.1 }], pos: 512 },
  '/api/sessions/s1b': { ...SESSION_DETAIL, meta: { ...SESSION_DETAIL.meta, session_id: 's1b' } },
  '/api/sessions/s1b/state': { layers: [], pos: 0 },
  '/api/data': [{ name: 'wikitext', dir: 'artifacts/data/wikitext', corpus: 'wikitext', vocab_size: 8192, splits: { train: 1, validation: 2, test: 3 } }],
  '/api/train/jobs': [],
  '/api/redteam': [],
};

beforeEach(() => {
  stopJobPolling();
  useStore.setState({ ...initialState });
});

afterEach(() => {
  stopJobPolling();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('single fetches', () => {
  it('stores health', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().refreshHealth();
    expect(useStore.getState().health).toEqual(HEALTH);
    expect(useStore.getState().error).toBeNull();
  });

  it('stores models', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().refreshModels();
    expect(useStore.getState().models.map((m) => m.model_id)).toEqual(['lm_1', 'phys_1']);
  });

  it('stores a model detail with its calibrated thresholds', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().loadModel('lm_1');
    expect(useStore.getState().modelDetail?.calibration?.thresholds.chunk_loss).toBe(4.8);
  });

  it('clears the loading flag when a request finishes', async () => {
    mockRoutes(FULL_ROUTES);
    const pending = useStore.getState().refreshModels();
    expect(useStore.getState().loading.models).toBe(true);
    await pending;
    expect(useStore.getState().loading.models).toBe(false);
  });
});

describe('session selection', () => {
  it('selects the first session when none is current', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().refreshSessions();
    expect(useStore.getState().currentSessionId).toBe('s1');
  });

  it('keeps the current session when it is still listed', async () => {
    mockRoutes(FULL_ROUTES);
    useStore.setState({ currentSessionId: 's1b' });
    await useStore.getState().refreshSessions();
    expect(useStore.getState().currentSessionId).toBe('s1b');
  });

  it('falls back to the first session when the current one disappeared', async () => {
    mockRoutes({ ...FULL_ROUTES, '/api/sessions': [SESSION_B] });
    useStore.setState({ currentSessionId: 'gone' });
    await useStore.getState().refreshSessions();
    expect(useStore.getState().currentSessionId).toBe('s1b');
  });

  it('reports no session at all when the store is empty', async () => {
    mockRoutes({ ...FULL_ROUTES, '/api/sessions': [] });
    await useStore.getState().refreshSessions();
    expect(useStore.getState().currentSessionId).toBeNull();
  });

  it('loads detail and per-layer state together', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().loadSession('s1');
    const s = useStore.getState();
    expect(s.sessionDetail?.transactions[0].decision.kind).toBe('commit');
    expect(s.sessionState?.layers).toHaveLength(1);
  });

  it('drops the previous session view when switching', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().loadSession('s1');
    useStore.setState({ chatResult: null, selectedTransaction: 3 });
    useStore.getState().setCurrentSession('s1b');
    expect(useStore.getState().selectedTransaction).toBeNull();
    expect(useStore.getState().sessionDetail).toBeNull();
  });
});

describe('bootstrap', () => {
  it('fills every top-level collection and opens the first session', async () => {
    const seen = mockRoutes(FULL_ROUTES);
    await useStore.getState().bootstrap();
    const s = useStore.getState();
    expect(s.health?.ok).toBe(true);
    expect(s.models).toHaveLength(2);
    expect(s.sessions).toHaveLength(2);
    expect(s.dataDirs).toHaveLength(1);
    expect(s.jobs).toEqual([]);
    expect(s.redteamRuns).toEqual([]);
    expect(s.sessionDetail?.meta.session_id).toBe('s1');
    expect(seen).toContain('/api/sessions/s1');
  });
});

describe('failures', () => {
  it('records the API detail and leaves the previous data alone', async () => {
    mockRoutes(FULL_ROUTES, { '/api/models': 500 });
    useStore.setState({ models: [MODEL] });
    await useStore.getState().refreshModels();
    const s = useStore.getState();
    expect(s.error).toContain('boom 500');
    expect(s.error).toContain('HTTP 500');
    expect(s.models).toHaveLength(1);
    expect(s.loading.models).toBe(false);
  });

  it('clears the error on the next successful request', async () => {
    mockRoutes(FULL_ROUTES, { '/api/models': 500 });
    await useStore.getState().refreshModels();
    expect(useStore.getState().error).not.toBeNull();
    mockRoutes(FULL_ROUTES);
    await useStore.getState().refreshModels();
    expect(useStore.getState().error).toBeNull();
  });

  it('refuses to chat without a session', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().sendChat({ prompt: 'hello' });
    expect(useStore.getState().error).toBe('no session selected');
    expect(useStore.getState().chatResult).toBeNull();
  });

  it('keeps a 400 from the wrong-domain route out of the result', async () => {
    mockRoutes(FULL_ROUTES, { '/api/sessions/s1/physics': 400 });
    useStore.setState({ currentSessionId: 's1' });
    await useStore.getState().runEpisode({ steps: 4 });
    expect(useStore.getState().episodeResult).toBeNull();
    expect(useStore.getState().error).toContain('HTTP 400');
  });
});

describe('mutations', () => {
  it('opens the child session after a fork', async () => {
    mockRoutes({ ...FULL_ROUTES, '/api/sessions/s1/fork': SESSION_B });
    const child = await useStore.getState().forkSession('s1');
    expect(child?.session_id).toBe('s1b');
    expect(useStore.getState().currentSessionId).toBe('s1b');
  });

  it('clears the view when the current session is deleted', async () => {
    mockRoutes({ ...FULL_ROUTES, '/api/sessions/s1': { deleted: true }, '/api/sessions': [SESSION_B] });
    useStore.setState({ currentSessionId: 's1', sessionDetail: SESSION_DETAIL });
    await useStore.getState().deleteSession('s1');
    expect(useStore.getState().sessionDetail).toBeNull();
  });

  it('reloads the model after calibrating it', async () => {
    const seen = mockRoutes({
      ...FULL_ROUTES,
      '/api/models/lm_1/calibrate': MODEL_DETAIL.calibration,
    });
    await useStore.getState().calibrate('lm_1', { chunks: 32 });
    expect(seen).toContain('/api/models/lm_1/calibrate');
    expect(seen).toContain('/api/models/lm_1');
    expect(useStore.getState().modelDetail?.record.model_id).toBe('lm_1');
  });

  it('refreshes jobs and models after starting a run', async () => {
    const seen = mockRoutes({ ...FULL_ROUTES, '/api/train': { model_id: 'lm_2', pid: 7 } });
    await useStore.getState().startTraining({ domain: 'text', steps: 10, batch_size: 2, seq_len: 64 });
    expect(seen).toContain('/api/train/jobs');
    expect(seen).toContain('/api/models');
  });
});

describe('training jobs', () => {
  it('fetches a status for every job it lists', async () => {
    mockRoutes({
      ...FULL_ROUTES,
      '/api/train/jobs': [RUNNING_JOB],
      '/api/train/lm_2': { model_id: 'lm_2', status: 'running', latest: { step: 40, loss: 4.2 }, eval: null },
    });
    await useStore.getState().refreshJobs();
    const s = useStore.getState();
    expect(s.jobs).toHaveLength(1);
    expect(s.trainStatus.lm_2?.latest?.step).toBe(40);
  });

  it('keeps the job list when a status call fails', async () => {
    mockRoutes({ ...FULL_ROUTES, '/api/train/jobs': [RUNNING_JOB] }, { '/api/train/lm_2': 500 });
    await useStore.getState().refreshJobs();
    expect(useStore.getState().jobs).toHaveLength(1);
    expect(useStore.getState().trainStatus.lm_2).toBeUndefined();
  });

  it('leaves a foreground error alone while polling', async () => {
    mockRoutes(FULL_ROUTES, { '/api/models': 500 });
    await useStore.getState().refreshModels();
    const message = useStore.getState().error;
    expect(message).not.toBeNull();

    // the poll loop runs every 2 s and must not wipe what the user is reading
    mockRoutes({ ...FULL_ROUTES, '/api/train/jobs': [RUNNING_JOB], '/api/train/lm_2': { model_id: 'lm_2', status: 'running', latest: null, eval: null } });
    await useStore.getState().refreshJobs(true);
    expect(useStore.getState().error).toBe(message);

    await useStore.getState().refreshJobs();
    expect(useStore.getState().error).toBeNull();
  });

  it('does not report its own failure when polling quietly', async () => {
    mockRoutes(FULL_ROUTES, { '/api/train/jobs': 500 });
    await useStore.getState().refreshJobs(true);
    expect(useStore.getState().error).toBeNull();
    await useStore.getState().refreshJobs();
    expect(useStore.getState().error).toContain('HTTP 500');
  });

  it('detects a running job', () => {
    expect(hasRunningJob([])).toBe(false);
    expect(hasRunningJob([{ ...RUNNING_JOB, status: 'finished', exit_code: 0 }])).toBe(false);
    expect(hasRunningJob([RUNNING_JOB])).toBe(true);
  });

  it('polls every 2 seconds while a job runs and stops when it ends', async () => {
    vi.useFakeTimers();
    const seen = mockRoutes({
      ...FULL_ROUTES,
      '/api/train/jobs': [RUNNING_JOB],
      '/api/train/lm_2': { model_id: 'lm_2', status: 'running', latest: null, eval: null },
    });
    useStore.setState({ jobs: [RUNNING_JOB] });
    startJobPolling();
    expect(isPolling()).toBe(true);
    await vi.advanceTimersByTimeAsync(2000);
    expect(seen.filter((p) => p === '/api/train/jobs')).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(2000);
    expect(seen.filter((p) => p === '/api/train/jobs')).toHaveLength(2);

    useStore.setState({ jobs: [{ ...RUNNING_JOB, status: 'finished', exit_code: 0 }] });
    await vi.advanceTimersByTimeAsync(2000);
    expect(isPolling()).toBe(false);
    expect(seen.filter((p) => p === '/api/train/jobs')).toHaveLength(2);
  });
});

describe('lineage forest', () => {
  it('nests a fork under its parent', () => {
    const forest = buildLineageForest([SESSION_B, SESSION_A]);
    expect(forest).toHaveLength(1);
    expect(forest[0].session.session_id).toBe('s1');
    expect(forest[0].children[0].session.session_id).toBe('s1b');
  });

  it('treats a session whose parent is missing as a root', () => {
    const orphan: SessionSummary = { ...SESSION_B, parent_session_id: 'vanished' };
    const forest = buildLineageForest([orphan]);
    expect(forest).toHaveLength(1);
    expect(forest[0].session.session_id).toBe('s1b');
  });

  it('orders roots and children by creation time', () => {
    const older: SessionSummary = { ...SESSION_A, session_id: 's0', created_at_unix: 1 };
    const forest = buildLineageForest([SESSION_A, older, SESSION_B]);
    expect(forest.map((n) => n.session.session_id)).toEqual(['s0', 's1']);
  });
});
