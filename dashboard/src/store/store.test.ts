import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hasRunningJob, initialState, isPolling, mergeSessionSummary, startJobPolling, stopJobPolling, useStore } from './index';
import {
  HARNESS_OVERRIDES,
  applyOverride,
  boolOverrideState,
  boolOverrideValue,
  buildLineageForest,
} from '../components/tabs/SessionsTab';
import { canaryPanelState } from '../components/tabs/SessionTab';
import { maxUnprotectedDamage } from '../components/tabs/RedTeamTab';
import { bestHeldoutLoss } from '../components/tabs/TrainTab';
import {
  countsFromSession,
  eligibleRejectionRate,
  interventionRate,
  rateSummaryText,
} from '../components/panels/RatePanel';
import {
  NO_VALID_PAYLOADS,
  UNAVAILABLE,
  UNBOUNDED,
  cleanErrorMessage,
  fmt,
  fmtThreshold,
  fmtValidOnly,
  fmtValidOnlyPercent,
} from '../utils/formatting';
import type {
  ChunkSignals,
  HarnessConfig,
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
    // log_write_norm is unbounded: a null threshold, which is not zero
    thresholds: { chunk_loss: 4.8, surprise_mean: 1.2, log_delta_norm: -1.4, log_write_norm: null, canary_delta_coherence: 0.05 },
    achievable_fpr: { chunk_loss: 0.004, surprise_mean: 0.004, log_delta_norm: 0.004, log_write_norm: null, canary_delta_coherence: 0.004 },
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

const ACCEPTED: TransactionRecord['accepted'] = {
  delta_norm: 0.31,
  budget_charge: 0.31,
  budget_used: 1.25,
  budget_remaining: 8.75,
  canary_coherence_after: 3.39,
  canary_poison_after: 8.15,
  canary_delta_coherence: -0.01,
  canary_delta_poison: 0.05,
};

const TRANSACTION: TransactionRecord = {
  index: 0,
  t_unix: 1_726_902_400,
  pos_start: 0,
  pos_end: 64,
  decision: { kind: 'commit', reasons: [], scale: 1 },
  requested: { kind: 'commit', reasons: [], scale: 1 },
  signals: SIGNALS,
  accepted: ACCEPTED,
  read_only: false,
  read_only_reason: null,
  seconds: 0.08,
};

/** A rolled-back chunk: a large proposal, nothing accepted. */
const ROLLED_BACK: TransactionRecord = {
  ...({} as TransactionRecord),
  index: 1,
  t_unix: 1_726_902_500,
  pos_start: 64,
  pos_end: 128,
  decision: { kind: 'rollback', reasons: ['budget_chunk(2.9>1.0)'], scale: 1 },
  requested: { kind: 'scale', reasons: ['z_chunk_loss(3.4)'], scale: 0.25 },
  signals: { ...SIGNALS, pos_start: 64, pos_end: 128, delta_norm: 2.9 },
  accepted: { delta_norm: 0, budget_charge: 0, budget_used: 1.25, budget_remaining: 8.75 },
  read_only: false,
  read_only_reason: null,
  seconds: 0.09,
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
  calibration: null,
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
  it('clears the previous completion and selected chunk after a reset', async () => {
    mockRoutes({ ...FULL_ROUTES, '/api/sessions/s1/reset': SESSION_A });
    useStore.setState({ currentSessionId: 's1', chatResult: { completion: 'old turn' } as never, selectedTransaction: 7 });
    await useStore.getState().resetSession('s1');
    expect(useStore.getState().chatResult).toBeNull();
    expect(useStore.getState().selectedTransaction).toBeNull();
    expect(useStore.getState().sessionDetail?.meta.session_id).toBe('s1');
  });
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

describe('proposed against accepted evidence', () => {
  it('keeps the two metric sets apart on a rolled-back chunk', async () => {
    mockRoutes({ ...FULL_ROUTES, '/api/sessions/s1': { ...SESSION_DETAIL, transactions: [TRANSACTION, ROLLED_BACK] } });
    await useStore.getState().loadSession('s1');
    const txs = useStore.getState().sessionDetail?.transactions ?? [];
    const rolled = txs[1];
    // the proposal was large; nothing was learned
    expect(rolled.signals.delta_norm).toBe(2.9);
    expect(rolled.accepted.delta_norm).toBe(0);
    expect(rolled.accepted.budget_charge).toBe(0);
    // and the accepted value is never the proposed one scaled
    expect(rolled.accepted.delta_norm).not.toBeCloseTo(rolled.signals.delta_norm * rolled.requested.scale);
  });

  it('carries the requested decision separately from the applied one', async () => {
    mockRoutes({ ...FULL_ROUTES, '/api/sessions/s1': { ...SESSION_DETAIL, transactions: [TRANSACTION, ROLLED_BACK] } });
    await useStore.getState().loadSession('s1');
    const txs = useStore.getState().sessionDetail?.transactions ?? [];
    expect(txs[1].requested.kind).toBe('scale');
    expect(txs[1].decision.kind).toBe('rollback');
    expect(txs[1].requested.reasons).not.toEqual(txs[1].decision.reasons);
    expect(txs[0].requested.kind).toBe(txs[0].decision.kind);
  });
});

describe('missing values never read as measurements', () => {
  it('renders an absent number as unavailable, not zero', () => {
    expect(fmt(null)).toBe(UNAVAILABLE);
    expect(fmt(undefined)).toBe(UNAVAILABLE);
    expect(fmt(Number.NaN)).toBe(UNAVAILABLE);
    expect(fmt(0)).toBe('0.0000');
  });

  it('separates an unbounded threshold from an unknown one', () => {
    expect(fmtThreshold(null)).toBe(UNBOUNDED);
    expect(fmtThreshold(undefined)).toBe(UNAVAILABLE);
    expect(fmtThreshold(1.25, 2)).toBe('1.25');
  });

  it('carries a null threshold and a null achievable rate through the store', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().loadModel('lm_1');
    const cal = useStore.getState().modelDetail?.calibration;
    expect(cal?.thresholds.log_write_norm).toBeNull();
    expect(cal?.achievable_fpr.log_write_norm).toBeNull();
    expect(cal?.achievable_fpr.chunk_loss).toBe(0.004);
    // the achievable rate is the honest one and differs from the request
    expect(cal?.target_fpr).toBe(0.01);
  });
});

describe('observed rejection rate', () => {
  it('counts rollback, scale and project as interventions but never readonly', () => {
    // readonly chunks committed as observations; they must not inflate the rate
    const counts = { n_transactions: 10, commits: 5, rollbacks: 1, scales: 2, projects: 1, readonly: 1 };
    expect(interventionRate(counts)).toBeCloseTo(0.4); // 4 of 10 chunks
    expect(eligibleRejectionRate(counts)).toBeCloseTo(4 / 9); // 4 of 9 eligible updates
  });

  it('does not read an all-generated session as fully intervened', () => {
    // one committed prompt, four read-only generated chunks
    const counts = { n_transactions: 5, commits: 1, rollbacks: 0, scales: 0, projects: 0, readonly: 4 };
    expect(interventionRate(counts)).toBe(0); // nothing was actually intervened
    expect(eligibleRejectionRate(counts)).toBe(0); // the one eligible update was accepted
    const text = rateSummaryText(counts);
    expect(text).not.toContain('intervened');
    expect(text).not.toContain('100');
    expect(text).toBe('0.0% rejected');
  });

  it('reports the rejection rate as unavailable, never 0%, when nothing proposed a write', () => {
    const counts = { n_transactions: 4, commits: 0, rollbacks: 0, scales: 0, projects: 0, readonly: 4 };
    expect(eligibleRejectionRate(counts)).toBeNull();
    expect(interventionRate(counts)).toBe(0);
    // the compact summary must not label a pure-observation session as intervened
    expect(rateSummaryText(counts)).toBe(UNAVAILABLE);
    expect(rateSummaryText(counts)).not.toContain('intervened');
  });

  it('counts an actual rollback among the eligible chunks', () => {
    const counts = { n_transactions: 8, commits: 5, rollbacks: 2, scales: 0, projects: 0, readonly: 1 };
    expect(eligibleRejectionRate(counts)).toBeCloseTo(2 / 7); // 2 of 7 eligible updates
    expect(interventionRate(counts)).toBeCloseTo(0.25); // 2 of 8 chunks
  });

  it('handles a mixed session with every decision kind', () => {
    const counts = { n_transactions: 10, commits: 4, rollbacks: 1, scales: 2, projects: 1, readonly: 2 };
    expect(eligibleRejectionRate(counts)).toBeCloseTo(0.5); // 4 of 8 eligible updates
    expect(interventionRate(counts)).toBeCloseTo(0.4); // 4 of 10 chunks
  });

  it('is unavailable rather than zero when nothing has run', () => {
    const empty = { n_transactions: 0, commits: 0, rollbacks: 0, scales: 0, projects: 0, readonly: 0 };
    expect(interventionRate(empty)).toBeNull();
    expect(eligibleRejectionRate(empty)).toBeNull();
  });

  it('reads the counts off a session summary', () => {
    expect(countsFromSession(SESSION_A)).toEqual({
      n_transactions: 8,
      commits: 6,
      rollbacks: 1,
      scales: 1,
      projects: 0,
      readonly: 0,
    });
    // no readonly here, so both denominators agree: 2 of 8
    expect(interventionRate(countsFromSession(SESSION_A))).toBeCloseTo(0.25);
    expect(eligibleRejectionRate(countsFromSession(SESSION_A))).toBeCloseTo(0.25);
  });
});

describe('error messages', () => {
  it('strips markup from an HTML error page', () => {
    const html = '<html><head><style>b{}</style></head><body><h1>502 Bad Gateway</h1><p>nginx</p></body></html>';
    expect(cleanErrorMessage(html)).toBe('502 Bad Gateway nginx');
  });

  it('keeps only the message from a Python traceback', () => {
    const raw = 'KeyError: model_id\nTraceback (most recent call last):\n  File "x.py", line 3, in f\n    boom()';
    expect(cleanErrorMessage(raw)).toBe('KeyError: model_id');
  });

  it('bounds the length and never returns an empty banner', () => {
    expect(cleanErrorMessage('x'.repeat(500)).length).toBeLessThanOrEqual(240);
    expect(cleanErrorMessage('   ')).toContain('no readable message');
  });

  it('sanitizes what reaches the store from a failing route', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 502,
        text: async () => '<html><body><h1>502 Bad Gateway</h1></body></html>',
      }) as Response),
    );
    await useStore.getState().refreshModels();
    const message = useStore.getState().error ?? '';
    expect(message).toContain('502 Bad Gateway');
    expect(message).not.toContain('<');
  });
});

describe('red team valid-only aggregates', () => {
  // A family where nothing met the plausibility constraint: the valid-only
  // fields are null, and null here means "nothing qualified", not "no damage".
  const NO_VALID = {
    n: 2,
    n_valid: 0,
    damage_mean: 0.0121,
    damage_max: 0.0257,
    unprotected_damage_mean: 0.1536,
    frozen_damage_mean: 0.0121,
    valid_damage_mean: null,
    valid_damage_max: null,
    provisional_damage_max: 0.0100,
    gated_fraction: 1,
    constraint_violated_fraction: 1,
    over_threshold_fraction: 0.5,
    valid_over_threshold_fraction: null,
  };

  it('distinguishes no qualifying payload from an unreported number', () => {
    expect(fmtValidOnly(NO_VALID.valid_damage_mean)).toBe(NO_VALID_PAYLOADS);
    expect(fmtValidOnlyPercent(NO_VALID.valid_over_threshold_fraction)).toBe(NO_VALID_PAYLOADS);
    expect(NO_VALID_PAYLOADS).not.toBe(UNAVAILABLE);
  });

  it('never renders a null valid-only aggregate as zero', () => {
    expect(fmtValidOnly(null)).not.toContain('0');
    expect(fmtValidOnly(0)).toBe('0.0000');
    expect(fmtValidOnlyPercent(0)).toBe('0%');
  });

  it('keeps the all-attempt numbers when the valid-only ones are absent', () => {
    expect(fmt(NO_VALID.damage_mean)).toBe('0.0121');
    expect(fmt(NO_VALID.unprotected_damage_mean)).toBe('0.1536');
  });

  it('measures what the harness prevented as unprotected minus accepted', () => {
    const prevented = NO_VALID.unprotected_damage_mean - NO_VALID.damage_mean;
    expect(prevented).toBeCloseTo(0.1415, 4);
  });

  it('carries the family stats through the store unchanged', async () => {
    const run = {
      run_id: 'rt_lm_1_1',
      model_id: 'lm_1',
      created_at_unix: 1_726_905_000,
      threshold_coherence: 0.0142,
      families: { repeat: NO_VALID },
    };
    mockRoutes({ ...FULL_ROUTES, '/api/redteam': [run], '/api/redteam/rt_lm_1_1': { summary: run, results: [] } });
    await useStore.getState().refreshRedteam();
    await useStore.getState().loadRedteamRun('rt_lm_1_1');
    const fam = useStore.getState().redteamDetail?.summary.families.repeat;
    expect(fam?.valid_damage_mean).toBeNull();
    expect(fam?.n_valid).toBe(0);
    expect(fam?.unprotected_damage_mean).toBe(0.1536);
    expect(fam?.frozen_damage_mean).toBe(0.0121);
  });

  it('takes the worst undefended attack as a per-attack max, not a max of family means', () => {
    // an individual unprotected attack of +0.078 must outrank a family mean of -0.14
    const results = [{ damage_unprotected: 0.078 }, { damage_unprotected: -0.1397 }, { damage_unprotected: 0.031 }];
    expect(maxUnprotectedDamage(results)).toBeCloseTo(0.078, 4);
  });

  it('reports no undefended worst case as null, never zero, when nothing carried a finite value', () => {
    expect(maxUnprotectedDamage([])).toBeNull();
    expect(maxUnprotectedDamage([{ damage_unprotected: null }])).toBeNull();
  });
});

describe('harness boolean overrides are tri-state', () => {
  it('keeps an explicit off as false instead of dropping it to inherit', () => {
    expect(applyOverride({}, 'enable_rollback', false)).toEqual({ enable_rollback: false });
  });

  it('stores an explicit on as true', () => {
    expect(applyOverride({}, 'enable_rollback', true)).toEqual({ enable_rollback: true });
  });

  it('clears a flag back to the model default when it inherits (null)', () => {
    expect(applyOverride({ enable_rollback: false }, 'enable_rollback', null)).toEqual({});
    expect(applyOverride({ enable_rollback: true }, 'enable_rollback', null)).toEqual({});
  });

  it('renders an inherited true as on (checked), an explicit false as off, an absent key as inherit', () => {
    expect(boolOverrideState(undefined)).toBe('inherit');
    expect(boolOverrideState(true)).toBe('on');
    expect(boolOverrideState(false)).toBe('off');
  });

  it('round-trips a control state back to the value it submits', () => {
    expect(boolOverrideValue('inherit')).toBeNull();
    expect(boolOverrideValue('on')).toBe(true);
    expect(boolOverrideValue('off')).toBe(false);
  });

  it('submits the three states as the right payload: on=true, off=false, inherit=absent', () => {
    let ov: Partial<HarnessConfig> = {};
    ov = applyOverride(ov, 'enable_rollback', boolOverrideValue('off'));
    ov = applyOverride(ov, 'enable_projection', boolOverrideValue('on'));
    ov = applyOverride(ov, 'log_only', boolOverrideValue('inherit'));
    expect(ov).toEqual({ enable_rollback: false, enable_projection: true });
    expect('log_only' in ov).toBe(false);
    // the create-session form sends the object only when at least one key is set
    expect(Object.keys(ov).length > 0).toBe(true);
  });

  it('documents that the rollback flag does not gate statistical or CUSUM rollbacks', () => {
    const flag = HARNESS_OVERRIDES.find((o) => o.key === 'enable_rollback');
    // the label must not read as a switch for all rollbacks
    expect(flag?.label.toLowerCase()).not.toBe('rollback enabled');
    // and the help text must state that the statistical / CUSUM rollbacks stay on
    const hint = flag?.hint?.toLowerCase() ?? '';
    expect(hint).toContain('cusum');
    expect(hint).toContain('statist');
    expect(hint).toMatch(/stay active|remain active|still active/);
  });
});

describe('canary panel empty state', () => {
  it('claims no canary suite only when the model summary says has_canary is false', () => {
    expect(canaryPanelState(false, false)).toBe('no-suite');
  });

  it('says measurements are missing, not the suite, when a calibrated model has no traffic yet', () => {
    // has_canary true, zero recorded measurements: an evidence-availability state
    expect(canaryPanelState(false, true)).toBe('no-measurements');
    // presence unknown (model detail not loaded) must not assert absence either
    expect(canaryPanelState(false, null)).toBe('no-measurements');
    expect(canaryPanelState(false, undefined)).toBe('no-measurements');
  });

  it('shows the populated view whenever measurements exist', () => {
    expect(canaryPanelState(true, true)).toBe('populated');
    expect(canaryPanelState(true, false)).toBe('populated');
    expect(canaryPanelState(true, null)).toBe('populated');
  });
});

describe('best held-out loss is split by domain', () => {
  const textA: ModelSummary = { ...MODEL, model_id: 't1', domain: 'text', eval: { ...MODEL.eval!, heldout_loss: 3.2 } };
  const textB: ModelSummary = { ...MODEL, model_id: 't2', domain: 'text', eval: { ...MODEL.eval!, heldout_loss: 2.9 } };
  const phys: ModelSummary = { ...MODEL, model_id: 'p1', domain: 'physics', eval: { ...MODEL.eval!, heldout_loss: 0.0001 } };

  it('never lets a physics MSE win the text ranking or vice versa', () => {
    expect(bestHeldoutLoss([textA, textB, phys], 'text')).toBe(2.9);
    expect(bestHeldoutLoss([textA, textB, phys], 'physics')).toBe(0.0001);
  });

  it('is null when a domain has no model carrying a finite loss', () => {
    const noEval: ModelSummary = { ...MODEL, model_id: 'p2', domain: 'physics', eval: null };
    expect(bestHeldoutLoss([textA, noEval], 'physics')).toBeNull();
  });
});

describe('health reachability', () => {
  it('drops health to offline when a check fails, then recovers on the next success', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().refreshHealth();
    expect(useStore.getState().health).toEqual(HEALTH);

    mockRoutes(FULL_ROUTES, { '/api/health': 500 });
    await useStore.getState().refreshHealth();
    expect(useStore.getState().health).toBeNull();

    mockRoutes(FULL_ROUTES);
    await useStore.getState().refreshHealth();
    expect(useStore.getState().health).toEqual(HEALTH);
  });

  it('invalidates health when any request hits a network failure', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().refreshHealth();
    expect(useStore.getState().health).toEqual(HEALTH);

    // fetch throwing surfaces as ApiError status 0 ("Failed to fetch")
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      }),
    );
    await useStore.getState().refreshModels();
    expect(useStore.getState().health).toBeNull();
  });

  it('keeps health online when a reachable API answers with an HTTP error', async () => {
    mockRoutes(FULL_ROUTES);
    await useStore.getState().refreshHealth();
    mockRoutes(FULL_ROUTES, { '/api/models': 500 });
    await useStore.getState().refreshModels();
    // the API answered (500), so it is reachable: a health/loading failure is not
    // the same as an empty-but-online collection, and neither ages the pill
    expect(useStore.getState().health).toEqual(HEALTH);
    expect(useStore.getState().error).toContain('HTTP 500');
  });
});

describe('summary tracks the loaded detail', () => {
  const detailAfterTraffic: SessionDetail = {
    ...SESSION_DETAIL,
    meta: { ...SESSION_DETAIL.meta, pos: 640, n_transactions: 10, commits: 8 },
    summary: { ...RUNNER, pos: 640, n_transactions: 10 },
  };

  it('refreshes the affected session summary row after chat traffic', async () => {
    mockRoutes({
      ...FULL_ROUTES,
      '/api/sessions': [SESSION_A, SESSION_B],
      '/api/sessions/s1': detailAfterTraffic,
      '/api/sessions/s1/chat': { prompt: 'hi', completion: 'there', transactions: [], n_tokens_in: 1, n_tokens_out: 1, summary: RUNNER },
    });
    useStore.setState({ currentSessionId: 's1', sessions: [SESSION_A, SESSION_B] });
    await useStore.getState().sendChat({ prompt: 'hi' });
    const s = useStore.getState();
    const row = s.sessions.find((x) => x.session_id === 's1');
    // the row now matches the detail, not the stale pos0/8tx it started with
    expect(row?.pos).toBe(640);
    expect(row?.n_transactions).toBe(10);
    expect(s.sessionDetail?.meta.pos).toBe(640);
    // the sibling row is left untouched
    expect(s.sessions.find((x) => x.session_id === 's1b')?.pos).toBe(512);
  });

  it('refreshes the summary row after physics traffic too', async () => {
    mockRoutes({
      ...FULL_ROUTES,
      '/api/sessions': [SESSION_A],
      '/api/sessions/s1': detailAfterTraffic,
      '/api/sessions/s1/physics': {
        mu: 0.1,
        steps: 4,
        per_step: [],
        transactions: [],
        means: { base_mse: 1, frozen_mse: 1, adaptive_mse: 1 },
        summary: RUNNER,
      },
    });
    useStore.setState({ currentSessionId: 's1', sessions: [SESSION_A] });
    await useStore.getState().runEpisode({ steps: 4 });
    expect(useStore.getState().sessions.find((x) => x.session_id === 's1')?.pos).toBe(640);
  });

  it('never appends a session that the list does not already carry', () => {
    const meta = { ...SESSION_DETAIL.meta, session_id: 'ghost' };
    expect(mergeSessionSummary([SESSION_A], meta)).toEqual([SESSION_A]);
  });
});
