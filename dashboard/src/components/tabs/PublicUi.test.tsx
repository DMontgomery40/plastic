// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import type { ModelConfig, ModelDetail, ModelSummary, SessionDetail, SessionSummary, TransactionRecord } from '../../api/types';
import { isPublicDemo, publicErrorMessage, visibleTabKeys } from '../../publicMode';
import { initialState, useStore } from '../../store';
import { hasFiniteSeriesData, LineChartPanel } from '../charts/LineChartPanel';
import { TabNav } from '../layout/TabNav';
import { ArchitectureTab, hasPlasticConfig } from './ArchitectureTab';
import { PublicSessionsTab } from './PublicSessionsTab';
import { PublicSessionTab, measuredSignals } from './PublicSessionTab';
import { SessionsTab } from './SessionsTab';
import { ChatTab } from './ChatTab';

const textSession = {
  session_id: 'demo_text', model_id: 'native', domain: 'text', pos: 96,
  n_transactions: 120, commits: 100, rollbacks: 20, updated_at_unix: 1_700_000_000,
} as SessionSummary;
const physicsSession = { ...textSession, session_id: 'demo_physics', domain: 'physics' } as SessionSummary;
const nativeModel = {
  model_id: 'native', backend: 'qwen', domain: 'text', status: 'completed',
  params: 800_000_000,
} as ModelSummary;
const plasticModel = { ...nativeModel, model_id: 'plastic_text', backend: 'plastic' } as ModelSummary;
const plasticConfig = {
  domain: 'text', d_model: 256, n_heads: 4, n_layers: 4, chunk: 64, scan_chunk: 16,
  conv_kernel: 4, vocab_size: 8192, tie_embeddings: true, rule: 'delta',
  memory: 'linear', memory_input: 'ssm_out', mlp_mult: 4, obs_dim: 4, act_dim: 2, ssm_c: 8,
} as ModelConfig;
const nativeDetail = {
  record: nativeModel, config: {}, eval: null, calibration: null, canary: null, log: [],
} as ModelDetail;
const plasticDetail = { ...nativeDetail, record: plasticModel, config: plasticConfig };

const nativeChunk = {
  index: 0, pos_start: 0, pos_end: 64, decision: { kind: 'commit', reasons: [], scale: 1 },
  requested: { kind: 'commit', reasons: [], scale: 1 },
  signals: {
    chunk_loss: 3.5, log_delta_norm: -1.2, delta_norm: 0.3,
    surprise_mean: null, surprise_max: null, beta_mean: null, alpha_mean: null,
    write_norm_sum: null,
  },
  accepted: { delta_norm: 0.3 },
} as unknown as TransactionRecord;
const sessionDetail = {
  meta: { ...textSession, harness: { log_only: true } },
  summary: { pos: 96, n_transactions: 120 },
  transactions: [
    nativeChunk,
    { ...nativeChunk, index: 1, decision: { kind: 'rollback', reasons: [], scale: 1 }, accepted: { delta_norm: 0 } },
  ],
  trace: [],
} as unknown as SessionDetail;

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  });
});

afterEach(() => {
  cleanup();
  delete document.body.dataset.publicDemo;
  useStore.setState({ ...initialState });
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('public route and action boundaries', () => {
  it('exposes only text routes and clamps hidden tabs; local research routes remain', () => {
    document.body.dataset.publicDemo = 'true';
    expect(isPublicDemo()).toBe(true);
    expect(visibleTabKeys()).toEqual(['chat', 'session', 'sessions']);
    useStore.setState({ ...initialState, activeTab: 'chat' });
    render(<TabNav />);
    expect(screen.getByRole('button', { name: /1 Chat/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /2 Measurements/ })).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Physics/ })).toBeNull();
    expect(screen.queryByRole('button', { name: /Train/ })).toBeNull();
    useStore.getState().setActiveTab('physics');
    expect(useStore.getState().activeTab).toBe('chat');
    cleanup();
    delete document.body.dataset.publicDemo;
    useStore.setState({ ...initialState });
    render(<TabNav />);
    expect(screen.getByRole('button', { name: /Physics/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /Architecture/ })).toBeTruthy();
  });

  it('lists text sessions without creation/destructive controls except confirmed reset', () => {
    document.body.dataset.publicDemo = 'true';
    useStore.setState({ ...initialState, sessions: [physicsSession, textSession], currentSessionId: 'demo_text' });
    render(<PublicSessionsTab />);
    expect(screen.getByText('demo_text')).toBeTruthy();
    expect(screen.queryByText('demo_physics')).toBeNull();
    for (const label of ['Create session', 'Fork', 'Delete', 'Resume']) {
      expect(screen.queryByRole('button', { name: label })).toBeNull();
    }
    expect(screen.getByRole('button', { name: 'Reset' })).toBeTruthy();
    useStore.getState().setCurrentSession('demo_physics');
    expect(useStore.getState().currentSessionId).toBe('demo_text');
    cleanup();
    delete document.body.dataset.publicDemo;
    render(<SessionsTab />);
    expect(screen.getByText('New session')).toBeTruthy();
    expect(screen.getAllByRole('button', { name: 'Fork' }).length).toBeGreaterThan(0);
  });

  it('confirms shared reset and keeps context-limit and busy errors actionable', () => {
    document.body.dataset.publicDemo = 'true';
    const resetSession = vi.fn(async () => {});
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    useStore.setState({ ...initialState, sessions: [textSession], resetSession });
    render(<PublicSessionsTab />);
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    expect(resetSession).not.toHaveBeenCalled();
    confirm.mockReturnValue(true);
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    expect(resetSession).toHaveBeenCalledWith('demo_text');
    expect(publicErrorMessage('This shared demo session reached its context limit. (HTTP 409)')).toMatch(/Reset it in Sessions/);
    expect(publicErrorMessage('Another visitor is running the shared CPU demo. (HTTP 503)')).toMatch(/Try again shortly/);
  });
});

describe('backend-specific architecture', () => {
  it('renders the native identity without Plastic equations or invalid numbers', () => {
    useStore.setState({ ...initialState, models: [nativeModel], modelDetail: nativeDetail });
    render(<ArchitectureTab />);
    expect(screen.getByText('qwen')).toBeTruthy();
    expect(screen.getByText('Research notes')).toBeTruthy();
    expect(screen.queryByText(/Memory branch/)).toBeNull();
    expect(screen.queryByText(/Transaction flow/)).toBeNull();
    expect(document.body.textContent).not.toMatch(/undefined|NaN/);
  });

  it('retains Plastic diagrams for complete configs and refuses incomplete ones', () => {
    expect(hasPlasticConfig(plasticConfig)).toBe(true);
    expect(hasPlasticConfig({ ...plasticConfig, n_heads: Number.NaN })).toBe(false);
    expect(hasPlasticConfig({})).toBe(false);
    useStore.setState({ ...initialState, models: [plasticModel], modelDetail: plasticDetail });
    render(<ArchitectureTab />);
    expect(screen.getByText('Transaction flow')).toBeTruthy();
    cleanup();
    useStore.setState({ modelDetail: { ...plasticDetail, config: { domain: 'text' } } });
    render(<ArchitectureTab />);
    expect(screen.queryByText('Transaction flow')).toBeNull();
    expect(document.body.textContent).not.toMatch(/undefined|NaN/);
  });

  it('does not draw a stale Plastic diagram while switching to Qwen', () => {
    useStore.setState({ ...initialState, models: [plasticModel, nativeModel], modelDetail: plasticDetail, loadModel: vi.fn(async () => {}) });
    render(<ArchitectureTab />);
    expect(screen.getByText('Transaction flow')).toBeTruthy();
    fireEvent.change(screen.getByRole('combobox', { name: 'Model' }), { target: { value: 'native' } });
    expect(screen.getByText('Loading model…')).toBeTruthy();
    expect(screen.queryByText('Transaction flow')).toBeNull();
    act(() => useStore.setState({ modelDetail: nativeDetail }));
    expect(screen.getByText('qwen')).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/undefined|NaN/);
  });
});

describe('measured signal rendering', () => {
  it('treats zero as measured and null, NaN and infinity as missing', () => {
    const series = [{ key: 'value', label: 'Value', color: '#58a6ff' }];
    expect(hasFiniteSeriesData([{ value: null }, { value: Number.NaN }, { value: Infinity }], series)).toBe(false);
    expect(hasFiniteSeriesData([{ value: 0 }], series)).toBe(true);
    render(<LineChartPanel data={[{ chunk: 1, value: null }]} xKey="chunk" series={series} ariaLabel="Missing metric" />);
    expect(screen.getByText('No measurements available.')).toBeTruthy();
    expect(screen.queryByRole('img', { name: 'Missing metric' })).toBeNull();
  });

  it('omits absent native surprise/write-rate charts without renaming loss or state change', () => {
    expect(measuredSignals([nativeChunk]).map((s) => s.key)).toEqual(['chunk_loss', 'log_delta_norm']);
    document.body.dataset.publicDemo = 'true';
    useStore.setState({ ...initialState, sessions: [textSession], currentSessionId: 'demo_text', sessionDetail });
    render(<PublicSessionTab />);
    expect(screen.getByText('Chunk loss')).toBeTruthy();
    expect(screen.getByText('Log state change')).toBeTruthy();
    expect(screen.queryByText('Memory surprise')).toBeNull();
    expect(screen.queryByText('Write rate β')).toBeNull();
    expect(screen.getByText('State change · proposed and accepted')).toBeTruthy();
    expect(screen.getByText(/latest 2 of 120 chunks/)).toBeTruthy();
    expect(screen.getByText('100')).toBeTruthy();
    expect(screen.getByText('20')).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/undefined|NaN/);
  });

  it('keeps the public Chat focused on output and links to measurements', () => {
    document.body.dataset.publicDemo = 'true';
    useStore.setState({
      ...initialState, sessions: [textSession], currentSessionId: 'demo_text', sessionDetail,
      chatResult: { completion: 'Hello', n_tokens_in: 3, n_tokens_out: 1, transactions: [nativeChunk], summary: sessionDetail.summary } as never,
    });
    render(<ChatTab />);
    expect(screen.getByText('Hello')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'View measurements' })).toBeTruthy();
    expect(screen.queryByText('Transactions of this turn')).toBeNull();
    expect(screen.queryByText('Runner after the turn')).toBeNull();
  });
});
