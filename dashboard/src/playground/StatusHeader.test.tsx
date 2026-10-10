// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { StatusHeader } from './StatusHeader';
import { WhyDrawer } from './WhyDrawer';
import { useStore } from './store';
import type { RunnerSummary, SessionDetail, TransactionRecord } from './types';

function tx(index: number, kind: TransactionRecord['decision']['kind'], reasons: string[] = [], proposed = 1, accepted = 1, readOnly: string | null = null): TransactionRecord {
  const decision = { kind, reasons, scale: 1 };
  return {
    index, t_unix: 0, pos_start: index * 16, pos_end: index * 16 + 16, decision, requested: decision,
    signals: { pos_start: 0, pos_end: 16, n_tokens: 16, chunk_loss: 4.2, delta_norm: proposed, z: {}, cusum_alarm: kind === 'rollback' && reasons.includes('cusum_alarm'), budget_used: 0, budget_remaining: null, log_delta_norm: 0 },
    accepted: { delta_norm: accepted, budget_charge: accepted, budget_used: accepted, budget_remaining: null },
    read_only: readOnly !== null, read_only_reason: readOnly, seconds: 0.1,
  } as TransactionRecord;
}

function detailOf(summary: Partial<RunnerSummary>, logOnly = false): SessionDetail {
  return {
    meta: { session_id: 's1', model_id: 'm', domain: 'text', harness: { log_only: logOnly } },
    summary: { pos: 0, pending: 0, budget_used: 0, budget_session: null, read_only: false, read_only_reason: null, n_transactions: 0, cusum: { k: 0.5, h: 5, s_hi: 0.4, s_lo: 0, alarms: 0 }, state_norms: {}, drift_from_anchor: 0, ...summary },
    calibration: null, lineage: [], transactions: [], trace: [],
  } as unknown as SessionDetail;
}

const initial = useStore.getState();
const caps = { create_session: true, fork: true, reset: true, delete: true, resume: true, calibrate: false, sleep: false };
beforeEach(() => {
  useStore.setState({ ...initial, health: { ok: true, artifacts_root: '', device: 'cpu', n_models: 1, n_sessions: 1, public: false, capabilities: caps } }, true);
});
afterEach(() => { cleanup(); useStore.setState(initial, true); vi.restoreAllMocks(); });

describe('status header', () => {
  it('renders nothing without a session', () => {
    const { container } = render(<StatusHeader />);
    expect(container.firstChild).toBeNull();
  });

  it('calm session: OK, last chunk, both gauges, and a timeline', () => {
    useStore.setState({ detail: detailOf({ budget_session: 10, budget_used: 2 }), transactions: [tx(0, 'commit'), tx(1, 'commit')] });
    render(<StatusHeader />);
    expect(screen.getByText('OK')).toBeTruthy();
    expect(screen.getByRole('status').textContent).toContain('Last chunk committed in full');
    expect(screen.getByRole('meter', { name: 'Update-size alarm' }).getAttribute('aria-valuetext')).toContain('0.40 of 5.0');
    expect(screen.getByRole('meter', { name: 'Session budget' }).getAttribute('aria-valuenow')).toBe('20');
    expect(screen.getByRole('list', { name: /Decision timeline/ })).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('locked session: banner with the reason, since-chunk, counter, and a Resume that calls the store', () => {
    const resume = vi.fn(async () => {});
    useStore.setState({
      resumeSession: resume as never,
      detail: detailOf({ read_only: true, read_only_reason: 'cusum_alarm', cusum: { k: 0.5, h: 5, s_hi: 0, s_lo: 0, alarms: 1 } }),
      transactions: [tx(0, 'commit'), tx(1, 'rollback', ['cusum_alarm'], 0.9, 0, 'cusum_alarm'), tx(2, 'readonly', ['session_read_only'], 0, 0, 'cusum_alarm'), tx(3, 'readonly', ['session_read_only'], 0, 0, 'cusum_alarm')],
    });
    render(<StatusHeader />);
    expect(screen.getByText('Locked')).toBeTruthy();
    const alert = screen.getByRole('alert');
    expect(within(alert).getByText('Update-size alarm')).toBeTruthy();
    expect(within(alert).getByText('chunk 1')).toBeTruthy();
    expect(within(alert).getByText(/2 chunks · 32 tokens/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Resume learning' }));
    expect(resume).toHaveBeenCalledWith('s1');
  });

  it('hides Resume when the server does not allow it', () => {
    useStore.setState({ health: { ...useStore.getState().health!, capabilities: { ...caps, resume: false } }, detail: detailOf({ read_only: true, read_only_reason: 'cusum_alarm' }), transactions: [tx(0, 'readonly', ['session_read_only'], 0, 0, 'cusum_alarm')] });
    render(<StatusHeader />);
    expect(screen.queryByRole('button', { name: 'Resume learning' })).toBeNull();
  });

  it('observational session says Not guarded', () => {
    useStore.setState({ detail: detailOf({}, true), transactions: [tx(0, 'commit')] });
    render(<StatusHeader />);
    expect(screen.getByText('Not guarded')).toBeTruthy();
    expect(screen.getByRole('status').textContent).toContain('every update is committed');
  });

  it('marks the reading as possibly out of date when the connection failed', () => {
    useStore.setState({ stale: true, detail: detailOf({}), transactions: [tx(0, 'commit')] });
    render(<StatusHeader />);
    expect(screen.getByRole('status').textContent).toContain('may be out of date');
  });
});

describe('timeline and why drawer', () => {
  const chunks = [tx(0, 'commit'), tx(1, 'rollback', ['canary_coherence(delta=+0.2000>0.0500)'], 0.8, 0), tx(2, 'scale', ['budget_chunk(delta_norm=0.9>0.15)'], 0.9, 0.35)];

  it('opens the drawer for a chunk from the keyboard and closes it with Escape', () => {
    useStore.setState({ detail: detailOf({}), transactions: chunks });
    render(<><StatusHeader /><WhyDrawer /></>);
    const cell = screen.getByRole('button', { name: /^Chunk 1: Rollback/ });
    fireEvent.click(cell);
    const dialog = screen.getByRole('dialog', { name: 'Why chunk 1' });
    expect(within(dialog).getByText('Canary coherence')).toBeTruthy();
    expect(within(dialog).getByText(/rose by 0\.2/)).toBeTruthy();
    expect(within(dialog).getByText('canary_coherence(delta=+0.2000>0.0500)')).toBeTruthy(); // raw code stays visible
    expect(within(dialog).getByText('What rollback did not undo')).toBeTruthy();
    expect(within(dialog).getByText(/roughly the next 64 tokens/)).toBeTruthy();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('shows accepted versus proposed for a scale, and no rollback note', () => {
    useStore.setState({ detail: detailOf({}), transactions: chunks, selectedChunk: 2 });
    render(<WhyDrawer />);
    const dialog = screen.getByRole('dialog', { name: 'Why chunk 2' });
    expect(within(dialog).getByText(/Kept 39% of the proposed change/)).toBeTruthy();
    expect(within(dialog).queryByText('What rollback did not undo')).toBeNull();
  });

  it('timeline cells are a single tab stop moved with the arrow keys', () => {
    useStore.setState({ detail: detailOf({}), transactions: chunks });
    render(<StatusHeader />);
    const cells = screen.getAllByRole('button', { name: /^Chunk \d/ });
    expect(cells.map((c) => c.getAttribute('tabindex'))).toEqual(['-1', '-1', '0']);
    cells[2].focus();
    fireEvent.keyDown(cells[2], { key: 'ArrowLeft' });
    expect(document.activeElement).toBe(cells[1]);
    fireEvent.keyDown(cells[1], { key: 'Home' });
    expect(document.activeElement).toBe(cells[0]);
  });

  it('says so when the chunk is not loaded', () => {
    useStore.setState({ detail: detailOf({}), transactions: chunks, selectedChunk: 99 });
    render(<WhyDrawer />);
    expect(screen.getByText('Chunk 99 is not in the loaded history.')).toBeTruthy();
  });
});
