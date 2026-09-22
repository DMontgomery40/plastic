// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';

import { ChatTab } from './ChatTab';
import { initialState, useStore } from '../../store';
import type { RunnerSummary, SessionDetail, SessionSummary } from '../../api/types';

afterEach(() => {
  cleanup();
  useStore.setState({ ...initialState });
});

// ASTRA-102 #1: the RENDERED Chat prompt subtitle must reflect the EFFECTIVE generation-write policy
// carried in the session summary (writes_generation), not the raw learn_from_generation flag -- a
// helper test alone copied the wrong assumption. Rendered here with a store-seeded native session.

function seed(writesGeneration: boolean | undefined, readOnly = false) {
  const summary = {
    pos: 10, pending: 0, budget_used: 0, budget_session: null, read_only: readOnly, read_only_reason: null,
    n_transactions: 3, cusum: { k: 0.5, h: 5, s_hi: 0, s_lo: 0, alarms: 0 },
    state_norms: { recurrent_norm_total: 5 }, drift_from_anchor: 0, backend: 'qwen',
    calibration: 'absent', writes_generation: writesGeneration,
  } as unknown as RunnerSummary;
  const detail = {
    meta: { domain: 'text', harness: { learn_from_generation: false } } as unknown as SessionDetail['meta'],
    summary, calibration: null, lineage: [], transactions: [], trace: [],
  } as unknown as SessionDetail;
  useStore.setState({
    ...initialState,
    sessions: [{ session_id: 's1', domain: 'text' } as unknown as SessionSummary],
    currentSessionId: 's1',
    sessionDetail: detail,
    chatResult: null,
  });
}

describe('ChatTab generation-write subtitle', () => {
  it('a native write-eligible session reads write-eligible, not "not learned"', () => {
    seed(true);
    render(<ChatTab />);
    expect(screen.getByText(/Memory writes enabled for prompts and generated tokens/i)).toBeTruthy();
    expect(screen.queryByText(/not learned/i)).toBeNull();
    expect(screen.queryByText(/read-only by default/i)).toBeNull();
  });

  it('a session whose generation is not write-eligible reads read-only, not learned', () => {
    seed(false);
    render(<ChatTab />);
    expect(screen.getByText(/generated tokens are read-only/i)).toBeTruthy();
  });

  it('an unknown effective policy stays unavailable, not a false claim', () => {
    seed(undefined);
    render(<ChatTab />);
    expect(screen.getByText(/Write policy unavailable/i)).toBeTruthy();
    expect(screen.queryByText(/write-eligible/i)).toBeNull();
  });

  it('a known read-only latch wins over an unknown capability (ASTRA-104)', () => {
    seed(undefined, true);
    render(<ChatTab />);
    expect(screen.getByText(/Read-only session: no memory writes/i)).toBeTruthy();
    expect(screen.queryByText(/write-eligible/i)).toBeNull();
  });
});
