// ILLUSTRATIVE, NOT MEASURED. A prototype of what a successful consolidation would look like, used only by the
// Anatomy view's "Illustrative" mode. These numbers are invented to show the intended outcome; they never enter the
// exported data, and the view labels every element that shows them.

import type { AnatomyModel, WakeChunk } from './anatomy';

function turn(chunks: number, flaggedAt: number | null, applied: WakeChunk['applied'] = 'commit'): WakeChunk[] {
  return Array.from({ length: chunks }, (_, i) => ({ applied, flagged: i === flaggedAt }));
}

const teachTurns: WakeChunk[][] = Array.from({ length: 24 }, (_, i) => turn(5 + (i % 3), i === 7 || i === 15 ? 1 : null));

export const ILLUSTRATIVE: AnatomyModel = {
  mode: 'illustrative',
  title: 'Illustrative: a successful consolidation',
  method: 'dream',
  target: 'w0',
  wake: {
    sessions: [
      { id: 'teach', forced: false, log_only: false, turns: teachTurns },
      { id: 'rolled', forced: false, log_only: false, turns: [turn(6, 1, 'rollback'), turn(5, 2, 'rollback')] },
    ],
  },
  harvest: { accepted: 24, rolled_back: 2, flagged: 2, flagged_excluded: 2, selected: 22, selected_source: 'illustrative', policy: 'exclude', all_turns: false, state_sessions: ['teach'] },
  consolidate: {
    steps: 20,
    lr: 0.0001,
    losses: [2.8, 2.5, 2.2, 1.95, 1.8, 1.66, 1.55, 1.47, 1.4, 1.35, 1.3, 1.27, 1.24, 1.22, 1.2, 1.19, 1.18, 1.17, 1.16, 1.16],
    loss_terms: 'reply_kl(frozen teacher) + replay_ce, unit weights',
    anchor_lambda: null,
    w0_change_total: 0.004,
    session_rows: 1,
    replay_rows: 1,
    dreams: { generated: 132, kept: 24, removed: { duplicate: 61, low_gain: 47 }, example: 'Your cat is called Marlowe, and the instrument you are learning is the cello.' },
  },
  gate: {
    checks: [
      { name: 'heldout_nll_mean_rise', label: 'held-out NLL rise', value: 0.012, limit: 0.05, passed: true, in_force: true },
      { name: 'reply_cluster_share', label: 'largest identical-reply share', value: 0.04, limit: 0.25, passed: true, in_force: true },
    ],
    nll_before: 1.612,
    nll_after: 1.624,
  },
  outcome: { outcome: 'committed', parent: 'parent', child: 'child' },
  recall: {
    taught_before: { n: 24, recalled: 0, n_paraphrase: 24, recalled_paraphrase: 0 },
    taught_after: { n: 24, recalled: 20, n_paraphrase: 24, recalled_paraphrase: 16 },
    rolled_after: { n: 2, recalled: 0, n_paraphrase: 2, recalled_paraphrase: 0 },
    general_before: { n: 7, recalled: 3, n_paraphrase: 7, recalled_paraphrase: 3 },
    general_after: { n: 7, recalled: 3, n_paraphrase: 7, recalled_paraphrase: 3 },
  },
};
