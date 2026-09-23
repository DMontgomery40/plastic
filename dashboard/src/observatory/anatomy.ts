// One normalized description of a sleep, from wake to outcome. The measured builder reads only exported data; the
// illustrative scenario lives in illustrative.ts and is marked as such wherever it renders.

import { isSleep } from './format';
import type { GateCheck, GroupCounts, Run, SleepArm, Trajectory } from './types';

export interface WakeChunk {
  applied: 'commit' | 'rollback' | 'scale' | 'project' | 'readonly' | string;
  flagged: boolean;
}

export interface WakeSession {
  id: string;
  /** rollbacks forced by the experiment's policy rather than detected */
  forced: boolean;
  log_only: boolean;
  turns: WakeChunk[][];
}

export interface AnatomyModel {
  mode: 'measured' | 'illustrative';
  title: string;
  method: string;
  target: string | null;
  wake: { sessions: WakeSession[] } | null;
  harvest: {
    accepted: number;
    rolled_back: number;
    flagged: number;
    flagged_excluded: number;
    selected: number | null;
    selected_source: string;
    policy: string | null;
    all_turns: boolean;
    state_sessions: string[];
  } | null;
  consolidate: {
    steps: number | null;
    lr: number | null;
    losses: number[] | null;
    loss_terms: string | null;
    anchor_lambda: number | null;
    w0_change_total: number | null;
    session_rows: number | null;
    replay_rows: number | null;
    dreams: { generated: number | null; kept: number; removed: Record<string, number>; example: string | null } | null;
  };
  gate: { checks: GateCheck[]; nll_before: number | null; nll_after: number | null };
  outcome: { outcome: 'committed' | 'pulled back' | 'unknown'; parent: string | null; child: string | null };
  recall: { taught_before: GroupCounts | null; taught_after: GroupCounts | null; rolled_after: GroupCounts | null; general_before: GroupCounts | null; general_after: GroupCounts | null };
}

export function wakeFromTrajectory(t: Trajectory): { sessions: WakeSession[] } {
  return {
    sessions: t.sessions.map((s) => {
      const byTurn: WakeChunk[][] = s.turns.length
        ? s.turns.map((turn) => s.chunks.filter((c) => c.i >= turn.tx[0] && c.i < turn.tx[1]).map((c) => ({ applied: c.applied ?? 'commit', flagged: c.flags.length > 0 })))
        : [s.chunks.map((c) => ({ applied: c.applied ?? 'commit', flagged: c.flags.length > 0 }))];
      // the experiment's rolled-back session uses a forced policy: its rollbacks are not detections
      const forced = s.session_id === 'rolled' && s.chunks.every((c) => c.applied === 'rollback');
      return { id: s.session_id, forced, log_only: s.log_only, turns: byTurn };
    }),
  };
}

export function buildAnatomy(run: Run, arm: SleepArm, trajectory: Trajectory | null): AnatomyModel {
  const h = arm.harvest;
  const tbr = h?.turns_by_reason ?? {};
  const cfg = arm.config ?? {};
  const w0 = arm.w0_relative_update?.total ?? null;
  const batch = (arm.batch ?? {}) as Record<string, unknown>;
  const dreams = arm.dreams;
  const floor = run.arms.find((a) => a.arm === 'floor');
  return {
    mode: 'measured',
    title: `${run.id} · ${arm.arm}`,
    method: arm.method,
    target: arm.target,
    wake: trajectory ? wakeFromTrajectory(trajectory) : null,
    harvest: h
      ? {
          accepted: tbr.accepted ?? 0,
          rolled_back: Object.entries(tbr).filter(([k]) => k !== 'accepted').reduce((a, [, v]) => a + (v ?? 0), 0),
          flagged: h.accepted_turns_flagged ?? 0,
          flagged_excluded: h.flagged_excluded ?? 0,
          selected: h.selected_turns.value,
          selected_source: h.selected_turns.source,
          policy: h.flagged_policy,
          all_turns: String(h.provenance ?? '').startsWith('all'),
          state_sessions: arm.method === 'replay' ? [] : h.source_sessions ?? [],
        }
      : null,
    consolidate: {
      steps: typeof cfg.steps === 'number' ? cfg.steps : null,
      lr: typeof cfg.lr === 'number' ? cfg.lr : null,
      losses: arm.losses,
      loss_terms: arm.loss_terms,
      anchor_lambda: arm.method === 'anchor' && typeof cfg.anchor_lambda === 'number' ? cfg.anchor_lambda : null,
      w0_change_total: w0,
      session_rows: typeof batch.session_rows === 'number' ? batch.session_rows : null,
      replay_rows: typeof batch.replay_rows === 'number' ? batch.replay_rows : null,
      dreams: dreams ? { generated: dreams.generated, kept: dreams.kept_count, removed: dreams.rejected_reasons, example: dreams.kept[0]?.text ?? null } : null,
    },
    gate: { checks: arm.gate?.checks ?? [], nll_before: arm.heldout_nll.before?.mean ?? null, nll_after: arm.heldout_nll.after?.mean ?? null },
    outcome: { outcome: arm.lineage.outcome, parent: arm.lineage.parent, child: arm.lineage.child },
    recall: {
      taught_before: arm.recall.before.by_group?.taught ?? (floor && !isSleep(floor) ? floor.recall.after.by_group?.taught ?? null : null),
      taught_after: arm.recall.after.by_group?.taught ?? null,
      rolled_after: arm.recall.after.by_group?.rolled ?? null,
      general_before: arm.recall.before.by_group?.general ?? null,
      general_after: arm.recall.after.by_group?.general ?? null,
    },
  };
}
