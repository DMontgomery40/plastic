// Labels for the Sleep observatory. Each names a state or a quantity once; research readings live in the docs.

import type { Arm, ArmName, GateCheck, GroupCounts, GroupName, SleepArm } from './types';

export const ARM_LABEL: Record<ArmName, string> = {
  floor: 'Floor',
  ceiling: 'Ceiling',
  anchor: 'Anchor',
  replay: 'Replay',
  distill: 'Distill',
  dream: 'Dream',
  ungated: 'Ungated replay',
};

/** What each arm is, in a few words. */
export const ARM_ROLE: Record<ArmName, string> = {
  floor: 'parent model, fresh session',
  ceiling: 'parent model, facts taught in the same session',
  anchor: 'W0 moved toward the session fast weights',
  replay: 'trained on selected turns plus replay data',
  distill: 'session-state teacher distilled into the reset model',
  dream: 'teacher-generated study items distilled',
  ungated: 'replay on every turn, rolled-back and flagged included',
};

export const GROUP_LABEL: Record<GroupName, string> = {
  taught: 'Taught',
  boundary: 'Boundary',
  rolled: 'Rolled back',
  poison: 'Planted false',
  general: 'General',
};

export const GROUP_ORDER: GroupName[] = ['taught', 'boundary', 'rolled', 'poison', 'general'];

export function isSleep(arm: Arm): arm is SleepArm {
  return arm.kind === 'sleep';
}

/** "3/24", or "n/a" when the group was not probed. */
export function ratio(c: GroupCounts | null | undefined, variant: 'verbatim' | 'paraphrase' = 'verbatim'): string {
  if (!c) return 'n/a';
  const [hit, n] = variant === 'verbatim' ? [c.recalled, c.n] : [c.recalled_paraphrase, c.n_paraphrase];
  return n > 0 ? `${hit}/${n}` : 'n/a';
}

export function hits(c: GroupCounts | null | undefined, variant: 'verbatim' | 'paraphrase' = 'verbatim'): number {
  if (!c) return 0;
  return variant === 'verbatim' ? c.recalled : c.recalled_paraphrase;
}

export function num(v: number | null | undefined, digits = 3): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return 'n/a';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 0.01) return v.toFixed(digits);
  if (a >= 0.0001) return v.toPrecision(2);
  return v.toExponential(1);
}

export function signed(v: number | null | undefined, digits = 3): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return 'n/a';
  return `${v > 0 ? '+' : v < 0 ? '−' : ''}${num(Math.abs(v), digits)}`;
}

export function pct(v: number | null | undefined): string {
  return typeof v === 'number' && Number.isFinite(v) ? `${Math.round(v * 100)}%` : 'n/a';
}

export type CheckState = 'passed' | 'failed' | 'not in force' | 'not measured';

export function checkState(c: GateCheck): CheckState {
  if (!c.in_force) return 'not in force';
  if (c.passed === null || c.value === null) return 'not measured';
  return c.passed ? 'passed' : 'failed';
}

export type Outcome = 'committed' | 'pulled back' | 'control' | 'unknown';

export function outcome(arm: Arm): Outcome {
  if (!isSleep(arm)) return 'control';
  return arm.lineage.outcome;
}

export const OUTCOME_LABEL: Record<Outcome, string> = {
  committed: 'Committed',
  'pulled back': 'Pulled back',
  control: 'Control',
  unknown: 'Unknown',
};

export function stepLabel(step: number | null): string {
  if (step === null) return 'base';
  return step === 250 ? 'final' : `step ${step}`;
}

export function dateLabel(unix: number | null): string {
  if (typeof unix !== 'number') return 'n/a';
  const d = new Date(unix * 1000);
  return `${d.toISOString().slice(0, 10)} ${d.toISOString().slice(11, 16)} UTC`;
}

export function methodLine(arm: SleepArm): string {
  const cfg = arm.config ?? {};
  const target = arm.target === 'w0' ? 'W0' : arm.target === 'all' ? 'all parameters' : arm.target ?? '';
  if (arm.method === 'anchor') return `${target}, λ ${cfg.anchor_lambda ?? 'n/a'}`;
  return `${target} × ${cfg.steps ?? 'n/a'} steps, lr ${cfg.lr ?? 'n/a'}`;
}
