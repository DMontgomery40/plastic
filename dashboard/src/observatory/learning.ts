// The learning-contract export: assets/learning/index.json, a byte-identical mirror of
// docs/research/results/learning-observatory/ written by scripts/export_contract_observatory.py. It loads separately
// from the Sleep export, so either can fail without taking the other down.

import { ObservatoryDataError } from './data';
import { num, signed } from './format';

export const LEARNING_SCHEMA = 'plastic.learning-observatory/1';

export interface Pair {
  adapt: number | null;
  no_adapt: number | null;
  elements: number | null;
}

export interface Speed {
  mean_ratio: number | null;
  steps_to_half: number | null;
  reached_half: boolean | null;
  horizon: number | null;
  episodes: number | null;
}

export interface Before {
  policies: (Pair & { policy: string })[];
  train: Pair & { policies: string[] };
  speed: Speed | null;
}

export interface AdaptationWindow {
  update_period: number | null;
  boundaries_per_episode: number | null;
  checked: boolean;
}

export interface Acceptance {
  accepted_good: number | null;
  n_good: number | null;
  refused_bad: number | null;
  n_bad: number | null;
}

export interface ModeResult {
  mode: string;
  transfer: { policy: string; delta: number | null; delta_no_adapt: number | null }[];
  forgetting_delta: number | null;
  poison_harm_vs_clean: number | null;
  poison_harm_vs_start: number | null;
  correction_residual: number | null;
  correction_residual_vs_start: number | null;
  revert: { ok: boolean | null; gap: number | null; tolerance: number | null };
  acceptance: Acceptance;
  decisions: { stream: string | null; beneficial: boolean | null; accepted: boolean | null }[];
  tokens_consumed: number | null;
  tokens_measured: number | null;
  tokens_measured_without_context: number | null;
  parameters: number | null;
}

interface Source {
  file: string;
  sha256: string;
}

export interface ReportSet {
  id: string;
  kind: 'report';
  tag: string | null;
  checkpoint: { model_id: string | null; digest_prefix: string | null; step: number | null };
  execution_commit: string | null;
  device: string | null;
  seed: number | null;
  contract_version: string | null;
  current: boolean;
  split: { id: string | null; train: string[][] | null; heldout: string[][] | null; bound_to_learner: boolean | null };
  adaptation_window: AdaptationWindow;
  spec: { seq_len: number | null; stream_episodes: number | null; heldout_policies: string[] | null; train_policies: string[] | null };
  stream: { episodes: number | null; tokens: number | null; policy: string | null; worlds_disjoint_from_measurement: boolean | null } | null;
  learner: { lr?: number | null; steps?: number | null } | null;
  no_adapt_label: string;
  before: Before;
  modes: ModeResult[];
  missing_modes: string[];
  sources: Source[];
}

export interface Variant {
  variant: string;
  config: Record<string, unknown> | string | null;
  parameters: number | null;
  size: { d_model: number | null; n_heads: number | null; n_layers: number | null; chunk: number | null };
  train_steps: number | null;
  s_per_step: number | null;
  final_train_loss: number | null;
  eta_per_layer: (number | null)[] | null;
  no_adapt_label: string | null;
  contract_version: string | null;
  current: boolean;
  split_id: string | null;
  adaptation_window: AdaptationWindow;
  before: Before;
}

export interface AblationSet {
  id: string;
  kind: 'ablation';
  tag: string | null;
  execution_commit: string | null;
  contract_version: string | null;
  contract_versions: string[];
  current: boolean;
  variants: Variant[];
  missing: string[];
  sources: Source[];
}

export type ContractSet = ReportSet | AblationSet;

export interface LearningIndex {
  schema: string;
  exporter_version: string;
  current_contract_version: string;
  archive: string;
  archive_digest: string;
  sets: ContractSet[];
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

export function isLearningIndex(v: unknown): v is LearningIndex {
  return isObject(v) && v.schema === LEARNING_SCHEMA && typeof v.current_contract_version === 'string' && Array.isArray(v.sets);
}

let pending: Promise<LearningIndex> | null = null;

export function loadLearning(): Promise<LearningIndex> {
  if (!pending) {
    const root = typeof document !== 'undefined' ? document.baseURI : 'http://localhost/';
    const url = new URL('assets/learning/index.json', root).toString();
    pending = (async () => {
      const res = await fetch(url);
      if (!res.ok) throw new ObservatoryDataError(`learning results: HTTP ${res.status}`);
      const value: unknown = await res.json();
      if (!isLearningIndex(value)) throw new ObservatoryDataError('learning results: unexpected format');
      return value;
    })().catch((err) => {
      pending = null; // a failed fetch can be retried
      throw err;
    });
  }
  return pending;
}

/** Test hook: forget the cached response. */
export function clearLearningCache(): void {
  pending = null;
}

// ---------------------------------------------------------------------------------------------------- labels
export const MODE_LABEL: Record<string, string> = { frozen: 'Frozen', continued: 'Continued training', in_context: 'Everything in context' };

export function modeRole(mode: string, learner: ReportSet['learner']): string {
  if (mode === 'frozen') return 'fast weights only, no lasting update';
  if (mode === 'continued') return `Adam, ${learner?.steps ?? 'n/a'} steps at lr ${learner?.lr ?? 'n/a'}`;
  if (mode === 'in_context') return 'the stream prepended to every measured episode';
  return mode;
}

export const VARIANT_LABEL: Record<string, string> = {
  full: 'Full candidate',
  no_fast: 'No fast updates',
  decay_only: 'Decay only',
  coords_only: 'Coordinates only',
  no_meta: 'No meta-gradient',
  fixed_z: 'Fixed-latent commit',
  delta_baseline: 'Delta-rule baseline',
};

/** A rate with its denominator. A rate over an empty set reads "n/a", never 0. */
export function rateCell(rate: number | null, n: number | null): string {
  const count = typeof n === 'number' ? n : 0;
  if (count <= 0 || typeof rate !== 'number' || !Number.isFinite(rate)) return `n/a (n=${count})`;
  return `${rate.toFixed(2)} (n=${count})`;
}

export function windowLabel(w: AdaptationWindow): string {
  if (!w.checked || w.update_period === null) return 'unchecked: the learner declared no update period';
  const every = w.update_period === 1 ? 'every step' : `every ${w.update_period} steps`;
  return `fast update ${every}, ${w.boundaries_per_episode ?? 'n/a'} boundaries inside each episode`;
}

export function speedLabel(s: Speed | null): string {
  if (!s || s.mean_ratio === null) return 'n/a';
  const half =
    s.reached_half === false
      ? `not below one half within ${s.horizon ?? 'n/a'} steps`
      : s.steps_to_half !== null
        ? `below one half at step ${s.steps_to_half}`
        : 'step to one half not recorded';
  return `mean ratio ${num(s.mean_ratio)}; ${half}`;
}

export function delta(v: number | null): string {
  return signed(v, 3);
}

export function count(v: number | null): string {
  return typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString('en-US') : 'n/a';
}

export type AblationState = { state: 'archived' | 'stale'; set: AblationSet } | { state: 'not archived'; set: null };

/** The coordinate ablation's state. "stale" means measured under a contract version other than the current one. */
export function ablationState(index: LearningIndex, setId: string | null = null): AblationState {
  const sets = index.sets.filter((s): s is AblationSet => s.kind === 'ablation');
  const set = sets.find((s) => s.id === setId) ?? sets[0];
  if (!set || set.variants.length === 0) return { state: 'not archived', set: null };
  return { state: set.current ? 'archived' : 'stale', set };
}

export function reportSets(index: LearningIndex): ReportSet[] {
  return index.sets.filter((s): s is ReportSet => s.kind === 'report');
}
