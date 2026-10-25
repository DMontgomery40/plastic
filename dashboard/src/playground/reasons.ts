// Turn the harness' machine reason strings into short plain sentences.
//
// The harness (plastic/harness/policy.py, transaction.py) writes reasons such as
//   canary_coherence(delta=+0.1234>0.0500)   chunk_loss(5.2>4.9)   surprise_mean_z(6.31>=6.0)   cusum_alarm
//   budget_session(1.2>=1.2)                 budget_retry(scale=0.25,delta=0.04)   final_candidate:canary_poison(...)
// The raw string stays available (the drawer shows it under "raw"); this module only adds a readable
// headline and, where the string carries them, the measured value and the limit.

export interface ReasonInfo {
  /** The raw string exactly as the harness wrote it. */
  raw: string;
  /** Short name of the check, e.g. "Update-size alarm (CUSUM)". */
  title: string;
  /** One sentence: what happened, with the numbers when the raw string carries them. */
  sentence: string;
  /** Observational mode: the policy would have acted, nothing was blocked. */
  would: boolean;
  /** Family used to group and to pick the consequence copy. */
  kind: 'cusum' | 'budget' | 'canary' | 'statistic' | 'projection' | 'state' | 'mode' | 'other';
}

const SIGNAL_NAME: Record<string, string> = {
  chunk_loss: 'chunk loss',
  surprise_mean: 'inner surprise',
  log_delta_norm: 'update size',
  log_write_norm: 'write size',
  fisher_update: 'Fisher-weighted update',
  fisher_drift: 'drift from the anchor (Fisher-weighted)',
};

const num = (s: string | undefined): string => {
  if (s === undefined) return '?';
  const v = Number(s);
  if (!Number.isFinite(v)) return s;
  const a = Math.abs(v);
  return a !== 0 && (a < 0.001 || a >= 10000) ? v.toExponential(2) : String(Number(v.toPrecision(3)));
};

function parse(raw: string): { name: string; args: string } {
  const m = /^([a-zA-Z_]+)(?:\((.*)\))?$/.exec(raw.replace(/^final_candidate:/, ''));
  return { name: m?.[1] ?? raw, args: m?.[2] ?? '' };
}

export function explainReason(rawIn: string): ReasonInfo {
  let raw = rawIn;
  let would = false;
  const w = /^would_(rollback|scale|project):(.*)$/.exec(raw);
  if (w) {
    would = true;
    raw = w[2];
  }
  const afterChange = rawIn.startsWith('final_candidate:') ? ' (checked again after scaling or projection)' : '';
  const { name, args } = parse(raw);
  const make = (kind: ReasonInfo['kind'], title: string, sentence: string): ReasonInfo => ({ raw: rawIn, title, sentence: sentence + afterChange, would, kind });

  if (name === 'cusum_alarm') {
    return make('cusum', 'Update-size alarm (CUSUM)', 'Update sizes have drifted upward across several chunks and crossed the alarm limit.');
  }
  if (name === 'session_read_only') return make('state', 'Session is read-only', 'The session is locked, so this chunk was read without learning.');
  if (name === 'learning_ineligible') return make('state', 'Nothing eligible to learn', 'No token in this chunk was allowed to write to memory, so nothing was proposed.');
  if (name === 'log_only') return make('mode', 'Observational mode', 'The guard records what it would do but never blocks an update.');

  if (name === 'canary_coherence') {
    const m = /delta=([+-]?[\d.eE+-]+)>([\d.eE+-]+)/.exec(args);
    return make('canary', 'Canary coherence', m ? `Loss on the benign canary text rose by ${num(m[1])} (limit ${num(m[2])}); the update made the model worse on text it should keep.` : 'Loss on the benign canary text rose past its limit.');
  }
  if (name === 'canary_poison') {
    const m = /delta=([+-]?[\d.eE+-]+)<([+-]?[\d.eE+-]+)/.exec(args);
    return make('canary', 'Canary poison', m ? `Loss on the poison canary text fell by ${num(String(-Number(m[1])))} (limit ${num(String(-Number(m[2])))}); the update made the model better at material it must not learn.` : 'Loss on the poison canary text fell past its limit.');
  }
  if (name === 'canary_alignment') {
    const m = /^([\d.eE+-]+)>([\d.eE+-]+)/.exec(args);
    return make('projection', 'Update points against the canaries', m ? `The update's direction overlaps the canary-damage direction (cosine ${num(m[1])}, limit ${num(m[2])}); that part was removed.` : 'The update overlaps the canary-damage direction; that part was removed.');
  }
  if (name === 'fisher_drift') {
    const m = /^([\d.eE+-]+)>([\d.eE+-]+)/.exec(args);
    return make('statistic', 'Drift from the anchor', m ? `Total drift from the anchor reached ${num(m[1])}, over the cap of ${num(m[2])}.` : 'Drift from the anchor went over its cap.');
  }

  // statistic thresholds: "<signal>(value>threshold)" and "<signal>_z(z>=limit)"
  const statZ = /^(.*)_z$/.exec(name);
  if (statZ && SIGNAL_NAME[statZ[1]]) {
    const m = /^([\d.eE+-]+)>=([\d.eE+-]+)/.exec(args);
    return make('statistic', `Unusual ${SIGNAL_NAME[statZ[1]]}`, m ? `The ${SIGNAL_NAME[statZ[1]]} is ${num(m[1])} robust standard deviations above this session's own recent history (limit ${num(m[2])}).` : `The ${SIGNAL_NAME[statZ[1]]} is far above this session's recent history.`);
  }
  if (SIGNAL_NAME[name]) {
    const m = /^([\d.eE+-]+)>([\d.eE+-]+)/.exec(args);
    return make('statistic', `High ${SIGNAL_NAME[name]}`, m ? `The ${SIGNAL_NAME[name]} was ${num(m[1])}, above the calibrated limit ${num(m[2])}.` : `The ${SIGNAL_NAME[name]} was above its calibrated limit.`);
  }

  // budgets
  if (name === 'budget_session') {
    const m = /^([\d.eE+-]+)>=([\d.eE+-]+)/.exec(args);
    return make('budget', 'Session budget spent', m ? `The session has used ${num(m[1])} of its ${num(m[2])} budget, so learning stopped.` : 'The session budget is spent, so learning stopped.');
  }
  if (name === 'budget_chunk') {
    const m = /delta_norm=([\d.eE+-]+)>([\d.eE+-]+)/.exec(args);
    return make('budget', 'Per-chunk cap', m ? `The proposed change ${num(m[1])} was over the per-chunk cap ${num(m[2])}, so it was scaled down.` : 'The proposed change was over the per-chunk cap and was scaled down.');
  }
  if (name === 'budget_retry' || name === 'budget_scaled_projection' || name === 'budget_recheck') {
    const m = /scale=([\d.eE+-]+)(?:,delta=([\d.eE+-]+))?/.exec(args);
    return make('budget', 'Budget re-check', m ? `Scaled to ${num(m[1])} of the proposal to fit the remaining budget${m[2] ? `; the change actually kept measured ${num(m[2])}` : ''}.` : 'Rescaled to fit the remaining budget.');
  }
  if (name === 'budget_unsatisfiable' || name === 'budget_unrepresentable') {
    const m = /delta=([\d.eE+-]+)>cap=([\d.eE+-]+)/.exec(args);
    return make('budget', 'No update fits the budget', m ? `Even after scaling, the change (${num(m[1])}) was over the cap (${num(m[2])}), because memory decay still moves the state. The chunk was refused.` : 'No scaled update fit the budget, so the chunk was refused.');
  }
  if (name === 'budget_exhausted') return make('budget', 'Session budget spent', 'Nothing is left in the session budget, so the projected update was refused.');

  // projection and numerical failures
  if (name === 'project_removed') {
    const m = /^([\d.eE+-]+)>([\d.eE+-]+)/.exec(args);
    return make('projection', 'Projection removed too much', m ? `Making the update safe would remove ${num(m[1])} of it (limit ${num(m[2])}), so the chunk was refused instead.` : 'Projection would remove too much of the update, so the chunk was refused.');
  }
  if (name === 'project_unavailable') return make('projection', 'Projection unavailable', 'No canary gradient was available to project against, so the chunk was refused.');
  if (name === 'removed_ratio') return make('projection', 'Share removed', `Projection removed ${num(args)} of the update.`);
  if (name === 'dot') return make('projection', 'Overlap before and after', `Overlap with the canary-damage direction went ${args.replace('->', ' to ')}.`);
  if (name === 'nonfinite_candidate') return make('state', 'Non-finite update', 'The proposed state contained NaN or infinite values, so it was refused.');
  if (name === 'nonfinite_frozen_recompute') return make('state', 'Non-finite state', 'Restoring the state produced NaN or infinite values; the session was locked.');

  return make('other', raw.replace(/_/g, ' '), args ? `${raw.replace(/_/g, ' ').replace(/\(.*$/, '')}: ${args}` : raw.replace(/_/g, ' '));
}

/** The one-line meaning of a read-only latch reason, for the lock banner. */
export function explainLock(reason: string | null | undefined): { title: string; sentence: string; kind: ReasonInfo['kind'] } {
  if (!reason) return { title: 'Locked', sentence: 'The session was set to read-only.', kind: 'state' };
  if (reason === 'cusum_alarm') {
    return { title: 'Update-size alarm', sentence: 'The update-size alarm (CUSUM) fired.', kind: 'cusum' };
  }
  if (reason.startsWith('budget_session')) {
    const e = explainReason(reason);
    return { title: 'Budget spent', sentence: e.sentence, kind: 'budget' };
  }
  const e = explainReason(reason);
  return { title: e.title, sentence: e.sentence, kind: e.kind };
}
