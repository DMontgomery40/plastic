# Sleep observatory

The playground's **Sleep** tab shows every archived Sleep run: what was taught, what was
consolidated, how the candidate child was checked, and whether it was kept or pulled back.
It reads static files that ship with the page, so it works even when the model service is
offline. Links to a view are shareable, for example
`#sleep/runs/final_step250_seed0_exclude/dream`.

The measurements and their interpretation are in the
[Sleep method note](2026-09-23-sleep-consolidation.md) and the
[results archive](results/sleep-2026-09-23/README.md). This page describes what the
interface shows and where each number comes from.

## The views

**Runs.** Runs are grouped by checkpoint. For each run:

- **Protocol:** checkpoint digest prefix, the code commit recorded at launch, facts, steps,
  target, learning rate, flagged-turn rule and ceiling mode.
- **Arm matrix:** one row per arm. It shows fresh-session recall per probe group, recall on
  unseen wording, held-out NLL before and after, the largest identical-reply share, and the
  outcome.
- **Arm detail:** lineage from parent to child, or "no child" when the arm was pulled back;
  every damage-gate check as a value against its limit; the turns the arm trained on; kept
  Dream items; every reply, filterable by group.

**Anatomy of a sleep.** One run and arm, in five stages:

1. Wake: every 16-token chunk of the teaching and rolled-back sessions, marked committed
   or rolled back, with advisory intervention flags.
2. Harvest: turns accepted online, rolled back, flagged, and the selected text turns. It
   also names the sessions whose full committed state the method loads.
3. Consolidate: the method, its loss over steps, the W0 change when recorded, and dreams.
4. Damage gate: each check against its limit.
5. Outcome: the child is committed, or pulled back with the parent unchanged. Fresh-session
   recall follows.

The **Illustrative** mode shows the same five stages with invented numbers for a
consolidation that works. Those numbers live in `dashboard/src/observatory/illustrative.ts`,
never in the exported data. Every stage in that mode carries an "Illustrative, not measured"
tag and a separate color.

**Weights and changes.** This view has two scopes:

- **Within one sleep:** a layers-by-chunks map of the proposed fast-weight change during the
  session Sleep read. It shows each tensor (W1, b1, W2, b2) or all four combined in
  quadrature, on a log color scale, with a terrain view as an alternative.
  - Below the map: the arm's loss per step, gradient norms by layer, child-versus-parent W0
    change per tensor, and Dream per-token gains.
- **Across runs:** every consolidation attempt as a dot plot. It shows held-out NLL change
  against its limit and same-reply share against 25%, with recall and outcome. Anchor W0
  changes appear on one shared scale.

## Labels and their meaning

| Label | Meaning |
| --- | --- |
| Damage gate | The Sleep gate, formerly called the locality gate (`gate_from_measurements` reports `kind: "damage gate"`). It checks for damage only: held-out chat NLL, reply collapse, and canary locality when a canary suite exists. It has no retention or contamination check, so passing it is not evidence that anything was learned or that the child stayed clean. Recorded reasons that say "locality gate failed" are shown as "damage gate failed"; the exported text is unchanged. |
| Committed | The damage gate accepted the child and registered it in the run's disposable experiment store. It is not published as a model. |
| Pulled back | A damage-gate check failed. No child was registered, and the parent is unchanged. |
| not in force | The run executed before this check existed. The value shown is rescored from the saved replies with the current rule, and the run's recorded outcome stands. |
| selected / derived | The selected-turn count is recorded by newer reports. For seed 0 it is derived from recorded harvest counts. Older runs show it as not recorded. |
| intervention flagged | The harness requested an intervention. In an observational session the chunk still commits, so the flag is advisory. |
| rolled back (rolled-back session) | The experiment forced these rollbacks to test provenance filtering. They are not detections. |
| proposed change | The norm of the working-minus-committed fast-weight change per layer and tensor. Per-layer retained values are not recorded; the chunk total shows both proposed and kept. |
| n/a, not recorded | The source did not record this value. Absent is never shown as zero. |

## How data reaches the page

`scripts/export_sleep_observatory.py` reads only JSON, JSON-lines and Markdown under
`docs/research/results/`. It never opens weights. It writes an index and one file per run to
[`results/sleep-observatory/`](results/sleep-observatory/). It mirrors the same bytes to
`dashboard/public/assets/observatory/`, which the dashboard build serves from `/assets` on the
Space.

- **Recall counts** are recounted with the experiment's own `group_counts` only when every
  saved probe row can be attributed to a probe. Otherwise the saved counts stand. The seed-0
  recount reproduces `sleep_controls_recounted.md`.
- **Session maps** come from each run's archived `sessions/` folder. Runs without one show
  "Session log not archived".
- **Identity:** each run file lists the sha256 of every source file. The index carries an
  archive digest and the exporter version. The output contains no timestamps and no local
  paths.

Refresh after adding a run folder, which needs `sleep_controls.json` and a README row:

```bash
.venv/bin/python -m scripts.export_sleep_observatory
.venv/bin/python -m scripts.export_sleep_observatory --check   # exit 1 when the committed export is stale
.venv/bin/python -m pytest tests/test_export_sleep_observatory.py -q
```

Commit both `docs/research/results/sleep-observatory/` and
`dashboard/public/assets/observatory/`. A test fails if the two copies differ.

## The Learning view

**Learning** (`#sleep/learning/<set>`) shows the learning-contract reports
([spec](../superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md),
[results](results/contract-2026-09-23/README.md)). The contract separates two kinds of change:

- **Temporary adaptation** happens within one episode, in the fast weights, and is cleared
  at reset. The before-stream table measures it. The contract does not count it as learning.
- **Lasting change** is what remains in the slow weights after the experience stream,
  measured from a fresh state with the stream removed. The after-stream table measures it.

For each report set the view shows:

- **Identity:** checkpoint digest prefix and training step, execution commit, contract
  version, split id and size, the adaptation window (the learner's declared update period and
  the number of update boundaries inside one scored episode), and the experience stream.
- **Before any stream:** held-out-pairing MSE per held-out intervention policy and on the
  training distribution, with adaptation and without it, plus adaptation speed. Speed is the
  adapting error as a fraction of the error without adaptation, averaged over the episode,
  and the first step where it drops below one half.
- **After the stream, per mode** (frozen, continued training, everything in context):
  transfer delta per policy, forgetting delta, poison harm against the clean stream and
  against the start, correction residual, the revert ablation, the acceptance pair with its
  counts, tokens consumed and measured, and parameters. Deltas are after minus before on
  identical inputs, so negative is improvement.
- **Coordinate ablation:** per variant, parameters, seconds per training step, within-episode
  MSE with and without adaptation per policy, speed, the learned inner step size per layer,
  and what "without" means for that variant.

The "without" measurement is each learner's own off-intervention, and the view never merges
them under one label. `DynamicsLearner` (the delta-rule model, including `phys_mps_3k` and the
ablation's `delta_baseline`) disables writes with `beta_scale=0`, and decay stays active.
Coordinate learners freeze fast parameters with `freeze=True`, which stops both writes and
decay. These are different interventions, so the two "without" columns do not compare
directly.

Reading the ablation (from the coordinate memo's falsification table): if `decay_only`
matches `full`, timescale adaptation explains the gain and the nonlinear coordinates are not
earning their place. If `no_meta` matches `full`, meta-training is not necessary. If
`delta_baseline` matches `full` at matched compute, coupling learning to the dynamics is not
useful. Every ablation score is taken before any stream, so none of it is a lasting-learning
result.

| Label | Meaning |
| --- | --- |
| n/a (n=0) | The acceptance rate has an empty denominator: no decision was recorded on that side. A measured rate of zero reads `0.00 (n=…)`. |
| unchecked | The learner declared no update period, so the contract could not confirm that an update boundary falls inside a scored episode. |
| not below one half within N steps | The adapting error never dropped below half the error without adaptation within the scored horizon. The contract's raw field reports the horizon in this case; the export records it as no step. |
| stale | The set was measured under a contract version other than the current one (`plastic.eval.contract.CONTRACT_VERSION`). Its numbers stay visible. |
| not yet archived | The coordinate ablation has no results in the archive yet. This differs from a load failure, which shows an error and a retry. |
| not archived (a variant or mode) | That variant or mode has no result file in the set. |

**Data.** `scripts/export_contract_observatory.py` reads the JSON and Markdown under
`docs/research/results/contract-2026-09-23/` and recognizes each set by its content. A
report set has `manifest.json` plus `frozen.json`, `continued.json` and `in_context.json`. An
ablation set has one JSON per variant, each carrying `variant`, `contract` and
`no_adapt_label`. The exporter refuses a report set whose modes disagree on the contract
version, split, adaptation window or the before-stream measurement. It writes
[`results/learning-observatory/index.json`](results/learning-observatory/index.json) and
mirrors it to `dashboard/public/assets/learning/`. That directory sits beside
`assets/observatory/` because the Sleep exporter replaces its own directory on every run. The
Learning view loads its index separately, so the Sleep views and the Learning view can fail
independently.

Refresh after adding or changing a set, including when the coordinate ablation is archived
or `CONTRACT_VERSION` changes, in the same commit:

```bash
.venv/bin/python -m scripts.export_contract_observatory
.venv/bin/python -m scripts.export_contract_observatory --check   # exit 1 when the committed export or its mirror is stale
.venv/bin/python -m pytest tests/test_export_contract_observatory.py -q
```

Not shown in the Learning view: the per-step speed curves (only their summary), the per-stream
decision list behind the acceptance pair, wall-clock time, the change in the
without-adaptation error after the stream, and the ablation's fast-path support diagnostic
(inner loss before and after the proposed step, proposed change per layer). The source
reports hold all of them.

## Not shown yet

- **Gradient norms and W0 change for gradient methods:** reports record these only from
  source `aae9b70` onward. Earlier runs show "not recorded". The anchor's own per-tensor
  update is shown for every anchor run.
- **Dream per-token gains:** the reports store gains without token text, so the bars are
  positional.
- **Ceiling sessions:** their session logs are not archived because of their size.
- **Theme:** the interface uses the playground's dark theme.
