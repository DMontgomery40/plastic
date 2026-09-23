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

## The three views

**Runs.** Runs are grouped by checkpoint. For each run:

- **Protocol:** checkpoint digest prefix, the code commit recorded at launch, facts, steps,
  target, learning rate, flagged-turn rule and ceiling mode.
- **Arm matrix:** one row per arm. It shows fresh-session recall per probe group, recall on
  unseen wording, held-out NLL before and after, the largest identical-reply share, and the
  outcome.
- **Arm detail:** lineage from parent to child, or "no child" when the arm was pulled back;
  every locality check as a value against its limit; the turns the arm trained on; kept
  Dream items; every reply, filterable by group.

**Anatomy of a sleep.** One run and arm, in five stages:

1. Wake: every 16-token chunk of the teaching and rolled-back sessions, marked committed
   or rolled back, with advisory intervention flags.
2. Harvest: turns accepted online, rolled back, flagged, and the selected text turns. It
   also names the sessions whose full committed state the method loads.
3. Consolidate: the method, its loss over steps, the W0 change when recorded, and dreams.
4. Gate: each check against its limit.
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
| Committed | The gate accepted the child and registered it in the run's disposable experiment store. It is not published as a model. |
| Pulled back | A locality check failed. No child was registered, and the parent is unchanged. |
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

## Not shown yet

- **Gradient norms and W0 change for gradient methods:** reports record these only from
  source `aae9b70` onward. Earlier runs show "not recorded". The anchor's own per-tensor
  update is shown for every anchor run.
- **Dream per-token gains:** the reports store gains without token text, so the bars are
  positional.
- **Ceiling sessions:** their session logs are not archived because of their size.
- **Theme:** the interface uses the playground's dark theme.
