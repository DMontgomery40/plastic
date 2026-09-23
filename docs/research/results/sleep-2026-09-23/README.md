# Sleep experiment outputs, 2026-09-23 (compact, reproducible identities)

Each folder is one `scripts/experiments/sleep_controls.py` run: `sleep_controls.md` (table),
`sleep_controls.json` (all arms, per-probe replies), and `sleep_<arm>_report.json` (the sleep run's
own report: harvest, before/after measurements, gate, dreams where applicable). No weights.

| Run | Checkpoint | Digest (prefix) | Code (see note) | Question |
| --- | --- | --- | --- | --- |
| `base_dryrun_*_report.json` | TTT-MLP-760M base (RetentionLabs conversion, public) | see report | ≈ 3992890 (unrecorded) | mechanics only |
| `sleep_controls_step50` | SFT step 50 of 250 (intermediate, not published) | 229e97b67875b732 | ≈ 6334247 (unrecorded) | null baseline, W0 × 10 steps |
| `sleep_controls_step100_all40` | SFT step 100 (intermediate) | 5d7f0b9c6c1a8d4f | ≈ 93d524d (unrecorded) | all parameters × 40 steps: collapse passed the NLL gate |
| `sleep_controls_step100_w0_40` | step 100 | 5d7f0b9c6c1a8d4f | ≈ a11157b (unrecorded) | W0 × 40, assistant-only labels: collapse |
| `sleep_controls_step100_w0_40_all` | step 100 | 5d7f0b9c6c1a8d4f | ≈ 9a1b502 (unrecorded) | W0 × 40, user tokens supervised |
| `sweep_step100_*` | step 100 | 5d7f0b9c6c1a8d4f | ≈ c564390 (unrecorded) | regime sweep, replay on raw turns |
| `study_step100_w0` | step 100 | 5d7f0b9c6c1a8d4f | ≈ 633a899 (unrecorded) | templated study set, prompt-loss weight 0.2 |
| `dream_step100_w0` | step 100 | 5d7f0b9c6c1a8d4f | ≈ 3828ac6 (unrecorded, pre ASTRA-175 fixes) | first dream run, free-form prompts |

**Code column.** Runs launched before 2026-09-23 15:40 UTC did not record their execution commit; the
values marked ≈ are the latest commit of the relevant code at launch time as reconstructed from the
session log and may be off by a documentation-only commit. From this point every `sleep_controls.json`
and `sleep_report.json` carries `code_commit` (short SHA, `+dirty` if the tree had changes) captured at
launch. The standalone floor arm now also records the expected-answer log-probability; in earlier
runs it is present only in each sleep report's `before` block, which measures the same parent from
the same fresh state.

The intermediate SFT checkpoints are training artifacts of job `6ab351f352d0dbd7f1d82429` and are not
published; the released chat checkpoint will be pinned in `deploy/huggingface/pretrained.py`
(`TTT_CHAT`) and all later runs use it. Readings of every table are in
`docs/research/2026-09-23-sleep-consolidation.md`; the community evidence that motivated the study set
and the dream method is in `docs/research/2026-09-23-sleep-community-research.md`.

Collapse artifacts: a "1/6 taught" cell whose report shows a large `max_cluster_share` is one sentence
containing a name, not recall. Read recall together with `max_cluster_share` and `mean_answer_logprob`.
