# Calibration replay on the saved WikiText model

**Latest follow-up:** the revised procedure (source matching `25baf43`) commits all 128 replayed chunks and rolls back 2/128 disjoint chunks (1.5625%), with no CUSUM alarms or read-only latching. This is a substantial improvement in this check, but 128 fresh chunks do not establish the requested 1% rate. The historical experiment below and the revised experiment at the end use different calibration sizes and evaluation initialization.

At source `23cb13e`, the default harness did not meet its requested 1% benign gating rate in this bounded check. It gated 16/128 chunks when replaying the calibration data and 13/128 chunks from a disjoint validation region. This is measured behavior of one lightly trained checkpoint, not a robustness result or a deployment-wide false-positive estimate.

## Procedure

- Model: `lm_smoke_mps2`, checkpoint step 60, width 256, four layers, four heads, vocabulary 8192; delta memory, transaction chunk 64, scan chunk 16.
- Device: CPU float32, PyTorch 2.14.0, one CPU thread. No training or paid compute was launched.
- Source: immutable export `/private/tmp/astra-m3-23cb13e`; per-file hashes saved. The checkpoint and source remained unchanged during the check.
- Canary suite: the default suite from the first 384 tokens of WikiText validation, with default synthetic poison probes.
- Calibration: 128 chunks, resetting state every four chunks, validation token range `[384,8576)`. Fisher computation used 16 chunks and the current batch-four procedure.
- Replay: same 128 chunks, fresh state every four chunks, with the complete default policy enabled.
- Fresh evaluation: another 128 chunks in `[8576,16768)`, also fresh state every four chunks. This range is disjoint from the threshold-calibration and coherence-canary ranges. Fisher sampling uses the whole validation split, matching the current CLI procedure; this is not a fully independent held-out study.
- Gating means an applied decision other than `commit`. Rollbacks and subsequent read-only chunks are reported separately.

## Results

| Evaluation | Commit | Rollback | Read-only | Gated fraction |
| --- | ---: | ---: | ---: | ---: |
| Same-data replay | 112 | 10 | 6 | 12.50% |
| Disjoint fresh sessions | 115 | 7 | 6 | 10.16% |

No scale or project decisions occurred in these two evaluation streams. Both evaluations recorded three CUSUM alarms, followed by six read-only decisions. Other triggers included calibrated upper thresholds and coherence-canary changes; replay also included one poison-canary trigger. Raw records retain every decision and reason.

Even before policy feedback, five of the 128 original log-only calibration records exceed the subsequently fitted upper thresholds: one each for loss, surprise, log delta norm, log write norm, and coherence-canary delta. These are five distinct chunks (3.91%). With only 128 observations, the interpolated near-maximum quantiles do not demonstrate a combined 1% operating point. CUSUM and the poison trigger introduce additional decisions that are outside that upper-quantile allocation.

The calibration artifact contains **no Fisher reference or threshold**, although Fisher is attached for later session execution. In `calibrate_model`, the observation runner receives `calibration=None`; its Fisher is consequently absent while reference signals are collected. Passing Fisher to `calibrate_from_runner` only attaches it to the returned artifact. Also, the known Fisher traversal and batch-normalization defects in ASTRA-014 are still present in this pinned source. This experiment reproduces that procedure rather than silently repairing it.

The full run took 119.20 seconds on CPU. That duration includes Fisher estimation, calibration and two evaluation passes; it is not GPU training throughput.

## Artifacts and reproduction

```sh
PLASTIC_AUDIT_SOURCE=/private/tmp/astra-m3-23cb13e .venv/bin/python docs/research/probes/check_calibration_replay.py --out artifacts/astra/calibration-23cb13e-20260922
```

The output folder contains `results.json` (source/data/checkpoint identities and summary), `records.json` (all calibration/replay/fresh transaction records), and the separate `calibration.json` and `fisher.pt` used by this audit. The original model folder was not modified.

Checkpoint SHA-256: `15e15f1a787e6123bd8530aa1fb89516bcceaafd191d2fd2a29440f74acaacbb`.

The next acceptance check should calibrate and evaluate the complete decision process on separate streams with the same reset and source-control behavior, reporting both direct interventions and read-only consequences. Larger samples and multiple sessions/checkpoints are needed to assess a 1% target; this small test is sufficient to show that the `23cb13e` settings failed it here.

## Revised procedure and results — 2026-09-22 04:36 UTC

The follow-up uses the same unchanged step-60 checkpoint, corpus, device and canaries. It captures `4fd7a6d` plus the then-uncommitted calibration/session changes in an immutable directory. Every captured Python source file under `plastic/` was subsequently compared byte-for-byte with Fable's `25baf43` commit and matched; the saved source-equivalence record and snapshot diff retain that provenance.

Changes from the original experiment:

- 512 calibration chunks rather than 128, tokens `[384,33152)`; state still resets every four calibration chunks.
- Corrected Fisher estimator; 16 Fisher chunks, attached to the runner **before** reference collection. All 512 Fisher observations and a threshold are present.
- Revised order-statistic thresholds, a lower poison threshold and fitted CUSUM threshold.
- 128 evaluation chunks in each pass. Same-data replay uses the first 128 calibration chunks, `[384,8576)`. Fresh evaluation uses `[33152,41344)`.
- Evaluation constructs a fresh runner for each four-chunk session. This deliberately avoids the reset bug found during this audit: initialization respects calibrated CUSUM h, while `reset()` at the pinned source reverts to the configured h. The fitted h here is **60.0362643**, versus configured **5**. These results therefore describe fresh sessions, not the defective reset path.

| Evaluation | Commit | Rollback | Scale/project | Read-only | Direct intervention fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-data subset replay | 128 | 0 | 0 | 0 | 0% |
| Disjoint fresh sessions | 126 | 2 | 0 | 0 | 1.5625% |

There are no CUSUM alarms in calibration or either evaluation. The two fresh refusals are one log-write-norm exceedance and one Fisher-update exceedance. The original source and checkpoint remain unchanged. Runtime **279.30 seconds** is CPU audit duration, not model throughput.

The reported achievable rate per each of seven thresholded signals is `1/513 = 0.19493%`; the sum is **1.36452%**, before considering sequential policy controls. This finite-sample reporting does not certify the whole policy's 1% operating point. The fresh observed rate is based on only 128 chunks in 32 sessions, and Fisher still samples the whole validation split, so this is not fully held out with respect to Fisher estimation. The changed calibration size, corrected estimator and fresh-runner initialization also prevent attributing the improvement to any one fix. The earlier fresh evaluation region is now inside the larger calibration region; the two fresh-stream percentages are not a paired comparison.

## Follow-up artifacts and reproduction

```sh
PLASTIC_AUDIT_SOURCE=/private/tmp/astra-sync-4fd7a6d-working-20260922 .venv/bin/python docs/research/probes/check_calibration_replay.py --out artifacts/astra/calibration-4fd7a6d-working-20260922 --chunks 512 --evaluation-chunks 128 --fisher-chunks 16 --fisher-before-calibration --session-init fresh
```

The output includes full records and source hashes, `calibration.json`, `fisher.pt`, `source-equivalence.json`, `base-commit.txt` and `snapshot.diff`. The new probe flags preserve the original reproduction defaults. Fable's later reset fix is outside this pinned experiment; it needs its own lifecycle regression check. This experiment establishes neither adversarial robustness nor a deployment-wide false-positive rate.
