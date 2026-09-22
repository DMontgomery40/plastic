# Calibration replay on the saved WikiText model

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

The next acceptance check should calibrate and evaluate the complete decision process on separate streams with the same reset and source-control behavior, reporting both direct interventions and read-only consequences. Larger samples and multiple sessions/checkpoints are needed to assess a 1% target; this small test is sufficient to show that the current settings failed it here.
