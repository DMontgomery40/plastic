# Copy-task memory-content audit

On the seed-0 copy model at commit `46f632e37840fa05e363c459f44c474bba93644f`, retained fast-memory content improves suffix prediction when compared with zeroed or shuffled memory, holding the prefix activation state and future write policy fixed. This is a baseline delta-memory result, not evidence for the proposed nonlinear coordinate architecture.

## Method

The probe trains the existing smoke-test recipe once: vocabulary 32, random segment length 8 repeated four times, batch 32, 150 steps, model width 64, two heads, two layers, chunk size 16, Muon matrix learning rate 0.02, AdamW learning rate 0.01, gradient clip 1. It uses CPU float32, PyTorch 2.14.0, one CPU thread. The observed training time was 10.79 seconds; this is not device throughput evidence for the default model or an L4/A10G training budget.

Evaluation uses 256 fresh examples generated with seed 260922. After consuming the first 8 or 16 tokens, the probe clones the prefix state into three branches. It retains S, zeros S, or rotates S across the batch, for every layer. It asserts h, convolution buffers, and momentum match across branches at the intervention. All three continuations use `freeze=True`; the probe asserts S is unchanged through each continuation. The other recurrent states then evolve normally in response to their respective memory reads.

Loss is next-token cross-entropy over target indices 9–31 for boundary 8 and 17–31 for boundary 16 (zero-based). The already-computed prefix prediction and the unknown target after the final input are excluded. Uninterrupted adaptation and writes disabled from the start are contextual controls; they do not hold the prefix activation state fixed.

## Results

| Evaluation condition | Boundary 8 NLL | Boundary 16 NLL |
| --- | ---: | ---: |
| Retained S, future writes frozen | 2.48546 | 2.25983 |
| Zero S, future writes frozen | 4.03509 | 4.17523 |
| Batch-shuffled S, future writes frozen | 4.41929 | 4.54924 |
| Uninterrupted adaptation | 2.23887 | 2.17908 |
| Writes disabled from the start | 4.10041 | 4.20961 |

Paired zero-minus-retained differences are 1.54963 ± 0.03526 and 1.91540 ± 0.04164 nats (mean ± standard error). Shuffled-minus-retained differences are 1.93383 ± 0.04001 and 2.28941 ± 0.04029. These standard errors describe variation across evaluation examples conditional on one trained model; they do not measure variation across training seeds.

This intervention supports a causal contribution from stored S content to this trained model's copy-task performance, beyond comparing an enabled versus disabled update gate. Resetting or shuffling S also introduces state combinations unseen during ordinary execution. The experiment does not identify the storage algorithm, prove generalization to natural text or physics, establish novelty, or validate transaction safety.

## Reproduction and artifacts

Run `docs/research/probes/check_copy_memory_content.py --out <new-output-directory>` with the repository Python environment. By default it imports the working tree. Set `PLASTIC_AUDIT_SOURCE` to an immutable export of the recorded commit to reproduce this specific result while implementation continues. The export used here is `/private/tmp/astra-copy-46f632e`, with `.audit-head` containing the full commit. The JSON records SHA-256 hashes of config, optimizer, and model sources and confirms they did not change during the probe.

```sh
PLASTIC_AUDIT_SOURCE=/private/tmp/astra-copy-46f632e .venv/bin/python docs/research/probes/check_copy_memory_content.py --out artifacts/astra/copy-memory-content-20260922
```

Local artifacts (git-ignored):

- `artifacts/astra/copy-memory-content-20260922/results.json`: source manifest, paired metrics, per-example losses, and timings.
- `artifacts/astra/copy-memory-content-20260922/copy_seed0_steps150.pt`: reusable trained checkpoint, configuration, source manifest, and endpoint training losses.

The probe reuses a cached checkpoint only if its source manifest matches; otherwise it requires a fresh output directory. A cached run records loading time rather than training time.
