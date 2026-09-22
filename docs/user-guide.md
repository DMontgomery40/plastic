# plastic

**A small sequence model that learns at inference, with inspectable, reversible memory updates.**

`plastic` combines a selective state-space recurrence, gradient-updated fast memory,
and an external transactional harness. Text prediction and hidden-friction physics
share the same block implementation with different input embeddings and output heads.
It runs in plain PyTorch on CPU, Apple MPS, or CUDA, without custom Triton kernels.

The project asks two separate questions: **does online memory help prediction, and
can we control what it retains?** Saved checkpoints show a useful memory contribution.
The current attack experiments do not establish adversarial robustness.

## Start locally

Requirements: Python 3.12+, `uv`, and Node.js/npm for the dashboard.
[pyproject.toml](../pyproject.toml) declares the Python dependencies, including torch >= 2.12.

```bash
git clone https://github.com/DMontgomery40/plastic.git
cd plastic
uv sync --extra dev
npm --prefix dashboard ci
uv run plastic --help
./start.sh
```

Open [the dashboard](http://127.0.0.1:5173). The API listens on
`http://127.0.0.1:13579`, with interactive documentation at `/docs`.
The launcher defaults to CPU and does not download data or train a model.
A fresh checkout needs model artifacts before session features become useful.

### Public pretrained checkpoints

Both trained baselines live in one public, ungated Hugging Face repository:
[dmontgomery40/plastic](https://huggingface.co/dmontgomery40/plastic).
Each model's folder contains a tested Python loading example and links to the
compatible source revision.

| Model | Parameters | Hugging Face |
| --- | ---: | --- |
| WikiText next-token prediction | 6.85M | [text/](https://huggingface.co/dmontgomery40/plastic/tree/main/text) |
| Hidden-friction physics | 3.56M | [physics/](https://huggingface.co/dmontgomery40/plastic/tree/main/physics) |

Packages include weights, configuration, saved evaluation, license, and a checksum
manifest; the text model also includes its tokenizer. They use this project's
custom PyTorch runtime. Calibration files and session state are separate, so direct
model loading does not apply the transactional harness. The existing noncommercial
license applies; public availability does not remove its commercial-use restriction.

For an isolated store or different device/ports:

```bash
DEVICE=mps API_PORT=13580 DASHBOARD_PORT=5175 ARTIFACTS_ROOT=/tmp/plastic-demo ./start.sh
```

### Physics: no corpus download

Train, calibrate, and open a session. Use unique model/session IDs for subsequent runs.
`--device auto` selects CUDA, then MPS, then CPU. Training time depends on the device
and configuration.

```bash
uv run plastic train physics --model-id phys_demo --steps 3000 --layers 3 --seq-len 512 --device auto
uv run plastic calibrate phys_demo --chunks 64 --device auto
uv run plastic session new --model phys_demo --session-id physics_demo --device auto
uv run plastic physics physics_demo --steps 256 --mu 0.12 --device auto
```

The environment is a 2D point mass with hidden friction `mu`. Inputs contain four
observation values, two actions, and a reset flag; the target is the next observation
delta. The session compares **base** (fresh state, frozen memory), **frozen** (current
session state without memory updates), and **adaptive** on the same trajectory.
Activation state still advances in both frozen controls.

### Text: prepare BPE data, then train

Data preparation downloads the corpus and writes the tokenizer and token files locally.

```bash
uv run plastic data prepare --corpus wikitext --out artifacts/data/wikitext --vocab 8192
uv run plastic train text --model-id lm_demo --data artifacts/data/wikitext --steps 3000 --device auto
uv run plastic calibrate lm_demo --data artifacts/data/wikitext --chunks 512 --device auto
uv run plastic session new --model lm_demo --session-id text_demo --device auto
uv run plastic chat text_demo "The history of computing" --max-new-tokens 32 --device auto
```

This is a small next-token model, not an instruction-tuned assistant. Prompt tokens
are eligible for learning; generated tokens are frozen by default. The dashboard
provides Sessions, Session, Chat, Physics, Train, Red team, and Architecture views.

## What is implemented

Each `PlasticBlock` contains short causal convolutions, a selective diagonal
recurrence, a fast-memory branch, and a residual MLP. `PlasticCore` carries state
through the stack; `PlasticLM` and `PlasticDynamics` supply the domain interfaces.

The default memory rule is gated online gradient descent on a linear associative loss.
For row-vector keys and values, one head updates its matrix `S` as follows:

```text
e_t = v_t - k_t (alpha_t S_(t-1))
S_t = alpha_t S_(t-1) + beta_t k_t^T e_t
read_t = RMSNorm(q_t S_t)
```

`beta` controls writing; `alpha` controls retention. Outer training differentiates
through the updates, and inference carries fast state between inputs. The default
`delta` rule has recurrent and chunk-parallel implementations. The optional
`--rule chunk` reads from chunk-start weights and applies minibatch updates with
momentum and optional Newton-Schulz orthogonalization.

The fast learner is **linear**, with normalization on its readout. It should not be
presented as an exact reproduction of Sun et al.'s TTT-Linear layer or a new nonlinear
inner model. `memory="mlp"` currently raises `NotImplementedError`.
The [coordinate-recurrence proposal](../docs/research/2026-09-21-plastic-coordinate-recurrence.md)
is experimental, not the default implementation.

## How the transaction harness works

The harness sits outside the differentiable model. It snapshots state, processes a
chunk provisionally, measures the proposed change, and decides what to retain.

| Decision | Effect |
| --- | --- |
| Commit | Accept the candidate state. |
| Rollback | Refuse the memory update and replay the chunk with memory frozen. |
| Scale | Reprocess with a smaller write rate; retention remains active. |
| Project | Remove a canary-damaging component of the memory delta, subject to checks. |
| Read-only | Stop memory learning while continuing to process inputs. |

Signals include prediction loss, associative prediction error, update norms,
Fisher-weighted changes, canary losses and alignment, robust statistics, and CUSUM.
The policy uses these numeric measurements rather than keyword or regex rules.
Accepted memory changes are checked against configured write budgets.

**Proposed and accepted are different measurements.** Transaction `signals` describe
the provisional update; `accepted` records what was actually kept. Rollback does not
erase activation-level influence or retract outputs already produced within the
chunk. A good adaptive physics error can coexist with a final rollback; it is not
proof of retained learning.

Requested calibration rates, observed interventions, and measured false-positive
rates are also different quantities. Finite-sample threshold ranks and in-sample
CUSUM alarm frequencies do not establish a fresh-data false-positive rate for the
complete sequential policy. Canary projection is a local approximation, not a
guarantee against harmful changes outside the probes' coverage.

## Recorded results and their limits

These are saved evaluation records from **22 September 2026**, not fresh evaluations
run by installing this checkout. Comparisons start from fresh state and change the
write multiplier to `beta_scale=0`; they are separate from harness evaluation.

| Saved model | Parameters / steps | Held-out metric | Writes enabled | Writes disabled |
| --- | --- | --- | --- | --- |
| `lm_wikitext_l4` | 6.85M / 6,000 | Text NLL, nats/token | 3.4704 | 5.0151 |
| `phys_mps_3k` | 3.56M / 3,000 | Observation-delta MSE | 0.00012885 | 0.20873034 |

Text evaluation covers 131,072 tokens; physics evaluation covers 131,072 target
elements. Text MQAR recall was 100% at 4 and 8 pairs and 97.66% at 16 pairs. The text
training recipe includes MQAR batches, so this is trained-task recall. These
single-run results support memory utility on these tasks, not general language
quality or a direct measurement of friction encoded in memory. NLL and MSE cannot
be ranked together.

The recorded text attack campaign had **7 valid payloads out of 40**; PGD had none
(0/8). The largest valid accepted canary-loss increase was +0.077995 against a
coherence threshold of +1.272153. Its matched frozen control was -0.000389, giving
an accepted-minus-frozen difference of +0.078384. Frozen controls measure activation
influence; their range across unrelated payloads is not statistical noise.
This campaign does not demonstrate defense against a strong valid attack.

See the [trained-model report](../docs/research/2026-09-22-trained-model-operating-point.md),
[calibration replay audit](../docs/research/2026-09-22-calibration-replay-audit.md), and
[shared review record](../SHARED_SCRATCHPAD.md). The shared review record tracks corrections and their disposition; the original ten UI findings were closed in ASTRA-038. Broader research claims remain subject to the limits above.

## Sessions and experiments

```bash
uv run plastic models
uv run plastic session list
uv run plastic session show text_demo
uv run plastic session fork text_demo text_branch
uv run plastic session reset text_branch
uv run plastic session resume text_branch
uv run plastic redteam lm_demo --data artifacts/data/wikitext --prefixes 4 --steps 30
uv run plastic sleep lm_demo --core artifacts/data/wikitext --sessions text_demo
```

Forks start from the parent's committed state. Reset clears fast state; resume lifts
a read-only latch but does not replenish a spent write budget. Red-team reports
include guarded, unprotected, and frozen endpoints from the same post-prefix state,
guarded-path NLL, and valid-only aggregates. Adding `--record` modifies the model's
poison-canary suite. Sleep attempts canary-gated consolidation into a child model.
`train text --adversarial` is experimental; improved robustness from it has not been
established by the reported campaign.

For cloud runs, see the [Hugging Face Jobs guide](../docs/research/2026-09-21-tooling-hf-jobs-torch.md).
The launcher exports **committed HEAD**, not uncommitted source, and starts paid compute.
Inspect its account, bucket, and export-path settings before using it.

## Development and artifacts

```bash
uv run pytest
npm --prefix dashboard test
npm --prefix dashboard run build
git diff --check
```

| Path | Responsibility |
| --- | --- |
| `plastic/model/`, `plastic/config.py` | Blocks, recurrence, memory rules, carried state |
| `plastic/data/`, `plastic/tokenizer/`, `plastic/train/` | Datasets, BPE, training and evaluation |
| `plastic/harness/` | Transactions, calibration, canaries, projection, budgets |
| `plastic/session/`, `plastic/store.py` | Persistence, signatures, forks and traces |
| `plastic/redteam/`, `plastic/sleep/` | Attack validation and consolidation |
| `plastic/api/`, `plastic/cli.py`, `dashboard/` | FastAPI, CLI, React/Vite interface |
| `tests/`, `docs/research/`, `scripts/` | Regression tests, research evidence, utilities |

The default store is `artifacts/`: checkpoints, configs, tokenizers, evaluation and
calibration files, session state/traces, and attack results. New generated artifacts
and `training_data/` contents are Git-ignored. A Git push is **not a checkpoint or
dataset backup**; copy those separately. Existing tracked historical artifacts are
an exception. A clean clone does not include the trained checkpoints reported above;
download them from the public Hugging Face repository linked above. That release
backs up the published model packages, not all local experiment or session artifacts.

Read [AGENTS.md](../AGENTS.md) before changing model or harness contracts. Coordinate
concurrent work through the append-only [shared scratchpad](../SHARED_SCRATCHPAD.md).

## License

The [license](../LICENSE) permits non-commercial use under its stated conditions.
Commercial use requires prior written permission from the copyright holder.
