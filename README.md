# plastic

A tiny test-time-training state-space model with a transactional safety harness.

`plastic` explores small models that learn while they read. Its recurrence is a selective state-space layer; its memory is a fast weight matrix that
takes one gradient step per token on a self-supervised objective, so context is
compressed into a fixed-size fast memory. The whole thing is meta-trained
end to end through that inner loop, so training shapes both the representations
being stored and the rules for updating them. The same block stack serves two
domains: next-token prediction on text, and a hidden-friction control task that
tests adaptation to changing latent dynamics.

Learning at inference is a security surface: an input can change how the
model behaves on later inputs. `plastic` wraps
the inner loop in a transactional harness. Each chunk of tokens is a transaction:
attempt the update, measure it, then commit, roll it back, scale it down, project it,
or refuse it. Every signal the harness reads comes from the model itself (loss,
surprise, the write rate the model chose, the size and curvature of the weight change,
how it moves a set of probe texts), never from pattern-matching the input. Sessions
are persisted and branchable, like version control for the plastic weights.

The research question is simple: **does online memory improve prediction, and can
we control what it learns?** The project brings the model, measurements, controls,
and experiments into one workbench, running in plain PyTorch on CPU, Apple Silicon,
or CUDA.

[Live playground](https://huggingface.co/spaces/dmontgomery40/plastic) ·
[Models and source on Hugging Face](https://huggingface.co/dmontgomery40/plastic) ·
[Research](docs/research/README.md) · [User guide](docs/user-guide.md)

## Install and run

```bash
git clone https://github.com/DMontgomery40/plastic.git
cd plastic
uv sync --extra dev
npm --prefix dashboard ci
uv run plastic --help
./start.sh
```

Requires Python 3.12+, PyTorch 2.12+, `uv`, and Node.js/npm. The dashboard opens at
[localhost:5173](http://localhost:5173), with the API on port 13579. A fresh checkout
needs model artifacts; the [user guide](docs/user-guide.md) covers downloading the
published checkpoints, training, calibration, and sessions.

## The architecture

The saved baselines stack one block four times for text and three for physics.
Each block combines a selective recurrence, fast-weight memory, and a residual MLP,
all in plain PyTorch.

**A selective state-space branch** provides activation-level context. It is a gated
linear recurrence with a per-channel, input-dependent decay, computed with a chunked
log-space scan that only ever forms decay ratios in `(0, 1]`.
The recurrent and chunk-parallel paths are tested for equivalence, including carried state.

**A fast-weight memory branch** is the test-time-training layer. Per head it holds a
matrix `S`, and per token it takes one gradient step on the associative loss
`½‖kS − v‖²` with a learned write rate `β` and a forget gate `α`:

```
e_t = v_t − k_t (α_t S_{t−1})          associative prediction error
S_t = α_t S_{t−1} + β_t k_tᵀ e_t       one gradient step per token, gated
m_t = RMSNorm(q_t S_t)                 normalized read after the token's own write
```

This is a linear gated-delta learner with a normalized readout. Its chunk-parallel
training form uses a unit-lower-triangular solve; the recurrent form processes
one token at a time. A second inner rule (mini-batch with momentum and
Newton-Schulz orthogonalization, in the style of LaCT and Atlas) is available behind
the same interface. `β` is the model's own answer to "should I learn from this?", and
the harness reads it.

**Meta-training** runs the full sequence with the fast state starting at zero, and
backpropagates the ordinary next-token loss through every inner update. Because the
inner step has a closed form, ordinary autograd can differentiate through it on
CPU, MPS, and CUDA.

The [literature review](docs/research/2026-09-21-ttt-ssm-literature.md) places the
design alongside TTT layers, Titans, LaCT, Gated DeltaNet, and related work. An
[architecture memo](docs/research/2026-09-21-architecture-memo.md) develops the block. The separate
[coordinate-recurrence proposal](docs/research/2026-09-21-plastic-coordinate-recurrence.md)
explores a nonlinear learner and remains experimental.

## The safety harness

Every input chunk is a transaction against three copies of the session state:
committed, working, and the pending chunk. At each chunk boundary the harness computes
signals entirely from the model and decides what to do with the update.

| Signal | What it is |
|---|---|
| chunk loss | the model's own confusion on the chunk (out-of-distribution proxy) |
| surprise | the inner-loop prediction error `‖e_t‖`, independent of the write rate |
| write rate `β` | the learned gate the model applied to each token |
| update norm | the actual change to the fast weights, and its Fisher-weighted size |
| canary suites | a coherence set whose loss must not rise, a poison set whose loss must not fall |
| canary alignment | cosine between the weight change and the gradient that would hurt the canaries |
| robust statistics | median/MAD z-scores against a reference or session history, plus CUSUM on log update size |

The decision is one of: **commit**, **rollback** (refuse the chunk as training signal,
read it but do not learn it), **scale** (retry with a smaller write rate), **project**
(remove the component that would raise the canary loss, an A-GEM style half-space
projection on the state delta), or **read-only** (the session's write budget is spent).
Budgets are hard: the actual weight change of the accepted candidate is checked against
the cap, non-finite states are refused, and a spent session stops learning. Resume
clears a read-only latch without replenishing the budget. Calibration supplies
empirical thresholds from a benign reference stream.
Sequential false-positive rates and probe coverage are evaluated separately.

The decision path uses numeric model evidence rather than keyword or regex rules.
Transaction records distinguish proposed updates from accepted changes. Rollback
controls retained memory; it does not retract emitted output or erase all activation
influence. The [safety survey](docs/research/2026-09-21-inference-time-learning-safety.md)
develops the threat model and the methods behind these controls.

## The two domains

**Text.** Byte-level BPE, wikitext-103 (or fineweb-edu). Next-token prediction. Recall
is measured directly with MQAR probes, and every checkpoint reports its held-out loss,
its held-out loss with the memory disabled (the difference is the value of the memory),
its MQAR accuracy, and the histogram of learned write rates.

**Physics.** A 2D point mass with a friction coefficient the model never observes. The
input is `[observation, action, reset flag]`; the target is the next observation delta.
Several episodes with different friction are packed into one sequence to test
adaptation to changing latent dynamics. Sessions compare three conditions on the
same trajectory: a fresh frozen state, the current session state with memory
frozen, and the session learning online.

Saved evaluations of the original research checkpoints show a substantial memory
contribution on both tasks:

| Checkpoint | Parameters | Held-out metric | Writes enabled | Writes disabled |
| --- | ---: | --- | ---: | ---: |
| Text | 6.85M | NLL, nats/token | 3.4704 | 5.0151 |
| Physics | 3.56M | Observation-delta MSE | 0.00012885 | 0.20873034 |

Each single-run evaluation covers 131,072 tokens or target elements. Disabling
writes sets `β=0` while retention remains active. These results measure memory
utility; the attack studies have not established adversarial robustness. Methods
and measurements are in the [evidence report](docs/research/2026-09-22-trained-model-operating-point.md)
and the Hugging Face [text](https://huggingface.co/dmontgomery40/plastic/tree/main/text)
and [physics](https://huggingface.co/dmontgomery40/plastic/tree/main/physics) bundles.

## Sessions, red team, and sleep

Sessions persist their plastic weights, harness state, and full transaction log, and
they branch: fork a session and the child starts from the parent's committed state,
so you can compare divergent learning histories from one point. `plastic chat` learns
each prompt through the harness; `plastic physics` runs an episode; `plastic session
fork|reset|resume|show` manage the tree.

`plastic redteam` attacks a model on the real token path: it optimizes a perturbation
of a suffix's embeddings to maximize canary damage subject to the model's own
perplexity staying plausible, snaps to real tokens, and re-validates the discrete
payload through the harness, so the reported damage is the damage of a payload that
was actually fed. `plastic train text --adversarial` meta-trains the write gate against
that attacker. `plastic sleep` consolidates what sessions learned into the slow
weights, accepted only if the canaries hold.

The workbench also supports pretrained models through a separate native backend.
The [public playground](https://huggingface.co/spaces/dmontgomery40/plastic) uses
Huihui's abliterated Qwen3.5-0.8B for text exploration and recurrent-state
measurements, in observational mode without automatic rollback. Its shared session
is public. This extends the harness experiments beyond the small meta-trained
PlasticCore models; available signals and controls depend on the backend.

See the [user guide](docs/user-guide.md) for training, sessions, and experiment
commands, and the [deployment guide](deploy/huggingface/README.md) to run the
hosted playground yourself.

## Layout

```
plastic/
  model/        selective scan, gated delta rule, chunk rule, fast-weight memory, blocks, models, state
  data/         wikitext/fineweb pipeline, MQAR recall probes, hidden-mu physics
  train/        outer loop, LR schedule, Muon/AdamW split, adversarial write-gate loss
  harness/      signals, robust stats, canaries, Fisher, projection, policy, calibration, transaction runner
  session/      persisted, branchable sessions for both domains
  redteam/      token-path attacker validated through the harness
  sleep/        canary-gated consolidation into the slow weights
  api/          FastAPI service over the artifact store
  store.py      models, sessions, transactions, forks, signatures
  cli.py        the `plastic` command
dashboard/      React UI
scripts/        Hugging Face Jobs launchers, experiments, throughput bench
docs/           the design spec, milestone plans, and the research surveys
```

## License

The [license](LICENSE) permits noncommercial use under its stated conditions.
Commercial use requires prior written permission from the copyright holder.
