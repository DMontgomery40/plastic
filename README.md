# plastic

A tiny test-time-training state-space model with a transactional safety harness.

`plastic` is one small model (about 6M parameters) that learns while it reads. Its
recurrence is a selective state-space layer; its memory is a fast weight matrix that
takes one gradient step per token on a self-supervised objective, so context is
compressed into weights instead of a growing cache. The whole thing is meta-trained
end to end through that inner loop, which is what makes it a real test-time-training
layer rather than an adapter bolted onto a frozen model. The same block stack serves
two domains: next-token prediction on text, and a hidden-friction control task where
the only way to predict well is to infer the latent dynamics into the fast weights.

Learning at inference is a security surface: every input is a gradient step, so a
hostile input is not just a bad answer, it is a bad thing learned. `plastic` wraps
the inner loop in a transactional harness. Each chunk of tokens is a transaction:
attempt the update, measure it, then commit, roll it back, scale it down, project it,
or refuse it. Every signal the harness reads comes from the model itself (loss,
surprise, the write rate the model chose, the size and curvature of the weight change,
how it moves a set of probe texts), never from pattern-matching the input. Sessions
are persisted and branchable, like version control for the plastic weights.

This is a research sandbox, not a product. It is deliberately small, runs in plain
PyTorch on an Apple Silicon laptop or a single cloud GPU, and is honest about what it
does and does not show.

## Install and run

```bash
uv sync --extra dev
uv run pytest -W ignore          # the test suite
uv run plastic --help
./start.sh                       # API on :13579, dashboard on :5173
```

Python 3.12, PyTorch 2.12 or newer. MPS on Apple Silicon, CUDA on Hugging Face Jobs.

## The architecture

One block, stacked four times for text and three for physics. Every block has three
parts, all in plain PyTorch that runs on CPU, MPS, and CUDA.

**A selective state-space branch** provides activation-level context. It is a gated
linear recurrence with a per-channel, input-dependent decay, computed with a chunked
log-space scan that only ever forms decay ratios in `(0, 1]`. (The scan the project
started with formed `a^t` directly and collapsed to zero after four tokens; the
replacement is exact against a sequential reference to within 2e-5.)

**A fast-weight memory branch** is the test-time-training layer. Per head it holds a
matrix `S`, and per token it takes one gradient step on the associative loss
`½‖kS − v‖²` with a learned write rate `β` and a forget gate `α`:

```
e_t = v_t − k_t (α_t S_{t−1})          prediction error (the inner-loop gradient direction)
S_t = α_t S_{t−1} + β_t k_tᵀ e_t       one gradient step per token, gated
m_t = q_t S_t                          read after the token's own write
```

This is TTT-Linear (Sun et al., 2024) with mini-batch one, plus Gated DeltaNet's
forget gate. It has a chunk-parallel form for training (an exact unit-lower-triangular
solve, not an approximate inverse, so repeated tokens do not blow it up) and a
recurrent form for inference; the two agree to a few parts in `1e-4` and that
equivalence is a permanent test. A second inner rule (mini-batch with momentum and
Newton-Schulz orthogonalization, in the style of LaCT and Atlas) is available behind
the same interface. `β` is the model's own answer to "should I learn from this?", and
the harness reads it.

**Meta-training** runs the full sequence with the fast state starting at zero, and
backpropagates the ordinary next-token loss through every inner update. Because the
inner step has a closed form, this needs only first-order autograd and runs on MPS at
about 8K tokens/second for the default model.

The design is grounded in a literature review current to September 2026 (TTT layers,
Titans, LaCT, TTT-E2E, Gated DeltaNet, Mamba-3) and an architecture memo verified on
CPU and MPS, both in `docs/research/`.

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
| robust statistics | median/MAD z-scores against a fixed benign reference, plus a CUSUM on drift |

The decision is one of: **commit**, **rollback** (refuse the chunk as training signal,
read it but do not learn it), **scale** (apply a fraction of the update), **project**
(remove the component that would raise the canary loss, an A-GEM style half-space
projection on the state delta), or **read-only** (the session's write budget is spent).
Budgets are hard: the actual weight change of the accepted candidate is checked against
the cap, non-finite states are refused, and a spent session stops learning until it is
explicitly resumed. Thresholds are calibrated on a benign held-out stream to a target
false-positive rate, using split-conformal order statistics that report the rate the
sample size can actually support.

There is no regex, no keyword list, and no string inspection anywhere in the harness.
The design and its threat model follow a survey of attacks on and defenses for models
that learn at inference (in `docs/research/`), including the 2026 result that test-time
training can strip safety guardrails and that a private-probe drift detector is the
defense that holds.

## The two domains

**Text.** Byte-level BPE, wikitext-103 (or fineweb-edu). Next-token prediction. Recall
is measured directly with MQAR probes, and every checkpoint reports its held-out loss,
its held-out loss with the memory disabled (the difference is the value of the memory),
its MQAR accuracy, and the histogram of learned write rates.

**Physics.** A 2D point mass with a friction coefficient the model never observes. The
input is `[observation, action, reset flag]`; the target is the next observation delta.
Several episodes with different friction are packed into one sequence, so the model has
to infer the latent dynamics into its fast weights within each episode. Sessions report
the three-way comparison on one trajectory: the base model with no memory, the session
with its memory frozen, and the session learning online. On a laptop the adaptive model
reaches an MSE of 0.0008 where the same model with its memory disabled sits at 1.04.

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
was actually fed. `plastic train --adversarial` meta-trains the write gate against
that attacker. `plastic sleep` consolidates what sessions learned into the slow
weights, accepted only if the canaries hold.

## Usage

```bash
# data and training
uv run plastic data prepare --corpus wikitext --out artifacts/data/wikitext
uv run plastic train text --data artifacts/data/wikitext --steps 3000 --device mps
uv run plastic train physics --steps 3000 --layers 3 --device mps
uv run plastic models

# a GPU run on Hugging Face Jobs (exports the working tree, no push needed)
scripts/hf_jobs/launch_text.sh l4x1 6000 BATCH=16 MODEL_ID=lm_wikitext_l4
hf jobs logs -f dmontgomery40/<job_id>
hf buckets sync hf://buckets/dmontgomery40/plastic-runs/artifacts/models/<id> artifacts/models/<id>

# calibrate the harness, then open a session
uv run plastic calibrate <model_id> --data artifacts/data/wikitext
uv run plastic session new --model <model_id>
uv run plastic chat <session_id> "your prompt here"
uv run plastic physics <session_id> --steps 256 --mu 0.12

# attack and consolidate
uv run plastic redteam <model_id> --data artifacts/data/wikitext --record
uv run plastic sleep <model_id> --core artifacts/data/wikitext
```

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

## Honest limits

- About 6M parameters on wikitext. Generations are fluent-ish, not coherent; this
  project demonstrates adaptation, recall, and the safety harness, not language quality.
- The harness's guarantees are first-order and probe-based. An adaptive attacker can
  still find directions the calibrated probes do not cover; the red team measures that
  residual rather than hiding it.
- The novelty is modest and stated as such: training the write gate for update safety
  against an adaptive attacker, using the delta rule's exact per-token write pressure
  as a signal, and projecting a TTT layer's state delta against a canary gradient. The
  underlying pieces (TTT layers, delta rule, canaries, A-GEM projection, adversarial
  training) are all prior work.

## License

See `LICENSE`. Research and educational use.
