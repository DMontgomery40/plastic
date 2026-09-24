# plastic

Test-time-training models with a transactional safety harness, and a study of what they
can learn lastingly from experience.

[Source on GitHub](https://github.com/DMontgomery40/plastic) ·
[models on Hugging Face](https://huggingface.co/dmontgomery40/plastic) ·
[live Space](https://huggingface.co/spaces/dmontgomery40/plastic)

`plastic` studies models that learn while they read: their memory is a set of fast
weights that take gradient steps on a self-supervised objective as tokens arrive, so
context is compressed into weights instead of a growing cache. Three substrates share
one harness:

- **PlasticCore**, a from-scratch model of about 6M parameters: a selective state-space
  recurrence plus a delta-rule fast-weight memory, meta-trained end to end through its
  inner loop. The same block stack serves next-token prediction on text and a
  hidden-friction control task where the only way to predict well is to infer the
  latent dynamics into the fast weights.
- **A pretrained TTT-MLP chat model**: the 760M TTT-MLP of Sun et al. (the RetentionLabs
  conversion of the released weights), whose sequence layer is a two-layer MLP updated
  by gradient steps on each mini-batch of 16 tokens, chat fine-tuned here for 250 steps
  on 8,000 SmolTalk conversations.
- **A coordinate block** (experimental): a selective recurrence whose nonlinear fast
  weights change the coordinates of its own state, studied on a physics mechanism
  testbed.

A Qwen model is available as an observational comparison; it has no
test-time-training layer.

Learning at inference is a security surface: every input is a gradient step, so a
hostile input is not just a bad answer, it is a bad thing learned. `plastic` wraps
the inner loop in a transactional harness. Each chunk of tokens is a transaction:
attempt the update, measure it, then commit, roll it back, scale it down, project it,
or refuse it. Every signal the harness reads comes from the model itself (loss,
surprise, the write rate the model chose, the size and curvature of the weight change,
how it moves a set of probe texts), never from pattern-matching the input. Sessions
are persisted and branchable, like version control for the plastic weights.

This is a research sandbox, not a product. It runs in plain PyTorch on an Apple Silicon
laptop or a single cloud GPU, and is honest about what it does and does not show.

## Where the research stands

The question is whether experience can change a model's weights so that it does better
on situations it has not seen, judged after the conversation and the temporary
fast-weight state are gone, with the benefit disappearing when the weights are
restored, and whether wrong lessons can be refused. The learning contracts measure
exactly that; [current status](docs/research/current-status.md) has the detail.

- On a text rule task, 20 gradient steps on the TTT chat model's initial fast weights
  from 17 worked lessons raised exact answers after one worked example from 0.31 to 1.0
  on held-out compositions, measured after a reset; restoring the weights removed the
  effect. One seed. The gradient runs through the model's own inner loop, so this is the
  TTT training objective applied online, a known mechanism. An untrained template
  copier, which never reads the rule names, is also exact on every answer after the
  first worked example, so this shows faster adaptation to the demonstration format,
  not knowledge of the named rules; only the first answer of an episode, where there is
  nothing to copy, can show that. On that first answer, lessons that pair every rule name
  with the wrong decoration raise the scores as much as the true lessons, so no stored
  rule knowledge is shown ([corrected runs](docs/research/results/text-rules-2026-09-24/README.md)).
  The update was a supervised training step outside the
  chat transaction path. Two of that report's readings (a refused poison and rule
  content stored only for trained compositions) were not supported as published; the
  archive records the corrections.
- Sleep, which consolidates session learning into the slow weights, retained none of
  24 facts taught once each across three seeds; the only transfers were planted
  falsehoods. It is paused on facts.
- The coordinate block's fast path adapts within an episode and matches the delta-rule
  baseline at lower cost, on one seed; lasting learning has not been tested on it.

Next: replicate the text result on further seeds and splits, then compare Sleep's
operators with it on the same stream.

## Install and run

```bash
uv sync --extra dev --extra pretrained   # pretrained: transformers, for the TTT and Qwen backends; add --extra jev for Jev grading
npm --prefix dashboard ci
uv run pytest                            # the test suite
uv run plastic --help
./start.sh                               # API on :13579, dashboard on :5173
```

Python 3.12, PyTorch 2.12 or newer. MPS on Apple Silicon, CUDA on Hugging Face Jobs.

## The from-scratch model (PlasticCore)

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

This is a delta-rule fast-weight memory in the family of TTT-Linear (Sun et al., 2024),
with Gated DeltaNet's forget gate and a normalized readout. It is not an exact
reproduction of TTT-Linear: the inner model, the normalization inside the loss and the
initialization differ. It has a chunk-parallel form for training (an exact unit-lower-triangular
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

## The pretrained TTT chat model

`plastic/backends/ttt_lm/` runs the 760M TTT-MLP under the same harness. Its inner
loop's reconstruction error (surprise), its learned per-token step size and the exact
per-token write norm are the transaction signals, and freezing holds the fast weights
exactly. The chat checkpoint learned the chat format and makes frequent factual errors
([evaluation](docs/research/results/chat-eval-2026-09-23/README.md)); it is not the
hosted model.

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
explicitly resumed. A candidate that scaling or projection changed after the decision is
checked again against the canary and drift limits before it is committed, and rolled
back if it fails them: a smaller or projected step is not guaranteed to do less damage. Thresholds are calibrated on a benign held-out stream to a requested
false-positive target, using split-conformal order statistics that report the rate the
sample size can actually support; a calibrated target is not a measured policy-level
false-positive rate.

There is no regex, no keyword list, and no string inspection anywhere in the harness.
The design and its threat model follow a survey of attacks on and defenses for models
that learn at inference (in `docs/research/`), including the 2026 result that test-time
training can strip safety guardrails, and that paper's private-holdout drift detector,
which works only while its holdout stays secret.

These signals measure how the model responds to an update; none is evidence that a new
statement is true. A genuine correction can be surprising and a fluent falsehood can look
ordinary, and a real rule change and an attacker's replacement that arrive as identical
examples cannot be told apart from the stream alone. The shipped chat canaries are 12
public true statements, 12 public contradictions and three repetition probes: a small,
public specification of what to preserve, not protection for behavior outside it.

## Domains and testbeds

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
That shows the network depends on its memory; it is not a competitive system
identifier. A causal scalar estimator that knows the simulator's equation family
reaches 2.8e-5 MSE with no training. Physics is an internal benchmark; the public
playground serves text.

**Learning testbeds.** The learning contracts use two tasks built so that lasting
learning can be told apart from temporary adaptation. A physics mechanism testbed
(six hidden mechanisms, held-out combinations and intervention policies) serves the
from-scratch learners. A text rule task (named decorations of a word list, taught
through worked examples and tested on held-out ordered pairs) serves the TTT chat model.

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
that attacker. `plastic sleep` consolidates accepted session learning into the slow
weights behind a damage gate (held-out loss, reply collapse, canaries where installed;
it has no retention or contamination check). Its product rule learns only from turns
whose every learning chunk the harness accepted; a turn with any rolled-back chunk is
excluded. The one exception is deliberate: the Sleep experiments' ungated control
(provenance "all") consumes refused turns to measure what that rule buys. For the
TTT backend it implements replay, fast-state distillation, anchoring and generated
dreams; anchoring and distillation read a session's saved state, which still carries
the writes of any excluded turn in that session. See where the research stands above for
what these methods have and have not retained.

## Usage

```bash
# data and training
uv run plastic data prepare --corpus wikitext --out artifacts/data/wikitext
uv run plastic train text --data artifacts/data/wikitext --steps 3000 --device mps
uv run plastic train physics --steps 3000 --layers 3 --device mps
uv run plastic models

# a GPU run on Hugging Face Jobs (exports committed HEAD; uncommitted edits are not included)
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

# the learning contracts
uv run python -m scripts.experiments.transfer_contract --model-id phys_mps_3k --out <dir>
uv run python -m scripts.experiments.text_contract_report --checkpoint <ttt chat checkpoint dir> --out <dir> --unstated-rules
```

## Layout

```
plastic/
  model/        selective scan, gated delta rule, chunk rule, fast-weight memory, coordinate block, blocks, models, state
  backends/     the harness's model backends: PlasticCore, the pretrained TTT-MLP, Qwen (observational)
  data/         wikitext/fineweb pipeline, MQAR recall probes, hidden-mu physics, mechanism testbed, text rules
  train/        outer loop, LR schedule, Muon/AdamW split, adversarial write-gate loss
  harness/      signals, robust stats, canaries, Fisher, projection, policy, calibration, transaction runner
  eval/         the learning contracts and their learners
  session/      persisted, branchable sessions for both domains
  redteam/      token-path attacker validated through the harness
  sleep/        consolidation into the slow weights behind a damage gate
  api/          FastAPI service over the artifact store
  store.py      models, sessions, transactions, forks, signatures
  cli.py        the `plastic` command
dashboard/      React playground (chat, signals, sessions) and the Sleep and learning observatory
scripts/        Hugging Face Jobs launchers, training, experiments
docs/           design specs, research notes and surveys, archived results
```

## Honest limits

- PlasticCore is about 6M parameters on wikitext. Its generations are fluent-ish, not
  coherent; it demonstrates adaptation, recall, and the safety harness, not language
  quality. The TTT chat model is small (760M), briefly fine-tuned and often wrong on
  facts, so claims about the meaning of its answers are qualified.
- One lasting-learning result so far, on one seed, from a known mechanism, on a task a
  template copier also solves after one example; consolidation by Sleep, protection
  against wrong lessons, and the two together through one operational path are not yet
  demonstrated.
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
