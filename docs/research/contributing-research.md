# Contributing to the research

Plastic studies whether a model can learn during use, retain useful changes across
sessions, and reject damaging updates using numerical evidence from the learner.
The current experiment is **Sleep**: transfer a pretrained TTT-MLP session's learning
into a new checkpoint, then test it from a fresh session.

Start with the [current status](current-status.md),
[Sleep methods and results](2026-09-23-sleep-consolidation.md), and
[saved experiment outputs](results/sleep-2026-09-23/README.md). The outputs include
individual answers and rejected runs, so contributions can begin with analysis
without running a model. The [backend note](2026-09-23-ttt-backend.md) explains the
inner learner; the [research briefing](README.md) covers the other research tracks.

## What is established, and what remains open

The TTT backend, four Sleep methods, provenance filtering, child checkpoints and
before/after measurements are implemented. Five tested raw-turn replay settings on
the intermediate step-100 checkpoint retained **0/6 taught facts**. More aggressive
settings produced repeated answers while held-out loss improved. Study-set training
also scored 0/6, with a rise in expected-answer likelihood. These are useful negative
results, not evidence that cross-session learning is impossible.

The current Dream method conditions a frozen session teacher on an accepted user
turn and trains a reset student on generated replies without that quoted turn.
A [conditioned-Dream run](results/sleep-2026-09-23/dream_step100_w0_conditioned/sleep_controls.md)
also retained 0/6 taught facts. The archive includes both free-form and conditioned
versions; match each result to its recorded configuration and source. We have not
yet demonstrated reliable consolidation or a protective
advantage over unfiltered replay. Weak chat competence limits behavioral conclusions
while leaving state, gradient and transaction experiments useful.

## Run the protocol

Use Python 3.12+, `uv`, and an MPS or CUDA device for practical model experiments.
CPU is supported but slow. From the repository root:

```bash
uv sync --extra dev --extra pretrained
uv run python -m scripts.experiments.sleep_controls --help
uv run plastic sleep --help
```

You need a local TTT-MLP checkpoint directory, including its tokenizer. The public
[760M base checkpoint](https://huggingface.co/RetentionLabs/TTT-MLP-760M-Base-Pile-8k)
can exercise the protocol now; it is not chat-tuned and does not reproduce the SFT
tables. This downloads the base revision checked on 23 September 2026:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "RetentionLabs/TTT-MLP-760M-Base-Pile-8k",
    revision="4a347bff5ebef31d5f89234f04a77cf6d16e7369",
    local_dir="artifacts/checkpoints/ttt-mlp-760m-base",
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
)
PY
```

The step-50 and step-100 SFT checkpoints used in the saved results are **not
published**. Their full digests are in the result JSON. Exact reruns of those tables
require those weights and the execution version; newer code changes the method.
The planned chat release will be pinned in
[`TTT_CHAT`](../../deploy/huggingface/pretrained.py); empty revision/digest fields
mean it has not been released. Do not substitute the hosted Qwen comparison model.

This example runs the current synthetic-fact protocol, with all arms starting from
the same parent. Choose a **new output directory**: the script removes an existing
`store/` inside `--out` before it runs. It also loads SmolTalk replay/held-out data.

```bash
uv run python -m scripts.experiments.sleep_controls \
  --checkpoint artifacts/checkpoints/ttt-mlp-760m-base \
  --out artifacts/experiments/my-sleep-comparison \
  --device mps --seed 0 --facts 6 \
  --arms floor,ceiling,anchor,replay,distill,dream,ungated \
  --target w0 --steps 20 --lr 3e-5 --seq-len 256 \
  --replay-ratio 0.8 --batch-size 5 \
  --replay-rows 16 --heldout-rows 8 --max-new-tokens 32 \
  --replay-revision 5feaf2fd3ffca7c237fc38d1861bc30365d48ffa \
  --session-loss all --prompt-loss-weight 1.0 --augment none \
  --teach-temperature 0.7 --dream-temperature 0.7 --dream-token-weighting uniform
```

Change `--device mps` to `cuda` for a CUDA machine. This is an exploratory example,
not an optimized recipe. For a study-set comparison, run again into a different
directory with `--augment study`, keeping the other settings fixed. Each fact then
appears in six templated teaching turns. `--arms` can select a smaller comparison.
The command pins both model and SmolTalk revisions. The historical archive did not
record a dataset revision, so this pin cannot recover its exact sampled data.

The expanded protocol defaults to `--facts 24`; `--poison` also teaches four planted
world-knowledge contradictions. Its paraphrase probes use wording held out of the
study set. These changes, and optional per-token Dream weighting, are described in
the [importance-weighting proposal](2026-09-23-importance-weighting-proposal.md).
The example explicitly keeps six facts and uniform Dream weights; it still uses
the current probes, not the historical protocol. No calibration or preregistration
is required to explore a hypothesis.

## Read the controls and measurements

| Arm or measurement | Meaning |
| --- | --- |
| `floor` | Parent checkpoint, fresh state for each question. |
| `ceiling` | Parent reteaches the eight raw teaching statements before each question. This tests in-session access; it is not a matched study-set ceiling when `--augment study` is used. |
| `anchor`, `replay`, `distill`, `dream` | Separate Sleep runs from the same parent; see the method definitions. |
| `ungated` | Replay includes rolled-back/read-only and flagged turns. The final Sleep locality gate still applies. |
| Probe groups | This example uses six taught facts, two benign chemistry facts, two forced-rollback facts, four poison-answer probes and seven general questions. The poison facts are taught only with `--poison`; their probes also measure the baseline without it. Each question has a paraphrase. Historical tables used five general questions and no poison group. |
| Recall | Normalized answer substring hits and exact matches, with separate verbatim/paraphrase counts. Inspect the replies: a repeated sentence containing a name can score a false-looking hit. |
| `mean_answer_logprob` | Mean of the verbatim probes' per-token expected-answer log probabilities, in nats/token. A likelihood rise alone does not locate stored knowledge or prove a readout failure. |
| `max_cluster_share` | Largest group of replies sharing a normalized first-12-word prefix, divided by all verbatim and paraphrase replies. Missing means unavailable. |
| Held-out NLL | Mean and median assistant-token loss on held-out SmolTalk; lower is better on that sample, not proof of retention. |
| Gate / status | Acceptance means the available locality checks passed. It does not require a recall gain. `accepted_unmeasured` means no locality check could run. |

The rolled-back session uses a deliberately forced policy. It tests the provenance
path, not the harness's ability to detect an attack. Observational teaching commits
everything; those commits do not establish that the content is safe to learn.
Current Sleep defaults additionally exclude accepted turns marked for intervention;
the `ungated` control explicitly includes them. Inspect the selected-turn and
source-session counts: these defaults differ from the historical runs, and a mixed
session's state can still contain indirect influence from excluded turns.
The 50% and 80% replay configurations use batches 2 and 5 respectively, so comparing
them changes batch size too. For distill/Dream, replay share changes row sampling;
mean session KL and mean replay CE are added with equal coefficients.

## Share a useful result

Open a [research issue](https://github.com/DMontgomery40/plastic/issues/new) with the
question, command, result and your interpretation, or submit a PR under
`docs/research/results/<experiment>/`. A compact result should include:

- Exact Git commit and any source patch; checkpoint repository, revision and full
  digest; dataset revision/subset/split, sampled rows and seed; device, dtype and
  Python/PyTorch/Transformers versions.
- Full command and realized settings, including teaching/Dream temperatures,
  actual replay/session batch counts and loss terms.
- `sleep_controls.json`, `sleep_controls.md` and each `sleep_<arm>/sleep_report.json`:
  denominators, before/after answers, exclusions, gates and Dream selections belong
  beside the summary, including null and rejected outcomes.
- What the controls rule out, what remains confounded, and one next comparison
  that could disprove your explanation.

The reports record a short execution commit (with a dirty marker when applicable),
but do not capture every item above automatically; supply a manifest for the rest.
Older reports predate that field, and their archive labels reconstructed source
versions as approximate. Their standalone `floor` arm lacks answer likelihood;
use each Sleep report's `before.recall` for paired historical comparisons. Current
code also records floor likelihood and cluster share.
Keep synthetic results public, but do not upload personal chat traces or the entire
`store/` (sessions and full child checkpoints). Publish model weights separately
with a retrieval location and digest when they are ready to reproduce a result.

## Useful contributions now

| Question | A bounded contribution | Start in |
| --- | --- | --- |
| Does fast state add anything beyond quoting the fact? | Compare conditioned Dream teachers with session state versus reset state, holding prompts, sampled targets and training budget fixed. The current `fastweight_gain` is diagnostic, not this ablation. | [`dream.py`](../../plastic/sleep/dream.py) |
| Does a child retain the fact, or just a phrase? | Add unseen paraphrases, conflicting facts and answer scoring cases; retain per-question replies and a parent control. | [`sleep_controls.py`](../../scripts/experiments/sleep_controls.py), [`recall.py`](../../plastic/sleep/recall.py) |
| Does filtering improve what is learned? | Extend accepted/all-turn comparisons while measuring both useful retention and contamination; zero retention in both arms is inconclusive. | `harvest_sessions` and `sleep_ttt` in [`ttt.py`](../../plastic/sleep/ttt.py) |
| Can another machine reproduce the protocol? | Record a pinned checkpoint, environment, runtime and compact outputs; report device differences without claiming speed from CPU contract tests. | [Saved results](results/sleep-2026-09-23/README.md), [`test_sleep_ttt.py`](../../tests/test_sleep_ttt.py) |

For implementation changes, follow [AGENTS.md](../../AGENTS.md): preserve state and
transaction contracts, add regression coverage for the affected behavior, and run
the relevant suites. Keep active method descriptions current in place; put dated
results in the results directory. Contributors should not need private team notes
to understand or reproduce the public work.
