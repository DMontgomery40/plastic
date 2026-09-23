# Sleep: consolidating accepted fast-weight learning into slow weights

Status: research proposal and build plan, 2026-09-23. Nothing below is implemented for the
TTT backend yet; `plastic/sleep/consolidate.py` is a toy-model prototype that was never
evaluated for retained improvement. Measured results will be added as they exist.

## Why this is the load-bearing piece

In the playground the TTT fast weights (per layer `W1, b1, W2, b2`) learn within a session and
the harness decides, per 16-token chunk, whether that learning is kept. At the end of the
session the fast weights are discarded: the next session restarts from the checkpoint's
learned initial fast weights `W0`. Without a consolidation step, nothing the harness accepted
ever changes the model. The system would be a policed scratchpad, not a model that improves.
Sleep is the step where accepted session learning becomes a durable change to slow weights,
under the same canary gate, producing a new model version.

## Research readiness note (checked 2026-09-23)

Local material read: the research briefing and corrections, the literature survey sections on
Titans, LaCT, Nested Learning/HOPE and the 2026 successors, the TTT backend note, the safety
review's continual-poisoning sections, and `plastic/sleep/consolidate.py` in full.

Primary sources checked today (arXiv abstract or HTML pages; versions as listed):

| Source | What it establishes for us |
| --- | --- |
| Behrouz, Hashemi, Javanmard, Mirrokni, *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*, [arXiv:2606.03979](https://arxiv.org/abs/2606.03979) v1 2026-06-02, v2 2026-07-10 | The closest mechanism. "Knowledge Seeding": fast-updating MLP blocks of a Continuum Memory System are distilled into slower blocks of the same model (new low-rank experts) with on-policy distillation plus RL imitation on self-generated data; triggered on the slow block's update period. "Dreaming": RL-selected synthetic curricula, LoRA SFT, binary improvement reward. Evaluated on Llama-3B/8B, Qwen3-1.7B/8B and HOPE; knowledge incorporation SQuAD 48.9 vs SEAL 46.7; fine-tune with no consolidation 33.4 vs 48.1 on single passage. |
| Lee, McLeish, Goldstein, Fanti, *Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference*, [arXiv:2605.26099](https://arxiv.org/abs/2605.26099) v2 2026-05-27 | A different meaning of "sleep": when the KV window fills, run N offline recurrent passes to write evicted context into gated-DeltaNet-style fast weights, then clear the cache. Fast weights stay fast; no slow-weight change, no cross-session retention measured. Useful as a contrast, not our target. |
| Song et al., *Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training*, [arXiv:2607.00368](https://arxiv.org/abs/2607.00368) 2026-07-01 | How to evaluate. One-step LoRA updates cut support and answer loss while free-form recall stayed at zero. Requires later recall, paraphrase robustness, locality, conflict handling, action after the support context is removed, matched explicit-memory baselines, and an evidence ladder (stream adaptation, bridge internalization, deployment-time learning). |
| Dennis, Shabahang, Guo, Patil, *Beyond Inference-Only Deployment: Comparing Weight-Based Consolidation Against Cascading Compaction*, [arXiv:2605.24657](https://arxiv.org/abs/2605.24657) 2026-05 | Nightly reflection plus LoRA fine-tuning on one consumer GPU retained 80.4% of session knowledge vs 36.8% for context compaction on 1,146 questions; median per-token CE tracked accuracy (r 0.99) while the mean was misleading. Evidence that weight consolidation of session content is worth doing at small scale. |
| Zweiger, Pari, Guo, Kim et al., *Self-Adapting Language Models (SEAL)*, [arXiv:2506.10943](https://arxiv.org/abs/2506.10943), NeurIPS 2025 | Model writes its own fine-tuning "self-edits"; RL on downstream gain. Reports catastrophic forgetting as an open problem; each self-edit costs a fine-tune plus eval. |
| Lin et al. (Meta/Berkeley), *Continual Learning via Sparse Memory Finetuning*, [arXiv:2510.15103](https://arxiv.org/abs/2510.15103) 2025-10-16; Goyal et al., *Improving Sparse Memory Finetuning*, [arXiv:2604.05248](https://arxiv.org/abs/2604.05248) 2026-04-06 | Update only the parameters most activated by the new knowledge relative to background data: NaturalQuestions F1 drop 11% vs 89% full fine-tune vs 71% LoRA. The principle transfers to our choice of *which* slow parameters to touch. |
| Wang et al., *Learning What to Remember: Test-Time Training via Context Distillation*, [arXiv:2608.01672](https://arxiv.org/abs/2608.01672) 2026-08-03 | Long-window teacher supervises a short-window student's fast weights; existing MLP weights as fast weights. No cross-session consolidation. |
| Behrouz et al., *Nested Learning* (HOPE), [arXiv:2512.24695](https://arxiv.org/abs/2512.24695); Ma et al., *Elastic TTT*, [arXiv:2604.07350](https://arxiv.org/abs/2604.07350) | Multi-timescale update levels; Fisher-weighted prior toward an anchor as the forgetting control. Both already in the survey. |

Search coverage: sleep/consolidation for LLMs, fast-to-slow weight transfer, test-time
learning surveys, SEAL follow-ups, sleep-time compute for agents, sparse memory finetuning.
Not found: any work that consolidates *TTT-layer* fast weights (Sun et al. layer) into the
layer's learned initial fast weights `W0`, or that gates consolidation on an external
transactional accept/reject record. Absence in a bounded search is not a priority claim.

Closest known mechanism: Knowledge Seeding (2606.03979). Distinction under investigation:
we consolidate only *harness-accepted* fast-weight learning, into the TTT layer's own
initial fast weights first, with an explicit accepted/rejected provenance and a canary gate,
and we measure retention behaviorally per 2607.00368. No RL, no self-generated curriculum in
the first version.

Unresolved assumptions: (a) 28.5M initial fast-weight parameters have enough capacity to
carry session facts across a reset; (b) session fast weights are transferable rather than
context-specific; (c) the chat checkpoint's harness signals will be calibrated well enough
that "accepted" means something. (a) and (b) are what the experiment measures; (c) is
handled by calibrating the chat model before the sleep runs.

## Design gate (AGENTS.md)

| Field | Sleep on the TTT backend |
| --- | --- |
| Inner objective (wake) | TTT layer reconstruction loss per 16-token mini-batch; unchanged. |
| Fast variables (wake) | `W1, b1, W2, b2` per layer plus pending mini-batch gradient `G`; the state the harness measures in effective coordinates. |
| Slow variables (sleep) | Target set selectable: `w0` = the TTT layers' initial fast weights (16 heads × (96×384 + 384 + 384×96 + 96) × 24 layers = 28.5M params, 3.7% of the model); `w0+lora` = `w0` plus LoRA on the SwiGLU down projections; `all` = every parameter (SFT-style). |
| Update rule (sleep) | Offline AdamW fine-tune of the target set; loss per method below. |
| Outer gradient path | Ordinary backprop through the full model from a reset fast-weight state; the inner loop runs inside the forward as in SFT. |
| Carried state | None across sleep; sleep consumes committed session states and traces, and emits a child checkpoint with a lineage record. Sessions of the parent are untouched. |
| Causal target availability | All targets are past data (accepted traces, saved committed fast weights, replay corpus). |
| Transaction boundary | Sleep is one transaction: canaries and held-out metrics are scored before and after from a zero state. A run whose locality checks pass is `accepted`; one whose checks fail is `rejected` and leaves only a report; one with no locality measurement available (no replay corpus, no canary suite) registers its child as `accepted_unmeasured`, an exploratory result that is never presented as verified. Nonfinite measurements fail their check. |

Provenance rule: sleep reads only chat turns whose every chunk's decision was `commit`,
`scale` or `project`; turns with a rolled-back or read-only chunk are excluded, and the report
states how many tokens were excluded and why. Turns are keyed by their transaction-index range
(positions restart after a reset; indices do not); traces without that range are grouped by
position only when positions never restart, otherwise marked ambiguous and excluded. Exclusion
is a deterministic input rule. It is not a guarantee that rolled-back content cannot surface
after sleep: a rollback replays the chunk frozen, so activation and conv state still carried
it, and later accepted chunks (and the committed fast weights that `anchor` and `distill`
read) can depend on that influence. Recall of rolled-back content after sleep is therefore
measured against the parent and matched controls as a contamination outcome.

## Three methods (the options exposed in the UI)

1. **Replay fine-tune (`replay`).** Harvest accepted chat traces (prompt and completion), mix
   with a replay sample of the SFT corpus at a chosen ratio, fine-tune the target set with
   next-token cross-entropy on assistant tokens. Closest prior: nightly LoRA consolidation
   (2605.24657), SEAL without self-edits. Simplest, and the baseline the others must beat.
2. **Fast-weight distillation (`distill`).** Teacher: the same model with a session's committed
   fast-weight state loaded (what the fast learner knew at the end). Student: the model from a
   reset state, target set trainable. Loss: KL(teacher ‖ student) on the session's own text and
   on fresh replay text, so the student's *prior* absorbs what the fast weights learned rather
   than the raw text. Closest prior: Knowledge Seeding (2606.03979) without RL imitation and
   without adding experts. This is the mechanism-native option.
3. **Fast-weight anchoring (`anchor`).** No gradient: `W0 ← W0 + λ · mean_s (W_s − W0)` over the
   accepted sessions' final effective fast weights. Closest prior: Reptile-style meta-updates.
   Cheap, probably weak; kept as the honest control that tells us whether session fast weights
   are transferable at all.

Common controls: learning rate, steps, replay ratio, target set, canary tolerance, which
sessions (default: all sessions of the model with at least one accepted chunk), a seed. Each
run writes `sleep_report.json` under the child (or under `artifacts/sleep/<run>` if rejected).

## Evaluation: what "actually improves" must mean here

Following 2607.00368, a run is judged behaviorally, from a fresh session with no context:

- **Later recall.** Facts or preferences taught during the source sessions are asked again,
  verbatim and paraphrased, in a fresh session of the child model. Score by exact and
  normalized match on the answer span; report the count, not only a mean.
- **Locality.** Held-out SmolTalk assistant NLL (median per-token CE as well as mean, per
  2605.24657) and the coherence canary must not degrade beyond tolerance; the poison canary
  must not improve toward the poison.
- **Provenance.** A source session with rolled-back chunks: recall of their content after
  sleep is measured against the parent floor. Direct replay of rolled-back turns is excluded
  by construction; indirect carry-over through activation state is the quantity reported.
- **Matched baselines.** (i) Parent model, fresh session: the floor. (ii) Parent model with
  the trace pasted into context: the explicit-memory ceiling. (iii) The three methods on the
  same sessions. (iv) Replay fine-tune on *all* chunks including rolled-back ones: shows what
  the gate buys.

- **Boundary content.** The point of the harness is to decide from numeric evidence, not
  wording. The evaluation therefore includes prompts that look like what a keyword filter would
  flag but are harmless in substance (how cellulose, soap or aspirin is made in molecular terms;
  which household cleaners not to mix). Two questions: do the learner's signals on these turns
  differ from neutral turns (surprise, write norm, proposed change; recorded per group by
  `scripts/train/eval_ttt_chat.py`), and does sleep treat an accepted boundary turn like any other
  accepted turn (it should, by construction; the gate and canaries are the check). The list is
  kept benign on purpose: it is the false-positive side of the evaluation, and a harness that only
  passes neutral text has not been tested.

Evidence ladder: within-session learning is stream adaptation (already visible in the
playground); recall in a fresh session after sleep is deployment-time learning. Only the
second counts as "the model improved". A perplexity drop alone does not.

Falsifying results: recall no better than the parent floor for every method at a locality
cost within tolerance means the 28.5M `w0` target cannot carry session knowledge, and the
target must widen (`w0+lora`, `all`) or the claim is abandoned for this checkpoint. Recall
gains that vanish under paraphrase mean memorized surface form, not knowledge.

## Playground

Sessions screen, per model: a **Sleep** action with the method, target set, sessions and a
few numeric controls, a running state ("consolidating N sessions, M accepted chunks"), and a
result card: accepted or rejected, child model id, recall before/after, held-out NLL
before/after, canary deltas, tokens used and excluded. Model lineage is shown on the model
picker (parent → child). A **Recall check** action runs the probe questions against any model
in a fresh, disposable session so a person can see the difference, not only read a number.
Copy stays as state labels; the explanation lives in this note and the README.

## Build order

1. Backend hooks: export/import committed fast-weight state for a session (already saved by
   the store), a teacher forward with a loaded state, a target-set selector, held-out NLL.
2. `plastic/sleep/ttt.py`: the three methods behind one `SleepJob`; report schema; child
   registration with lineage; `plastic sleep` for TTT records.
3. Recall probe: a small file of taught facts per source session (the playground's chat
   traces plus an optional user-provided list) and the scorer.
4. API route and the Sessions screen actions; visible browser pass on the local TTT model.
5. First measured run on the chat checkpoint after SFT and calibration; results appended
   here and to current status, including negative ones.

Cost: methods 1 and 2 on the 760M model over MPS at a few hundred steps are minutes to an
hour; an A100 job is not needed for the first measurement.

## Measured so far

### 2026-09-23, dry run on the 760M base (not chat-tuned): mechanics only

Setup: local MPS, disposable store, one teaching session (3 turns, 172 accepted tokens, log-only)
stating a cat's name, a city and an instrument; 3 recall probes with paraphrases; `w0` target;
20 steps, seq 256, batch 2, replay ratio 0.5 from 16 SmolTalk conversations; held-out 8
conversations (887 assistant tokens); tolerance 0.5 nats so nothing would be rejected.

| Method | Time | Held-out NLL mean / median before → after | Recall verbatim | Recall paraphrase |
| --- | --- | --- | --- | --- |
| anchor λ=0.5 | 92 s | 2.085 / 1.422 → 2.051 / 1.379 | 0/3 → 0/3 | 0/3 → 1/3 |
| replay | 125 s | 2.085 / 1.422 → 1.605 / 0.829 | 0/3 → 0/3 | 0/3 → 0/3 |
| distill | 182 s | 2.085 / 1.422 → 1.626 / 0.875 | 0/3 → 0/3 | 0/3 → 0/3 |

Reading. This run verifies that all three methods execute end to end, register a loadable
child with lineage, and produce the report; it does not measure consolidation. The base model
cannot chat, so "recall" before is zero by construction and the large held-out NLL drop under
replay and distill is the model learning the chat *format* from the replay corpus, not the
facts. The replies after replay moved to the topic without the content ("The name of your
pet.", "The city you're in."). The single anchor paraphrase hit ("The name of my cat is named
Marlowe.") is one sample from a model that otherwise echoes the prompt; it is suggestive that
the moved `W0` carries session content, and nothing more until it is reproduced on the chat
checkpoint with matched controls. The chat checkpoint is the first real measurement.

### 2026-09-23, step-50 SFT checkpoint (of 250): first chat samples and boundary signals

`scripts/train/eval_ttt_chat.py` on the step-50 checkpoint (MPS, temperature 0.7, top-k 40, 80
tokens). Held-out assistant NLL: everyday-conversations 1.697 (4,647 tokens, 40 rows),
smol-magpie-ultra 1.425 (51,890 tokens, 40 rows). The chat format is learned (answers start,
stay on the question, end); content is often wrong at this stage ("The capital of France is La
Havilland"). Learner signals per prompt group, log-only, means over 8 prompts each:

| Group | surprise mean | chunk loss | write norm sum | proposed change sum |
| --- | --- | --- | --- | --- |
| neutral | 9.66 | 3.43 | 1.09e3 | 28.6 |
| boundary (benign chemistry/pharma wording) | 9.62 | 3.63 | 1.16e3 | 30.2 |

Reading: at this checkpoint the fast learner writes about as hard on boundary-worded prompts as
on neutral ones (ratios 1.00 to 1.07). That is the expected behavior of a wording-blind learner
and the baseline against which any later "boundary content writes harder / gets rolled back
more" claim has to be measured. Eight prompts per group is a smoke test, not a study.

## Hosted sleep on the Space (delivery requirement)

A hidden Sleep control on the public Space is not a delivered feature. Implemented in
`deploy/huggingface/app.py` and `pretrained.py` (2026-09-23), active as soon as the public model
is a TTT record:

1. The public gate derives its catalog from the store: the pinned model plus every model sleep
   derived from it (lineage order), and one demo session per model (`demo_text` for the root,
   `demo_<child>` for each child, created with the shared observational controls the first time
   the child is seen). Unrelated models and private sessions stay hidden.
2. `sleep` is advertised only when the public model has fast weights (backend `ttt`), decided
   from the store per request. Then `GET /api/sleep[/run]` and `POST /api/models/<root>/sleep`
   are open. The gate validates the visitor's body and forwards the bounded body it validated,
   never the visitor's body plus server defaults: anchor by default or replay, target `w0`,
   steps ≤ 10 (default 5), seq_len ≤ 256, batch 1, replay rows ≤ 8, held-out rows ≤ 2, up to 6
   short probes, sessions ⊆ the root model's own public sessions (default all of them). One run
   at a time; a second start is refused while one runs.
3. The playground's Sleep form on the shared demo offers exactly that set with the gate's
   defaults, and only the root model.
4. The store is ephemeral; children vanish on a Space restart.
5. Release path: publish the chat checkpoint under `text-chat/` of the model repo, pin its
   revision and digest in `TTT_CHAT`, build with `PUBLIC_MODEL=ttt`, push. `active_spec()` refuses
   to build or boot an unpinned spec.
6. Hardware: `cpu-basic` (2 vCPU) cannot host this honestly: a local anchor run mirroring it
   spent over 11 minutes in the "before" measurement alone (92 s for the whole run on MPS).
   David approved GPU hardware (2026-09-23); the image is CUDA-capable (46d0a06). The hardware
   change itself needs a token or a settings click this machine's token does not have.

### 2026-09-23, step-50 SFT checkpoint: matched-controls dry run (null baseline)

`scripts/experiments/sleep_controls.py`, MPS, 10 steps, target `w0`, seq 256, batch 2, 8 replay
rows, 4 held-out rows, greedy 24-token probes, 27 minutes. Verbatim recalled / n, p = paraphrase.

| Arm | taught | boundary | rolled (contamination) | general (locality) | held-out NLL | status |
| --- | --- | --- | --- | --- | --- | --- |
| floor (parent, fresh session) | 0/6 (p 0/6) | 0/2 | 0/2 | 1/5 (p 1/5) | | |
| ceiling (parent, teaching turns in session) | 5/6 (p 4–5/6) | 0/2 | | | | |
| anchor λ 0.5 | 0/6 (p 1/6) | 0/2 | 0/2 | 1/5 | 1.766 → 1.766 | accepted |
| replay, w0 | 0/6 (p 0/6) | 0/2 | 0/2 | 2/5 | 1.766 → 1.615 | accepted |
| distill, w0 | 0/6 (p 0/6) | 0/2 | 0/2 | 1/5 | 1.766 → 1.627 | accepted |
| ungated replay, w0 | 0/6 (p 0/6) | 0/2 | 0/2 | 2/5 | 1.766 → 1.620 | accepted |

Reading. The model uses its session (ceiling 5/6) but nothing any method wrote to `W0` in 10
steps survived a reset. After replay the fresh-session replies take the *form* of an answer
("Your cat is called 'Pinkie'", "Your sister's name is 'Mary'") without the content: the small
update taught the response pattern, not the facts. The held-out NLL drops under replay, distill
and ungated are the SFT replay corpus continuing to train a half-trained checkpoint, which is
why the locality gate passed; they are not consolidation evidence. Rolled-back facts stayed at
the floor in every arm including ungated, so at this scale the experiment cannot yet separate
gated from ungated: nothing was retained either way. Boundary probes scored 0/2 even in context
because their expected strings were too specific; they now expect the distinguishing token.

This is the null baseline. The chat checkpoint run sweeps steps {10, 40} × target {w0, all} for
replay and distill with anchor as the control; only a taught-recall gain above the floor with
rolled-back recall still at the floor and locality within tolerance counts as consolidation.
