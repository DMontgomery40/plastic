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
| Transaction boundary | Sleep is one transaction: canaries and held-out metrics are scored before and after from a zero state; the child is registered only if the gate passes. Rejected runs leave a report and nothing else. |

Provenance rule: sleep reads only chunks whose transaction decision was `commit`, `scale` or
`project` (accepted metrics nonzero), never rolled-back or read-only chunks. The report states
how many tokens were excluded and why.

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
- **Provenance.** A source session with rolled-back chunks: their content must not be
  recalled after sleep. This is the harness contract carried into sleep.
- **Matched baselines.** (i) Parent model, fresh session: the floor. (ii) Parent model with
  the trace pasted into context: the explicit-memory ceiling. (iii) The three methods on the
  same sessions. (iv) Replay fine-tune on *all* chunks including rolled-back ones: shows what
  the gate buys.

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
