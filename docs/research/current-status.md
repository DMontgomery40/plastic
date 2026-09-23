# Current research status

Updated 23 September 2026. This page describes active work. Dated reports preserve
experiment history and do not override this direction.

For reproduction commands, available checkpoints and ways to help, see
[Contributing to the research](contributing-research.md). Public experiment
[outputs](results/sleep-2026-09-23/README.md) include per-probe replies and reports.

## Active investigation

Build useful chat on a pretrained, gradient-updated TTT-MLP model, with its own
inner-learning signals available to the transactional harness. Chat fine-tuning of
the 760M base model is complete (250 steps, 23 September 2026); the final checkpoint
is a release candidate undergoing evaluation: held-out assistant loss, sampled answers
through the transaction path, calibration, and the Sleep controls. The first record, the
[chat evaluation at three temperatures](results/chat-eval-2026-09-23/README.md), shows the chat
format learned, the lowest measured within-reply repetition at temperature 0.7 (16 replies), and frequent factual errors. Qwen remains the
default hosted chat model until the evaluation is complete. The backend implementation and
recorded checks are described in the [TTT backend note](2026-09-23-ttt-backend.md).

The backend can process and persist base-model sessions. That is distinct from a
verified chat-tuned checkpoint. Switching the public model awaits verification of
the fine-tuned checkpoint and its affected user flow. The playground offers Qwen in
observational mode and the published 6.85M PlasticCore WikiText model with the
transactional harness enabled. PlasticCore is a text-continuation baseline, not
chat-tuned. Its shared session learns from prompts and generated tokens, and a
CUSUM alarm rolls back the current chunk without latching future chunks read-only.
These controls make the baseline available for exploration; they do not establish
protection or useful retention. Source identity is recorded in each host's
`source_snapshot.json`. Qwen remains a comparison backend, not the active TTT chat target.

**Sleep is paused for reassessment (23 September 2026).** On the final chat checkpoint, 15
consolidation attempts across three seeds passed the gate 13 times and retained none of 24 taught
facts; the only transfers were verbatim planted sentences, and selection by surprise chose a
planted falsehood. The probes measured phrase recall and each fact was taught once as one sentence.
The queued comparisons were stopped. The [reassessment memo](2026-09-23-reassessment-concepts-not-phrases.md)
(draft, source-checked) sets out the concept-level measurement battery and the recurrence-and-consistency
criterion to test, and the smallest experiment that could falsify it; no new run before it is agreed.
Details in the [Sleep note](2026-09-23-sleep-consolidation.md).

In parallel, **Sleep** tests whether accepted session learning can become a durable
checkpoint change. Replay, fast-state distillation, anchoring and generated Dream
distillation are implemented; their current objectives and controls are in the
[Sleep note](2026-09-23-sleep-consolidation.md). Five tested raw-turn replay settings
on an intermediate step-100 checkpoint retained 0/6 taught facts. The first study-set
run also retained 0/6, with higher expected-answer likelihood. Aggressive earlier
runs showed why falling held-out loss alone can miss repeated-answer collapse.
These results motivate new comparisons; they do not close the research question.

Current Dream generation quotes an accepted turn to a frozen session teacher and
trains a reset student without the quote. The first
[conditioned run](results/sleep-2026-09-23/dream_step100_w0_conditioned/sleep_controls.md)
also retained 0/6 taught facts; its locality checks passed. The saved free-form
predecessor is a different method. Clean cross-session retention and a protective benefit from
provenance filtering remain unestablished. The intermediate checkpoints behind
the historical tables are not public; the contributor guide distinguishes those
tables from the protocol runnable on public base weights.

The [importance-weighting proposal](2026-09-23-importance-weighting-proposal.md)
extends the experiment to 24 taught facts, unseen probe wording and optional
planted contradictions. Per-token Dream weights are implemented but unmeasured;
uniform remains the default. Adaptive sampling and classifier comparisons remain
research proposals. Historical six-fact results do not evaluate these additions.

Compare retained learning, frozen processing, and complete snapshot restoration
on matched inputs and starting states. Try guarded comparisons early; calibration
and prior signal separation are not prerequisites for exploratory runs. Claims of
useful adaptation or protection require measured outcomes. Freezing memory and
restoring an entire turn remain different interventions.

## Measurements and interpretation

The TTT backend reports inner reconstruction-error norms, effective inner step
sizes, per-token update-contribution norms, prediction loss, and fast-weight
changes. `surprise` is an error norm, not the squared reconstruction loss itself.
Canary alignment uses a probe-loss gradient with respect to fast weights. A
state-change norm is not a gradient difference; a gradient comparison must identify
both gradients and their base points. Fisher measurements remain unavailable here.

The currently published Qwen adapter lacks the inner-memory surprise, write-rate,
and write-norm measurements. Missing measurements stay missing. Its earlier
loss/state experiments do not establish inner-learning detection or protection.
Keep proposed measurements separate from the changes accepted by the harness.

Weak baseline competence limits conclusions drawn from output meaning. It does
not stop alpha-stage state, weight, gradient, retention, or transaction research.
The separate requirement for useful chat is now being addressed through TTT-model
fine-tuning.

## Research scope and interface

The original PlasticCore experiments and nonlinear coordinate proposal remain
separate research tracks; adding the pretrained backend does not establish their
architectural equivalence or resolve the proposed coupling. Physics stays as an
internal benchmark and CLI workflow, outside the public playground.

The playground has Chat, Signals, and Sessions screens. Fast-weight signals are
available with the TTT backend; the hosted Qwen model exposes its own supported
measurements. Desktop browser checks cover the local TTT and hosted Qwen paths;
responsive and failure-recovery checks remain specific to each release. Keep concise controls, measurements, and actionable status in the UI;
put implementation and experiment detail in research documentation.

Publish methods, useful evidence, and reproducible findings openly. Preserve
compact evidence, including useful negative results. Private coordination, session
data, and incidental artifacts stay out of public source. Keep active training and
checkpoint inputs available; archive useful inactive bulk artifacts separately.
