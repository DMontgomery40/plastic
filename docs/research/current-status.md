# Current research status

Updated 23 September 2026. This page describes active work. Dated reports preserve
experiment history and do not override this direction.

## Active investigation

Build useful chat on a pretrained, gradient-updated TTT-MLP model, with its own
inner-learning signals available to the transactional harness. The 760M base model
is undergoing chat fine-tuning. Evaluation will compare held-out assistant loss
and sampled answers through the transaction path. The backend implementation and
recorded checks are described in the [TTT backend note](2026-09-23-ttt-backend.md).

The backend can process and persist base-model sessions. That is distinct from a
verified chat-tuned checkpoint. Switching the public model awaits verification of
the fine-tuned checkpoint and its affected user flow. The new playground is deployed
with Qwen in observational mode; its source identity is recorded in
each host's `source_snapshot.json`. Qwen is a comparison backend, not the active
TTT chat target.

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
