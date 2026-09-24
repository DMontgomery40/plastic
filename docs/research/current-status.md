# Current research status

Updated 24 September 2026. This page describes active work. Dated reports preserve
experiment history and do not override this direction.

For reproduction commands, available checkpoints and ways to help, see
[Contributing to the research](contributing-research.md). Contract reports are indexed in
[text-rules results](results/text-rules-2026-09-23/README.md) and
[mechanism-testbed results](results/contract-2026-09-23/README.md); the Sleep outputs are in
[Sleep results](results/sleep-2026-09-23/README.md).


## Active investigation

The question is whether accumulated experience can change a model's weights so that it does
better on situations it has not seen, judged after the conversation and the temporary fast-weight
state are gone, with the benefit disappearing when the weights are restored, and whether wrong
lessons can be refused. The measurement is the learning contract: the
[mechanism-testbed contract](../superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md)
for the from-scratch recurrent learners and the
[text rule contract](../superpowers/specs/2026-09-23-text-rule-contract.md) for the pretrained
TTT-MLP chat checkpoint (760M, chat fine-tuning finished 23 September).

**One lasting update has passed the contract's first test, on one seed.** On the text rule task
(decorations of a copied word list, rules not stated), 20 gradient steps on the TTT layers'
initial fast weights W0 from 17 worked lessons raised exact answers after a single worked example
from 0.31 to 1.0 on held-out compositions, measured after a reset, and restoring W0 removed the
effect. The gradient flows
through the model's own inner-loop updates, so this is the TTT training objective applied online
to a small stream (closest prior work: online meta-learning and the meta-learned initialisation of
end-to-end TTT), not a new mechanism. Review found that this update trained 11 of the 17 lesson
compositions, that two of the eight held-out pairs repeat trained answers, and that the report's
"first refused poison" and "content stored only for trained compositions" readings are not
supported; each archive carries the corrections. The measurement now trains in full passes, keeps
per-item records, pairs the sequential poison arm with a clean-again control, and adds a
first-situation choice score over every composition's output.

**Next, in order:** replicate that update on further seeds and splits under the corrected
measurement; then compare Sleep's consolidation operators (anchor, distillation, dream) with it on
the same stream at matched exposure; then, only after those comparisons, recurrence
(dose-response over how often a composition recurs) and a physics testbed whose experience stream
carries a persistent regime.

**Sleep on facts is paused (23 September).** On the final chat checkpoint, 15 consolidation
attempts across three seeds passed the damage gate 13 times and retained none of 24 facts taught
once each; the only transfers were verbatim planted falsehoods. Why is not established: replay
trained on the saved teaching text and Dream received the quoted statement, so the session's loss
of the facts does not explain those arms, and no run showed that the write path can store a fact
under the most favourable exposure. The [Sleep note](2026-09-23-sleep-consolidation.md) keeps the
record; its operators are next measured on the rule task, where the model can do the task in
context.

**The from-scratch coordinate learner** adapts within an episode once its inner step ceiling is
lifted, with the nonlinear coordinates doing the work, and matches the delta-rule baseline at lower
cost ([ablation](results/contract-2026-09-23/README.md), one seed). Nothing there is lasting
learning yet: the experience stream is drawn from the model's own training distribution, so a
lasting update cannot be separated from further pretraining. The
[slow-update specification](../superpowers/specs/2026-09-23-slow-update-rule.md) waits for a
testbed with a persistent regime and a matched no-regime control.

Qwen remains the hosted chat model in observational mode; the published 6.85M PlasticCore
WikiText model runs with the transactional harness. The TTT chat checkpoint is not hosted.

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
