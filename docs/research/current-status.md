# Current research status

Updated 22 September 2026. This page gives the current direction. Dated notes are
experiment and design history; update this page when decisions or evidence change.

## Active investigation

Explore the refusal-ablated Qwen3.5-0.8B checkpoint under the transactional harness.
Compare observational and experimental guarded behavior early with matched prompts,
seeds and starting state. Keep the detector unchanged for the first comparison.
An uncalibrated guard is an experiment, not validated protection; calibration and
prior signal separation are not prerequisites for trying it. Original Qwen is a
selective reference where it adds information, including benign signal comparisons.

Record loss, proposed and accepted recurrent-state changes, both CUSUM sides,
actual versus hypothetical decisions, outputs and later-turn behavior. Compare
retained turns with omission and complete snapshot/restore controls. Frozen replay
and complete turn restoration are different interventions; neither retracts emitted
answers. The co-leads can narrow a run for a concrete experimental reason, without
turning an exploratory comparison into another prerequisite process.

Gradient instrumentation is an open implementation question. The native backend
currently reports loss and state changes, not measured gradient differences. A
local gated-delta objective gradient and a task-loss derivative with respect to
recurrent state are different quantities. Any gradient metric must identify its
objective, differentiation variable, base point and comparison. Do not rename a
state-change norm to a gradient or assume no gradient interpretation exists.

The current small model can produce fluent factual errors. Baseline competence
limits semantic interpretations: an incorrect output alone cannot establish
poisoning, guard damage, or protection. This does not block alpha-stage state and
weight research, nor require a model replacement before testing update, retention,
gradient, and transaction mechanics.

The public native backend does not yet report inner-memory prediction-error norms,
write rates, or retention rates. Its loss and state-change measurements remain
usable; the missing inner-memory signals are not zero. Native residual
instrumentation is separate work from the current observational/guarded comparison.

Publish methods, useful examples and reproducible results openly. Basic safety
and alignment probes are sufficient; publication does not require dangerous
procedural outputs. Keep private coordination and incidental session data out of
public source. Put scientific and implementation detail in documentation, not UI
banners. The interface needs concise controls, results and actionable status.

## Implemented and measured

- The public demo uses `qwen3_5_0_8b_abliterated` in observational mode.
  Each host's `source_snapshot.json` identifies its published source revision.
- T0's four recorded chains passed full-cache restoration/omission comparisons.
  This supports restoration mechanics, not detector efficacy.
- T1 at `979a542` completed with a negative fit result: no declared CUSUM threshold
  from 3 to 20 met the benign-turn criterion. DEV was not run. Keep this useful
  negative result; it does not block other exploratory comparisons.
- PlasticCore remains linear delta memory. The nonlinear, end-to-end meta-trained
  coordinate proposal is unintegrated and is not native Qwen's mechanism. Its
  original research question remains open.
- Physics remains an internal benchmark available in the local dashboard; the
  public interface is scoped to text chat and session measurements.

## Continuation

Fable and Astra co-lead; reuse the existing Sol session for dashboard implementation
and connected-browser validation. File ownership and active session identities belong in the private
brief and scratchpad. Use the canonical checkout on main. Review each new artifact
set once; reopen only for new evidence or a material change.

Keep compact evidence supporting reported findings, including useful negative
results. Archive useful inactive bulk artifacts privately on HF, verify them before
local removal, and retain a retrieval manifest. Disposable attempts and obsolete
copies need not accumulate. Keep active checkpoint/runtime inputs available.
