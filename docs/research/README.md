# Research guide and briefing

**Want to help?** Start with [Contributing to the research](contributing-research.md)
for runnable experiments, metric definitions and open comparisons. Read
[current status](current-status.md) for the active questions and
[Sleep results](results/sleep-2026-09-23/README.md) for the saved tables and individual
answers. The [Sleep method note](2026-09-23-sleep-consolidation.md) describes the
implemented consolidation methods and what the experiments have found. The playground's
Sleep tab shows every archived run interactively; [the observatory page](sleep-observatory.md)
explains what it displays and where each number comes from.

The [importance-weighting proposal](2026-09-23-importance-weighting-proposal.md)
describes the next comparisons: per-token consolidation weights, adaptive sampling
and external-classifier baselines. Its implementation and efficacy status are
separate from the historical Sleep results.

This is the entry point for agents working on `plastic`. It is a reading map and
claim-checking protocol, not a new survey or a certification that every linked
statement remains correct. Initial index: 22 September 2026. Read the repository
[AGENTS.md](../../AGENTS.md) for the mandatory readiness and verification gates.

## Current work

Read [current research status](current-status.md) first. Dated notes below preserve
their original experiments and proposals; their schedules, approval requests and
next-step instructions do not override current direction. Update current status
and affected active documents when decisions change. The scratchpad is a private
history, not a public dependency.

## The question to preserve

The research objective is more than implementing a known fast-memory layer: investigate
a coupling of genuinely gradient-updated, end-to-end meta-trained fast learning and
a selective activation recurrence, with an external non-differentiable transactional
harness. The target is one shared block stack for text and hidden-friction physics,
originally roughly 1–10M parameters, plain PyTorch usable on MPS, and a one-hour
L4/A10G training target. These describe the original architecture study, not a
restriction on the current pretrained TTT chat work or authorized compute.

The implemented linear delta-memory stack is the baseline. The nonlinear coordinate
proposal is an unintegrated experimental candidate. Do not confuse delivering a useful
baseline with answering the architectural research question, or a mathematical probe
with demonstrated learned behavior. Existing methods remain necessary comparisons.

## Corrections to carry into the reading

- The original architecture memo's identification of its delta rule with exact
  TTT-Linear is too strong. Check the inner model, normalization inside the loss,
  learned initial weights, and minibatch base points rather than equating names.
- Zero write rate does not freeze retention. Proposed write pressure is not total
  accepted mutation. Rollback does not erase activation influence or emitted output.
- Normalization/nonlinearity alone proves neither novelty nor useful adaptation.
  An attention-equivalence result must be read with its objective, terminal-layer,
  normalization, and feature-dependence assumptions intact.
- Canary projection is local; an amplitude bound is not a robustness guarantee.
  Nominal calibration ranks are not observed sequential false-positive rates.
- Frozen controls are matched activation controls, not a cross-payload noise floor.
  Invalid attacks and missing measurements cannot establish strong defense.

These are guardrails against known misreadings. If new evidence contradicts one,
document the evidence and revise the claim; do not treat this briefing as dogma.

## Required reading by task

Read this file, the current [project overview](../../README.md), and the relevant
documents below. Agents in the shared checkout also read the latest private
`SHARED_SCRATCHPAD.md` for ownership and pending corrections; corrections affecting
public methods or results must be carried into the public documents.

| Task | Required local material |
| --- | --- |
| Architecture, inner learning, equivalence, novelty | [Literature survey](2026-09-21-ttt-ssm-literature.md), [original candidate](2026-09-21-architecture-memo.md), and [coordinate proposal](2026-09-21-plastic-coordinate-recurrence.md), including limitations, nearest prior art and falsification criteria |
| Harness, calibration, safety experiments | [Safety review](2026-09-21-inference-time-learning-safety.md), [calibration replay audit](2026-09-22-calibration-replay-audit.md), [the Qwen shock pass (negative)](2026-09-23-qwen-shock-pass-negative.md), and the coordinate proposal's transaction semantics if that candidate is involved; for the native Qwen backend also the [turn-boundary retention candidate](2026-09-22-qwen-turn-boundary-retention.md) (not adopted); for the pretrained TTT chat backend the [TTT backend note](2026-09-23-ttt-backend.md) (signals, controls, what was measured) |
| Learning utility or system identification | [Copy-memory intervention](2026-09-22-copy-memory-content-audit.md), [trained-model study](2026-09-22-trained-model-operating-point.md), relevant model/evaluation code and saved configuration |
| UI, API, documentation | The above source for every scientific metric or mechanism being exposed; verify proposal/accepted, units, controls and missing-data semantics against actual payloads; the [TTT backend note](2026-09-23-ttt-backend.md) for which signals each backend produces; the [Sleep method note](2026-09-23-sleep-consolidation.md) and saved outputs for consolidation claims |
| Compute or performance | [Tooling notes](2026-09-21-tooling-hf-jobs-torch.md), actual launch configuration and measured logs; do not extrapolate a CPU identity probe into a GPU throughput result |

Historical test counts and pinned experiments describe their recorded source, not
the current checkout. Read the corrections in this briefing and each method note
before repeating a dated claim.

## Primary-source refresh and decision record

For research-dependent work, use the survey and candidate's reference lists to find
primary papers and author code, then check revisions and relevant newer work at the
start of the session. Read the sections that determine your decision. Search by
mechanism as well as project terminology so differently named prior art is considered.
Do not assume all cited identifiers, dates, summaries, or priority claims are correct
because an earlier agent marked them verified. Record access failures explicitly.

The short readiness note must let another agent answer: **what did you read, what
changed your assumptions, how does the proposed mechanism differ, and what result
would make you abandon the claim?** Include source links/versions/check date and
separate verified facts, local observations, hypotheses, and unresolved questions.
Keep the note decision-focused; do not paste entire papers into the scratchpad.

Before finishing, compare the result with this note and the actual experiment
criteria. If a known baseline explains it, say so. If only algebra was checked,
report algebra. Update the shared record with new evidence and explicit corrections,
not a stronger claim derived from repeated wording.
