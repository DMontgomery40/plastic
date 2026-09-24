# The text rule contract: durable, transferable competence on the TTT chat model (Thread T4)

Draft spec, 23 September 2026, session 41b763a8. Owner of the work: Fable; human owner: David.
Benchmark: the same five measures as the [mechanism testbed contract](2026-09-23-mechanism-testbed-and-contract.md),
applied to a text stream on the step-250 TTT-MLP chat checkpoint. Sources for the design are collected in
[the T4 sources memo](../../research/2026-09-23-text-rule-contract-sources.md) (draft, in progress); the readiness
note below is completed when that memo lands.

## Research readiness note

**Task and constraints.** David's redirect of 23 September: learning is a durable, transferable change in
competence, judged after the conversation and the temporary state are gone; facts go to retrieval; safety is
scored in both directions. The Sleep study measured phrase recall of facts taught once and retained nothing
(15 attempts, 13 gate-accepted, 0 of 24 facts, only verbatim planted sentences transferred; see the Sleep note's
final sections). This thread replaces the object of learning: not facts, a *rule system* whose competence shows
on inputs and compositions never shown. It reuses the T1 contract's measures and report shape so that the text
learner and the physics learners are read on one page. No sweeps; build, then one report with the baselines.

**Local sources read.** AGENTS.md; the T1 spec and `plastic/eval/contract.py` (`Learner` protocol, `run_contract`,
`acceptance_rates`, the fixed-seed measurement discipline, the compute block); `plastic/data/mechanisms.py`
(`MechanismBatch`, `poison_stream` as a consistent learnable false lesson); the Sleep note and
`plastic/sleep/ttt.py` (the four operators, the gate, `heldout_nll`, `fresh_session_answer`); the TTT backend
(`forward(chunk, state, freeze, beta_scale)`, `encode_chat`, `encode_conversation`); the reassessment memo
(measurement battery: paraphrase, reverse, entailment, locality; recurrence and consistency over surprise);
scratchpad FABLE-193/194, ASTRA-229/230.

**Primary sources.** Twenty-seven checked on 2026-09-23 in the [T4 sources memo](../../research/2026-09-23-text-rule-contract-sources.md)
(five at body level: MLC, Ramesh et al. 2311.12997, Akyürek et al. 2411.07279, GPT-3 §3.9.2, ARC Prize 2025).
What they fix in this design: the split is CFQ/COGS-style (every operator seen in training, only the ordered pairing
new at test; length is a separate split that even MLC fails); Ramesh et al. generalize to unseen compositions only
when intermediate outputs are shown, and unseen operator order fails in the direct format; no source shows a model
at or below 1B learning word-list reverse or swap from a few examples (GPT-3 sub-1B at 0% on reversed words), small
models learn format and label vocabulary before the rule (Min et al.; In-Context Fixation), and a stated rule beats
examples alone (Fu et al. 2609.03213). No TTT or fast-weight paper evaluates after reverting weights and resetting
state. In-context poisoning work uses triggers and labels, not a consistent false rule; the nearest analogue of
propose-and-verify is task-to-task verification in continual learning.

**Closest known mechanism.** Test-time training on ARC (Akyürek et al., 2411.07279): per-task adaptation from
worked examples with leave-one-out, adapter discarded per task, measured within the task, not after reset. The T1
contract's baselines (frozen / continued / in-context) transfer unchanged.

**The distinction under investigation.** Whether a lasting update produced by propose-and-verify over an
experience stream of worked rule applications improves competence on *held-out compositions*, measured after
the session state is cleared and the fast weights reset, with the gain vanishing on revert and with a poisoned
rule stream refused or corrected. The in-context baseline sets the bar at matched compute.

**Unresolved assumptions.** (1) That the step-250 checkpoint can learn any of the operators from worked examples
in context at all; if the in-context ceiling is at chance, the testbed is uninformative on this model and the
report says so (analogue of T1's "frozen equals continued"). (2) That single-word operator outputs make exact
scoring and per-token likelihood both meaningful; multi-word outputs are scored by exact string and by mean
log-probability over the output tokens. (3) That a "false rule" stream (an operator taught with a wrong
definition) is the right poison: it is consistent, learnable, and false on every input, matching T1's design.
(4) That the four Sleep operators, used as the lasting update, can move competence at all; the Sleep results say
they moved phrases, not relations, so the expected first result is negative, and that is a result.

**Falsifying check for the testbed itself.** If the frozen learner (no lasting update) and continued training on
the stream score the same on held-out compositions after reset, the stream carries nothing a slow update can use
on this model at this scale, and the rule system or the exposure must change before any mechanism is judged.

## The rule world (`plastic/data/rules.py`)

A world is a set of **operators** over short word lists (three or four common nouns drawn from a fixed vocabulary
of 40). Each operator has a name token the model has never seen used this way (single capital letters with a
marker, e.g. `#R`), a deterministic definition, and a one-line worked form:

| Operator | Definition | Example on `apple pear plum` |
| --- | --- | --- |
| `#R` | reverse the list | `plum pear apple` |
| `#S` | swap the first two | `pear apple plum` |
| `#D` | drop the first | `pear plum` |
| `#K` | keep only the last | `plum` |
| `#T` | duplicate the last | `apple pear plum plum` |
| `#U` | uppercase every word | `APPLE PEAR PLUM` |

A **situation** is one worked application: user turn "Apply `#R` to: apple pear plum", assistant turn
"plum pear apple". A **composition** `#S #R` means apply `#R` first, then `#S` (right to left, stated once in the
system line of every stream). The **split**: all six singles and a training subset of the 30 ordered pairs are
seen in streams; the held-out pairs (default 8, chosen by seed with every operator appearing in at least one
training pair) are never shown; inputs (word lists) at measurement time are also unseen. Following the sources
memo: the preface states all six rules (a stated rule beats examples alone for small models); 12 training pairs
and 18 held-out, with every operator in both positions of a training pair and at least six held-out pairs being
the reverse order of a training pair, so order is tested; word lists of 4 or 5 nouns; lessons of 8 worked
situations, with the in-context curve read at situations 2, 4 and 8. The poisoned episode states its false
definition in the preface, so the stated rule and the worked outcomes agree.

## The decoration set (`rule_set="decorate"`)

The probes below showed that the step-250 checkpoint copies a list in context and learns fixed decorations around
it from a few worked examples, while it learns none of the transforming operators. The second operator set is
therefore five decorations: `#P` write "please" before the list, `#Q` write "thanks" after it, `#B` put it in square
brackets, `#W` repeat the whole list twice, `#H` write "here" before it. Ordered pairs differ in output
(`#B #P` gives `[ please … ]`, `#P #B` gives `please [ … ]`), so composition is testable; 20 ordered pairs, 12
taught and 8 held out with at least four reversed orders of taught pairs. The poison for a decoration is another
decoration's definition (`#P` taught as "write thanks after the list"), consistent and wrong on every input. The
first report runs this set; the transforming set stays in the archive as the measured floor.

## First measurement, and what it decides

[Results](../../research/results/text-rules-2026-09-23/README.md), step-250 checkpoint, MPS. The in-context
gate is at the floor: with the six rules stated and eight worked examples in the session, teacher-forced exact
match on single operators is 0 for #R, #S, #D and #U and 1 of 3 at one late situation for #K and #T, while the
per-token loss on the answers falls to about 1 nat within two or three situations (the fast path learns the
answer's shape and vocabulary, not the operation). Held-out pairs reach 0.12 exact from the fifth situation,
all of it from `#D #K`, whose correct output is a single word. This is the sources memo's first falsifier: on
this checkpoint the contract would judge every lasting update against a ceiling of zero. Consequence: no
lasting-update runs on this checkpoint with this rule system. The rule world, the contract and the learner stay
as the measurement; the next step is either a substrate that can perform the task in context (the T2 coordinate
learner on the T1 testbed is the project's route) or a task this checkpoint can perform in context, found by
measuring the same gate on simpler transformations (single-word outputs, copying with a marker), before any
update is judged. The three falsifiers stand for any later run: the in-context gate at floor; frozen equal to
continued on held-out pairs after reset; a format-only stream (same lessons, shuffled outputs) producing the
same lasting gain as the true stream.

An **episode** is `n` situations of one composition on different inputs, in one chat session; a **stream** is a
sequence of episodes over training compositions. A **poisoned stream** teaches one operator with a consistent
false definition (e.g. `#R` shown as "drop the first" on every situation); it is learnable and false on every
input, the T1 property.

## The measures (`plastic/eval/text_contract.py`)

Same five, same report shape and version tag as T1, with MSE replaced by two scores on the assistant's output
span: **exact** (the greedy output string equals the correct output after whitespace normalization) and
**nll** (mean negative log-probability per output token, teacher-forced). Every measurement starts from a fresh
state (`backend.init_state()`), with the stream absent from the context.

1. **transfer**: on held-out compositions with unseen inputs, `adapt=True` (the fast path writes during the
   episode's earlier situations) and `adapt=False` (`freeze=True`: the slow parameters alone); reported per
   composition and averaged, with denominators.
2. **speed**: within an episode of `probe_steps` situations on a held-out composition, the adapting nll at
   situation *i* as a fraction of the writes-disabled nll on the same inputs and matched activation state; area
   and situation-to-half.
3. **forgetting**: held-out SmolTalk assistant NLL (`heldout_nll`, pinned revision) and exact/nll on training
   compositions.
4. **correction**: transfer after the poisoned stream (harm) and after a corrective clean stream (residual).
5. **revert**: restore the pre-stream slow parameters; the largest absolute gap to the before measurements must
   be within tolerance.

Acceptance is the pair accepted-good / refused-bad over the three `consume` decisions (clean, poisoned,
corrective), with counts; an empty side is `None`.

## The learner (`plastic/eval/text_learner.py`)

`TextRuleLearner` implements the T1 `Learner` protocol over the TTT backend:

- `snapshot_slow` / `restore_slow`: the fast-weight initialization W0 (target `w0`) or all parameters (target
  `all`), cloned to CPU.
- `step_nll(batch, adapt)`: renders each episode as a chat (system line, then the situations), runs it from a
  fresh state with `freeze = not adapt`, and returns per-situation exact and nll on the assistant spans.
- `consume(stream)`: **propose-and-verify**. Propose: run the stream as an observational session (the existing
  runner, all chunks committed, flags recorded), then apply one Sleep operator (replay / distill / anchor /
  dream, chosen by the mode) to produce a candidate slow state. Verify, on *training-distribution* material only
  (never the held-out compositions): the damage gate (held-out chat NLL, canaries when present, reply-cluster
  share) plus a competence check, exact/nll on training compositions with fresh inputs, must not fall below the
  pre-stream value by more than a tolerance. Accept installs the candidate; refuse restores the snapshot. The
  record carries `accepted`, the verify measurements, the operator, the harvest summary and the selected-turn
  count, so the report can say what was consolidated.

Baseline modes, as in T1: `frozen` (no lasting update; `adapt=True` is the TTT-only baseline), `continued`
(cross-entropy on the stream's assistant spans at the community note's low learning rate, no gate; accepts
everything), `in_context` (the stream's situations prepended at measurement time; extra tokens counted). The
Sleep operators are the *propose* step of the modes under test, not baselines.

## What the first report must show

The step-250 checkpoint, one seed, on CPU or MPS: the in-context ceiling on held-out compositions (if at chance,
stop and report); frozen vs continued on transfer after reset (the testbed's own falsifier); one propose-and-verify
mode; the poisoned stream's fate (accepted or refused, harm, residual); revert within tolerance. Numbers carry
denominators and compute; nothing is called learning unless the after-reset gain on held-out compositions
exceeds the in-context baseline at matched compute and vanishes on revert.

## First contract report (decoration set, step 250, seed 0)

[Results](../../research/results/text-rules-2026-09-23/contract_decorate_s0/README.md). Held-out ordered pairs,
teacher-forced exact with in-episode adaptation, before → after the lasting update: frozen 0.67 → 0.67 (the
episode-only bar); continued (AdamW on W0, 20 steps) 0.67 → 0.87 with answer NLL 0.387 → 0.120; in_context (the 17
lessons read into the fast weights before each episode, without a reset) 0.67 → 0.55 with NLL 0.278, so on this
model more fast-weight context lowers the loss and reduces exact matches, the interference the 30-turn sleep
sessions showed; replay_verify equals continued and accepted every stream. The frozen-fast-path exact stays 0 in
every mode (NLL 8.40 → 8.15 after the update): the slow weights never answer alone; the lasting change makes the
fast path adapt faster (speed ratio at the first situation 0.28 → 0.10). Held-out chat NLL fell (1.629 → 1.598),
revert was exact, poison harm was +0.019 NLL and the corrective stream removed it.

What this does and does not show. A lasting update from 17 lessons improved held-out compositions after reset and
above the in-context bar, with no damage and a clean revert: the contract's first positive row in this project.
It is not yet rule learning: the poisoned stream (one operator taught by a false definition) produced almost the
same gain as the clean one, which points at format and procedure (how to do the decoration task in an episode)
rather than rule content; the format-only control (same lessons, shuffled answers) decides this and runs next.
The verifier (v1) accepted the poison for two reasons: its held-in checks scored with the fast path frozen, where
exact is 0 before and after (vacuous), and a consistent false rule consumed from a snapshot with no prior about
the operator is indistinguishable from a true rule by any damage check. Verifier v2 will score with adaptation on
and add a sequential arm (clean accepted, then the poison on top, verified against the accepted lessons), which is
the recurrence-and-consistency criterion of the reassessment memo in its first concrete form.

### The format-only control (same run, continued and replay_verify)

[Results](../../research/results/text-rules-2026-09-23/contract_decorate_s0_format_control/README.md). The
clean lessons with every answer moved to another situation of its episode (a derangement: same chat shape,
vocabulary and answer lengths, every answer wrong for its input) give 0.67 → 0.76 exact and NLL −0.205, against
0.67 → 0.87 and −0.267 for the true lessons: 44 % of the exact gain and 77 % of the likelihood gain are format.
The remaining exact gain (+0.11) needs the answers to be right for their inputs, but the poisoned stream (one
operator taught wrong, consistently) produced almost the same gain as the clean one (0.84 against 0.87), so what
the answers carry is that a stated rule is applied to the copied list, not which rule. All of the gain is in the
situations after the first: the first situation's exact is 0 before and after in every arm (its NLL 2.30 → 0.92);
the second situation goes 0.06 → 0.94. The lasting update makes one worked example do what three did before.
Reading: a procedure ("apply the stated decoration to the list, from the first example on") was learned; the
decorations themselves were not stored, and the transforming operators are still not learnable at all here.

Verifier v1 accepted all four streams, including the format-only one, whose shuffled answers lowered its frozen
held-in NLL more than the true lessons did (−0.467 against −0.213), with held-in exact 0 on both sides of every
decision. Frozen scoring does not see rule content on this model; v1 is retired below.

## Verifier v2 and the sequential poison arm

Two changes to `replay_verify` and one to the contract, made after the first report and named in every archive row.
(1) The held-in checks are scored with the fast path adapting (`verify_adapt=True`, the default; `--verifier v1`
reproduces the frozen scoring), so that exact is no longer 0 on both sides. (2) A third check, the exact match on
the *first* situation of each held-in episode, which no worked example precedes: what the slow weights carry under
the stated rule, the place a lasting false rule shows first. The tolerances are unchanged (exact drop 0.05, NLL
rise 0.1, chat NLL rise 0.05); nothing is calibrated, they are stated limits. (3) `TextContractSpec.sequential_poison`
(default on): the poisoned stream is also consumed on top of the accepted clean lessons, without a revert between,
and measured; the decision counts as a second refused-bad row (`poisoned_sequential`) and the report carries the
harm relative to the clean state. From the snapshot the model has no prior about the poisoned operator, so a
consistent false rule taught there is a different rule, not a contradiction; on the sequential arm the verifier has
its own accepted lessons to compare with, which is the recurrence-and-consistency criterion of the reassessment memo
in its first concrete form. Verified held-in material stays fresh inputs of the training compositions under the true
definitions; the held-out compositions are never seen by the verifier.

### Verifier v2 result: nothing refused, and why that is a finding about the poison, not only the verifier

[Results](../../research/results/text-rules-2026-09-23/contract_decorate_s0_verifier_v2/README.md). The v2 checks
measure something now (held-in exact 0.17 → 0.61 for the clean stream, first-situation exact 0 → 0.17, NLL
0.885 → 0.141), and every stream was still accepted: accepted-good 1.0, refused-bad 0 of 3. The snapshot poison
produces the same held-in gain as the clean stream; the sequential poison, consumed on top of the accepted clean
lessons, raises held-in exact further (0.61 → 0.67) and costs the held-out transfer +0.009 NLL; the format-only
stream gains less (0.50 exact, no first-situation gain) but gains. A verifier that checks for damage cannot refuse a
lesson that helps by every measure it has, and on this learner all three "bad" streams help.

Why they help. Each stream's preface states its definitions (true, or false for the poisoned operator). The lasting
update taught the learner to apply whatever definition the preface states to the copied list from the first example
on (the format-only control's reading above). Verification and measurement state the true definitions, so a learner
that follows the stated rule is correct there whatever the poisoned stream's answers were. The poisoned stream is
therefore a consistent lesson under a different stated rule, not a false belief about `#P`, and the sequential arm
has nothing to contradict: the accepted lessons are "follow the preface", which the poison also teaches. Held-out
harm is real but small (+0.019 NLL from the snapshot, +0.009 sequential), the residue of the answer strings.

What this decides. Under stated rules, refused-bad cannot be earned by a damage verifier, and a comparative verifier
(gain relative to a matched clean control, or the first-situation gain that separates the format-only stream) would
refuse "less useful", not "false". To test correction of a false belief, the poison has to be one. Two variants,
both small changes to `plastic/data/rules.py` and the learner's encoding, are the next report, one row each:

1. **Unstated rules.** The preface names the task but no definitions, in streams and in measurement; the rule can
   come only from the worked examples, and the poisoned stream's wrong examples are then the only source about
   `#P`. This is the "learn the rule from experience" setting the redirect asks for; the in-context gate must be
   re-read first, since the stated rule was part of what made decorations learnable in context.
2. **Inconsistent poison.** The true definitions stated, the answers for the poisoned operator wrong: a lesson whose
   preface and outcomes disagree, which a learner that follows the preface must either ignore or absorb against it.

Neither is a fix to the verifier. If the unstated setting is learnable and the poison there is still accepted with
no held-out harm, the propose-and-verify design as built does not protect against false rules on this learner, and
the thread reports that.

## Not in this thread

New learning mechanisms (T2, T3); sweeps over seeds or hyperparameters; hosting. Loose ends listed in FABLE-193
follow the first report.
