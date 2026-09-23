# Sleep: consolidating accepted session learning into slow weights

Updated 23 September 2026. **Implemented, experimentally unresolved.** The pretrained
TTT backend supports replay, distillation, anchoring and Dream consolidation in
[`plastic/sleep/ttt.py`](../../plastic/sleep/ttt.py). No tested configuration has yet
established reliable fact retention across a reset without damaging behavior.
The original PlasticCore prototype remains a separate implementation.

[Run or contribute an experiment](contributing-research.md) ·
[Inspect saved outputs](results/sleep-2026-09-23/README.md) ·
[Current project status](current-status.md)

## Research question

A TTT session carries fast weights between turns and can save or resume that state.
A new independent session starts from the checkpoint's learned initialization,
`W0`; persisting one session does not change that initialization. Sleep asks whether
useful session learning can transfer into a child checkpoint that answers correctly
without the source conversation, while limiting unrelated damage and contamination
from rejected turns.

The harness accepts or rejects proposed fast-state changes at chunk boundaries.
Those decisions provide provenance for consolidation. They are not labels proving
that accepted text is true or safe. In observational mode, every proposal commits.

## Implemented mechanism

Contract checked against source `c6a96a5` on 23 September 2026. The
[TTT backend note](2026-09-23-ttt-backend.md) describes the wake-time inner objective
and state coordinates.

| Field | Current implementation |
| --- | --- |
| Wake objective and fast variables | TTT-MLP reconstruction; per-layer `W1, b1, W2, b2`, plus pending mini-batch gradients. Inner mini-batch size is 16. |
| Sleep targets | `w0`: learned initial fast weights, about 28.5M parameters in the 760M model; `all`: all model parameters for gradient methods. Anchoring changes only `W0`. No LoRA target is implemented. |
| Sleep update | AdamW through the student's full forward, starting from reset fast state; anchoring instead interpolates initial weights directly. |
| Carried state | Saved committed sessions supply traces and teacher states. Parent files and sessions are unchanged; an accepted run registers a child checkpoint and lineage. |
| Targets and timing | Past accepted turns, frozen session teachers and sampled replay data. Dreams are generated before student optimization. |
| Transaction | One Sleep run, with available held-out NLL, canary and reply-cluster checks before/after. A failed check rejects the child; no available checks yields `accepted_unmeasured`. |

### Provenance

Candidate turns must have **every** chunk committed, scaled or projected. A turn
containing a rollback or read-only chunk is excluded. Transaction-index ranges
distinguish turns even after positions restart; ambiguous legacy traces are excluded.

The default `flagged_policy=exclude` then removes accepted turns whose chunks were
scaled/projected or recorded a requested intervention, including observational
`would_*` decisions. The same selection supplies direct training text, Dream quotes,
and the set of sessions allowed to supply an anchor or teacher state. A session
with no selected turns supplies no state. Reports record this selection separately
from the harvest's online acceptance counts.

`include` retains those flagged candidates. `downweight` gives them lower relative
weight in replay CE or in Dream KL when a Dream quotes the flagged turn; it is not
supported for anchor or distill. These options are experiment settings, not measured
evidence that the filtering protects consolidation. Historical results below
predate this default exclusion rule.

**Measured consequence in an uncalibrated observational session (2026-09-23, final
checkpoint, seed 0).** The experiment's teaching session runs log-only without a
calibration, so the policy's `would_rollback` reasons come from the runner's own
running statistics. Of 30 teaching turns, 29 were flagged: one chunk per turn, the
chunk in which the new fact arrives, crossed the chunk-loss and surprise z-thresholds.
Under `exclude` the selected text was therefore one turn: the replay rows and the
Dream quote pool came from one teaching statement. State is a separate channel:
anchor, distill and Dream load the teach session's full committed state once that
session has one selected turn, so the seed-0 anchor null is a test of the whole
session's state, while the replay and Dream nulls are one-turn results. This is the
tension the [importance proposal](2026-09-23-importance-weighting-proposal.md) names:
importance by surprise is highest for new facts, so a rule that excludes surprising
turns excludes the facts from the text pool. The experiment script now takes
`--flagged-policy` (default `include`, so every harness-accepted turn is selected and
the product rule is a comparison arm) and every result table states the selected-turn
count from the selection itself. The hosted Qwen Space does not offer Sleep and has no
inner surprise, so this finding applies to the future hosted TTT configuration: under
the shipped default, a chat that teaches new facts would supply little selected text
to Sleep. Whether that hosted default should stay `exclude` or become `include` is an
open decision, recorded here and not changed in the interface. Rollback exclusion
(provenance), the advisory `would_*` flags (this rule) and the post-sleep locality
checks (the gate) are three distinct controls; changing this rule leaves the other two.

This does not remove every influence of rejected content. Frozen replay still
advances activation and convolution state, which can affect subsequent accepted
updates. A mixed session's final committed state can also contain writes from
accepted-but-flagged turns, even when their text is excluded from Sleep.
Anchor/distill/Dream read selected sessions' states with that history.
The experiment measures recall of rolled-back facts against controls rather than
assuming provenance filtering guarantees their absence.

### Four methods

| Method | What trains the child |
| --- | --- |
| `replay` | Cross-entropy on accepted turns mixed with SFT replay. Session loss defaults to all tokens after BOS, with configurable user-token weight; `assistant` masks user tokens. SFT replay remains assistant-only. |
| `distill` | KL from a frozen session-state teacher on packed session text, plus assistant-token CE on replay. A loaded teacher state is sampled for each session row. |
| `anchor` | No gradient: `W0 ← W0 + λ · mean_s(W_eff,s − W0)` over compatible committed states from sessions with selected turns. |
| `dream` | Generate study items with a frozen session teacher conditioned on an accepted user turn. Train a reset student, without the quoted turn, using reply-token KL plus replay CE. |

For distill and Dream, the mean KL and mean replay CE have unit coefficients;
`replay_ratio` controls row sampling, not their relative loss weights. Batches keep
at least one session row and preserve the requested batch size. Reports contain
both requested and realized ratios: batch 2 realizes 50%, batch 5 realizes 80%.
Prompt-loss weighting affects replay CE, not the distillation KL terms.

Teachers remain fixed throughout student updates. The `all` target uses a separate
frozen model copy; for `w0`, the loaded session state overrides the changing initial
weights. These contracts, rendering alignment and skipped-state identities have
regression coverage in [the Sleep tests](../../tests/test_sleep_ttt.py).

### Dream generation and its two scores

For each accepted user turn, three templates request a restatement, a question and
answer, or a future reply. The teacher receives the turn (whitespace normalized,
limited to 400 characters) and its saved session state. The student prompt omits
that turn. The same sampled reply tokens, including EOS, are scored in three ways:

- `teacher_logprob`: session state plus the quoted turn;
- `student_logprob`: reset state, without the turn;
- `fastweight_logprob`: session state, without the turn.

Each is a mean log probability in nats/token. Per-token values are also recorded.
`gain` is teacher minus student;
`fastweight_gain` is the no-quote session score minus student. The first therefore
includes information from the quote. The second still compares different full
session states/renderings; it is not a controlled intervention on fast weights
alone. Neither score proves that a generated statement is correct.

The filter rejects short/repetitive replies, duplicate eight-word prefixes,
nonfinite or insufficient gain (default 0.2 nats/token), items over the keep cap
(default 24), and items whose teacher or student rendering exceeds `seq_len`.
An empty usable set rejects the run. Reports retain selected items and rejection
reasons; rejected text excerpts can be truncated. Teacher/student prefixes differ,
so the KL aligns the identical reply tokens rather than matching absolute positions.
See [`dream.py`](../../plastic/sleep/dream.py) for the templates and selection rule.

The default KL gives each reply token equal weight. New `gain` and `fw_gain`
options use floored, normalized token weights from the respective score differences.
They are implemented but have not established a retention benefit; the
[importance-weighting proposal](2026-09-23-importance-weighting-proposal.md) describes
the intended comparisons. The historical tables below do not evaluate them.

## What the experiment measures

The [protocol](../../scripts/experiments/sleep_controls.py) teaches synthetic facts
and two benign chemistry facts in observational sessions, and forces rollback of
two other facts separately. Historical runs used six taught facts and five general
questions. The expanded protocol defaults to 24 taught facts, adds four
poison-answer probes and matching true-answer controls, and optionally teaches
those contradictions with `--poison`. Current paraphrase probes are held out of the
study set; historical study runs reused a teaching paraphrase. Report the protocol
version and each group's denominator with every comparison.

| Control or outcome | Interpretation |
| --- | --- |
| Parent floor | Fresh parent, no teaching context. |
| In-session ceiling | Parent reteaches the selected raw teaching statements before each probe; includes ongoing adaptation. It does not replay the augmented study-set turns. |
| Sleep arms | Separate children from the same parent and source sessions. |
| All-turn replay (`ungated`) | Includes directly rejected and flagged turns; retains the final Sleep locality gate. |
| Recall | Normalized substring and exact matches; inspect replies alongside counts, including incorrect associations and repeated-answer hits. |
| Expected-answer likelihood | Per-token log probability, averaged across verbatim probes. A rise without recall is a hypothesis-generating observation, not proof of storage or a unique readout diagnosis. |
| Locality | Held-out assistant NLL, canaries when installed, and largest identical 12-word reply-prefix cluster across verbatim and paraphrase probes. |

Default locality limits are an NLL rise of 0.05 nats/token, canary changes of 0.1,
and cluster share of 0.25. A cluster above 0.25 also passes if it does not worsen
from its own baseline. Only available checks run; acceptance does not require a
recall improvement. Read the report's actual configuration, checks and denominators.
A deliberately forced rollback tests input selection, not attack detection efficacy.

Exploratory runs do not require calibration first. A successful retention claim
needs fresh-session improvement over the parent, paraphrase checks and tolerable
locality. A protective benefit requires useful learning while reducing unwanted
retention against an all-turn control. Zero recall in both arms is inconclusive.
A finite null sweep rejects those tested settings, not the capacity of `W0` or the
possibility of consolidation. Failed paraphrase transfer narrows the demonstrated
behavior to the tested wording.

## Relation to prior work

Primary method sections checked on 23 September 2026:

| Source and version | Relationship and distinction |
| --- | --- |
| Sun et al., [TTT layers](https://arxiv.org/html/2407.04620v4), v4, 31 August 2025, §2 and learned initialization | A sequence-specific learner starts from shared learned initial weights. This supplies our backend; transferring deployment sessions into a new initialization is the question tested here. |
| Padmanabhan et al., [Propagating Knowledge Updates to LMs Through Distillation](https://arxiv.org/html/2306.09306v2), v2, 31 October 2023, §3 / Algorithm 1 | Generates continuations from a definition and distills a definition-conditioned teacher into an unconditioned student. This is the closest comparison for current fact-conditioned Dream. Our teacher also carries a selected TTT session state; that state's additional value still needs an ablation. |
| Behrouz et al., [Language Models Need Sleep](https://arxiv.org/html/2606.03979v2), v2, 10 July 2026, §3.2–3.3 | Knowledge Seeding transfers faster memory into slower blocks using new low-rank experts and distillation/imitation. Our implementation targets existing `W0` or all weights, has no RL or expert expansion, and uses external transaction provenance. |
| Eyuboglu et al., [Cartridges](https://arxiv.org/html/2506.06266v3), v3, 13 June 2025, §3–4 | Self-study uses synthetic conversations and context distillation to train a compact virtual KV cache. It motivates study data; that cache is a different target from our child checkpoint weights. |

The [community research note](2026-09-23-sleep-community-research.md) records the
broader search, including practitioner reports and neighboring mechanisms. Those
lead to experiments; their results do not establish what this architecture can or
cannot do. No bounded search establishes priority. The distinction under test is
whether transaction-selected session learning improves a reset model, and whether
its fast state contributes beyond ordinary context distillation.

## Running and hosted availability

Use the [contributor guide](contributing-research.md) for setup, commands, output
files and checkpoint availability. The local TTT Sessions screen offers Sleep;
CLI/API runs produce reports and child models. A separate interactive Recall-check
action is not implemented; probes are supplied to the experiment or Sleep job.

The public Space still serves Qwen. Hosted TTT/Sleep remains an open delivery
requirement: publish the chat checkpoint, pin `TTT_CHAT` in
[`pretrained.py`](../../deploy/huggingface/pretrained.py), select `PUBLIC_MODEL=ttt`,
and verify the actual hosted run and child-session journey on suitable hardware.
The public gate implements bounded anchor/replay jobs and child catalog access for
that release; it does not expose all local experiment options. Hosted children use
an ephemeral store and disappear on restart.

## Measured results

The [compact result archive](results/sleep-2026-09-23/README.md) contains reports and
probe replies. Intermediate SFT step-50/100 weights are not public, so the saved
outputs can be inspected but their exact tables cannot yet be rerun externally.
Historical gate results below use the rules in force at execution; later rescoring
is identified separately. Current methods must be evaluated on their own version.

### 2026-09-23, dry run on the 760M base (not chat-tuned): mechanics only

[Reports](results/sleep-2026-09-23/README.md). Setup: local MPS, disposable store, one teaching session (3 turns, 172 accepted tokens, log-only)
stating a cat's name, a city and an instrument; 3 recall probes with paraphrases; `w0` target;
20 steps, seq 256, batch 2, replay ratio 0.5 from 16 SmolTalk conversations; held-out 8
conversations (887 assistant tokens); tolerance 0.5 nats so nothing would be rejected.

| Method | Time | Held-out NLL mean / median before → after | Recall verbatim | Recall paraphrase |
| --- | --- | --- | --- | --- |
| anchor λ=0.5 | 92 s | 2.085 / 1.422 → 2.051 / 1.379 | 0/3 → 0/3 | 0/3 → 1/3 |
| replay | 125 s | 2.085 / 1.422 → 1.605 / 0.829 | 0/3 → 0/3 | 0/3 → 0/3 |
| distill | 182 s | 2.085 / 1.422 → 1.626 / 0.875 | 0/3 → 0/3 | 0/3 → 0/3 |

Reading. This run verifies that all three methods execute end to end, register a loadable
child with lineage, and produce the report; it does not establish reliable consolidation. Baseline recall was measured as zero; weak
chat competence and additional SFT replay confound interpretation of the held-out NLL drop. The replies after replay moved to the topic without the content ("The name of your
pet.", "The city you're in."). The single anchor paraphrase hit ("The name of my cat is named
Marlowe.") is one sample from a model that otherwise echoes the prompt; it is suggestive that
the moved `W0` carries session content, and nothing more until it is reproduced on the chat
checkpoint with matched controls. Chat-tuned checkpoints and matched controls are needed to interpret the behavioral effect.

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
on neutral ones (ratios 1.00 to 1.07). This gives a small comparison for later boundary-content measurements; it does not
establish how a calibrated policy will treat either group. Eight prompts per group is a smoke test, not a study.

### 2026-09-23, step-50 SFT checkpoint: matched-controls dry run (null baseline)

[Outputs](results/sleep-2026-09-23/sleep_controls_step50/sleep_controls.json).
`sleep_controls`, MPS, 10 steps, target `w0`, seq 256, batch 2, 8 replay
rows, 4 held-out rows, greedy 24-token probes, 27 minutes. Verbatim recalled / n, p = paraphrase.

| Arm | taught | boundary | rolled (contamination) | general (locality) | held-out NLL | status |
| --- | --- | --- | --- | --- | --- | --- |
| floor (parent, fresh session) | 0/6 (p 0/6) | 0/2 | 0/2 | 1/5 (p 1/5) | | |
| ceiling (parent, teaching turns in session) | 5/6 (p 5/6) | 0/2 | | | | |
| anchor λ 0.5 | 0/6 (p 1/6) | 0/2 | 0/2 | 1/5 | 1.766 → 1.766 | accepted |
| replay, w0 | 0/6 (p 0/6) | 0/2 | 0/2 | 2/5 | 1.766 → 1.615 | accepted |
| distill, w0 | 0/6 (p 0/6) | 0/2 | 0/2 | 1/5 | 1.766 → 1.627 | accepted |
| ungated replay, w0 | 0/6 (p 0/6) | 0/2 | 0/2 | 2/5 | 1.766 → 1.620 | accepted |

Reading. The model uses its session (ceiling 5/6), but no method improved verbatim
taught-fact recall after a reset in this 10-step run. Replies such as "Your cat is
called 'Pinkie'" have the answer format without the taught content. Falling NLL
coincides with continued SFT replay, so it does not isolate consolidation.
Rolled-back recall stayed at the floor even for ungated replay; with no useful
retention, the run does not establish a protective advantage from filtering.
Boundary probes scored 0/2 even in context. Later runs shortened their expected
answer strings, so those scores are not directly comparable across versions.

This is a null baseline for these settings. Later experiments below vary steps, target,
loss masks and study data; none should be treated as a rerun of an unchanged method.

### 2026-09-23, step-100 SFT checkpoint: the perplexity gate passed a collapsed model

[Outputs](results/sleep-2026-09-23/sleep_controls_step100_all40/sleep_controls.json).
`sleep_controls` with 40 steps on **all** parameters (lr 5e-5), floor / replay / ungated arms:

| Arm | taught | boundary | rolled | general | held-out NLL | gate |
| --- | --- | --- | --- | --- | --- | --- |
| floor | 0/6 | 0/2 | 0/2 | 0/5 (p 1/5) | | |
| replay, all × 40 | 1/6 (p 1/6) | 0/2 | 0/2 | 2/5 (p 2/5) | 1.683 → 1.629 | passed |
| ungated replay, all × 40 | 1/6 (p 1/6) | 0/2 | 0/2 | 3/5 (p 2/5) | 1.683 → 1.628 | passed |

The 1/6 hit in each arm comes from a repeated sentence containing the cat's name.
In the archived verbatim replies, replay repeats "A pleasure to meet you, Marlowe.
I'm proud to meet you…" on 7/15 questions and has 8/15 distinct replies. Ungated
replay repeats its "I'm glad to meet you…" variant on 6/15 questions and has 10/15
distinct replies. The parent has 15/15 distinct replies. Session-turn loss fell to
0.02 while held-out NLL improved; the loss-only gate missed this behavioral damage.

The current gate checks the **largest identical-reply cluster share**, rather
than the original distinct-reply ratio. Recomputing it from the saved replies gives
0.033 before and 0.467 / 0.433 after for replay / ungated. Both fail the 0.25 limit;
the original recorded `accepted` statuses are left intact in the archived reports.
The [saved-reply regression fixture](../../tests/fixtures/sleep_step100_all40_replies.json)
covers this failure. This demonstrates rejection of these cases, not general
collapse detection. The earlier distinct-ratio rule missed the ungated case.

### 2026-09-23, step-100 checkpoint: regime sweep of replay on raw turns (W0, user tokens supervised)

[Outputs and configurations](results/sleep-2026-09-23/README.md).
Five points, replay arm only, exact batch composition (batch 2 realizes 0.5, batch 5 realizes 0.8),
cluster-share gate in force:

| lr | steps | replay share | taught | rolled | held-out NLL | largest reply cluster | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3e-5 | 10 | 0.5 | 0/6 | 0/2 | 1.683 → 1.632 | 0.03 | accepted |
| 3e-5 | 20 | 0.8 | 0/6 | 0/2 | 1.683 → 1.541 | 0.07 | accepted |
| 1e-4 | 10 | 0.8 | 0/6 | 0/2 | 1.683 → 1.500 | 0.03 | accepted |
| 1e-4 | 20 | 0.8 | 0/6 | 0/2 | 1.683 → 1.476 | 0.03 | accepted |
| 3e-5 | 40 | 0.8 | 0/6 | 0/2 | 1.683 → 1.482 | 0.03 | accepted |

Reading. None of these five settings retained a taught fact. The four 80%-replay
settings did not collapse under the cluster measure; earlier 40-step/50%-replay
settings did. Since batch size also differs, this is not an isolated replay-ratio
ablation. Falling held-out NLL coexists with zero recall. These results motivated
study-set augmentation and likelihood measurements, without ruling out other
raw-turn settings or identifying a storage/readout mechanism.

### 2026-09-23, step-100 checkpoint: study-set augmentation and expected-answer likelihood

[Outputs](results/sleep-2026-09-23/study_step100_w0/sleep_controls.json).
Each fact taught as six templated turns (48 accepted turns, 3,481 tokens), W0, lr 3e-5, 20 steps,
80% replay (batch 5), prompt-loss weight 0.2:

| Arm | taught | rolled | general | held-out NLL | answer log-prob (mean/token) | largest cluster | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| parent (Sleep before) | 0/6 | 0/2 | 0/5 | | −6.89 | 0.03 | |
| replay | 0/6 | 0/2 | 1/5 | 1.683 → 1.529 | −6.89 → −6.05 | 0.03 | accepted |
| distill | 0/6 | 0/2 | 1/5 | 1.683 → 1.576 | −6.89 → −6.04 | 0.07 | accepted |

Reading. Taught recall remained zero, cluster share stayed low and expected-answer
likelihood rose by about 0.84 nats/token. The mean covers all verbatim probe groups,
not just the six taught facts. This does not establish storage or explain missing
recall. The model's generated teaching replies were often degenerate, while the
facts were supplied in the user text; a prompt-loss weight of 0.2 down-weighted
that text for replay. This weight does not change distill's KL objective.
A full-user-weight replay comparison is a separate experiment. Wrong answers such
as "Pinkie Pie", "San Francisco" and "Buddy" persisted across these arms.

### 2026-09-23, step-100 checkpoint: predecessor Dream method

[Outputs](results/sleep-2026-09-23/dream_step100_w0/sleep_controls.json), associated
with source `3828ac6` (execution SHA not captured), before the teacher/rendering fixes and fact-conditioned prompts. Fifteen
free-form dreams repeated the model's greeting; fourteen were removed as duplicates
and one trained the student. Taught recall was 0/6, held-out NLL 1.683 → 1.582, and
the run passed its locality gate. The positive likelihood gap selected a repeated
response pattern rather than demonstrating useful fact transfer.

That run motivates the current conditioned method but does not evaluate it. The
current method also fixes teacher isolation for all-parameter updates, aligns
teacher/student reply tokens and explicitly rejects over-length items. New results
belong with the exact method version and outputs in the results archive.

### 2026-09-23, step-100 checkpoint: fact-conditioned Dream (current method)

[Outputs](results/sleep-2026-09-23/dream_step100_w0_conditioned/sleep_controls.json) and the
[sleep report](results/sleep-2026-09-23/dream_step100_w0_conditioned/sleep_dream_report.json);
source recorded as `08e726a+dirty` (the working-tree patch at launch was not captured). Six taught
facts, W0 target, 20 steps, lr 3e-5, dream temperature 0.7, uniform token weights. The teacher was
shown each accepted turn; it still produced the same greeting in 47 of 48 dreams, which the
duplicate filter removed. The one kept dream had a gain of +2.21 over the reset model, while its
no-quote session-state score difference (the same session state scoring the reply without the
quoted turn, minus the reset model; see the definition above, which is not a controlled
intervention on fast weights alone) was −0.60. The positive score gap disappears without the quote;
what the student distilled therefore came with the quoted turn in the prompt, and this run
demonstrated no taught-fact transfer. Causal attribution to the fast weights remains unresolved.
Taught recall 0/6 before and after; held-out NLL 1.683 → 1.574; cluster share 0.03; gate passed.
The single new verbatim hit was the general-knowledge probe (capital of France), not a taught
fact. Reading: in these 48 sampled dreams the step-100 teacher did not articulate a taught fact even
with the fact in its prompt, so this run gave Dream nothing to consolidate; the two scores are
recorded per dream and are the numbers to read on the final checkpoint, with the caveat above.

### 2026-09-23, step-100 checkpoint: inner surprise versus output entropy (sampling question)

Recorded in the [importance-weighting proposal, section 3](2026-09-23-importance-weighting-proposal.md)
with [outputs](results/sleep-2026-09-23/surprise_vs_entropy_step100/surprise_vs_entropy.json). The
fast learner's per-token surprise adds about 0.002 R² and 0.004 AUC to output entropy as a predictor
of the next token's error on 60 held-out conversations, and neither signal separates looping tokens
from fresh ones in free generation. Entropy-adaptive sampling is the baseline; no surprise-driven
sampler is built. To be repeated once on the final checkpoint.

### 2026-09-23, final chat checkpoint (step 250), seed 0: 24 facts, planted contradictions, every arm, product rule

[Outputs](results/sleep-2026-09-23/final_step250_seed0_exclude/sleep_controls.json), the
[recounted table](results/sleep-2026-09-23/final_step250_seed0_exclude/sleep_controls_recounted.md)
and one sleep report per arm; checkpoint digest `29e0f855…`; source recorded `ccfa446+dirty`
(no experiment-code difference from ccfa446). Protocol: 24 taught facts, 2 boundary facts, 4
planted contradictions taught in one observational session of 30 turns at temperature 0.7,
then floor, in-context ceiling, anchor, replay, distill, Dream and the ungated control, all
W0 × 20 steps, lr 1e-4, replay ratio 0.5, 16 replay and 8 held-out SmolTalk rows. The gated
arms ran under the product rule `flagged_policy=exclude`. The launch-time table carries the
poison/general attribution collision; the recounted table is authoritative.

Findings, per arm:

- **Floor** (fresh session): 0/24 taught. Baseline hits on the general probes (France, Earth)
  and one containment hit on a planted contradiction (the reply to the Moon/Earth question
  contains "Moon"), so poison 1/4 is the baseline, not uptake.
- **Ceiling** (all 30 statements re-taught before each probe): 3/24 verbatim, 2/24 unseen
  phrasing (lighthouse keeper, Nebraska, the Elbe, grandmother Yolanda). Every reply began
  "That's great. I've …": all 52 replies shared that prefix after 30 turns of teaching. The
  archived 6-fact ceiling on the step-50 checkpoint was 5/6. Whether the drop is in-context
  capacity under load, the changed checkpoint, or the thirty-versus-six teaching statements
  is the hypothesis the single-fact ceiling was run to separate; it is not established by
  this run alone. **Single-fact ceiling, same checkpoint and seed
  ([outputs](results/sleep-2026-09-23/final_step250_seed0_ceiling_single/sleep_controls.json)):**
  teaching only the probed fact before each probe gives 22/24 verbatim and 18/24 on unseen
  phrasing (boundary 1/2). So this checkpoint answers from one taught fact in context almost
  every time, and greedy recall fails under the 30-statement session. Whether the stored
  information is lost from the state or present but blocked at retrieval or response is
  unresolved by these greedy answers. Two protocol differences also keep this from being a
  direct probe of the saved Sleep teacher state: the ceiling re-teaches with 8-token
  acknowledgements while the Sleep teach session generates up to 32 tokens per turn, and
  replay consumes the selected raw user text rather than anything the teacher articulates.
  The step-100 Dream runs showed a similar output failure (a teacher repeating one greeting);
  whether the mechanism is shared is not established.
- **Selection under the product rule:** 29 of 30 teaching turns were flagged (would-rollback on
  the chunk where each fact arrives), so anchor, replay, distill and Dream selected ONE turn of
  text. Anchor, distill and Dream still loaded the full committed state of the teach session.
- **Anchor** (λ 0.5, full session state): accepted; held-out NLL 1.612 → 1.603; 0/24 taught.
  It reproduced one planted contradiction verbatim that the floor did not ("At sea level,
  water boils at 50 degrees Celsius"); the second poison hit is the baseline containment. One
  falsehood of four, no true fact of twenty-four, one seed: a signal to watch, not a result.
- **Replay** (one selected turn): accepted; NLL 1.612 → 1.466; 0/24.
- **Distill** (one turn of text, full teacher state): accepted; NLL 1.612 → 1.467; 0/24; general
  verbatim fell 3/7 → 1/7.
- **Dream** (one turn → 6 dreams, 2 kept): rejected by the collapse gate, cluster share 0.31 >
  0.25, with NLL 1.612 → 1.460: the perplexity gate alone would have accepted a collapse again.
- **Ungated** (all 32 turns including the rolled-back ones and every flagged turn): accepted;
  NLL 1.612 → 1.483; 0/24 taught; the one unseen-phrasing hit is a containment artifact ("the
  birds and the bees" for expected "bees"); no rolled-back fact recalled on the two rolled probes.

Reading: on the final checkpoint no method retained a taught fact across a reset, and the
in-context ceiling itself held only 3 of 24 after 30 turns of teaching. **Rerun with every
accepted turn selected** ([outputs](results/sleep-2026-09-23/final_step250_seed0_include/sleep_controls.json),
`--flagged-policy include`, 30 of 30 turns): replay accepted, 0/24, NLL 1.612 → 1.485; distill
accepted, 0/24, → 1.474; Dream generated 180 candidates from the 30-turn teacher state of which
178 were duplicates of one acknowledgement ("That's great. I'm glad to hear that you're enjoying
your notes…"), kept 2, and was rejected by the collapse gate (0.31 > 0.25). The two kept dreams score about
+2.2 nats per token above the reset model under the session state with the quoted turn and, within
0.05, the same without it: the gain is the session state's own frame, not the fact. So the one-turn
text selection was not what hid retention, and the saved teacher state itself, prompted with each
fact, produces the same sentence for every fact: an output failure observed at the Sleep teacher
directly, with storage versus retrieval still formally open. The product rule
turned the gated arms into one-turn runs, which is a finding about the rule in observational
sessions (see Provenance), not about the methods; the ungated control, which consumed every
turn, still retained nothing. The next runs consolidate every accepted turn (`--flagged-policy
include`) on seeds 0, 1 and 2, and measure the single-fact ceiling.
