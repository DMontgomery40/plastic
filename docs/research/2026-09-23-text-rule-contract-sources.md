# Text rule contract (T4): what prior work says about teaching string operators in a session

Draft, 23 September 2026, for Fable's review before commit. A source-checked reading memo, not a
result. "Abstract" means the arXiv abstract page was read; "body" means the named section of the
HTML or PMC rendering was read. Nothing here was run.

## Research readiness note

**Task and constraints.** Inform the T4 design: the T1 learning contract
(`docs/superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md`) applied to a text stream on
the 760M TTT-MLP checkpoint (`2026-09-23-ttt-backend.md`): an invented operator system taught by
worked examples in a chat session, tested on unseen operator pairs after the session state and fast
weights are cleared, with a propose-and-verify lasting update, a poisoned stream and a revert
ablation. No runs; no other file edited.

**Local sources read.** `AGENTS.md`, `docs/research/README.md`, the T1 spec, the reassessment memo
(`2026-09-23-reassessment-concepts-not-phrases.md`), the TTT backend note.

**Primary sources checked.** All on 2026-09-23; the closing table gives id, version and level.

**Closest known mechanisms.** MLC (Lake and Baroni 2023): meta-trained, frozen at test, solves
SCAN/COGS lexical splits from ten in-context study examples. Akyürek et al. (2411.07279): per-task
LoRA test-time training on ARC, adapter discarded per task. Ramesh et al. (2311.12997): in-context
composition of bijections and permutations on six-token strings, the nearest analogue of our pairs.

**Specific distinction.** None of these evaluates after the adapted state is removed, none reverts
weights to show the gain vanishes, and none pairs a held-out-composition split with a poisoned
stream and correction. Ouellette (2507.15877) comes closest by asking whether test-time fine-tuning
learns anything new. Bounded search; no priority claim.

**Unresolved assumptions.** (1) That a 760M model learns any word-list operator from a few worked
examples; the nearest evidence (GPT-3 below 1B, character tasks) is near floor. (2) That sequential
composition of two operators on one list is learnable in context at this scale; Xu et al. find scale
helps only "separable" compositions. (3) That TTT-MLP fast weights carry an operator rule at all; no
source tests this.

**Falsifying check for the testbed.** Section 5, last paragraph.

## 1. Compositional generalization benchmarks and their splits

- **SCAN** (1711.00350 v3, abstract): primitive splits (a verb seen alone, tested in composed
  commands) and a length split; RNNs "fail spectacularly" when systematic composition is required.
  **COGS** (2010.05465 v1, abstract): lexical versus structural generalization; Transformers and LSTMs
  reach 96 to 99% in distribution and 16 to 35% out, with 6 to 8 points of seed variance. **CFQ**
  (1912.09713 v2, abstract): maximum compound divergence with low atom divergence; accuracy falls as
  compound divergence rises. The standard split: every primitive seen in training, only the pairing
  new at test; length is a separate, harder split.
- **MLC** (Lake and Baroni, Nature 2023, PMC body): episodes of 10 study and 2 query examples;
  weights frozen at test. SCAN add-jump 0.22% error, around-right 0.04%, opposite-right 0.06%; COGS
  lexical 0.87%. It fails the SCAN length split and scores 100% error on COGS structural splits. The
  compositional skill lives in meta-trained slow weights, used through context. **Han and Padó**
  (2403.11834 v1, abstract) find the same direction on SCAN, COGS and GeoQuery.
- **Ramesh et al.** (2311.12997 v2, body): bijections (token lookup) and permutations (reorder six
  tokens) over vocabulary 10, composed by task tokens. With intermediate outputs, a 12-layer nanoGPT
  trained on 30 to 100 compositions generalizes to 3125 unseen ones; the direct format fails unless
  about 64% of compositions are seen. Compositions whose operator order never appeared in training
  fail.
- **Abedsoltan et al.** (2502.08991 v2, abstract): T operations from D subtasks; about Õ(D) training
  tasks suffice for D^T in theory, shown on sparse parity with ICL plus chain of thought. The snippet
  claim that ICL without chain of thought fails is unverified. **Xu et al.** (2407.15720 v2,
  abstract): LLMs compose in-context tasks when the composition is "separable" (different mappings
  on different input segments); sequential multi-step composition does not improve with scale.
  **Lampinen et al.** (2505.00661 v3, abstract): ICL generalizes reversals and deductions more
  flexibly than finetuning; finetuning on in-context reasoning traces closes the gap. That is the
  published form of propose-and-verify: consolidate the model's own in-context inferences, not the
  raw stream.

**Failure modes to design against.** Length extrapolation fails even for MLC. Direct-format
composition is learned only by covering most of the pair table (surface memorization). An operator
never seen in second position is not applied there. The held-out set must therefore contain unseen
ordered pairs, both orders of non-commuting pairs, and a separately labeled length probe.

## 2. What small models learn about strings from worked examples in context

- **GPT-3** (2005.14165 v4, body §3.9.2): character tasks with K = 100 examples. At 175B few-shot:
  cycle letters 37.9%, anagram A1 15.1%, A2 39.7%, random insertion 67.2%, reversed words 0.44%.
  Reversed words is 0% at every smaller size; at 760M the other four read off Figure 3.11 at roughly
  4 to 20% (approximate). One-shot halves performance or worse. These are within-word character
  tasks, harder for a tokenizer than our word lists, but they are the only sub-1B numbers found.
- **Embers of Autoregression** (2309.13638 v1, abstract): accuracy tracks output probability even
  on deterministic tasks; GPT-4 decodes a shift cipher at 51% for high-probability output and 13%
  otherwise; word-sequence reversal is one of its tasks.
- **Wei et al.** (2303.03846 v2, abstract): small models ignore flipped labels and follow priors;
  arbitrary input-label mapping emerges with scale. **In-Context Fixation** (2605.08295 v1,
  abstract): six models from 0.8B to 8B put 42 to 67% of probability mass on tokens seen in the
  label position. **Min et al.** (2202.12837 v2, abstract): random labels barely hurt across 12
  models; label space, input distribution and format carry the effect.
- **Fu et al.** (2609.03213 v1, abstract): across five tasks, models learn more reliably from a
  stated rule than from examples; more examples add no consistent gain; the rule advantage is
  largest on formal, algebraic tasks.

**Bound on the in-context baseline.** No source found reports a model at or below 1B learning a
word-list reverse or swap from a handful of worked examples. The 760M checkpoint may fail to bind
the operator names (Wei), reproduce the format without the rule (Min), or favor permutation
operators, whose outputs reuse the demonstrated vocabulary, over uppercase and duplicate, which add
new tokens (Fixation). The in-context curve on single operators is the gating measurement.

## 3. Test-time training and fast weights on compositional or algorithmic tasks

- **Akyürek et al.** (2411.07279 v2, body): a per-task LoRA trained at inference on leave-one-out
  tasks from the demonstrations plus rotations and flips, about 2 epochs over 40 shuffles, from a
  base fine-tuned on synthetic ARC-like tasks. On the 80-task development set a 1B model goes from
  about 5% to about 29%; the fine-tuned 8B goes from 47.1% to 53.0% on public validation. Adapters
  are discarded per task. No evaluation after removing the adapter.
- **ARC Prize 2024** (2412.04604 v2, abstract): private-set state of the art 33% to 55.5%, with
  test-time training among the drivers. **ARC Prize 2025** (2601.10904 v1, body): NVARC (24.03%)
  builds on the ARChitects test-time-training entry with heavy synthetic data; MindsAI (12.64%) is a
  test-time fine-tuning pipeline. Model sizes and per-task reset are not stated in the report.
- **Ouellette** (2507.15877 v2, abstract): a controlled compositional experiment in the ARC domain;
  execution-guided program synthesis composes novel solutions best, and test-time fine-tuning's
  success "appears to stem primarily from activating preexisting knowledge".
- TTT layers (2407.04620 v4) are evaluated on perplexity; In-Place TTT and TTT-NTP ablate state and
  chunk sizes on held-out loss (T1 spec, not re-read). Searches for test-time training on SCAN, COGS
  or string manipulation in 2025 to 2026 found nothing on point.

**Gap.** No TTT or fast-weight paper found evaluates after resetting state and reverting weights,
and none uses a held-out-pair split. That TTT gains are per-task and discarded, and possibly
activation rather than learning (Ouellette), is exactly the hypothesis the T4 revert ablation and
reset measurement separate.

## 4. Poisoned lessons: definitions of harm and correction

- **ICLAttack** (2401.05949 v6, abstract): correctly labeled poisoned demonstrations act as
  triggers; about 95% attack success on OPT 1.3B to 180B; no defense evaluated. **ICLPoison**
  (2402.02160 v3, abstract): discrete perturbations that steer hidden states and degrade accuracy.
  **advICL** (2305.14950 v2, abstract): robustness falls as demonstrations increase; adversarial
  demonstrations transfer to unseen inputs.
- **ICLShield** (2507.01321 v1, abstract): the model learns a task concept and a backdoor concept
  from the same demonstrations; vulnerability is governed by their preference ratio; demonstration
  selection by confidence and similarity gives +26.02 points. **In-context unlearning**
  (2310.07579 v4, abstract): label-flipped instances in context remove a training point's influence.
- Continual learning: **PACOL** (2311.10919 v1, abstract) defines label flipping of earlier-task
  samples and a clean-label poison; harm is forgetting; defenses are sanitization and outlier
  detection. **Hu and Duan** (2606.29841 v1, abstract): no defense succeeds when a linear proportion
  of tasks is poisoned with unbounded noise; for infrequent attacks a task-to-task verification step
  detects poison. That step is the nearest published form of propose-and-verify.

**What is missing.** ICL poison work uses triggers and labels, not a consistent false rule that is
learnable everywhere, and measures attack success rather than uptake of the false semantics, loss of
clean competence and recovery after a clean stream. The T1 `poison_stream` (bias whenever the x
action is positive) is the template. Wei et al. is the confound: a small model that ignores the
poison may be unable to learn any rule in context, so refused-bad must be read next to
accepted-good, as the T1 spec requires.

## 5. What this means for T4

**Operator set and split.** Keep the six operators. They give 30 ordered pairs; most do not
commute, so order is testable. Teach all six singles and 12 ordered pairs chosen so every operator
appears in both positions (low atom divergence); hold out the other 18, including the reverse order
of at least six taught pairs. Lists of 4 or 5 unrelated common nouns, fixed across arms; a separate
length-6 probe labeled as a length split and expected to fail. Report uppercase and duplicate last,
which add tokens absent from the input, separately from the four permutation-like operators.

**Examples per episode.** MLC used 10 study examples; GPT-3 one-shot halved performance; Fu et al.
find a stated rule beats examples. Each lesson: one rule sentence plus worked examples, with the
in-context curve measured at 2, 4, 8 and 16 examples per single operator before any lasting-update
run. For pairs, two arms: direct output, and intermediate output shown (Ramesh); the intermediate arm
is the one expected to generalize and it changes the scoring target.

**Scoring.** Exact match on the normalized output string. Continuous: mean per-token
log-probability of the correct output, teacher-forced, and the margin against the strongest
distractor from the same input (the other composition order and each single-operator output), in
the ROME-style form the reassessment memo adopts. Report the fraction of items where the correct
output beats every distractor; that separates rule from format (Min). Every number with its
denominator; the no-context floor for the same output strings (Embers) as the baseline row.

**Contract mapping.** Transfer: held-out pairs after context cleared and fast weights reset, with
and without fast adaptation. Speed: exact and log-prob at example k within a session, adapting
versus writes-disabled. Forgetting: singles and taught pairs after the update versus before.
Correction: a poisoned stream that consistently redefines one operator (for example "reverse"
demonstrated as rotate-by-one throughout); harm is uptake of the false semantics on held-out pairs
containing that operator plus the clean-accuracy drop; then a clean stream; accepted-good and
refused-bad with counts. Revert: restore slow parameters; a gap above tolerance is a bug.

**Falsifying checks for the testbed.** First, the gate: if 16-shot single-operator exact match sits
at the no-context floor, the checkpoint cannot be measured and the work moves to the in-context
substrate. Second, the T1 analogue: if frozen (TTT-only, then reset) and continued training score
the same on held-out pairs after reset, the stream carries nothing a slow update can use. Third, the
Ouellette and Min check: a format-only stream (same lessons, shuffled outputs) that yields the same
lasting gain as the true stream means the update consolidates format, not rule.

## Sources checked (2026-09-23)

| Source | Version, date | Level |
| --- | --- | --- |
| [SCAN 1711.00350](https://arxiv.org/abs/1711.00350) | v3, 2018-06-06 | abstract |
| [COGS 2010.05465](https://arxiv.org/abs/2010.05465) | v1, 2020-10-12 | abstract |
| [CFQ 1912.09713](https://arxiv.org/abs/1912.09713) | v2, 2020-06-25 | abstract |
| [MLC, Lake and Baroni, Nature 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620072/) | 2023 | body (PMC) |
| [Han and Padó 2403.11834](https://arxiv.org/abs/2403.11834) | v1, 2024-03-18 | abstract |
| [Ramesh et al. 2311.12997](https://arxiv.org/html/2311.12997v2) | v2, 2024-02-05 | body |
| [Abedsoltan et al. 2502.08991](https://arxiv.org/abs/2502.08991) | v2, 2025-06-09 | abstract |
| [Xu et al. 2407.15720](https://arxiv.org/abs/2407.15720) | v2, 2024-08-11 | abstract |
| [Lampinen et al. 2505.00661](https://arxiv.org/abs/2505.00661) | v3, 2025-11-10 | abstract |
| [GPT-3 2005.14165](https://arxiv.org/html/2005.14165v4) | v4 | body §3.9.2; sub-1B values are figure readings |
| [Embers 2309.13638](https://arxiv.org/abs/2309.13638) | v1, 2023-09-24 | abstract |
| [Wei et al. 2303.03846](https://arxiv.org/abs/2303.03846) | v2, 2023-03-08 | abstract |
| [In-Context Fixation 2605.08295](https://arxiv.org/abs/2605.08295) | v1, 2026-05-08 | abstract |
| [Min et al. 2202.12837](https://arxiv.org/abs/2202.12837) | v2, 2022-10-20 | abstract |
| [Fu et al. 2609.03213](https://arxiv.org/abs/2609.03213) | v1, 2026-09-02 | abstract |
| [Akyürek et al. 2411.07279](https://arxiv.org/html/2411.07279v2) | v2, 2025-03-25 | body |
| [ARC Prize 2024 2412.04604](https://arxiv.org/abs/2412.04604) | v2, 2025-01-08 | abstract |
| [ARC Prize 2025 2601.10904](https://arxiv.org/html/2601.10904v1) | v1 | body (results) |
| [Ouellette 2507.15877](https://arxiv.org/abs/2507.15877) | v2, 2025-09-21 | abstract |
| [ICLAttack 2401.05949](https://arxiv.org/abs/2401.05949) | v6, 2024-10-09 | abstract |
| [ICLPoison 2402.02160](https://arxiv.org/abs/2402.02160) | v3, 2025-06-02 | abstract |
| [advICL 2305.14950](https://arxiv.org/abs/2305.14950) | v2, 2023-10-14 | abstract |
| [ICLShield 2507.01321](https://arxiv.org/abs/2507.01321) | v1, 2025-07-02 | abstract |
| [In-context unlearning 2310.07579](https://arxiv.org/abs/2310.07579) | v4, 2024-06-06 | abstract |
| [PACOL 2311.10919](https://arxiv.org/abs/2311.10919) | v1, 2023-11-18 | abstract |
| [Hu and Duan 2606.29841](https://arxiv.org/abs/2606.29841) | v1, 2026-06-29 | abstract |

Cited via the T1 spec or reassessment memo without re-reading: 2407.04620 v4, 2604.06169,
2606.21803. Seen only as search snippets: 2511.20194, 2509.24510, 2410.16531, 2311.11995. Not
searched: program induction (DreamCoder and successors) and the psychology of rule learning.
