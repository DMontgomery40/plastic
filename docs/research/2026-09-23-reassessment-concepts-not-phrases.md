# Reassessment: concepts, not phrases. What the literature says about measuring, selecting and consolidating facts

Draft, 23 September 2026. Written for Fable's review before commit. This is a source-checked
reading memo, not a result. Claims marked "abstract only" were checked against the arXiv abstract
page; "body" means the relevant section of the HTML rendering was read. Numbers quoted from the
[community research note](2026-09-23-sleep-community-research.md) are marked "(community note)".

## Research readiness note

**Task and constraints.** David paused Sleep after three seeds: 15 attempts, 13 accepted, 0 of 24
taught facts retained, only verbatim planted falsehoods transferred (sleep note, final sections;
FABLE-178 to 186). His critique: learn relations, not sentences; one instance should not become
permanent. This memo answers "what does the research say" from primary sources. No code, runs or
GPU; no other file edited.

**Local sources read.** Sleep note (last four sections and "Relation to prior work"), community
research note, importance-weighting proposal, FABLE-178 to 186, and `plastic/sleep/recall.py`,
whose probe is a verbatim question plus at most one paraphrase scored by substring containment.

**Primary sources checked (all on 2026-09-23).** Listed in the final section with versions.

**Closest known mechanisms.** Padmanabhan et al. context distillation (already compared in the
sleep note); Behrouz et al. Sleep (wholesale transfer plus gradient-scored dreams); ConsistencyGate
(self-consistency admission control for agent memory); Sun et al. 2023 generalization-optimized
systems consolidation (neuroscience theory); EntiGraph and SEAL (relation- and implication-generating
augmentation).

**Distinction under investigation.** Whether a selection rule of recurrence and consistency,
measured by a paraphrase/reverse/entailment/locality battery, consolidates concept-level facts where
surprise or gain selection consolidated strings and planted contradictions.

**Unresolved assumptions.** (1) That the 760M TTT-MLP session state holds any relation-level
information at all; no source tests this for any TTT or fast-weight model, and our in-context
ceiling under a 30-turn load was 3/24. (2) That the neuroscience accounts transfer beyond analogy.
(3) That a self-consistency score from a 760M model is informative rather than noise.

**Falsifying check.** Section 5, step 3: if the recurrence-and-consistency selector retains no more
battery-level facts than the gain selector on matched sleep runs, or also selects the planted
contradictions, the hypothesis fails on this checkpoint.

## 1. How the field measures "learned as a concept" rather than "stored as a string"

The knowledge-editing literature separated these two decades ago in metric form, and its
consistent finding is that parameter edits pass the string metrics and fail the concept metrics.

- **ROME** (2202.05262 v5, 2023-01-13, body §3.3) defines Efficacy Score as the share of cases where
  the edited object beats the original, P[o*] > P[o_c], on the edit prompt; Paraphrase Score as the
  same test on "rephrased prompts equivalent to (s, r)"; Neighborhood Score as P[o_c] > P[o*] holding
  on nearby subjects (specificity); plus Reference Score (TF-IDF consistency of generations) and
  Generation Entropy (fluency). It reports that fine-tuning reaches "high generalization at the cost
  of making mistakes on most neighboring entities". All are probability comparisons, not greedy
  substring hits.
- **KnowEdit** (2401.01286 v5, 2024-11-17, body §3.5, §4.2) adds Portability with three parts:
  Alias (subject replaced by a synonym), Compositionality and Reasoning (reason with the changed
  fact), and Logical Generalization (semantically related facts that should change). Its result:
  ROME and MEMIT have strong edit success but "their portability is unsatisfactory".
- **RippleEdits** (2307.12976 v2, 2023-12-20, abstract and body summary) evaluates Logical
  Generalization, Compositionality I and II, Subject Aliasing, Preservation and Relation
  Specificity. Editing methods "fail to introduce consistent changes"; a plain in-context baseline
  beat them; two-hop composition is worst.
- **Why ripples are messy** (2407.12828 v3, 2025-07-20, abstract): GradSim, the cosine between the
  gradient of the edited fact and of its related knowledge, correlates strongly with ripple success
  across models and methods; negation, over-ripple and multilingual failures sit at low GradSim.
  This is the nearest measurable form of David's "shift the vectors between Moon and Earth and big
  and small": whether the taught fact's gradient points the same way as its entailments.
- **Long-form evaluation** (2402.09394, 2024, abstract): edits judged in long generations show
  factual drift and internal inconsistency, and the protocol has "very little relationship with
  previous short-form metrics".
- **2025 to 2026 successors.** Logical-rules benchmark (2606.10554 v1, 2026-06-09, abstract): ROME
  and FT show "a substantial performance gap, up to 24%, between evaluations on directly edited
  knowledge and on entailed knowledge". KUP (2504.12523 v1, 2025-04-16, abstract): the best
  continued-pretraining models score under 2% on indirect (reasoning) probes. Pressure-aware
  neighborhood optimization (2606.01610, 2026-06-01, abstract) adds a semantic pre-execution gate.
  CODE (2605.28303, 2026-05-27, abstract): fact overwriting gives a 95.6% self-refutation rate;
  causal-narrative grounding drops it to 6.6%. The bilinear-representation paper (2509.21993 v3,
  ICLR 2026, abstract): editing consistency "depends not only on the choice of algorithm but on the
  underlying representational geometry of the knowledge itself"; the reversal curse (2309.12288 v4,
  2024-05-26) is the canonical string learned in one direction only, reversible in context.

**Battery to adopt.** For each taught fact, six probe classes, all asked from a fresh state:
verbatim (kept, labeled a memorization control); at least three held-out paraphrases; the reverse
direction ("Who owns a cat called Marlowe?"); one entailment or composition ("Does my cat's name
start with M?"; for Moon/Earth, "Could the Earth fit inside the Moon?"); one locality neighbor in
the same frame ("water freezes at ... degrees"); and, for planted contradictions, the same reverse
and entailment probes, since a concept-level uptake of "the Moon is larger" must flip its
entailments while a string uptake flips only the verbatim frame. Score each with the ROME-style
margin (log-prob of the taught answer minus log-prob of the pre-teaching answer) and greedy
containment, reported separately. Define the concept score as the minimum over paraphrase, reverse
and entailment, and the string score as verbatim alone. Seed 2's Dream result (verbatim flipped,
unseen phrasing lost even containment, FABLE-184) is already a string-score-only signature.

## 2. Exposure: how many, what kind, and the one-instance rule

Every positive parametric result I found multiplies exposures across diverse frames; none reports
a single bare exposure yielding a paraphrase-robust fact.

- **Physics of LMs 3.1** (2309.14316 v3, 2024-07-16, body): bioS with one biography per person
  gives 9.7% out-of-distribution QA after fine-tuning; five templated rewrites plus sentence
  permutation give 96.6%. Adding augmented "celebrities" lifts unaugmented minorities from 4.4% to
  86.8%. Linear probes show that with augmentation the attribute is encoded at the entity-name
  positions; without it the knowledge is spread over the text. Extractability is a property of how
  the fact was tied to the entity across contexts, not of whether the string was memorized.
- **Mecklenburg et al.** (2404.00213 v2, 2024-04-02, body): fact-based generation, one QA set per
  atomic fact, scales without regressions; token-based rewriting gains "drop off ... with 10x
  scaling" and leaves about 20% of facts uncovered. The often-repeated "saturates around 10
  paraphrases" figure I could not locate in the body; treat it as unverified.
- **O'Neill** (2607.11020, 2026-07, abstract; community note): after twenty sequential writes,
  bare-statement facts retain 1%, study-set facts (about 24 diverse items) 46%; the
  recitation-to-use gap shrinks from 27.4 to 5.4 points; 70% of wrong answers under bare training
  contain the most recently written fact.
- **SEAL** (2506.10943 v2, 2025-09-18, body): SQuAD no-context accuracy 32.7 base, 33.5 passage
  only, 39.7 passage plus self-generated implications, 46.3 with GPT-4.1 implications, 47.0 SEAL;
  "performance on earlier tasks gradually declines as the number of edits increases".
- **EntiGraph** (2409.07431 v2, 2024-10-03, abstract): models "trained on hundreds to thousands of
  diverse representations" of a fact learn it; synthetic text "drawing connections between the
  sampled entities" rearranges knowledge for data-efficient learning.
- **Priming** (2504.09522, body): on Outlandish, "keyword probability had the most robust
  correlation with amount of priming", with a threshold near 1e-3 below which the new fact bleeds
  into unrelated contexts; "surprising training data will bleed more into unrelated knowledge".
  Stepping-stone rewrites cut priming by a median 75%, pruning the top 8% of updates by 96%, both
  while keeping memorization.
- **Knowing-Using gap** (2607.08393, 2026-07-09, abstract): memorization is fast, generalization
  lags; the authors attribute it to memorized representations "not routed to computation-effective
  layers". From Style to Facts (2503.05919, 2025-03-07, abstract): QA format beats documents,
  numbers are hardest, multi-step use fails even when trained on comparable examples.

**Implication.** "One instance should not become permanent" is the empirical regime, not only a
value: a single surprising exposure stores a string, primes unrelated contexts most, and is
overwritten by the next write. Multiplication can come from self-generated diversity (study sets,
dreams) or from recurrence across the user's turns and sessions. Seed 2 shows the danger of the
first alone: the generator multiplied the planted contradiction 22 times. Recurrence should gate
whether a fact is a candidate; generated diversity should govern how it is written once admitted.

## 3. What to consolidate: criteria other than surprise

**Neuroscience.** Updated CLS theory (Kumaran, Hassabis, McClelland, TiCS 2016, abstract) holds that
replay "allows goal-dependent weighting of experience statistics" and that "neocortical learning can
be rapid for information that is consistent with known structure"; Tse et al. (Science 2007,
abstract) showed one-trial associations become hippocampus-independent quickly only when a schema
already exists. Sun et al. (Nature Neuroscience 2023, abstract) formalize the selection problem:
"unregulated neocortical memory transfer can cause overfitting and harm generalization in an
unpredictable world", so "memories only consolidate when it aids generalization", which in their
model means the predictable components; the unpredictable remainder stays episodic. Replay
selection is frequency- and reward-weighted: Yang, Buzsáki et al. (Science 2024, abstract) find
sleep "continued to replay those trial blocks that were reactivated most frequently during waking";
Huelin Gorriz et al. (Nature Communications 2023, abstract) find sleep replay rate rises with the
number of laps run, falls with familiarity, and is predicted by cumulative awake replay; Michon et
al. (Current Biology 2019, abstract) find replay proportion tracks reward size. Together: recurrence
of the not-yet-known, schema consistency, and value, not surprise alone.

**Machine learning instantiations.**

- **ConsistencyGate** (2607.22962 v1, 2026-07-25, body): a write-time gate queries the model K = 5
  times for a soft 0 to 1 support score of candidate m given context c, admits when the mean is at
  least tau = 0.7. On synthetic MemContam, contamination falls from 50% to 1.2% and QA F1 rises from
  0.474 to 0.840; on real conversations the gain is smaller (34.1% and 36.7% contamination) and the
  cost lands on facts "stated only implicitly" (recall 0.58 on LoCoMo). Its diagnosis applies to
  our gate: "utility-based criteria (relevance scoring, novelty filtering, recency management) are
  structurally blind to contamination". This is a non-parametric memory, and the score is a
  self-judgment; in our terms it is numeric model evidence, not a keyword heuristic, but it is a new
  decision signal and must be labeled as one.
- **Sleep** (2606.03979 v2, 2026-07-10, body §3.3 to 3.4): consolidation is wholesale ("transfer the
  knowledge of MLP(f_{l*-1}) ... into the expanded set of parameters") from teacher-sampled data, with
  no content gate. Dreams are scored by the gradient of the SFT loss, top-k kept, and rewarded 1 if
  an isolated update improves performance. That is the same family as our gain gate, and it rewards
  what the model finds least expected. Their evaluations are class-incremental and long-context
  benchmarks, not paraphrase retention.
- **Titans** (2501.00663 v1, body §3.1) adds "past surprise" momentum so the memory does not stall
  after "a big surprising moment"; that addresses missing the aftermath, not admitting anomalies.
  SuRe (2511.22367 v1, 2025-11-27, abstract) prioritizes replay by high NLL and reports no downside;
  absence of a reported downside is not evidence of none. EVAF (2606.29916 v1, 2026-06-29, body)
  gates LoRA consolidation on valence times surprise with an external valence oracle; its
  "test-retest" is a before/after surprise delta, not recurrence, and its downstream effect is
  "54% ± 44% vs 1% frozen ... not significant at n=4". SDFT (2601.19897 v2, 2026-08-07, abstract)
  is the on-policy relative of our distill arm, not a selection rule.

**Where surprise is known to prefer anomalies.** A teacher-minus-student log-ratio is largest
where the base model finds content least plausible, which includes true novelties and
contradictions alike (importance proposal §5). On our protocol it chose the planted contradiction
22 of 24 times (FABLE-183), and the anchor transferred only the falsehood in a fluent prior frame
(FABLE-180). The priming paper shows low-probability keywords spill over most, and Amnesia
(2606.12655, community note) shows replay selection is an attack surface. No source found argues
that surprise alone selects for truth or recurrence.

## 4. Do fast weights carry relations, and how could relations reach slow weights

**Evidence on fast-weight content.** The TTT inner objective reconstructs one token's views,
l(W; x_t) = ||f(theta_K x_t; W) - theta_V x_t||^2 (2407.04620 v4, body §2.3), and Theorem 1 (§2.6)
shows TTT-Linear with batch GD equals linear attention: a key-to-value associative map. TTT-MLP is
the nonlinear regression of the same target; the paper evaluates perplexity only. Titans evaluates
needle recall, not paraphrase. Facts as First Class Objects (2603.17781 v1, 2026-03-18, body): Titans
reaches "100% training loss convergence" with free-form recall 0 to 40% and "interference during
retrieval even when facts are successfully encoded"; no paraphrase or relation test. SR-TTT's
corrected exact match of 0% (community note) points the same way. Our anchor analysis found the
fold lifts every short factual completion by 1.2 to 2.6 nats and the taught facts by 0.3: answer
shape, not entity-attribute binding (FABLE-180). No source shows relation-level (reverse, entailed)
information in a TTT or fast-weight state; none has tested it; the mechanism does not target it.
Untested, not disproven, and testable on our saved states without training.

**Mechanisms for relational change in slow weights, with failure modes.**

| Mechanism | Evidence | Known failure |
| --- | --- | --- |
| Context distillation on generated continuations (Padmanabhan 2306.09306 v2) | +31.8 inference accuracy at -1.6 specificity vs fine-tuning's -15.9 (community note) | Depends on generator quality; our 760M teacher loops |
| Self-generated implications (SEAL) | 39.7 vs 33.5 passage-only; stronger generator 46.3 | Sequential forgetting; weak generator, weak implications |
| Entity-relation synthesis (EntiGraph) | data-efficient scaling from relation-connecting text | Hallucinated relations become targets; needs a capable generator |
| Causal-narrative on-policy distillation (CODE) | self-refutation 95.6% to 1.8%, multi-hop 83.5% (abstract only) | Unverified beyond abstract |
| Fact-based QA sets per atomic fact (Mecklenburg; O'Neill study sets) | 46% vs 1% retention after 20 writes | Recency interference remains |
| Representational geometry (bilinear) | consistent edits need bilinear structure, induced by relational pretraining | Not available post hoc on a pretrained 760M model |
| Update pruning / stepping-stone rewrites (priming paper) | 96% / 75% less spillover, memorization kept | Untested for retention across resets |

Shared failure modes across all of them: the reversal curse (a direction learned by gradient is not
its reverse), the knowing-using gap, priming from surprising items, recency overwrite, and, specific
to us, training on the model's own degenerate generations.

## 5. What this means for plastic

**Measurement.** Replace the two-probe recall with the six-class battery of section 1, scored by
log-prob margin and containment, with concept and string scores reported separately for taught
facts and for planted contradictions. Apply it first to the saved teacher states in context, then
to children. Keep the current gate, but name it a damage gate; it has no retention or contamination
check and passed 13 of 15 runs that retained nothing.

**Selection criterion to test.** Recurrence times consistency, both numeric from the model:
recurrence is the number of distinct turns or frames in which the session asserted the fact;
consistency is the mean over K = 5 paraphrased support queries to the session state (ConsistencyGate
form), or the agreement of the state's answers across paraphrases. Surprise and gain stay logged and
plotted, not deciding. Planted contradictions should fail consistency against the model's prior;
personal facts have no prior and must earn admission by recurrence.

**Smallest honest experiment, no new model training beyond sleep runs.**

1. Read-only probe of the three archived 30-turn teacher states with the battery. If no fact
   answers a reverse or entailment probe in context, consolidation cannot be expected to produce
   one; the bottleneck is upstream of Sleep and the next work is the in-context state, not the
   fold. This step needs inference only.
2. One protocol variant of `sleep_controls.py`: 12 facts taught twice in different frames, 12
   taught once, 4 planted contradictions once, three seeds.
3. Two selectors on the same candidate pool (taught statements plus generated paraphrases): gain,
   and recurrence times consistency, plus a count-matched random selector. One consolidation
   method each (replay or distill at the community note's lower learning rate). Prediction on the
   record: gain selects the plants; recurrence-consistency selects repeated facts and rejects plants.
   Falsifier: the recurrence-consistency child retains no more concept-score facts than the gain
   child, or also admits the plants. About nine sleep runs; Dream took 1804 s, so this is an
   afternoon on MPS.

The result is bounded above by step 1. A negative there is itself the finding David asked for.

## Sources checked (2026-09-23)

| Source | Version, date | Level |
| --- | --- | --- |
| [ROME 2202.05262](https://arxiv.org/html/2202.05262v5) | v5, 2023-01-13 | body §3.3 |
| [KnowEdit 2401.01286](https://arxiv.org/html/2401.01286v5) | v5, 2024-11-17 | body §3.5, §4.2 |
| [RippleEdits 2307.12976](https://arxiv.org/abs/2307.12976) | v2, 2023-12-20 | abstract, summary |
| [GradSim 2407.12828](https://arxiv.org/abs/2407.12828) | v3, 2025-07-20 | abstract |
| [Long-form eval 2402.09394](https://arxiv.org/abs/2402.09394) | 2024-03 | abstract |
| [Logical rules 2606.10554](https://arxiv.org/abs/2606.10554) | v1, 2026-06-09 | abstract |
| [KUP 2504.12523](https://arxiv.org/abs/2504.12523) | v1, 2025-04-16 | abstract |
| [Pressure-aware 2606.01610](https://arxiv.org/abs/2606.01610) | 2026-06-01 | abstract |
| [CODE 2605.28303](https://arxiv.org/abs/2605.28303) | 2026-05-27 | abstract |
| [Bilinear 2509.21993](https://arxiv.org/abs/2509.21993) | v3, ICLR 2026 | abstract |
| [Reversal curse 2309.12288](https://arxiv.org/abs/2309.12288v4) | v4, 2024-05-26 | abstract |
| [Physics 3.1 2309.14316](https://arxiv.org/html/2309.14316) | v3, 2024-07-16 | body |
| [Mecklenburg 2404.00213](https://arxiv.org/html/2404.00213v2) | v2, 2024-04-02 | body |
| [O'Neill 2607.11020](https://arxiv.org/abs/2607.11020) | 2026-07 | abstract; community note |
| [SEAL 2506.10943](https://arxiv.org/html/2506.10943) | v2, 2025-09-18 | body |
| [EntiGraph 2409.07431](https://arxiv.org/abs/2409.07431) | v2, 2024-10-03 | abstract |
| [Priming 2504.09522](https://arxiv.org/html/2504.09522) | 2025 | body |
| [Knowing-Using 2607.08393](https://arxiv.org/abs/2607.08393) | 2026-07-09 | abstract |
| [Style to Facts 2503.05919](https://arxiv.org/abs/2503.05919) | 2025-03-07 | abstract |
| [PASTA 2606.28898](https://arxiv.org/abs/2606.28898) | v1, 2026-06-27 | abstract; no paraphrase counts given |
| [Kumaran et al. TiCS 2016](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1016/j.tics.2016.05.004&resultType=core&format=json) | 2016 | abstract (Europe PMC) |
| [Tse et al. Science 2007](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1126/science.1135935&resultType=core&format=json) | 2007 | abstract (Europe PMC) |
| [Sun et al. Nat Neurosci 2023](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:37474639%20AND%20SRC:MED&resultType=core&format=json) | 2023 | abstract (Europe PMC) |
| [Yang, Buzsáki et al. Science 2024](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE:%22Selection%20of%20experience%20for%20memory%20by%20hippocampal%20sharp%20wave%20ripples%22&resultType=core&format=json) | 2024 | abstract (Europe PMC) |
| [Huelin Gorriz et al. Nat Commun 2023](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41467-023-43939-z&resultType=core&format=json) | 2023 | abstract (Europe PMC) |
| [Michon et al. Curr Biol 2019](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE:%22Post-learning%20Hippocampal%20Replay%20Selectively%20Reinforces%20Spatial%20Memory%20for%20Highly%20Rewarded%20Locations%22&resultType=core&format=json) | 2019 | abstract (Europe PMC) |
| [ConsistencyGate 2607.22962](https://arxiv.org/html/2607.22962v1) | v1, 2026-07-25 | body |
| [Sleep 2606.03979](https://arxiv.org/html/2606.03979v2) | v2, 2026-07-10 | body §3.3 to 3.4 |
| [Titans 2501.00663](https://arxiv.org/html/2501.00663v1) | v1 | body §3.1 |
| [SuRe 2511.22367](https://arxiv.org/abs/2511.22367) | v1, 2025-11-27 | abstract |
| [EVAF 2606.29916](https://arxiv.org/html/2606.29916v1) | v1, 2026-06-29 | body |
| [SDFT 2601.19897](https://arxiv.org/abs/2601.19897) | v2, 2026-08-07 | abstract |
| [SCoL 2605.07076](https://arxiv.org/abs/2605.07076) | 2026-05 | abstract; no consistency filter described |
| [When CL requires learning 2607.07847](https://arxiv.org/abs/2607.07847) | v1, 2026-07-08 | abstract; no single-exposure claim |
| [TTT layers 2407.04620](https://arxiv.org/html/2407.04620v4) | v4, 2025-08-31 | body §2.3, §2.6 |
| [Facts as First Class Objects 2603.17781](https://arxiv.org/html/2603.17781v1) | v1, 2026-03-18 | body |
| [Relational recall tracing 2604.19934](https://arxiv.org/abs/2604.19934) | v2, 2026-04-23 | abstract; distributed across heads, no compositional structure claimed |
| [Surprise as plasticity signal 2606.31495](https://arxiv.org/abs/2606.31495) | 2026-06-30 | abstract; no anomaly discussion |

Not re-read today and cited only via the community note: Padmanabhan 2306.09306 v2, SR-TTT
2603.06642, Amnesia 2606.12655, TTT-E2E issue #8. Inaccessible today: cell.com and nature.com
full texts (403), PMC (captcha); abstracts were taken from Europe PMC instead. Not searched: the
psychology literature on spaced repetition and testing effects, and any 2026 work on TTT-state
probing beyond the searches recorded here. No priority claim is made.
