# Proposal: per-memory and per-token importance, adaptive sampling, and external classifiers as oracles

Status: research proposal, 2026-09-23. Every item below is labeled **measured**, **built (unmeasured)**
or **proposed**. Sources were checked on 2026-09-23 by the review partner (OPUS-004 in the private
scratchpad) and are summarized here with their dates; the private memo's reasoning is reproduced in
substance. No priority claim is made anywhere in this document.

## The direction, in David's words

"Classify on a per-memory or even per-token basis sometimes, and have variant temp and top_p and
top_k based on the context and importance of the variance and token / memory context. Real conscious
humans don't classify every weight and memory and encoding the same way. Multiple classifier layers
may be necessary, like even classifying if a full memory should be classified down to the token."

## What the learner already provides

The transaction harness records, for every token, the fast learner's own surprise, write norm and
inner step size, and per chunk a decision (commit, rollback, scale, project). Sleep's dream method
(see [the sleep note](2026-09-23-sleep-consolidation.md)) records for every generated study item a
gain over the reset model and, since 08e726a, a per-token gain map. Since the current commit that map
is split into a fast-weight part (teacher state without the quoted turn, minus the reset model) and a
turn part, and each dream carries the top-3 concentration of its fast-weight gain. These are numeric
quantities of the model's own state, which is the class of evidence the harness is allowed to decide on.

## 1. Per-token weighting of consolidation

**Built (unmeasured):** `dream_token_weighting` = `uniform` | `gain` | `fw_gain`. Weights are the
clamped per-token gain, floored at 0.2 and normalized to mean 1, applied to the dream KL.

Closest prior methods: token selection by excess loss against a reference model (Rho-1, arXiv
2404.07965, v4 2025-01-08) and token-importance distillation that keeps high-entropy and
high-divergence tokens (TIP, arXiv 2604.14084, v4 2026-05-21; ToDi, arXiv 2505.16297, v2 2025-09-28).

The prediction that must be respected: **KL distillation already weights itself.** Where teacher and
student agree, the per-position KL and its gradient are near zero. Rho-1's gains come from cross-entropy,
whose gradient does not vanish on noisy tokens, and TIP reports that training on under 10% of tokens
matches, not beats, full-token training. Soft gain weighting is therefore expected not to change
retention measurably, except where the divergence comes from the wrong source: the step-100 run in
which the session's fast weights had raised the model's own greeting (FABLE-140) is exactly that case.
Hence the split: weighting by the fast-weight part tests fast-to-slow consolidation; weighting by the
turn part is context distillation. Reports state which ran.

Falsifier: identical kept dreams and seeds, three weightings; read recall on unseen phrasings,
cluster share and poison uptake. A known circularity: the answer log-probability probe scores the
answer tokens, which are the tokens gain weighting up-weights, so that metric cannot compare
weightings. Greedy recall on unseen phrasing can.

## 2. A two-level gate: memory, then token

**Proposed; log-only first.** The gate already exists in levels: harness chunk decision → accepted turn
→ dream gain → token weight. "Whether a memory should be classified down to the token" has a numeric
form: the concentration of a dream's per-token fast-weight gain. Concentrated gain (a name, a place)
argues for token weights; diffuse gain (style, paraphrase) argues for uniform weights or rejection.
Titans' "past surprise" (arXiv 2501.00663) is the caution: a fact spans tokens that are not
individually surprising ("my cat is called" before "Marlowe"), so hard top-k selection would cut the
carrier phrase; the floor in the shipped weights exists for that reason. Concentration is now logged
per dream; the gate is built only if concentration predicts retention per fact.

## 3. Sampling that adapts to the learner's signals

**Proposed; measure first.** Closest methods drive temperature from output entropy (EDT, arXiv
2403.14541; AdapT, arXiv 2309.02772) or from a learned policy on the hidden state (arXiv 2602.13035,
2026-02-13). The distinct version here would drive temperature and top-k from the inner learner's
surprise. Before building it: on held-out chat, does per-token inner surprise predict next-token error
or loops beyond output entropy? If not, entropy-adaptive sampling is the honest baseline and the
TTT-specific version adds nothing.

Trap specific to this architecture: generated tokens write to the fast weights. Changing the sampling
policy changes the learner's own training data and shifts the signal distributions the harness is
calibrated on. Every sampling policy is a separate operating condition with its own calibration.
Measured so far (FABLE-144, step 100): temperature 0.3 loops within replies (repeated 4-gram share
0.185, 4 of 16 replies) where 0.7 does not (0.040, 0 of 16); accuracy similar and poor at both.

**Measured (step 100, 2026-09-23, `scripts/experiments/surprise_vs_entropy.py`, results in
`results/sleep-2026-09-23/surprise_vs_entropy_step100/`).** On 60 held-out SmolTalk test conversations
(7,367 assistant tokens, 14,148 in all), output entropy alone has Spearman 0.85 with the true next token's
NLL and AUC 0.85 for the top-1 miss. The inner surprise (mean over layers and heads) has 0.19 and 0.60, and
adding it to entropy raises R² by 0.002 and AUC by 0.004; the bootstrap over conversations puts both
increments above zero but at that size. The last layer's surprise, the first layer's, the inner step size and
the write norm are each better than the mean but still weaker than entropy and mostly redundant with it
(increments at most 0.005 R²). In free generation at temperature 0.3 (18% of tokens inside a repeated 4-gram)
and 0.7 (6%), neither signal separates looping tokens from fresh ones: both have AUC below 0.5, since a
loop is low-entropy and no more surprising to the learner than fresh text. **On this checkpoint the
answer to the question above is no: entropy-adaptive sampling is the honest baseline and a
surprise-driven sampler is not justified.** Repeated on the final chat checkpoint (step 250, digest
29e0f855…, [outputs](results/sleep-2026-09-23/surprise_vs_entropy_step250/surprise_vs_entropy.json)): entropy
0.86 / 0.85, surprise 0.16 / 0.57. Assistant-token point estimates of the increment: +0.000015 R² and
+0.00021 AUC; the bootstrap means are +0.00011 and +0.00025, the R² interval [0.0000001, 0.00052] stays
above zero and the AUC interval [−0.00007, 0.00093] crosses it. Loop AUC 0.42 / 0.39 for entropy and
0.50 / 0.38 for surprise (below 0.5 means the reverse direction discriminates: looping tokens have lower
entropy and lower surprise; a joint entropy-plus-surprise increment was not measured for loops). **For these
two checkpoints, this feature set and these probes, the increment is tiny and the sampling question has an
entropy answer; no surprise-driven sampler is built.** Adaptive-decoding effects and other configurations
were not evaluated.

## 4. External classifiers (Jev / typesafe.ai, diffusion-based classifiers)

**Proposed as an oracle, not a decision signal.** A classifier is a learned semantic judgment, not the
learner's numeric evidence; used in a decision it is a new decision signal and changes the research
claim from "the model's own inference-time learning signals control retention" to "an external
classifier controls retention". A veto is not neutral either: an asymmetric filter still decides what
is consolidated. Therefore mechanism claims come only from the numeric arm; a classifier arm is a
system variant, default off.

Best research use: feature discovery. Read the dreams on which the numeric gate and the classifier
disagree; if the classifier rejects the style dreams the numeric gate keeps, that points to the
fast-weight-gain share as the numeric signal to add, after which the classifier is removed.

Honest experiment: one candidate-dream pool per session; arms numeric-only (N), numeric plus
classifier veto (N+C), classifier-only (C), and random selection matched to each arm's kept count
(R_N, R_C), plus the floor. Measures: per-dream agreement (Cohen's κ), the disagreement cases read by
hand, recall on unseen phrasings with confidence intervals over facts and seeds, collapse (cluster
share), uptake of planted poison facts, latency and cost. The classifier must never see the probe
questions or the fact list. Granularity is per memory, offline at sleep time; per-token online calls
are infeasible at 12–18 tok/s on MPS. Dream texts contain session content; sending them to a remote
API is stated in every report that does so. Jev returns typed choices, scores and yes/no probabilities
with calibrated confidence, batched (docs.typesafe.ai, read 2026-09-23); a 25B diffusion classifier
would compete for memory with the 760M model on one MPS box.

## 5. Where the harness and consolidation must disagree by design

Importance as a teacher-minus-student log-ratio is highest for content the base model finds least
plausible: new true facts and poison alike. Weighting by it amplifies what the harness should treat
most carefully. Design rule, **built (unmeasured)**: online acceptance is necessary but not sufficient for
consolidation. Harvesting marks a turn *flagged* when any of its chunks was scaled or projected, had
a requested intervention overridden, or carried a would-have-intervened reason in observational mode;
`flagged_policy` excludes such turns from sleep by default, or down-weights them, or includes them
for comparison. The selection is made once and governs every path: the direct rows, which sessions may
lend a committed state to the anchor or the teacher (only sessions with at least one selected turn),
and which turns dreams may quote. Down-weighting scales the replay method's cross-entropy rows and
the KL rows of dreams that quote a flagged turn; distill has no per-turn row and takes exclude or
include only. What exclusion cannot remove is indirect: inside a mixed session the committed state
already carries the flagged turn's writes, so a teacher built from that state is influenced by it.
The post-sleep gate (canaries, cluster share, held-out NLL) remains the second transaction. A new benign fact looks like high importance with clean canaries; a
contradiction of world knowledge looks like high importance with canary coherence damage. Amnesia
(arXiv 2606.12655, 2026-06-10) shows that the choice of replay items is an attack surface even under
auditable budgets; a user controls session content and therefore the gain distribution, so selection
histograms are logged per session. The degenerate and duplicate filters in dream selection are
lexical data hygiene and are named as such in reports.

## Build order (from OPUS-004, adopted)

0. Grow the fact set to 24 with known-fact controls, planted contradictions and unseen phrasings; at
   least 3 seeds. **Built** in `scripts/experiments/sleep_controls.py` (this commit).
1. On the chat checkpoint: re-baseline the in-session ceiling and uniform dream.
2. Log per-token fast-weight and turn gains and concentration for every dream. **Built.**
3. A/B uniform vs gain vs fw_gain on identical dreams and seeds. **Built, unmeasured.**
4. Measure surprise vs entropy as predictors of error and loops on held-out chat; build a sampler only
   if surprise adds information. **Measured on step 100 and on the final checkpoint: a tiny increment
   (section 3); no sampler built. Closed for these runs.**
5. Offline classifier labels per dream on the same pool, blind to probes; agreement analysis with the
   random-matched arms.
6. The memory-then-token gate, only if concentration predicts retention.
