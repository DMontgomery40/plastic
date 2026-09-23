# Community and literature evidence on consolidating TTT fast weights (compiled 2026-09-23)

Compiled by a research agent from ~45 searches and ~50 page fetches across GitHub issues, Hugging
Face discussion tabs, Hacker News, Reddit, X and arXiv, at Fable's request after the sleep null
results (see the sleep design note). Sources are graded by specificity, not venue: A = exact
recipe/numbers, B = concrete but partial, C = anecdotal. Claims below are the agent's readings of
the linked pages; before a research decision rests on one, read the primary source.

## 1. Has anyone persisted or consolidated TTT-layer state across sessions?

No. Every primary source in the TTT-layer ecosystem resets fast weights at sequence or document
boundaries, and no fine-tuning or consolidation report exists for the public checkpoints.

- `test-time-training/ttt-lm-pytorch` issues (12): none on fine-tuning or persistence; #31
  (loading the HF checkpoint) unanswered since Nov 2024; #38 (Feb 2026, "why add the previous
  mini-batch gradient to the current one") unanswered; #36 reports a conv prefill padding bug with
  caching when seq_len > conv_kernel. Grade B for #36, C otherwise.
- `ttt-lm-jax` issues: #16 notes η is a mini-batch × mini-batch decayed matrix, not the per-token
  vector of eqs. 6–7 (unanswered); #6 fine-tuning with TTT layers, no maintainer answer. Grade C.
- HF discussion tabs on `Test-Time-Training/ttt-mlp-760m-pile-8k`: none. RetentionLabs org:
  "Pytorch Conversion", 30–47 downloads each, no notes. Grade C (absence).
- TTT-E2E issue #8: a user reports no exact needle retrieval once the needle leaves the
  sliding-window attention span; unanswered. Grade B. Fast-weight recall of specific facts is
  weak even within a session in the strongest TTT paper.
- In-Place TTT (arXiv 2604.06169): "At document boundaries, the fast weights are reset to their
  pre-trained state." TTT-NTP (2606.21803): "restore the original weight after each sample." Grade A
  that neither consolidates.
- Elastic TTT / Fast Spatial Memory (arXiv 2604.07350, blog): the only TTT work with an explicit
  anchor-and-consolidate rule at test time, θ_{c+1} = θ'_c − λF_c(θ'_c − θ*_c), Fisher-weighted
  around an EMA anchor; still resets per scene. Grade A for the mechanism.
- Consolidator (arXiv 2608.11701): persistent memory across context boundaries with parameters
  fixed; a learned slot operator over non-parametric memory. Grade B.
- Learning, Fast and Slow (arXiv 2605.12484): distilling the fast channel into slow weights
  "plateaus well below" co-training (Appendix H). Grade B.
- Facts as First Class Objects (arXiv 2603.17781): Titans MAC memorizes facts to 100% training
  loss yet free-form recall is 0–40%; "continuous parametric memory suffers interference during
  retrieval even when facts are successfully encoded." Grade A. The same storage-versus-access
  pattern as ours.
- SR-TTT post-mortem (arXiv 2603.06642): a claimed +23% TTT retrieval gain was an evaluation
  artifact, corrected exact match 0%; fast-weight memory needs "a stronger readout path". Grade A.
- lucidrains/titans-pytorch issues: engineering bugs only. HN Titans/TTT threads: no
  consolidation discussion. X threads by the TTT authors: inaccessible (402) or empty for
  "sleep"/"consolidation". Reddit: nothing surfaced.

## 2. Recipes that inject a few facts into a ~1B model with later recall

- arXiv 2607.11020 (Qwen3-4B, LoRA r16 α32, lr 2e-4, 24–192 steps per fact; full FT at 3e-5):
  "bare-statement" writing (two framings) versus a "study" set (24 diverse items: paraphrases, QA,
  implications, contrasts). After 20 sequential writes, bare retains 1%, study 46%. Forgotten facts
  keep 57–67% of their log-probability lift (access lost, not erased); in 70% of bare failures the
  model answers with the most recently written fact. Reverse-KL distillation through the model's
  own merges produced looping in 9.4% rising to 85.2% of cases; "sequential distillation from a
  frozen teacher preserves capability." Grade A.
- Physics of LMs 3.1 (arXiv 2309.14316): no augmentation gives 99% first-token memorization and
  zero QA accuracy; 5 diverse rewrites per entity lift QA fine-tune accuracy 9.7% → 96.6%. Grade A.
- Padmanabhan et al., NeurIPS 2023 (arXiv 2306.09306), GPT-Neo-1.3B context distillation: 5
  generated continuations per entity (p 0.9, T 1.0, 40 tokens), teacher conditioned on the
  definition, KL on tokens after the entity mention, lr 3e-6, 5 epochs. +31.8% inference accuracy
  at −1.6% specificity, versus +23.6% / −15.9% for fine-tuning on the definition. Grade A. The
  closest published analogue of our `distill` arm; far lower lr and generated transfer text.
- Cartridges (arXiv 2506.06266): plain next-token training reaches near-perfect perplexity yet
  fails non-memorization queries; self-study (synthetic conversations from 5 seed prompt types)
  plus context-distillation KL beat it by 8.6 chrF; LoRA parameterization cost MMLU 54.7 → 45.3
  while prefix/KV held. Grade A.
- arXiv 2502.14502 (Llama-3.1-8B-Instruct): LoRA r1, lr 1e-3, 10 epochs, QA format, 100%
  reliability to 500 facts; best with a mixture of known and new facts; damage signature is "a
  twofold decrease in the number of unique answers". Grade A.
- SEAL (arXiv 2506.10943): LoRA r8 α16, lr 2e-4, 1 epoch, 5 generated implications per passage;
  SQuAD no-context 33.5 → 47.0%; retention of edit 1 falls to ~40% after 5 passages. Grade A.
- ARC TTT (Akyürek, arXiv 2411.07279): LoRA r128 α16, lr 5e-5 (1B/3B), 2 epochs, ≤250 examples
  per task, leave-one-out, loss on demonstrations and outputs. ARChitects 2025 (2505.07859):
  LoRA r32, lr 1e-4, 64 steps per task, 16 augmentations. Grade A.
- Sparse Memory Finetuning (arXiv 2510.15103), 1.3B: full FT "learning rates above 5e-6 break
  the model". Grade A.
- Continual Learning Mechanisms Compose (arXiv 2609.06986): generative replay + self-distillation
  + online EWC + merged LoRA: retention 1.2% → 34.9% after 100 tasks. Grade B.

## 3. Repeated-sentence collapse on tiny SFT sets

- HF forum, Llama-3.2-1B QLoRA: causes named were truncation removing EOS, training on the full
  template, prompt-format mismatch, greedy decoding. Grade C.
- Does Prompt Loss Matter (arXiv 2401.13586): a nonzero prompt-loss weight helps when completions
  are short; optimal ~0.15–0.24; larger values lengthen outputs and regularize. Grade A. Consistent
  with our user-token supervision reducing collapse (53% → 33%).
- Memorized knowledge fails to generalize (arXiv 2607.08393): memorization saturates in a few
  epochs while use lags. Grade B.

## 4. TTT-MLP-760M specifics

Nothing beyond the repos: Llama-2 tokenizer; no fine-tuning reports; no quality complaints; the
PyTorch README says training in PyTorch is not recommended. Open unanswered issues: conv-cache
bug (#36), gradient accumulation across mini-batches (#38), η shape (jax #16). Grade C.

## What to try on our setup (each traced to a source above)

1. Teach each fact as a study set (~20 items: paraphrases, QA both directions, implications) and
   measure recall on unseen phrasings (2607.11020; 2309.14316).
2. Distill from a frozen teacher onto generated transfer text, not the transcript: ~5 continuations
   per fact, KL after the fact mention, lr ~3e-6, ~5 epochs (2306.09306). Never chain distillation
   through the merged student (2607.11020).
3. Weight replay ~80/20 and generate it from the model rather than the SFT corpus (Cartridges'
   self-study seed types as the template).
4. Full-parameter learning rate an order of magnitude lower (≤ 5e-6) (2510.15103; 2306.09306);
   keep 1e-4 only for low-rank updates.
5. Mix known facts into the fact set and track unique-answer count as the collapse metric
   (2502.14502).
6. A Fisher/EWC anchor on W0 around the pre-sleep weights, Elastic TTT's rule (2604.07350).
7. Report storage and access separately: log-probability lift of the answer under a fixed probe
   versus greedy recall (2607.11020; 2603.17781). Lift without recall means a readout problem
   (2603.06642).
8. Fractional prompt-loss weight (~0.2) rather than full user-token supervision (2401.13586).

## Searched and found nothing

Fine-tuning or chat-SFT of the RetentionLabs / Test-Time-Training checkpoints; carrying TTT
state across sessions; TTT author statements on sleep or consolidation; Reddit threads on TTT
hands-on use or on personal-fact collapse; ttt-video-dit fine-tuning issues. PDFs not parsed:
2312.05934, "Fine-Tuning Done Right in Model Editing" (ICLR 2026), 2605.10537 beyond its abstract.
