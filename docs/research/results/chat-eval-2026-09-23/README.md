# Chat evaluation of the fine-tuned TTT-MLP checkpoint (2026-09-23)

Checkpoint: `ttt_mlp_760m_chat_v1` at step 250 of 250, the final save of Hugging Face job
`6ab351f352d0dbd7f1d82429` (a100-large, 23822 s wall). Backend checkpoint digest
`29e0f8558d157cbfeaa9670334a38766448c9bdfe2ba651f286db22300aaf8c0`. Base: RetentionLabs TTT-MLP-760M-Base-Pile-8k.
Recipe from `sft_meta.json`: 8,000 SmolTalk conversations (seed 20260923, seven subsets), seq 2048,
batch 6 x accum 3, lr 2e-05, warmup 20, assistant-only loss.

Token units, stated once: the job processed 9,216,000 input positions (masked positions included;
the trainer counts `x.numel()`). The packed corpus holds 6,814,486 unmasked label positions in total;
the run's supervised exposure is neither number and was not logged separately. Last logged training loss
1.3768 (one step, noisy; steps 200-250 ranged 1.31-1.45).

Eval: `scripts/train/eval_ttt_chat.py`, 16 prompts (8 neutral, 8 boundary), top-k 40, 96 new tokens, seed 0,
MPS, through the transaction path. Held-out assistant NLL is nats per assistant token on 100 SmolTalk test rows per
subset (one pass, at temperature 0.7 only; sampling temperature does not affect it). Lower NLL means higher
likelihood under this model on that sample, not general quality.

| File | Temperature | Distinct replies (share of 16) | Within-reply repeated 4-gram share | Held-out assistant NLL |
| --- | --- | --- | --- | --- |
| [`step250_t0.3.json`](step250_t0.3.json) | 0.3 | 1.00 | 0.218 | not run (sampling only) |
| [`step250_t0.5.json`](step250_t0.5.json) | 0.5 | 1.00 | 0.111 | not run (sampling only) |
| [`step250_t0.7.json`](step250_t0.7.json) | 0.7 | 1.00 | 0.008 | everyday-conversations 1.588 (12,065 assistant tokens, 100 rows), smol-magpie-ultra 1.410 (133,708 assistant tokens, 100 rows), openhermes-100k 1.341 (20,478 assistant tokens, 100 rows) |

Reading. Temperature 0.7 is the setting that does not loop (repeated 4-gram share 0.008 against 0.111 at 0.5
and 0.218 at 0.3, matching the step-100 measurement that set the default). The model answers in the chat
format and stays on topic; factual content is often wrong (at 0.7: "a virus is a type of bacterium",
ibuprofen "acts as a pro-inflammatory agent"; the capital of Australia is right at 0.5 and 0.7, "Adelaide" at
0.3). Boundary and neutral prompts produce indistinguishable learner signals (surprise 9.47 vs 9.53). For the
step-50 and step-100 intermediate evaluations see the dated Sleep note; those files recorded no held-out
NLL in this format.
