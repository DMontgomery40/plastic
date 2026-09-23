# A pretrained test-time-training model as the playground's chat backend

Status: implemented backend, base weights only; chat fine-tuning in progress. Date: 23 September 2026.
Scope: engineering record and readiness note for `plastic/backends/ttt_lm/`. No safety, detection or
learning-utility claim is made here.

## Why

The user's objective for the playground is legitimate chat on a test-time-training architecture with
the harness reading the learner's own signals. The Qwen3.5 backend has no TTT layer: its gated-delta
state is a closed-form recurrence and exposes neither an inner loss, an inner learning rate nor a write
norm, which is why those charts were empty (scratchpad ASTRA-144) and why intervening on it suppressed
context (FABLE-091, F2). The toy `lm_wikitext_l4` has every signal but cannot chat. A pretrained model
whose sequence layer IS a gradient-updated, meta-trained fast learner closes that gap.

## Sources checked (2026-09-23)

- Sun et al., *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*,
  [arXiv:2407.04620](https://arxiv.org/abs/2407.04620) (v2, 11 Aug 2024); official PyTorch code
  [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch) (MIT).
  Read: the TTT-MLP inner loop (two-layer GELU MLP fast weights, reconstruction target `V − K`, LayerNorm
  inside the loss, learned per-token inner learning rate `η = base · σ(w·x)/d`, mini-batch 16 with the
  dual form for whole mini-batches and the primal form with accumulated gradients inside a partial one,
  learned `W0`), the Mamba-style backbone with `pre_conv`, shared-QK convolutions and a gate.
- Weights: [RetentionLabs/TTT-MLP-760M-Base-Pile-8k](https://huggingface.co/RetentionLabs/TTT-MLP-760M-Base-Pile-8k)
  and [TTT-MLP-1.3B-Base-Pile-8k](https://huggingface.co/RetentionLabs/TTT-MLP-1.3B-Base-Pile-8k) (MIT),
  conversions of the official JAX train states `Test-Time-Training/ttt-mlp-*-pile-8k`. Base models on
  the Pile, 8k context, Llama-2 32k tokenizer. The conversion's fidelity to the JAX weights is the
  publisher's claim; we checked that the model produces coherent English and sensible factual
  continuations, not numeric parity with JAX.
- Alternatives found and set aside: E²-TTT ([zeyun-zhong/e2-ttt-mlp-*](https://huggingface.co/zeyun-zhong/e2-ttt-mlp-1.3B-15B),
  chunk-512 closed-form TTT-MLP, fused ops), In-Place TTT (4B on VeOmni), TTT-E2E (JAX, GCS). No
  instruct or chat model with a TTT layer exists on the Hub as of this date; chat requires our own SFT.

## What the backend does

`TTTBackend` implements the harness `Backend` protocol over the reference `TTTCache`.

| Item | This backend |
| --- | --- |
| Inner objective | Sun et al.'s per-token reconstruction loss with LayerNorm, unchanged |
| Fast variables | `W1, b1, W2, b2` of every layer (24 layers × 4 tensors = 96 memory units) |
| Update rule | The model's own inner SGD step, mini-batched 16 tokens, learned per-token η; no decay |
| Outer gradient path | None at inference; the pretrained `W0`, projections and η gates are frozen slow weights |
| Carried state | Fast weights `W`, pending mini-batch gradient `G`, conv windows, position. Harness quantities use `W_eff = W − c15·G` (OPUS-001 F1): otherwise a chunk straddling a 16-token boundary sees the previous chunk's pending gradient committed during its own forward, even frozen |
| Transaction boundary | The harness chunk (16 tokens, aligned with the mini-batch) |

Signals, all computed inside the inner loop and reported per token, layer and head:

- `surprise` = ‖LN(f_W(k)) − (v − k)‖, the inner loss the update descends;
- `beta` = the step applied to the token's own read, `token_eta[t] · η[t]`;
- `write_norm` = the exact norm of the token's rank-one contribution to the committed update,
  `token_eta[last] · η[t] · sqrt((‖x1‖²+1)‖g1‖² + (‖x2‖²+1)‖g2‖²)`; the committed change per layer is
  bounded by the sum of these (triangle inequality), which the tests check;
- `delta_norm` = the committed fast-weight change over all 96 units; `alpha` = 1 (no decay);
- frozen canary gradients ∂NLL/∂W on a disposable copy with the inner step disabled.

Controls: `freeze` and `beta_scale` multiply the learned inner learning rate through one thread-local
scale (0 = genuine no-write: fast weights and pending gradients stay bit-identical, conv/position
advance). Because the learner has no decay, `beta_scale=0` and `freeze` coincide here.

Reference-code changes, all marked in `modeling_ttt.py`: transformers-5 compatibility (tied-weight
mapping, stateful generation), the thread-local scale and collector, a probe mode that skips the cache's
in-place copy, and a cached causal convolution that accepts more than one token (upstream's cached path
handled one token at a time, which forced single-token prompt ingestion after the first turn).

## What was measured

- Contract tests (`tests/test_ttt_backend.py`, 760M base, MPS): chunked logits agree with a single pass
  across mini-batch alignment (max abs diff below 5e-2, argmax agreement above 90%; the dual and primal
  forms differ by float error); snapshots reproduce continuations exactly; freeze leaves fast weights and
  pending gradients unchanged while conv and position advance; `beta_scale=0.5` scales the committed
  change to between 0.2× and 0.8× of the unscaled one; per-layer committed change ≤ Σ write norms;
  state dicts round-trip and reject a mismatched identity; canary gradients are finite, nonzero and
  read-only; a chat turn runs through `TransactionRunner` with every signal populated.
- Speed (760M, float32): 12–18 tokens/s decoding on Apple M4 Pro MPS; 17 tokens/s on two CPU threads.
- Training throughput on MPS is about 100 tokens/s (fp32, batch 1, sequence 512): a smoke path only.
  GPU throughput is measured by a job before any budget is committed.

## Semantics to remember

The fast weights are the layer's only context across mini-batches (the convolutions see four tokens).
Freezing therefore still removes those tokens from the layers' memory, as on Qwen (F2). The difference
from Qwen is that here the write IS an explicit gradient step with an explicit loss and step size, so the
harness measures what it was designed to measure. Whether chunk-level interventions are usable in chat,
or only turn-boundary retention (T0), is a measurement to make on the chat-tuned model, not an assumption.

## Physics

Decision recorded here for the playground: the hidden-friction physics benchmark leaves the public UI and
the README headline. It is a separate 3.56M model demonstrating in-context system identification by fast
weights; it stays as an internal benchmark (`plastic physics`, saved evaluation in the model bundle) with
this paragraph as its public description.

## Open

- Chat SFT of the 760M (then 1.3B if needed) on SmolTalk subsets; the rendering is fixed in the backend.
- Calibration of the full signal set on chat traffic for this backend; no thresholds exist yet.
- Fisher over the fast weights (the one STAT_SIGNAL not produced).
- The chunk-level vs turn-boundary policy question, measured on the chat model.
