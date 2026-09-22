# Pretrained-backend integration: driving the harness on Qwen3.5

**Status:** design, pre-implementation. Foundation (`plastic/backends/qwen.py`) is built and
verified (parity, genuine freeze, snapshot, frozen canary). This spec is the plan for
generalizing the transaction harness to drive it, for Astra + user review before the refactor.

**Goal:** run the transactional safety harness (commit / rollback / scale / project / read-only,
calibration, canaries) on a *pretrained* model — Qwen3.5-0.8B, whose layers are mostly Gated
DeltaNet, the same recurrent-memory family as plastic — so the playground offers a genuinely
usable assistant under the same governance, without distillation or from-scratch training.

## 1. What the harness needs from a model + state (the Backend interface)

Enumerated from `TransactionRunner`'s actual calls:

- `forward(chunk, state, *, freeze, beta_scale) -> (logits, new_state, signals)` — advance a chunk;
  `freeze` = genuine no-write (Qwen: zero β+g in the kernel, verified); `beta_scale` scales the
  write (plastic: β; Qwen: **not available** without kernel instrumentation — see §3).
- `init_state()`, and state ops `clone()`, `state_delta(a, b) -> list[Tensor]` (per-memory-unit
  change, for the update-norm signals and projection), finiteness check, persistence
  (`state_dict`/`load_state_dict`).
- Canary: `score_suite(state)` and `canary_gradient(state)` — both **frozen** (Qwen uses the
  disposable grad-safe cache + frozen kernel, verified).
- Projection support: apply a corrected delta to the memory units (plastic: `layers[].S`; Qwen:
  `recurrent_states` — optional, §3).

Plan: a `Backend` Protocol capturing exactly this, with `PlasticBackend` (wrapping today's model +
`SessionState`) and `QwenBackend` conforming. The runner is refactored to hold a `Backend` instead
of a bare model, so its control flow (which is backend-agnostic) is unchanged.

## 2. The signal set differs by backend (the main design consequence)

`STAT_SIGNALS = (chunk_loss, surprise_mean, log_delta_norm, log_write_norm, fisher_update)`.
`surprise_mean` and `log_write_norm` come from plastic's `MemorySignals` (the memory's own
prediction error and write norm), which **Qwen's kernel does not expose**. So a Qwen session gates
on a *reduced* set:

| signal | plastic | Qwen |
| --- | --- | --- |
| chunk_loss (NLL) | ✓ | ✓ |
| log_delta_norm (recurrent-state change) | ✓ | ✓ (norm over the 18 recurrent tensors) |
| canary coherence/poison delta + gradient/projection | ✓ | ✓ (frozen, verified) |
| surprise_mean, log_write_norm | ✓ | ✗ (would need per-session kernel instrumentation) |
| fisher_update / fisher_drift | ✓ | later (needs a Fisher estimate on Qwen's state) |

This is honest and acceptable — the harness already renders missing signals as `null` and calibrates
per-signal — but it must be **stated**, not hidden: the Qwen operating point is governed by
loss + state-drift + canaries, a weaker net than plastic's full memory-signal suite. Recovering
`surprise`/`write_norm` for Qwen (instrument the gated-delta kernel per session) is a possible later
step, evaluated on its own, not assumed.

## 3. Generation is a real write on Qwen (the key semantic difference)

Astra measured that truly freezing Qwen during generation makes it degenerate (repetition); its
recurrent state *is* its language context. So, unlike plastic (where generated tokens are frozen
read-only observations), **Qwen generation writes to its recurrent state and those writes persist
and must be accounted** — labeled as native generative writes, subject to budgets and canary
probes, never counted as free read-only observations. The runner's `_token_freeze`/eligibility
logic is generalized: a backend declares whether a source writes. This preserves the
native / guarded / read-only distinction honestly rather than mislabeling Qwen's generative writes.

## 4. Calibration on the real conversational protocol

Qwen's native tokenizer (248k vocab) can't use the uint16 BPE format, and short chat prompts flush
partial chunks (unlike calibration's four full windows). So: a native-tokenizer calibration path,
run over the actual `Session.chat` sequence protocol (partial-chunk flush, reset, multi-turn) on a
conversational corpus (the dolly set from `build_chat_calibration.py`), producing Qwen-specific
thresholds. Do not reuse the toy model's thresholds.

## 5. Phasing (each phase testable, harness tests stay green)

1. **Backend Protocol + PlasticBackend** wrapping today's model/state; runner refactored to use it,
   with all 271 existing tests unchanged (pure refactor, no behavior change).
2. **QwenBackend conforms**: `forward` with the reduced signal set, `state_delta` over recurrent
   tensors, frozen canary score/gradient. Gated on transformers + checkpoint.
3. **Generation-write accounting**: the eligibility generalization (§3), with tests for
   Qwen-generation-writes vs plastic-frozen-generation.
4. **Native-tokenizer calibration** (§4) + a real short-conversation operating-point measurement.
5. **Dashboard/API**: surface the backend + its signal availability honestly.

## 6. Open decisions (for the user / Astra)

- **Dependency:** `transformers` as an optional project dependency + `huggingface_hub` download of
  the checkpoint (default recommendation), vs the isolated feasibility runtime.
- **Deployment role:** Qwen as the model locally/on MPS (0.44s/reply) with the toy still serving the
  free hosted CPU demo, vs paid GPU hosting — Astra owns the hosted-CPU latency measurement.
- **Signal recovery:** whether to instrument Qwen's kernel to recover `surprise`/`write_norm`, or
  accept the reduced set. Recommend accept-for-now, revisit if the reduced net proves too weak.
