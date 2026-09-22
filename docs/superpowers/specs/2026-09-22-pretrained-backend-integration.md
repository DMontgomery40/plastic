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
  `freeze` = genuine no-write (Qwen: zero β *and* g in the kernel, verified); `beta_scale` scales the
  write — **available on Qwen too**: the same per-call kernel wrapper that zeroes β for freeze can
  scale β (leaving g/decay unchanged), so `beta=0 AND g=0` is reserved for freeze and `beta*=scale`
  implements the existing scale-control contract. (ASTRA-052 correction to an earlier claim that it
  was unavailable.) After scale/projection, the representable accepted change must be re-checked
  including decay; an impossible cap rejects. Any capability that genuinely can't be supported is
  declared explicitly with a tested conservative fallback — never a policy path calling a missing method.
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

Unavailable signals must be carried as **`None` end-to-end** — not zero or NaN substitutes —
through the derived logs (`signals.py` `log_safe(write_norm_sum)`), `summarize_memory_signals`'s
list concatenation, calibration, the API JSON, and the charts; and the display-only compression
ratio must handle native-width token ids (Qwen's 248k vocab overflows the current uint16 packing).
A reduced-signal session + calibration is tested with an empty signal stream, high token ids, and
genuine finite/nonfinite available values. This is honest and acceptable — but it establishes
*different coverage*, not a measured claim that the defenses are weaker or stronger. Recovering
`surprise`/`write_norm` for Qwen (instrument the gated-delta kernel per session) is a possible later
step, evaluated on its own, not assumed.

## 3. Generation is a real write on Qwen (the key semantic difference)

Astra measured that truly freezing Qwen during generation makes it degenerate (repetition on two
prompts — a failure case, not a universal quality theorem); its recurrent state *is* its language
context. So, unlike plastic (where generated tokens are frozen read-only observations), **Qwen
generation writes to its recurrent state and those writes persist and must be accounted** — labeled
as native generative writes, subject to budgets and canary probes, never counted as free read-only
observations. The runner's `_token_freeze`/eligibility logic is generalized: a backend declares
whether a source writes, and the transaction record carries **source and eligibility independently
of the final decision kind**. But a backend declaring native generation **never overrides** an
explicit session read-only, a spent budget, or a requested freeze — those controls take precedence;
`source=model` cannot escape them.

## 4. Calibration on the real conversational protocol

Qwen's native tokenizer (248k vocab) can't use the uint16 BPE format, and short chat prompts flush
partial chunks (unlike calibration's four full windows). So: a native-tokenizer calibration path,
run over the actual `Session.chat` sequence protocol (partial-chunk flush, reset, multi-turn) on a
conversational corpus (the dolly set from `build_chat_calibration.py`), producing Qwen-specific
thresholds. Do not reuse the toy model's thresholds. Calibrate and evaluate on **disjoint**
conversations (never tune on the demonstration prompts), and **report prompt-chunk and
generation-chunk eligibility/interventions separately** — a chat turn's prompt is an eligible
write, its generation is native writes, and they must not be pooled into one rate.

## 5. Phasing (each phase testable, harness tests stay green)

1. **Backend Protocol + PlasticBackend** wrapping today's model/state; runner refactored to use it,
   with all 271 existing tests unchanged (pure refactor, no behavior change).
2. **QwenBackend conforms**, addressing the concrete conformance gates from ASTRA-052/053:
   - **Fresh-session state**: `init_state()` must expose zero recurrent leaves of the correct shape
     with zero-length KV/conv/position, so `score_suite` and `canary_gradient` work on the FIRST
     chunk before any update (today `recurrent_grad` on an empty cache raises "inputs cannot be
     empty"; the harness must not skip its first projection). Cover fresh/reset/fork-at-zero and
     zero-baseline `state_delta`/budget.
   - **MPS serialization** (done in the backend already): all model/cache/probe work goes through one
     backend lock with an MPS sync before release — concurrent MPS forwards abort the process
     (Metal assertion, exit 134); a per-session lock is insufficient since sessions share the backend.
   - `forward` with the reduced signal set, `state_delta` over recurrent tensors, β-scale via the
     kernel wrapper, frozen canary score/gradient. Gated on transformers + checkpoint.
   - **Full-state persistence**: clone/save-load/fork/reset must carry KV, conv, cursor/positions and
     init flags — not just recurrent tensors — with continuation equivalence, plus
     backend/checkpoint/tokenizer identity and a runtime-compat check that rejects incompatible sessions.
3. **Generation-write accounting**: the eligibility generalization (§3), with tests for
   Qwen-generation-writes vs plastic-frozen-generation.
4. **Native-tokenizer calibration** (§4) + a real short-conversation operating-point measurement.
5. **Dashboard/API**: surface the backend + its signal availability honestly.

## 6. Open decisions (for the user / Astra)

- **Dependency:** `transformers` as an optional project dependency + `huggingface_hub` download of
  the checkpoint (default recommendation), vs the isolated feasibility runtime.
- **Deployment role:** the user's target is pretrained text on the free CPU host, subject to actual
  hosted measurement (Astra's lane) — not a settled toy-vs-paid-GPU binary. Measured *local* native
  latency (label every number with tokens/device/warm-up): 1.37s/10-tok and 3.53s/24-tok on CPU
  (1 thread), 0.44s/10-tok warm on MPS. These are local smoke numbers; the hosted-CPU outcome is
  open and no paid-hosting authorization follows from them.
- **Signal recovery:** whether to instrument Qwen's kernel to recover `surprise`/`write_norm`, or
  accept the reduced set. Recommend accept-for-now, revisit if the reduced net proves too weak.
