# Turn-boundary retention for the native Qwen backend (design-gate note)

> Historical candidate and test protocol. T0 passed its recorded mechanics check;
> T1 completed with a negative fit result and did not reach DEV. The original
> sequencing and calibration prerequisites below describe that study, not gates
> on new exploration. Follow [current status](current-status.md) for continuation.

Status: **candidate, not implemented, not adopted.** Scope per scratchpad ASTRA-123/124 and
CODEX-003: take this candidate through the design gate, specify every gate's timing, and pass a
matched omitted-turn control before any policy code or larger comparison. It is not a replacement
for the Plastic fast-memory baseline, and it makes no safety, learning-utility or novelty claim.
Date: 22 September 2026.

## 1. Why (recorded evidence, DEV split only)

The pinned operating-point development diagnostic ran on the actual Qwen3.5-0.8B checkpoint,
MPS float32, on 16 DEV prompts, with identical seeds and calibration in every arm. Records are in
`artifacts/experiments/qwen-oppoint-dev-0cafa60-mps/` and `…-rollback-only/`; observations are in
scratchpad FABLE-091 and FABLE-095.

| Arm (sole difference) | Empty | Cap | EOS | Unusable |
| --- | --- | --- | --- | --- |
| Guarded: rollback + absorbing latch (`freeze_on_alarm=True`) | 9 | 6 | 1 | 12 of 12 latched sessions |
| Rollback-only (`freeze_on_alarm=False`) | 2 | 12 | 2 | none degenerate; see below |
| Log-only | 0 | 14 | 2 | none |

- **Latch.** The persistent read-only latch accounts for the degenerate outputs.
- **Single rollback.** A single within-turn frozen-replay rollback is still not neutral. Two of 12 sessions were empty. After an early prompt rollback, several fluent answers appear to miss the question's content (a subjective reading, not a metric). Late single rollbacks (2 of 2) left the output unchanged.
- **Mechanism, verified in code** (`plastic/backends/qwen.py:44-68, 494-512`). Freeze and frozen replay zero the gated-delta write β and decay g. The replayed tokens' **recurrent context writes are suppressed** in the 18 gated-delta layers. Prior recurrent context survives, and conv/KV retain representations of those tokens. This is context-update suppression, not deletion (correction in FABLE-094).
- **Gates.** Every within-turn gate carries this cost, not only the CUSUM. On the recorded log-only trajectory of pair 1818, `chunk_loss` alone would have rolled back generation chunk 11 (CODEX-003, ASTRA-124).

## 2. Readiness note

- **Local sources read.** AGENTS.md; this directory's README; the safety review §6, whose Layer 5/6 recipe is the closest prior design; the calibration replay audit; scratchpad ASTRA-083/088-090/109/114-116/123-125, CODEX-002/003, FABLE-088-095. Code at the pinned commit `0cafa60`: `plastic/harness/transaction.py`, `policy.py`, `config.py`, `stats.py`, `calibrate.py`, `plastic/backends/qwen.py`, `plastic/session/runner.py`.
- **Primary sources.** All re-checked 2026-09-22 by the Fable specialist, CODEX-003 and ASTRA-123; I rely on their shared reading:
  - Gated DeltaNet, [arXiv:2412.06464v3](https://arxiv.org/html/2412.06464v3) (latest listed v3, 6 March 2025), Eq. 10: S_t = S_{t-1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ, zero-initialized in the installed Transformers 5.17.0 kernel.
  - The [NIST tabular CUSUM](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm).
- **Not searched.** No literature search was made for prior transactional retention of recurrent LLM context. No priority is claimed.
- **Closest known mechanism.** Mechanically, restoring a saved context is ordinary checkpoint/restore, the same as deleting a turn from a chat context. Conceptually, it is the safety review's Layer 5/6 "alarm → freeze learning, revert to anchor". That recipe assumes an auxiliary learner whose freezing leaves input processing intact, which native Qwen does not have.
- **Distinction under investigation.** Where to place the retention decision for a backend whose recurrent state IS its working context.
- **Unresolved.**
  - The cause of the per-session startup spike: empty state vs first position vs identical chat-template prefix (ASTRA-123 #1).
  - Detection power: no Qwen positive control exists.
  - The per-turn calibration unit.
- **Falsifier.** See §7.

## 3. Design gate

| Item | This candidate |
| --- | --- |
| Inner objective | None added. The gated-delta update is Qwen's native context mechanism with pretrained, input-dependent α, β. The harness optimizes nothing. |
| Fast variables | The full native cache: 18 gated-delta recurrent states and their conv windows (length 4), the KV of 6 full-attention layers, and the position (cache length). |
| Update rule | The unmodified native forward inside a turn: no freeze, no β scaling, no frozen replay. |
| Outer gradient path | None. Weights are frozen and nothing is meta-trained. This is a runtime retention policy, not a learning rule. |
| Carried state | The committed cache between turns, and one turn-start snapshot while a turn is open. |
| Causal target availability | The boundary decision uses only signals already computed on this turn's processed tokens: prompt, generation and assistant closure. Nothing from future turns. |
| Transaction boundary | The chat turn (user prompt + generation + closure) is one transaction: commit or discard. |

Comparison with the Plastic baseline:
- **Plastic.** Plastic's fast memory is a gradient-updated learner, so freezing it pauses learning while activation still progresses, and chunk-level transactions are meaningful.
- **Native Qwen.** Here there is no learner separate from context, so a chunk-level intervention is an input edit.
- **Scope.** The candidate applies to the native Qwen backend only. Plastic keeps its semantics.

## 4. Semantics

1. **Turn start.** Snapshot the full runner state: committed/working/anchor caches, position, last logits, detector state. At a boundary the pending buffers are empty by construction.
2. **Inside the turn.** Native processing only. Every gate computes and records its evidence as a *proposal* (`would_*`, as in log-only), and no gate acts on the state.
3. **Turn end.** A boundary decision from the turn's recorded evidence. The rule is declared separately (§6) and calibrated per turn.
   - **Commit:** keep the post-turn cache.
   - **Discard:** restore the turn-start snapshot for every component at once: recurrent, conv, KV, position.
4. **Output.** Emitted output is never retracted. The user saw it, and the audit log keeps it with a `discarded` marker.

Contract mapping (AGENTS):
- **#2 freeze:** not used for native Qwen under this policy.
- **#3 transaction boundary:** the boundary moves from the chunk to the turn. For this backend, rollback becomes snapshot restore, not frozen replay. Emitted outputs are still not retracted.
- **#4 proposal vs accepted:** unchanged. Signals stay per-chunk proposals. Accepted change is the turn's measured change on commit, and exactly zero in every component on discard. The zero-on-discard claim is what the control in §7 must show.
- **#5 budgets:** any budget counts committed turns only; a per-turn budget is an open choice.
- **#6 canaries, #8 numeric evidence only:** unchanged. The Qwen signal set is reduced (no canaries, projection or Fisher).
- **#7:** forks, saves and resets use committed state only. A turn-start snapshot never persists past its boundary.

**"Retention" on native Qwen means the next turn conditions on this turn.** That is context carry, not weight learning, so it must not be reported as learning utility.

## 5. Every gate's timing (ASTRA-124)

| Gate (current code) | Today on Qwen | Under this candidate |
| --- | --- | --- |
| `chunk_loss` calibrated threshold (policy.py stats) | Per-chunk rollback by frozen replay | Per-chunk evidence; can only trigger a turn-level discard |
| `log_delta_norm` calibrated threshold | Per-chunk rollback | Evidence only. Remains confounded by the startup spike (F1). Exclusions/refits (FABLE-092 option A) are a separately declared ablation, not adopted. |
| z-based rollback/scale for signals without a threshold | Per-chunk rollback/scale | Evidence only; no β scaling inside a turn |
| CUSUM alarm (`cusum_alarm`) | Rollback + absorbing latch | Evidence only; no latch inside a turn. Treatment of the detector state on discard: see below. |
| Canary coherence/poison, projection, Fisher drift | Not produced on Qwen (reduced signals) | Not produced; must stay disabled, never silently enabled |
| `budget_chunk` / `budget_session` | Scale/latch (none set in the screen) | Per committed turn only, if declared |
| Nonfinite working state | Rollback by frozen replay | Discard (restore snapshot) and mark the turn |
| Session read-only latch | Absorbing, per chunk | Replaced by an optional declared session rule: after N discards, discard all later turns (amnesic mode); not a within-turn freeze |

**Detector state on discard.** Proposed default:
- The model state and the CUSUM statistic are restored together, so the statistic always describes the *retained* stream.
- The audit log keeps the discarded turn's evidence.
- A separate session-level discard count persists, so repeated suspicious turns are not invisible.
- The alternatives are declared and open: keep the CUSUM across the discard (the detector then remembers content the model no longer holds), or reset it.

## 6. Boundary rule (open; must be declared and calibrated before any policy run)

- **Candidate rule:** discard the turn if any calibrated gate would have fired on any of its chunks.
- **Calibration unit:** today's thresholds are calibrated per chunk, so an any-gate-per-turn rule has a per-turn benign rate far above the per-chunk rate. It must be recalibrated per turn on fit data before use. This is a calibration change, not a threshold loosening.
- **Current status:** under the current (F1-confounded) detector, most benign DEV turns would be discarded. So this candidate cannot be used as configured until the startup confound is resolved as a separate question.

## 7. Tests, in order

**T0: matched omitted-turn control (mechanics; first and blocking).** Chains of three DEV prompts with the same per-position seeds, run log-only so no gate acts inside a turn. From the same post-turn-1 state:

- **O (omitted):** turn 1, then turn 3.
- **D (discarded):** turn 1, snapshot, turn 2, full restore, turn 3.
- **N (kept):** turn 1, turn 2, turn 3. This is the non-vacuity control.

**Pass requires all of:**
1. D's state after restore is bit-identical to O's state before turn 3 in every cache tensor (recurrent, conv, K, V) and in position.
2. D's turn 3 is identical to O's turn 3 in generated token ids and per-chunk signals and decisions.
3. N's turn 3 differs from O's in at least one chain. Otherwise turn 2 carried nothing and the test is vacuous.

T0 checks restoration completeness only. It does not test a policy, a detector or safety.

**T1 (only after T0 passes and this note is reviewed).** Implement the declared boundary rule with per-turn calibration, and compare log-only, current semantics and this candidate on DEV chains. Report:
- per-turn evidence for every gate, and discard counts;
- coherence relative to the omission control;
- for each discarded turn, whether the next turn matches the omission control.

No 48–64-turn matrix before T0 passes. The locked 32 prompts and the follow-up fixtures stay closed. A revised screen needs a declared policy and fresh calibration.

## 8. What would make us abandon or revise the candidate

- **T0 fails (D ≠ O).** Restoration is incomplete or the protocol is wrong: fix the restore, not the policy.
- **T0 passes but turns after a discard differ from the omission control** under the implemented policy. The candidate then leaks discarded content.
- **Benign per-turn discard rates stay high** after the startup question is resolved. The candidate is then unusable at this operating point, just as the current screen is.
- **Evidence** that a positive control (a demonstrably damaging retained turn) is not caught at the boundary.
  - This limits the candidate as a guard; it does not affect coherence.
  - No such control exists yet, so no detection claim can be made either way.

## 9. Known limitations

- **Unguarded current turn.** The current turn's output is never protected, because it is emitted before the decision (as today).
- **Visible divergence.** After a discard, the transcript the user saw diverges from the model's context. Interfaces must label a turn as discarded, as distinct from absent or failed.
- **Reintroduction.** A later prompt can reintroduce discarded content, for example by quoting it. The harness cannot prevent this; it is stated here, not solved.
- **Cost.** Each turn needs one full cache snapshot, which grows with KV length. The deep-copy cost has been measured only at short prefixes.
