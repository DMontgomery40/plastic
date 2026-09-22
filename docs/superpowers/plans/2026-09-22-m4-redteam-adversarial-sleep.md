# M4 Red Team, Adversarial Write Gate, and Sleep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An attacker that targets the real token path and reports the payload it actually tested; the experiment that meta-trains the model's write gate against that attacker; and sleep consolidation gated by the canary suites.

**Architecture:** `plastic/redteam/attack.py` optimizes a continuous perturbation of a suffix's token embeddings by projected gradient ascent on canary damage (coherence-canary loss after learning the suffix from a benign prefix, minus before), subject to a perplexity constraint expressed with the model's own next-token loss and an embedding-norm ball; it snaps to nearest tokens and re-validates through `TransactionRunner` so the reported damage, decisions, and signals are those of the discrete payload. Sampled attack families (random tokens, single-token runs, shuffled benign text, topic switches) share the same evaluation. `plastic/train/adversarial.py` adds `λ_adv · Damage(Φ, x_adv)` to the outer loss on 1 of every 10 steps with the attack treated as constant, and `λ_rec · CE_MQAR` keeps recall honest. `plastic/sleep/consolidate.py` replays session traces into the blocks (embeddings and head frozen), accepts the candidate only if the coherence canary does not rise and the poison canary does not fall beyond calibrated tolerances, and registers it as a child model.

**Tech Stack:** torch, the M1 to M3 modules. Pure PyTorch.

**Spec:** `docs/superpowers/specs/2026-09-21-plastic-design.md` sections 7 and 8.

## Global Constraints

- Every attack result records the discrete token ids that were validated, the harness decision on each chunk, and both the provisional (pre-decision) and accepted canary deltas.
- No text heuristics; the "benign-looking" constraint is the model's own NLL on the payload.
- Sleep never touches the embedding or head; a rejected candidate leaves no registered model.
- Every task ends with `uv run pytest` green and a commit on `fuse`. Never push.

---

### Task 1: Attack objective and PGD (`plastic/redteam/attack.py`)

**Interfaces:**
- `@dataclass AttackConfig`: `suffix_len=64`, `steps=50`, `lr=0.05`, `radius=1.0` (embedding-norm ball per token, relative to the embedding RMS), `nll_max: float | None` (payload NLL ceiling; default: benign prefix NLL + 1.0), `nll_weight=1.0`, `seed=0`, `harness: HarnessConfig | None`, `families: tuple[str, ...] = ("pgd", "random", "repeat", "shuffle", "topic_switch")`.
- `damage(model, prefix_ids, suffix_emb, suite, *, device) -> tuple[Tensor, dict]`: run the prefix through the model from zero state (learning on), then the suffix as continuous embeddings through `model.core` with learning on, score the coherence canary from the resulting state minus from the prefix state (differentiable in `suffix_emb` through the delta-rule inner loop), and return also the payload NLL under the model.
- `pgd_attack(model, cfg, prefix_ids, suite, *, device) -> AttackResult` with `AttackResult(payload_ids, prefix_ids, family, damage_continuous, damage_validated, nll_payload, decisions: list[str], signals: list[dict], canary_before, canary_after_provisional, canary_after_accepted)`. Snap: nearest embedding by cosine to each perturbed token embedding; validate: fresh `TransactionRunner` with `cfg.harness` (default `HarnessConfig()`), feed prefix then payload, flush, read the decisions and canary deltas from the transaction records; `damage_validated` is the coherence canary delta between the state after the prefix and the accepted state after the payload.
- `sampled_attack(model, family, cfg, prefix_ids, suite, *, device, rng) -> AttackResult` for the other families.
- `run_redteam(store, model_id, *, cfg, n_prefixes=8, data_dir, device) -> dict` picks benign prefixes from the held-out file, runs every family, writes `artifacts/redteam/<run_id>/{config.json, results.jsonl, summary.json}` and returns the summary (per family: mean and max validated damage, fraction of chunks rolled back, fraction of payloads whose validated damage exceeds the calibrated coherence threshold), appends the top payloads to the model's poison canary set when `--record` is set.

**Tests (`tests/test_redteam.py`)** on the tiny fixture: the validated payload ids are the ids that were fed (recorded in the transaction log positions); PGD damage does not decrease over 5 steps on a fixed seed (monotone non-decreasing within tolerance is too strong; assert final continuous damage ≥ initial − 1e-6 after 10 steps with a small lr); the NLL constraint is honored (payload NLL ≤ `nll_max` + 1e-3) or the result is flagged `constraint_violated`; sampled families run and produce finite numbers; `run_redteam` writes the files.

---

### Task 2: Adversarial write-gate training (`plastic/train/adversarial.py`)

**Interfaces:**
- `TrainConfig` gains `adversarial: bool = False`, `adv_every=10`, `adv_lambda=1.0`, `adv_steps=5`, `adv_suffix_len=32`, `adv_radius=1.0`, `mqar_lambda=1.0`.
- `adversarial_loss(model, batch_prefix_ids, suite, cfg, *, device) -> Tensor`: runs a short PGD (attack treated as constant: the perturbation is optimized with the slow weights detached, then the damage is recomputed with gradients flowing to the slow weights); returns the damage.
- `train` calls it every `adv_every` steps and logs `adv_damage`.
- Evaluation additions in `eval.json`: `attack_damage_mean`, `attack_damage_max`, `beta_auroc` (β at attack tokens vs benign tokens, computed from the memory signals), `rollback_fraction`.

**Tests:** 6 adversarial steps on the tiny fixture run and log `adv_damage`; a `--adversarial` CLI flag is accepted.

---

### Task 3: Sleep (`plastic/sleep/consolidate.py`)

**Interfaces:**
- `consolidate(store, model_id, *, sessions: list[str] | None, core_data_dir, steps=200, lr=1e-4, core_ratio=0.8, seq_len, device, tolerance: dict | None) -> dict` harvests `trace.jsonl` chat records from sessions of the model, encodes prompt+completion, mixes with core windows, trains blocks only, evaluates the canary suites before and after from zero state (coherence must not rise by more than `tolerance["coherence"]`, default the calibrated `canary_delta_coherence` threshold or 0.1; poison must not fall by more than `tolerance["poison"]`), registers `sleep_<id>` with `parent_model_id` on acceptance, and returns the manifest either way.
- CLI `plastic sleep <model_id> [--sessions ...] [--core artifacts/data/wikitext] [--steps N]`.

**Tests:** consolidation on the tiny fixture with two chat sessions produces a registered child model when tolerances are loose, and no model when the coherence tolerance is set to −1 (forced rejection).

---

### Task 4: Experiment runner and report

- `scripts/experiments/adversarial_gate.py`: trains baseline and hardened models on the same seed and data (CLI wrapper around `train` with `--adversarial`), runs `run_redteam` on both, and writes `docs/research/2026-09-22-adversarial-gate-results.md` with the numbers from the spec (max damage, fraction over threshold, harness residual, β AUROC, LM loss, MQAR accuracy, transfer). This script is run on the L4 (or MPS for the small config) once M4 lands; the report is committed with whatever it shows.

## Self-review

- Spec coverage: 7 (Tasks 1, 2, 4), 8 (Task 3). Poison canary augmentation from recorded attacks (Task 1 `--record`).
- Type consistency: `AttackConfig`, `AttackResult`, `run_redteam`, `consolidate`, `TrainConfig.adversarial` used identically.
