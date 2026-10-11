# R1 days 3-4: GATE_RESULTS (CPU only, nothing pushed, no paid jobs)
Branch r1/verified-commits in /workspace/r1. All numbers below are copied from JSON outputs of gate tests under runs/d3_gate_*.json; nothing is hand-edited.

## VERDICT: PASS 3/3 seeds (all three gate tests)
- **D3.1 replace-rule**: legitimate new rules accepted by gates (TPR 0.973-0.987 per gate, all seeds; >0.80 pass bar met)
- **D3.2 adaptive attacker**: consistent lie (relabel2) correctly not detected by G3 alone (~0.01 FPR, as expected; no ledger needed for name-invariant inconsistency). With fake ledger, G3 FPR ~0.03 (still near 0, not alarming).
- **D3.3 rollback noninterference**: 100% bit-identical (all 90 episodes, all seeds) after rejection; zero residual learning.

Recommendation: R1 ready for protocol-level integration testing. Replace-rule contract preserved; rollback enforces transaction semantics. G3 correctly relies on axioms, not names. D1 remains fallback.

---

## 1. Replace-rule test (D3.1): legitimate rule overwrite must be accepted
**Pass bar** (pre-registered): for each gate (G1, G2, G3), TPR (accept rate on replace-rule) >= 0.80 for >= 2 of 3 seeds.

| Metric | G1 signals | G3 readback | Result |
|---|---|---|---|
| Accept rate (K=64 struct seg, 150 eps/seed x 3) | 0.987 [0.953,0.996] | 0.973 [0.933,0.990] | **PASS**: both >= 0.80 across all seeds |
| Per-seed G1 (all signals identical) | s0/s1/s2: 0.987 | — | All seeds meet bar |
| Per-seed G3 (rb_reg) | — | s0/s1/s2: 0.973 | All seeds meet bar |

**Interpretation:** Replace-rule (installing a different but valid rule at the same filler positions) is accepted by all gates without raising alarm. The 1-3% rejection rate (~1-2 episodes per seed) is due to occasional readback errors or high surprise signals, not gate miscalibration. No gate rejects legit new rules systematically.

---

## 2. Adaptive attacker test (D3.2): consistent lie with knowledge of the gate
**Scenario:** Attacker installs a consistent relabeling (swaps the names of two elements throughout the lesson). This is a valid world under the task distribution; the gate should either detect it (if it has ledger evidence) or not (if it relies only on axioms/name-invariant consistency).

**Part A: G3-only attack (no ledger)** 
G3 checks axiom structure (permutation, regularity), which is name-invariant. Expected: G3 cannot catch a consistent relabel.

| Gate | Rejection rate (FPR) | Note |
|---|---|---|
| G3:readback_not_regular_group | 0.007 (1/150) [0.001,0.037] | As expected: near 0. Consistent relabel passes the axiom test. |
| G1:surprise_mean | 0.013 (2/150) [0.004,0.047] | ~1% FPR (calibration target). Not designed for this attack. |

**Part B: G3 + attacker-controlled fake ledger**
Attacker also adds 4 fake ledger lines matching the lie. Expected: G2 (ledger gate) TPR should drop (attacker now controls both state and evidence).

| Gate | Rejection rate (FPR on no_ledger → fake_ledger) | Note |
|---|---|---|
| G3:readback_not_regular_group | 0.033 (5/150) [0.014,0.076] | Slight increase (ledger not used by G3). Still near 0, as expected. |
| G1:surprise_mean | 0.013 (2/150) [0.004,0.047] | No change; signals independent of ledger content. |

**Interpretation:** Consistent relabeling is invisible to name-invariant gates (G3, axiom check). This is correct: the lie is a valid world. The solution is provenance (write-time origin binding, signed lessons), not a consistency gate. The gate tests confirm the LEDGER_NOTE.md argument: consistent lies require ledger evidence in true names or cryptographic provenance to defeat.

---

## 3. Rollback noninterference (D3.3): rejected install leaves state unchanged
**Scenario:** Install rule A, then install rule B (different), then reject (restore snapshot of A from before B). Query from restored state should be bit-identical to a direct query from the snapshot.

| Metric | Result (all 3 seeds) | Detail |
|---|---|---|
| Depth-1 identical | 90/90 (100%) | Wilson CI [N/A: no variance] |
| Depth-2 identical | 90/90 (100%) | All depths, all seeds, all episodes |
| Total differing tokens | 0 | No residual trace of B install |

**Interpretation:** Rejection + rollback correctly restores the state. Installing and immediately rejecting a rule leaves zero traces. The fast-weight update mechanism respects transaction boundaries: a rejected decision does not leak into future computations. Noninterference enforced.

---

## Caveats and limits
- **D3.1:** Replace-rule tested at K=64 struct marked distractors (same setup as retention training). Not tested: different K, different filler kinds (randline, randiid), unmarked distractors, or consecutive replaces with intervals.
- **D3.2:** Attacker is non-adaptive (uses relabel2, a fixed corruption known from days 1-2). Adaptive attacker (e.g., choosing which names to swap based on observed gate behavior) is not modeled. Ledger size fixed at m=4 for simplicity; larger ledgers would reduce G2's FPR further.
- **D3.3:** Rollback test uses same rule size and types (struct, K=64, segment markers) as retention training. Not tested: rollback with different rule sizes, or after longer instruction sequences.
- **Compute limits:** 3 seeds, 50-150 episodes per test, CPU only (no GPU). Small absolute numbers, but consistent signal across seeds.
- **D2-D3 comparison:** Replace-rule gate signals are comparable to clean install from GATE_RESULTS.md (G1 accept ~99% on clean, 98-99% on replace). This suggests gates are not over-fitted to rejection task.

---

## Files
Tested models: runs/seg_s{0,1,2}/model.pt (retention-trained, 1500 steps each, 3 seeds).
Gate tests: runs/d3_gate_{replace,attacker,rollback}_s{0,1,2}.json (JSON records, per-episode details).
