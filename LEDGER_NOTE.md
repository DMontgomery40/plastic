# Ledger / provenance note: consistent relabeling is invisible to every consistency gate
(R1, exp4 group-action task. Argument is exact for this task; measurements are in GATE_RESULTS.md.)

**Setup.** Each episode draws a group, a generating pair, and a uniformly random injective naming of the elements/generators (`task.make_world`). A lesson is the full right-Cayley graph written with those names.

**Claim.** Let pi be any permutation of the element names and L a valid lesson. pi(L) is also a valid lesson (the same Cayley graph with other names), and under the task distribution P(L) = P(pi(L)) because names are drawn uniformly at random. Hence any gate whose decision depends only on the *name-invariant structure* of the lesson or of the state's readback (bijectivity, group closure/regularity, isomorphism type, internal surprise on first-seen random names, write norms) outputs the same decision for L and pi(L). Its TPR on "consistently relabeled" lessons therefore equals its FPR on clean ones (AUROC 0.5 by construction), whatever the model. This is the `derange` corruption (every name moved) and `relabel2` (the minimal lie: the names of just two elements swapped).

**What can detect it.** Only evidence that carries *information about the true names from outside the lesson*:
1. a trusted **ledger** of m true facts (e.g. m lesson lines, or held-out depth-2 compositions, in true names) checked against the model's readback of S (the G2 gates below *are* this when the held-out evidence is trusted ground truth: they catch `derange` for that reason, not because held-out loss is a better consistency test);
2. **provenance**: write-time origin binding / signed lessons (cf. TMA-NM, arXiv:2606.24322 as summarised in the novel-direction report: content- and lineage-based defences are malleable, write-time origin binding is necessary).
No claim is made that this is new; it is the fast-weight instance of "the set of consistent lies is the set of valid worlds".

**Sizing the ledger.** `derange` moves every element, so every true line is contradicted by the readback: one correct ledger probe suffices *if the readback is accurate*; with readback errors the detection rate follows the readback accuracy. `relabel2` changes only the lines whose x or y is one of the two swapped elements (<= 8 of the 2n lines, since each element appears as x in 2 lines and as y in 2 lines, for 2 elements): a random m-line ledger misses it with probability about C(2n-8, m)/C(2n, m), so m ~ ln(1/delta)/e with e = 8/(2n) of the lines damaged. Measured on relabel2 in GATE_RESULTS.md.

**Cost / caveat.** On this toy task the ledger is as expensive as the lesson itself (the trusted source *is* the world); the point is the taxonomy: caught by consistency (wrong_y: bijectivity; swap_y: group regularity, not bijectivity) / needs a ledger / needs provenance.
