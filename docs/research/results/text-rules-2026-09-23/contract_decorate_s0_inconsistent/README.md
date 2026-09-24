

**Provenance.** Run by the Fable session 41b763a8 (started 2026-09-24 02:07 UTC), archived by the lead on 2026-09-24 (OPUS-LEAD-001). Stated rules with
the inconsistent poison: the true definitions are stated in every preface while the poisoned stream's `#P` answers
follow the false definition ("thanks after the list"). Code 1c8bc6c plus the uncommitted report-script refactor and
`--verify-episodes` option then in the tree (default 6, no behaviour change; committed in 65225ee) and a teammate's
AGENTS.md edit. The clean arm reproduces the stated-rule reports (0.67 → 0.87, NLL 0.387 → 0.120), as it must: the clean
stream is the same.

**Reading.** Every stream was accepted in both modes: the poison from the snapshot (harm +0.020 NLL) and on top of the
accepted clean lessons (+0.006 NLL), so refused-bad is 0 of 3 with the definitions stated. With the true rule stated, a
stream whose answers contradict it still passes every damage and held-in check the verifier has.

**Correction (OPUS-LEAD-001/002, 2026-09-24).** (1) Coverage: the lasting update drew 20 stream episodes with replacement from
`Random(7)`; on this 17-episode stream it trained 11 compositions (`#P`, `#P #Q`, `#Q #W`, `#Q #H`, `#H #P`, `#H #B`
never), and the poisoned stream differs from the clean one in 2 of the 20 steps (`#P #W`, `#W #P`), so the poison rows
compare updates that share 18 of 20 steps. (2) Held-out repeats: `#H #Q` and `#Q #P` write the same answer as the
training compositions `#Q #H` and `#P #Q` on every input, so two of the eight held-out pairs are not new behaviour.
