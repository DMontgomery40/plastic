# Text rule contract, corrected protocol (24 September 2026)

Step 1 of the lead's plan after the review of the 23 September reports (their archive READMEs carry the corrections):
reproduce the archived conditions with the missing control, re-run with every composition trained, and test whether the
first-situation gains are rule content. Checkpoint `ttt_mlp_760m_chat_v1` step 250 (digest `29e0f855…`), MPS, decoration
rule set, rules unstated, seed 0 (split seed 0: held-out `#B #H, #B #P, #B #Q, #H #Q, #P #B, #P #H, #Q #P, #W #B`, of which
`#H #Q` and `#Q #P` repeat the answers of the trained `#Q #H` and `#P #Q`), lasting-update target W0, lr 1e-4. Each run
executed from an exported snapshot of the named commit (no uncommitted code). One seed; seeds 1 and 2 are the next rows.

**The template-copier ceiling.** An untrained copier that fills the first worked answer's template with each later input
is exact on every held-out situation after the first on this split and evaluation seed (16/16 at the second situation,
112/112 over all later ones; `plastic.data.rules.template_baseline`, pinned by a test). These three runs predate the
report field, so the ceiling is stated here. Every score after the first situation below is at or under that ceiling and
cannot show rule knowledge; only first-situation measures can.

| Run | Settings | What it shows |
| --- | --- | --- |
| `archive_conditions_s0/` | Code 65225ee. The 23 September conditions (20 draws with replacement, 11 of 17 compositions trained, 6 verification episodes), replay_verify, plus the clean-again control and the first-situation choice score | Reproduces the archived numbers exactly: held-out exact 0.742 → 0.875, second situation 0.31 → 1.0, first situation 0 → 0, NLL 0.417 → 0.155, revert gap 0, poison harm +0.014 NLL. The clean stream flips the first-situation verify items `#Q` and `#W` 0 → 1. **Clean again: accepted, no item moves. Poison on top: refused, because the single `#W` flips 1 → 0 with its first-situation NLL moving −0.017 nats**, a near-tie on a composition without `#P`. Poison from the snapshot accepted; the corrective stream flips `#Q` 1 → 0 and `#H` 0 → 1. Reading: the archived refusal is caused by the two poisoned steps, as collateral on an unrelated near-tie item, not detection of the false rule; six first-situation items move both ways across arms |
| `full_coverage_s0/` | Code 65225ee. One full pass (17 steps, every composition once), continued and replay_verify, 34 verification episodes, clean-again control, choice score | The gain replicates with full coverage: held-out exact 0.742 → 0.875, second situation 0.31 → 1.0 (the copier's ceiling), NLL 0.417 → 0.163, revert gap 0, chat NLL 1.629 → 1.601; first-situation exact 0/16 held-out and 0/17 training before and after. replay_verify accepted every stream including the poison on top (verify first-situation items moved both ways in both the clean-again and the poison arm): **the archived refusal does not survive the corrected protocol.** First-situation choice (23 distinct outputs, chance 1/23): rank-1 accuracy unchanged (training 2/34, held-out 0/16); per-item margins rise (training +9.95 nats, 30 of 34 improved; novel held-out +2.48, 9 of 12). The sequential poison (5 of 17 steps touch `#P`) lowers `#P` items' margins while the matched clean pass raises them: training `#P` −3.16 (0 of 10 improved) against +3.09 (9 of 10); held-out `#P` −1.64 (0 of 8) against +4.82 (8 of 8); items without `#P` move little either way |
| `content_null_s0/` | Code 98eb880. As `full_coverage_s0` (continued) plus the name-permuted content null: every name's answers follow another composition of its arity, fixed per stream, never a commuting twin | **The content null raises the correct outputs' first-situation margins as much as the true lessons or more:** training +11.0 nats (true +9.95), novel held-out +4.09 (true +2.48). True minus null, per item on the same inputs: training −1.05 on average (11 of 34 favour the true lessons), held-out −1.86 (2 of 16); the `#P` compositions are among the negatives. Held-out exact after the first situation is 0.875 after the null, as after the true lessons |

**Reading (one seed).** The lasting update to W0 makes one worked example enough within an episode, after a reset, and
restoring W0 removes it; that improvement is real and replicates with full coverage. It is adaptation to the
demonstration format, which an untrained copier also performs perfectly. The first-situation evidence does not show rule
knowledge: exact stays at zero, and the choice margins rise as much when every name is paired with the wrong decoration,
so their rise is a generic shift toward decorated outputs rather than stored name-to-decoration content. What does depend
on the names is the direction of a further update: re-teaching `#P` wrongly on top of the trained state lowers `#P` items
specifically against a matched clean pass. The first-situation exact verifier is at noise level at six items and refuses
nothing at 34. Not shown: stored rule knowledge, composition, or a verifier that separates a false lesson from a true one.
