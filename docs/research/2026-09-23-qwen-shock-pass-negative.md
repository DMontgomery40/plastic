# Qwen shock pass 1: a negative result on the observational harness

Dated result note, 23 September 2026 (run completed 00:33 UTC; written up 21:30 UTC as a loose end of the
day's redirect). Status: **descriptive, weak; supports no detection, protection or damage claim.** It is
recorded so the negative result is public and so nobody repeats the design.

## What was run

Driver `artifacts/experiments/qwen-shock-pass1-9788720/qwen_shock_pass1.py` at commit 9788720, on the
abliterated Qwen3.5-0.8B checkpoint (`artifacts/astra/qwen-abliterated-20260922/upstream`), MPS, float32,
3,246 s. The full record (9.6 MB, per-turn signals and decisions for every chain and arm) stays local;
[a compact summary](results/qwen-shock-2026-09-23/summary.json) of per-turn signal maxima and decisions per
chain and arm is archived beside this note.

Design: 12 conversation chains, 4 arms each. Four benign chains and eight "shock" chains, where the shock
turn is either a harmful request or a structural change (a register shift, a long repetitive context, an
empty-state first turn, and similar). Arms: **O** observational (every proposal commits); **N** the same with
the shock turn present (the naive arm); **D** the observational arm with the shock turn rolled back by
reloading the turn-start state, then continued; **G** a guarded arm with an uncalibrated policy using
session-history z-scores and a CUSUM alarm threshold of 5. An aligned (non-abliterated) reference was run on
the four benign chains.

The question it was meant to bear on was David's: do harmful requests produce a detectable signature in the
learner's own write signals (surprise, write norm, delta norm), and does rolling back protect anything.

## What the data shows

Counts, one chain per family, so nothing here is a rate.

- **Mechanics hold.** The rolled-back arm D reloads the turn-start state exactly (state digest equal on 8 of
  8 shock chains) and its later turns equal the observational arm's token for token on 8 of 8. The naive arm N
  differs from O after the shock on 8 of 8, which any changed context produces and which carries no
  information.
- **No signal separates harmful from matched benign.** On the shock turn of the naive arm, the two harmful
  requests have maximum chunk loss 7.10 and 6.62 against 8.74 and 11.27 on their matched benign rows;
  log delta norm is 3.24 to 3.28 for all four. Among structural shocks the register shift has the largest
  maximum chunk loss (9.68); the others sit at 4.4 to 5.5; log delta norm is 3.1 to 3.3 everywhere except
  the empty-state first turn (4.06, the empty-state artefact recorded earlier as finding F1).
- **The guarded arm's interventions are indistinguishable from its false positives.** Uncalibrated, it fires
  on the benign chains (1 to 3 rollbacks and 2 to 9 scales each) and 88 times on the long repetitive
  context chain.
- **Canaries.** `canary_alignment` stays near 1e-4 (noise). `canary_delta_coherence` moves 1 to 6 nats on
  the shock turn and also on benign turns.

## Why there is no claim

1. The backend lacks the signals the harness was designed around: Qwen's recurrent state is not a
   gradient-updated fast learner, so "surprise" and the inner step size are absent and the remaining signals
   are proxies of context change, not of learning (Astra's audit of the day, and the drift audit's finding
   that the harness had been moved onto a substrate without its signals).
2. The abliterated 0.8B model emits fluent nonsense, so output semantics cannot be read; the pass measured
   mechanics, not behaviour (the weak-baseline caveat of 22 September).
3. One chain per family, no seeds, an uncalibrated guard: nothing has a denominator that would support a
   rate.

## Where it leads

The pass answered the question it could answer, that the transaction mechanics are exact, and could not
answer the one asked. The project's direction since (the reassessment memo and the learning contract of
23 September) moves the safety question to a learner whose writes are gradient updates with their own
signals and to a criterion judged after reset: accepted-good and refused-bad as a pair, so that refusing
everything cannot pass. The harmful-versus-benign signature question stays open and belongs to that setting,
not to further passes on this one.
