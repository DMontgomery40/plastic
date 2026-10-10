# plastic dashboard: UI review and redesign

Branch `ui/redesign` (local; nothing pushed). Scope: `dashboard/` (React 18, Vite, Tailwind, zustand, Recharts) and how it is served by `plastic serve` / `start.sh` / `deploy/huggingface`. Reviewed 10 October 2026 against `main` at `a24d8b7`.

How this was checked: the real API (`plastic serve`, CPU) over a disposable store with a 48K-parameter toy model (`scripts/ui_demo_store.py`), real chats through the real harness (`scripts/ui_demo_sessions.py`), the dashboard in Chrome via Playwright for screenshots only (`scripts/ui_screens.mjs`). The toy model is far too small to say anything about harness quality; it only produces real transactions, locks and rollbacks to look at. The Qwen / TTT chat backends were not run (no GPU, 760M checkpoint not on the box); their screens were reviewed from code and from the type contract in `dashboard/src/playground/types.ts`.

Measured facts from David's experiments used to prioritise (not re-measured here):

| Fact | Source | What the UI must therefore do |
|---|---|---|
| With the shipped calibration (target FPR 0.01) 9.44% [8.04, 11.06] of benign chunks were intervened on, 7.08% refused; 47.8% of 16-chunk benign sessions saw at least one intervention | `experiments/exp3_invariants/README.md` | A flag is not evidence of an attack. Never word a rollback as "attack detected"; show what fired and by how much. Expect watch states often. |
| The CUSUM alarm latches the session read-only (81 of 1440 benign chunks, 5.6%); flag rate climbs with position (2-4% early, ~18% at chunks 13-15) | same | A locked session needs a prominent "why locked", "since when" and a counter of chunks read without learning. |
| Rollback restores fast weights bit-exactly (360/360) but a rolled-back chunk leaks through transient activation state; effect is concentrated in the first 16 tokens and about 0 after 64 tokens; restoring (not zeroing) activations removes it | `report/exp2_tables.md`, README "Exp 2" | Never imply a rollback is a full undo. State what it did not restore. |
| `scale` is not a guaranteed fraction: at scale 0.25 accepted/proposed had median 0.358, range [0.272, 0.522]; on a tiny random model a 0.5 scale gave a 4x larger change because decay is applied unchanged | `experiments/exp3_invariants/README.md` (I5) | Show measured accepted vs proposed norm, never only the requested scale. |
| `signals` describe the proposal, `accepted` describes the retained update | `AGENTS.md` contract 4 | Label both everywhere; never merge. |

Project UI rules respected (AGENTS.md): no emoji, nothing under 11px, colour only through tokens, no opacity-based text de-emphasis, no developer notes or disclaimer banners in the product UI (those live in this file).

## 1. What exists today: screens and the data behind them

Four tabs, number keys 1-4 switch. A header strip (model picker, session picker, mode chip, connection chip) sits above all of them.

| Screen | Panels | Data shown |
|---|---|---|
| **Chat** (`chat/ChatScreen.tsx`) | One card per turn: prompt, completion, learning strip, per-turn totals; composer (prompt, max tokens, temperature, top-k, seed); right rail "Session" and "Harness" | Session: model, learner, mode, position, chunk count, writable/read-only + reason. Harness: threshold source, whether generated tokens write memory, CUSUM `s_hi / s_lo`, `h`, alarm count, budget used and cap, drift from anchor, decision signals the backend reports. Per turn: chunks, committed, intervened, would-intervene, read-only, summed proposed and accepted change, prompt vs generated tokens. Learning strip: one cell per chunk (fill = applied decision, dashed outline = guard would have acted, height = write norm, white line = accepted share). |
| **Signals** (`signals/SignalsScreen.tsx`) | Five stat tiles; "Memory change" line chart (proposed vs accepted); up to six line charts; "Memory state"; "Chunks" table | Position, chunks, commits, rollbacks, scale/project/read-only counts. Charts (each only if the backend reports it): chunk loss, inner surprise, inner step size (log), write norm, canary delta coherence, canary gradient alignment. Memory state: per-layer S norm per head, ‖h‖, drift (PlasticCore) or per-unit fast-weight norms with drift ticks (TTT/Qwen). Chunk table (newest 60): decision, source, tokens, loss, surprise, write, proposed, accepted, signals fired; expanding a row shows z-scores, reasons, requested reasons, eligibility, read-only reason, budget used/remaining, positions, seconds, canary deltas, step/alpha. |
| **Sessions** (`sessions/SessionsScreen.tsx`, `SleepPanel.tsx`) | New session; Sleep (consolidate accepted learning into a child model); session list | New session: model, mode (observational default), calibrate. Sleep: model, method, target, steps, source sessions, recall probes, run cards with harvest counts, held-out NLL before/after, recall, damage gate checks. List: id, model, learner, position, chunks, committed, rolled back, last update; Chat / Signals / Fork / Reset / Delete. |
| **Sleep** (`observatory/*`, static archived JSON in `public/assets/observatory`) | Runs, Anatomy of a sleep, Weights and changes, Learning | Archived experiment results: per-arm consolidation outcomes, damage gate, probes, per-layer W0 change heatmaps, learning-contract tables. Read-only research archive, works without the service. Reviewed for consistency only; not redesigned. |

Backend data available but not surfaced before this work: calibrated thresholds per signal (`detail.calibration.thresholds`), the transaction's own reason strings in plain language, whether a lock was cusum/budget/numerical, when it began, how many chunks it has refused, the share of the proposed change actually kept.

## 2. Findings

Severity: **P0** = the user can reach a wrong conclusion about the model's safety from what is on screen. **P1** = important information is hard to find, jargon-bound or inaccessible. **P2** = polish and consistency. "Fixed" = implemented on this branch; "Open" = recommended, needs a decision or backend work.

### P0

| # | Finding (before) | Evidence | Status |
|---|---|---|---|
| P0-1 | **No at-a-glance answer to "is memory safe right now?"** The only state indicators are a small chip next to the composer and "State: read-only (cusum_alarm)" in a side card. A locked session looks almost identical to a healthy one. | `screens/before_demo_calm_chat.png`: session locked 26 chunks ago, the screen leads with chat cards; lock is an amber chip beside a greyed Send. | Fixed: status header (traffic light + one sentence) on every session screen. |
| P0-2 | **Lock-in is invisible and the alarm gauge reads calm while locked.** `Cusum.update` resets `s_hi/s_lo` to 0 on alarm and skips updates while read-only, so the rail shows `CUSUM 0 / 0 · 1 alarms` for a frozen session. No "since when", no counter of refused chunks, no reason in words. With 5.6% of benign chunks locked this way in exp3, this is the most common bad state. | same screenshot; `plastic/harness/stats.py` `Cusum.update`; exp3 README "Where it comes from" | Fixed: locked banner (why / since chunk / chunks and tokens read without learning), alarm gauge shows "reset after alarm · N alarms" at limit while locked, Resume button with the budget caveat. |
| P0-3 | **Reasons are raw harness strings** (`canary_coherence(delta=+0.1234>0.0500)`, `chunk_loss(5.2>4.9)`, `cusum_alarm`), visible only after expanding a table row on a different tab. Nothing says what the threshold means or whether a flag is likely a false alarm. | `signals/SignalsScreen.tsx` chunk table detail row; `plastic/harness/policy.py` | Fixed: `reasons.ts` turns every known reason into a title and a sentence with value and limit, raw string kept beneath; Why drawer reachable from the chat strip, the header timeline, and the table. |
| P0-4 | **Rollback implies a full undo.** The chunk shows red `x` and "Rollback" with no statement of what was not restored; per experiments, activation state keeps a short-lived trace and emitted text stays. | exp2 report; the UI had no copy about it | Fixed: every rolled-back chunk's drawer has "What rollback did not undo". Wording is measured, short and scoped ("roughly the next 64 tokens (measured on a small model)"). |
| P0-5 | **Scale shown as a decision with no actual norm.** Only a thin white line in the strip hints at the kept share; the table has proposed/accepted columns but they are summed in turn totals without the ratio. A scale can keep more than asked (0.358 median for 0.25) or even more than proposed. | exp3 I5 | Fixed: `normOutcome` + `NormCompare` bars on one scale; sentences like "Kept 36% of the proposed change; the policy asked for 25%" and "more than the proposed (…)"; ratio in the status header and timeline cell. |
| P0-6 | **Long sessions show the wrong chunks.** `GET /transactions` pages from the start; the dashboard requests `limit=400, offset=0`, so after 400 chunks the newest decisions (and the lock) are not loaded. The chat view has the same property server-side (`read_trace(limit=50)` returns the first 50 turns). | `plastic/api/service.py` `transactions_page`, `store.ts` `reloadCurrent`; `sessions.py` line 91 | Fixed for chunks: the store fetches the last 400. Open: the 50-turn trace limit is server-side (needs a tail read). |
| P0-7 | **Mode chip colour is inverted.** The header chip is amber ("warn") for `guarded` and neutral for `observational`, the one mode that blocks nothing. | `Header.tsx` | Fixed. |

### P1

| # | Finding | Status |
|---|---|---|
| P1-1 | **No decision timeline.** The only chunk-by-chunk view is a per-turn strip in Chat (cells shrink with token share, 6px minimum, tooltips only) and a 60-row table. Nothing shows the last N decisions across turns. | Fixed: `DecisionTimeline` (glyph + fill + kept-share bar, single tab stop with arrow keys, opens the drawer). |
| P1-2 | **"What changed since the last chunk" is not answerable.** | Fixed in the header: last chunk's decision, proposed/accepted, this turn's tally. Open: a per-chunk budget delta line chart. |
| P1-3 | **Budget and alarm distance are text-only and split across places** ("Budget used 3.54 · no session cap" in a rail; CUSUM in another row). No notion of how close either is. | Fixed: two gauges with a level word (OK / Rising / At limit) and `role="meter"`. Open: per-chunk budget cap (`budget_chunk`) is not in the summary payload, so only the session cap is gauged. |
| P1-4 | **Green/red decision palette fails for deuteranopia.** Measured with Machado 2009 simulation (`scripts/cvd_check.py`): commit vs rollback min CIE76 ΔE 8.2 under deuteranopia (below ~10 is hard to tell apart). Glyphs already helped. | Fixed: new tokens (commit #4ad6a7, rollback #ff8466, scale #f5d547, project #8ab4ff, read-only #94a3b4): minimum pairwise ΔE 32.4 normal, 20.8-24.0 protan, 21.9 deutan, 21.4 tritan; all ≥ 7:1 on surface and raised card. Charts' accepted/proposed also differ by dash. |
| P1-5 | **New sessions default to observational mode** (`SessionsScreen` `guarded` state false). A visitor who clicks New session gets a session that blocks nothing, and the header chip did not say so. | Partly fixed (status header says "Not guarded"). Open: David to decide the default. |
| P1-6 | **Table rows open only on mouse click** (`<tr onClick>`); no keyboard path to z-scores or reasons. Fragment in `.map` without key. | Fixed: row number is a real button with `aria-expanded`; key fixed. |
| P1-7 | **Charts have no thresholds, no decision markers, no CUSUM trace.** Calibrated thresholds exist in `detail.calibration` but are not drawn; the CUSUM statistic is not in the transaction records (only its final value), so its trajectory cannot be drawn. | Open (backend): add `cusum.s_hi/s_lo` and the calibrated threshold used to each transaction's `signals`. Then draw reference lines and mark refused chunks on every chart. |
| P1-8 | **Stale / offline states are a small chip.** After a failed request the data on screen may be old, but only a "stale" chip hinted at it (and its own word was jargon). | Fixed: "connection lost" chip, and the status sentence appends "Connection lost; this may be out of date." |
| P1-9 | **Jargon in primary labels:** "proposed Δ", "accepted Δ", "fired", "intervened", "CUSUM 0 / 0", "Decision signals", "drift from anchor". | Fixed in the status header, chat totals, chart and table headers; remaining jargon (z-scores, Fisher) is confined to the drawer, with the plain sentence above it. |
| P1-10 | `bg-surface-sunken` is not a defined token (3 uses in SleepPanel), so probe box and log tails had no background. | Fixed (`bg-surface-inset`). |

### P2

* Chat rail "Session" repeats what the header and status now show; keep but shorten (Open).
* Six 220px charts at equal weight; the two that matter for safety (canary coherence, update size) are not first and not annotated. Reorder once thresholds are available (P1-7).
* Public Space (`health.public`): a shared visitor session is described only by a chip; the status header's "Locked" state is exactly what a visitor will hit after a few turns, so the Resume wording ("Clears the lock. Spent budget stays spent.") matters there.
* The Sleep archive pages carry long prose leads; fine as a research archive, but they are not product status. Left unchanged.
* Fonts load from Google Fonts: offline boxes fall back to the system sans (screenshots were taken that way).
* Keyboard: `1-4` tab switching is good; the new timeline is one tab stop with arrows, the drawer closes on Esc and returns focus to the chunk.
* Responsive: layout collapses to one column below `lg`; the chunk table scrolls horizontally inside `overflow-x-auto`. The new header wraps to a stacked layout (checked at 390px, see screenshots).
* Empty states are good (they name the command that fills them). The status header adds "Ready: no chunks yet" for a fresh session and tells the user what a chunk is.

## 3. Information architecture

```
+-------------------------------------------------------------------------------------------------+
| plastic [connected]  1 Chat  2 Signals  3 Sessions  4 Sleep           MODEL lm  SESSION s1  [mode] |
+-------------------------------------------------------------------------------------------------+
| [ LIGHT  OK | Watch | Locked | Not guarded ]  one sentence: why, in words          [Resume learning] |
|   (locked only)  LOCKED BECAUSE  | LOCKED SINCE chunk 14 [Why it locked] | READ WITHOUT LEARNING SINCE  |
|                  Update-size alarm|                                     | 9 chunks · 144 tokens        |
| LAST CHUNK #17        | UPDATE-SIZE ALARM     | SESSION BUDGET        | RECENT CHUNKS                 |
|  x Rollback           | ####------  At 60%    | ######---- Rising     | 3 of 16 intervened            |
|  proposed 0.9 acc 0   | 3.0 of 5.0 · 0 alarms | 8.0 of 10 used        |                               |
| [timeline: oldest ................................................ newest]  (each cell = a chunk) |
+-------------------------------------------------------------------------------------------------+
|  screen body (Chat / Signals / Sessions)                                      |  WHY DRAWER (on click) |
|                                                                               |  Why chunk 17          |
|                                                                               |  x Rollback: Refused.  |
|                                                                               |  WHY  (title, sentence,|
|                                                                               |       raw code)        |
|                                                                               |  PROPOSED vs ACCEPTED  |
|                                                                               |  WHAT ROLLBACK DID NOT |
|                                                                               |  UNDO                  |
|                                                                               |  CANARY / CHUNK DETAIL |
+-------------------------------------------------------------------------------------------------+
```

Traffic light semantics (derived in `status.ts`, no new backend data):

| Light | Condition | Words |
|---|---|---|
| **Locked** (vermillion, square glyph) | `summary.read_only` | "Learning is frozen since chunk N. {reason}. K chunks have been read without learning since." |
| **Watch** (yellow, triangle) | last chunk rolled back / scaled / projected; or alarm reading ≥ 60% of limit; or budget ≥ 80% | The reason of the last intervention in plain words, or the gauge reading |
| **OK** (teal, circle) | guarded, last chunk committed in full, gauges below thresholds | "Last chunk committed in full. n of the last 16 chunks were intervened on." |
| **Not guarded** (blue, diamond) | `harness.log_only` | "Observational mode: every update is committed. The guard would have intervened on n of the last m chunks." |
| **Ready** (grey, ring) | guarded, no chunks yet | "Guard is on. No chunks yet." |

"OK" does not mean safe; it means the guard has not flagged anything recent. Given the measured benign flag rate, **Watch will appear often on ordinary text**; it is deliberately a yellow, not a red.

Principles used: state first, then reason, then numbers; every colour paired with a glyph and a word; proposed and accepted are always two labelled things; absent data says "not reported" and never 0; raw strings stay one click away so nothing is hidden by the plain-language layer.

## 4. Implemented on this branch

* `playground/reasons.ts`: reason string to title + sentence (values and limits), lock explanations.
* `playground/status.ts`: traffic light, lock info (since chunk, chunks and tokens refused), alarm and budget gauges, proposed-vs-accepted outcome, verdict and rollback-limit sentences. Pure functions.
* `playground/StatusHeader.tsx`, `DecisionTimeline.tsx`, `WhyDrawer.tsx`, `statusParts.tsx`: the new UI.
* `App.tsx`, `store.ts` (selected chunk; fetch the newest 400 chunks), `Header.tsx` (mode chip, connection wording), `ChatScreen.tsx` and `LearningStrip.tsx` (cells open the drawer; clearer totals), `SignalsScreen.tsx` (keyboard rows, keys, labels, dashed proposed series, long-session timeline), `SleepPanel.tsx` (token fix).
* Colour tokens: `tailwind.config.js`, `format.ts`, `charts/theme.ts`; `scripts/cvd_check.py` reproduces the colour-vision numbers.
* Tests: `status.test.ts` (reason wording, norm outcomes incl. scale larger than proposal, lock detection, status transitions, gauges, rollback note) and `StatusHeader.test.tsx` (render states, Resume, keyboard timeline, drawer open/Esc, missing chunk).
* Demo tooling (no product code): `scripts/ui_demo_store.py`, `scripts/ui_demo_sessions.py`, `scripts/ui_screens.mjs`.

## 5. Open items and decisions for David

1. **Push / deploy.** Nothing has been pushed. Merging to `main` triggers the sync to both Hugging Face repos; the Space rebuild is the only place the Qwen path can be checked live. Needs your go-ahead and a live check of the Qwen and TTT Space journeys.
2. **Backend fields for time-series legibility (P1-7).** Add to each transaction's `signals`: the CUSUM statistic after the chunk (`s_hi`, `s_lo`) and the calibrated threshold applied per signal. Also `budget_chunk` in the summary, and a tail read for the chat trace. Small, additive; touches `plastic/harness/transaction.py` and `plastic/api/service.py` and their tests, so it was left to you.
3. **Default mode for new sessions** (observational today).
4. **Lock auto-clear.** `alarm_cooldown` exists but the UI cannot set it; given 5.6% of benign chunks end in a latched lock in exp3, consider a default cooldown or a clearer "Resume" flow on the public Space.
5. **Wording of "OK".** Chosen to avoid "Safe"; change if you prefer another word, it is one constant (`LIGHT_LABEL`).
6. The 64-token figure in the rollback note comes from a ~6M-parameter PlasticCore model (exp2); for Qwen/TTT the true figure is unmeasured. The sentence says "measured on a small model" instead of claiming it generally.

## 6. How to run and re-take the screenshots

```bash
# store + API (CPU, toy model)
python scripts/ui_demo_store.py /workspace/plastic-ui-store
plastic serve --artifacts-root /workspace/plastic-ui-store --port 13601 --device cpu &
python scripts/ui_demo_sessions.py http://127.0.0.1:13601
# dashboard (proxies /api to the API)
cd dashboard && VITE_API_URL=http://127.0.0.1:13601 npx vite --port 5273 &
node scripts/ui_screens.mjs http://127.0.0.1:5273 screens after demo_locked,demo_budget
```
