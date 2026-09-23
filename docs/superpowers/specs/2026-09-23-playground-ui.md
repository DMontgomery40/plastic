# The playground UI: one product, one mode, the learner's own signals

Spec, 23 September 2026. Status: approved by default under the user's direction ("completely overhaul the
UI and get rid of dead slop code" during the chat fine-tune), one review pass by the Opus reviewer.

## What it is for

A person opens the Space, types into a chat box, and watches a test-time-training model learn from the
conversation: which chunks were committed, how surprised the inner learner was, how hard it wrote, and
what the harness did about it. Locally the same app drives the same sessions against any registered text
model. There is no second "public mode" UI: the server states what the client may do, the client renders
exactly that.

## Principles

1. One product. Chat, Signals, Sessions. No physics, training, red-team or architecture surfaces in the
   app; those are CLI and scripts for research, documented in `docs/`.
2. The data is the light source. Signals are plotted, not explained. One short label per quantity, the
   explanation lives in `docs/user-guide.md`. No developer notes, provenance dumps or disclaimers in the UI.
3. Honest labels, always: proposed vs accepted change; committed vs rolled back vs read-only; observational
   vs guarded mode; unavailable signal = the panel is absent, never an empty axis.
4. Legibility floor: nothing under 11px, body 14px, contrast per `~/.claude/rules/design-legibility.md`,
   no opacity-dimmed text, no grain, Archivo for type, JetBrains Mono for numbers and ids.
5. Small code. Zustand store with three slices (sessions, chat, signals); Recharts line charts; Tailwind
   tokens. Every remaining file is reachable from `App.tsx`.

## Server contract the UI relies on

- `GET /api/health` gains `capabilities`: `{create_session, fork, reset, delete, resume, calibrate}` booleans
  and `public: bool`. The Space deployment sets the restricted set; local runs get everything.
- `GET /api/models`: text models with `backend` (`ttt` | `qwen` | `plastic`), `status`, `params`,
  `calibrated`.
- `GET /api/sessions`, `POST /api/sessions {model_id, harness?}`, `GET /api/sessions/{id}`
  (summary incl. `signals_available`, `backend`, `calibration`, `read_only`, `cusum`, `state_norms`, trace),
  `GET /api/sessions/{id}/transactions`, `GET /api/sessions/{id}/state` (kinds `fast_weight` | `recurrent`
  | `plastic`), `POST /api/sessions/{id}/chat`, `/fork`, `/reset`, `/resume`, `DELETE`.
- Removed from the HTTP API: `/api/train*`, `/api/redteam*`, `/api/sleep`, `/api/data`,
  `/api/sessions/{id}/physics`, `/api/models/{id}/log`. The CLI keeps training, red team, sleep and physics.

## Screens

### Chat
- Left: transcript of the session's turns (prompt, completion) newest last, then the composer (textarea,
  Send, Enter to send, Shift+Enter newline) and a compact sampling row (max tokens, temperature, top-k, seed).
- Under each turn a **learning strip**: one cell per chunk, colored by decision (commit / rollback / scale /
  project / read-only), height by proposed change, a thin marker where accepted differs from proposed;
  hover shows chunk loss, surprise, step, write norm. Totals: chunks, committed, intervened, proposed vs
  accepted change for the turn.
- Right: session card (model, backend, mode, position, read-only state with a Resume button when allowed),
  harness card (calibration installed / absent, signals available).

### Signals
- Time series over the session's transactions: chunk loss; inner surprise; inner step size; write norm;
  fast-weight change (proposed and accepted); decision markers. Each chart appears only if its signal is in
  `signals_available`.
- Fast-weight state: per-layer norm bars (grouped units) and drift from anchor; for `recurrent` the same
  with its own label; for `plastic` the existing per-layer view.
- Recent chunks table: index, decision with reasons, loss, surprise, proposed, accepted, sources
  (prompt/model token counts), eligibility.

### Sessions
- Table: id, model, backend, mode, position, chunks, updated. Actions by capability: Chat, Signals,
  Reset, Fork, Delete. Create form (model picker from text models; observational / guarded toggle where the
  model has a calibration; guarded is disabled with a plain reason otherwise).

## Code plan

Keep and trim: `api/client.ts`, `api/types.ts`, `store` (rewritten, three slices), `charts/LineChartPanel`,
`charts/theme`, `panels/{Panel,KeyValue,DecisionBadge,Empty}`, `utils/formatting` (trimmed),
`layout/{Header,TabNav}` (rewritten), `index.css`, Tailwind config.

New: `components/chat/{Transcript,Composer,LearningStrip,TurnTotals}`,
`components/signals/{SignalCharts,FastWeightPanel,ChunkTable}`,
`components/sessions/{SessionTable,CreateSession}`.

Remove: `tabs/{TrainTab,RedTeamTab,PhysicsTab,ArchitectureTab,PublicSessionTab,PublicSessionsTab,PublicUi.test}`,
`architecture/BlockDiagram`, `charts/{ScatterChartPanel,GroupedBarChartPanel,BarChartPanel}`,
`panels/RatePanel`, `publicMode.ts`, the training/redteam/physics/data slices, their types and tests.

Python: remove the HTTP routers for train, redteam, sleep, data and the physics chat endpoint and their
service helpers and tests; add `capabilities` to health; deployment passes the restricted set. Research
code (`plastic/model`, `train`, `data`, `tokenizer`, `redteam`, `sleep`, physics) stays for the CLI and the
published checkpoints; scripts and docs are pruned per the dead-code inventory.

## Verification

Vitest for the store and each component's honest-label behavior; production build; then the visible
Chrome journey on the local app and, after release, on the Space: create or select a session, send two
prompts, read the learning strip and Signals charts, reset, confirm the capability gating. Screenshots or a
GIF recorded for the user each time.

## Decisions from the OPUS-002 review (2026-09-23)

1. Charts and strip cells are gated on **field presence in the transaction records**, not on
   `signals_available` (which lists decision signals only and omits `beta_mean`). Three states stay
   distinct: absent field = no panel; null or nonfinite value for a chunk = gap marker, never zero; stale or
   offline = last data kept with a stale label.
2. Observational mode shows the **requested** decision (`requested.kind`, the policy's choice) as an outlined
   "would roll back / scale / project" cell beside the applied `commit`. The applied decision fills the cell.
3. The harness card states the **threshold source**: "calibrated thresholds" or "session-relative
   thresholds" (robust z against the session's own history when no calibration is installed). Locally,
   guarded mode stays available without a calibration under that label; the Space disables it.
4. Cell height uses the chunk's own `write_norm_sum` where present (TTT), else `delta_norm`; the
   proposed-vs-accepted marker uses `accepted.delta_norm` vs `signals.delta_norm` (both in effective
   coordinates after the F1 fix). Cell width is proportional to `n_tokens`; hover shows the count.
5. Source marking in the strip: prompt-token chunks and generated-token chunks get distinct glyphs
   (`sources.user` / `sources.model`), and mixed chunks show both.
6. The chunk table carries, compactly, with detail on expand: per-signal z and which fired, CUSUM alarm,
   `budget_used` / `budget_remaining`, `read_only_reason`, eligibility, and canary Δcoherence / Δpoison /
   alignment when a suite exists (absent otherwise).
7. `calibrate` is a real local action: the Sessions screen's model picker offers "Calibrate on chat traffic"
   for a text model (runs the real-chat calibrator with a pinned prompt set); the Space reports the
   capability as false and hides it.
8. One-line **state labels** are not developer notes and stay: "observational", "session-relative
   thresholds", "shared public session", "base model, not chat-tuned" (for any model without chat SFT).
9. Legibility: decision states use hue plus a glyph; the accepted-vs-proposed marker is at least 2px wide at
   3:1 contrast; Recharts ticks and grid use ink tokens (no default grays).
10. Verification adds three journeys: offline/recovery (stop the API mid-session, then restart), keyboard-only
    chat (type, Enter to send, Tab through controls), and a 390px-wide layout pass.
