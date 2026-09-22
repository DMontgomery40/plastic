# Working on plastic

`plastic` is a research sandbox for selective recurrence, gradient-updated fast
memory, and an external transactional harness. Text and hidden-friction physics
share `PlasticCore`; their embeddings and heads differ. Keep implemented behavior,
experimental proposals, and measured results distinct.

## Start here and coordinate

- Read `README.md`, `plastic/config.py`, and the code/tests for the affected surface.
- Before concurrent work, read the latest `SHARED_SCRATCHPAD.md` entries. Respect
  the recorded ownership and agree on overlapping edits there. The user's current
  instructions can change scope or ownership.
- The scratchpad is append-only. Reread its tail before appending a timestamped
  entry with findings, exact file/line references, verification, and open requests.
  Keep transient status and test counts there rather than in this file.
- Design and implementation plans live in `docs/superpowers/`; research proposals
  and evidence live in `docs/research/`. Neither substitutes for checking current
  code, artifacts, or unresolved review findings.

## Setup and verification

Python 3.12+, torch >= 2.12, `uv`, and Node.js/npm are required for the full stack.

```bash
uv sync --extra dev
npm --prefix dashboard ci
uv run plastic --help
uv run pytest
npm --prefix dashboard test
npm --prefix dashboard run build
git diff --check
```

There is no single cross-language quality gate. Use the relevant gates:

- Python changes: run focused tests and the full Python suite. The existing
  `.venv/bin/python -m pytest` is an equivalent runner when the environment is ready.
  Do not suppress additional warnings just to make output look clean.
- Model/state changes: cover scan and delta equivalence, chunk streaming, blocks,
  carried state, and freeze behavior. Include repeated keys, nonzero initial state,
  full/partial chunks, and different input partitions where applicable.
- Harness/persistence changes: cover decision transitions, calibration, budgets,
  canaries, sessions, signatures, and affected API/CLI contracts.
- Dashboard changes: run Vitest and the production build, then exercise the affected
  connected-browser journey. A build does not prove rendering or backend behavior.
- Bug fixes need a regression plus realistic coverage of the broader bug family,
  preferably in an existing suite. Include failure/recovery and boundary cases.
- Documentation changes: check local links and validate documented commands with
  parser/help checks. Follow any broader verification required by the user. Do not
  download datasets or launch training merely to check command syntax.
- Report failures and untested devices explicitly. A backup commit is not a passing
  release; do not change unrelated tests to hide an existing failure.

`./start.sh` starts a CPU API on port 13579 and Vite on 5173. It supports `DEVICE`,
`ARTIFACTS_ROOT`, `API_HOST`, `API_PORT`, `DASHBOARD_HOST`, and `DASHBOARD_PORT`.
Use separate ports and a disposable artifact store for concurrent UI checks; copy
needed checkpoints rather than mutating another agent's sessions or experiment data.

## Architecture and state contracts

Key paths: `plastic/model/` (recurrence, fast memory, blocks, carried state),
`plastic/train/`, `plastic/data/`, `plastic/tokenizer/`, `plastic/harness/`,
`plastic/session/`, `plastic/store.py`, `plastic/redteam/`, `plastic/sleep/`,
`plastic/api/`, `plastic/cli.py`, and `dashboard/`.

The default is linear memory with the `delta` rule and `ssm_out` memory input.
The optional `chunk` rule uses minibatch updates, momentum, and optional
Newton-Schulz orthogonalization. `memory="mlp"` is not implemented. Physics inputs
contain observation, action, and reset information; hidden friction is not an input.

Preserve these contracts and test them when changing their implementation:

1. Chunk-parallel and recurrent paths remain numerically equivalent, including
   carried state and arbitrary supported input partitions.
2. `freeze=True` disables memory writes **and decay**; activation and convolution
   state still advance. `beta_scale=0` disables writes only. Pending chunk statistics
   and momentum are part of the state, not disposable implementation details.
3. The harness changes controls at transaction boundaries. Rollback uses frozen
   replay; it does not retract emitted outputs or remove all activation influence.
4. Transaction `signals` describe the proposal. `accepted` describes the retained
   update. Keep this distinction in logs, API payloads, charts, and budget accounting.
5. Budgets constrain the actual representable accepted change, including retention.
   Recheck after scaling/projection, reject nonfinite persisted state, and distinguish
   a per-chunk cap from exhaustion of a session's cumulative budget.
6. Canary probes do not mutate the session being measured. Handle masks and ragged
   probes correctly. Projection is a local first-order safeguard, not a global proof.
7. Preserve `model_signature` compatibility checks and fork/load/reset semantics.
   Forks use committed state. Resume clears a read-only latch without replenishing
   a spent budget. Calibration settings must survive the relevant persistence paths.
8. Harness decisions use numeric model/state evidence, not keyword or regex safety
   heuristics. Display-only proxies must not silently become decision signals.

## Research evidence and interface claims

- The current fast learner is linear with a normalized readout. Do not call it an
  exact reproduction of Sun et al.'s TTT-Linear or a novel nonlinear inner model.
  The coordinate-recurrence memo is an experimental candidate requiring ablations;
  it is not the default block or the mechanism behind the implemented scan.
- Keep mathematical equivalence, gradient checks, learned behavior, device timing,
  and safety evidence separate. CPU tests do not establish MPS/CUDA timing or
  under-an-hour training. Training loss alone does not establish memory utility.
- Compare memory-enabled and writes-disabled behavior on matched held-out data.
  Distinguish provisional within-chunk prediction from retained learning after the
  harness decision. Record checkpoint/config identity, evaluation denominators,
  metric units, and training exposure such as MQAR batches.
- Do not pool text NLL and physics MSE into a single best-loss ranking. Lower NLL
  means greater likelihood under the specified assessor, not general text quality.
- Requested false-positive targets, nominal order-statistic rates, in-sample alarm
  frequencies, and fresh sequential intervention rates are different measurements.
  State relevant assumptions; calibration alone does not prove a policy-level FPR.
- Red-team comparisons need guarded, unprotected, and frozen controls from the same
  post-prefix state, guarded-path constraint checks, and explicit valid counts.
  Missing/undefined aggregates are not zero. An all-invalid attack arm is not strong
  defense evidence. Matched frozen controls are not a cross-payload noise floor.
- UI labels must distinguish proposed/accepted changes, enabled/accepted writes,
  inherited/explicit boolean settings, and absent/failed/stale data. Verify lifecycle
  updates, offline/recovery behavior, keyboard use, and narrow layouts when affected.
  Preserve the dashboard floor: no emoji, no text below 11px, color tokens throughout,
  and no opacity-based text de-emphasis.

## Artifacts, compute, and Git

- Generated `artifacts/` and `training_data/` contents are generally ignored; a Git
  push does not back up checkpoints or datasets. Some historical files are tracked.
  Preserve original experiment artifacts and use disposable stores for destructive
  test operations.
- Cloud jobs require user authorization. The HF launcher exports committed HEAD,
  not dirty source, and uses account-specific bucket/export settings. Inspect them
  before launching; paid training is not a routine verification step.
- Check Git status before editing or committing. Stage explicit paths when another
  agent is working. Do not reset, discard, or silently include their unfinished work.
- Follow the user's commit/push scope. Verify the remote and destination branch;
  do not force-push without authorization. If replacement is authorized and needed,
  use an explicit lease and verify the resulting remote SHA.
- `CLAUDE.md` imports this file. Keep shared project instructions here so Claude and
  Codex do not acquire conflicting copies.
