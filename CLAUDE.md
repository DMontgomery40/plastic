# plastic: Claude Code instructions

`plastic` is a tiny test-time-training state-space model (TTT-SSM) with a transactional safety harness. One package, one artifact store, one API, one dashboard, two domains (text and hidden-mu physics) sharing the same block stack.

## Commands

Everything runs through `uv` (Python 3.12, torch >= 2.12; MPS on Apple Silicon, CUDA on Hugging Face Jobs).

```bash
uv sync --extra dev                      # environment
uv run pytest -W ignore                  # the whole suite (CPU + MPS when available)
uv run plastic --help                    # CLI

# data and training
uv run plastic data prepare --corpus wikitext --out artifacts/data/wikitext --vocab 8192
uv run plastic train text --data artifacts/data/wikitext --steps 3000 --batch-size 16 --device mps
uv run plastic train physics --steps 3000 --layers 3 --seq-len 512 --device mps
uv run plastic train text ... --adversarial     # meta-train the write gate against an attacker
uv run plastic models                            # registry with the three checkpoint numbers
scripts/hf_jobs/launch_text.sh l4x1 6000 BATCH=16 MODEL_ID=lm_wikitext_l4   # cloud run (exports git HEAD, no push needed)
hf jobs logs -f dmontgomery40/<job_id>; hf buckets sync hf://buckets/dmontgomery40/plastic-runs/artifacts/models/<id> artifacts/models/<id>

# harness, sessions, attacks, sleep
uv run plastic calibrate <model_id> --data artifacts/data/wikitext      # canaries, Fisher, thresholds
uv run plastic session new --model <model_id>                            # prints the session id
uv run plastic chat <session_id> "prompt"                                # learns the prompt through chunk transactions
uv run plastic physics <session_id> --steps 256 --mu 0.12                # base / frozen / adaptive on one trajectory
uv run plastic session list | fork <parent> [child] | reset <id> | resume <id> | show <id>
uv run plastic redteam <model_id> --data artifacts/data/wikitext --record
uv run plastic sleep <model_id> --core artifacts/data/wikitext

# service and dashboard
uv run plastic serve --artifacts-root artifacts --port 13579
./start.sh                                       # API + dashboard (http://127.0.0.1:5173)
npm -C dashboard run build && npm -C dashboard test
```

## Layout

- `plastic/model/`: `scan.py` (chunked log-space selective scan), `delta.py` (gated delta rule, recurrent and chunk-parallel with an exact triangular solve), `chunk_rule.py` (mini-batch rule with momentum and Newton-Schulz), `memory.py` (FastWeightMemory: `delta` and `chunk` rules, `freeze`, `beta_scale`), `block.py` (PlasticBlock: conv, SSM branch, memory branch, MLP), `lm.py` (PlasticLM, PlasticDynamics on PlasticCore), `state.py` (SessionState per layer: h, S, M, conv buffers, pending chunk statistics).
- `plastic/data/`: wikitext/fineweb pipeline, MQAR recall batches, hidden-mu physics episodes.
- `plastic/train/`: outer loop (Muon on block matrices, AdamW elsewhere), schedule, adversarial write-gate loss.
- `plastic/harness/`: signals, robust stats and CUSUM, canary suites, Fisher, projection, policy, calibration, and `transaction.py` (the chunk transaction runner: commit / rollback / scale / project / readonly with hard budgets).
- `plastic/session/`: persisted branchable sessions for both domains. `plastic/store.py`: models, sessions, transactions, traces, forks, model signatures.
- `plastic/redteam/`, `plastic/sleep/`, `plastic/api/`, `plastic/cli.py`, `dashboard/` (React), `scripts/hf_jobs/`, `scripts/experiments/`.
- `docs/superpowers/specs/2026-09-21-plastic-design.md` is the design spec; `docs/superpowers/plans/` the milestone plans; `docs/research/` the surveys, the architecture memo, the coordinate-recurrence proposal, and results.
- `SHARED_SCRATCHPAD.md`: the append-only board shared with the Codex reviewer ("Astra"); entries are `ASTRA-nnn` and `FABLE-nnn`.

## Rules that matter here

- No string heuristics in anything safety-related; every harness signal comes from the model (loss, surprise, β, state deltas, Fisher, canaries, statistics). Compression ratio is display only.
- `freeze=True` means no write and no decay; `beta_scale` scales writes only. The harness changes controls only at chunk boundaries.
- Chunk-parallel and recurrent paths must stay numerically equivalent; the equivalence tests (scan, delta rule incl. identical keys, block, model with carried state, chunk-rule streaming) are permanent.
- Never write "legacy", "v1", "backward compatibility", or schema-version fields. Session compatibility is guarded by `model_signature`.
- Keep the design floor for the dashboard: no emoji, no text below 11px, tokens for every color, no opacity for de-emphasis.
- Commit on a working branch; never push; never `rm -rf` (use `git rm`).
