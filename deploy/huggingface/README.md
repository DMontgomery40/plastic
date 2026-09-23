# Public CPU demo

## Keep GitHub and Hugging Face in sync

Every push to GitHub `main` runs [the publication workflow](https://github.com/DMontgomery40/plastic/actions/workflows/sync-to-hub.yml)
for both the [model repository](https://huggingface.co/dmontgomery40/plastic) and
[Space](https://huggingface.co/spaces/dmontgomery40/plastic). It uses the official
[Hugging Face sync action](https://huggingface.co/docs/hub/repositories-github-actions).
The GitHub `HF_TOKEN` secret must have write access to both repositories.

The workflow stages committed source and the shared README body, preserves each
destination's README metadata and existing `text/` and `physics/` checkpoint bundles,
and supplies the Space Dockerfile. It excludes local artifacts and the private
scratchpad. Both jobs verify every exported file against the published bytes;
`source_snapshot.json` records the originating GitHub commit. GitHub is the source
for normal source/documentation edits; checkpoint releases remain explicit.

Check both jobs in GitHub Actions after publishing. Runtime changes also require a
successful Space build and a check of the affected live flow. A development-branch
push is a backup and does not trigger deployment. To retry delivery, use the workflow's
**Run workflow** control on `main`.

## Deployment

The Space serves the existing React dashboard and Python model API on one port.
The deployment adapter adds a visible public-session notice and restricts the API
to bounded operations on `demo_text`. The text session uses pinned
`huihui-ai/Huihui-Qwen3.5-0.8B-abliterated`, a refusal-ablated Qwen3.5-0.8B derivative, through the
existing native backend. It is **log-only / observational**: proposed context
updates are recorded and retained, without rollback protection. It supports exploratory
prompts without a fixed attack list or prior calibration. No original-checkpoint thresholds
are installed as calibrated decisions for this derivative. Physics and the other
research workflows remain available in the local dashboard.

The session is shared by all visitors. Prompts and outputs are public; storage
is disposable and resets when the Space restarts. Chat permits up to 1,024 prompt
characters and 128 generated tokens. One mutation
runs at a time. Sessions reaching position 4,096 need an explicit reset. Training,
calibration, red-team execution, consolidation, and session creation/forking/deletion
are disabled in the Space; use the local project for those workflows.

The served HTML marks public mode with `data-public-demo="true"` on its body.
The dashboard uses that marker to show text Chat, session measurements, and session
controls. Model/session catalogs and health counts include only this public model
and session; existing local artifacts are preserved. The unrestricted local API
and dashboard have no public-mode marker.

The published `text/` and `physics/` folders preserve the original research
checkpoints, tokenizers, saved calibration/canaries/Fisher references, and training
logs. The Space registers only the `text/` bundle at boot; the physics checkpoint
stays published as an internal adaptation benchmark and is not part of the public
playground. `prepare.py` verifies manifest checksums before registering a bundle and
refuses to overwrite changed files.
Saved calibration is an operating point, not a promise of a policy-level false-positive
rate or adversarial robustness.

## Run locally from the complete Hugging Face download

```bash
uv sync --extra dev --extra pretrained
uv run python -m deploy.huggingface.pretrained --download artifacts/qwen
npm --prefix dashboard ci
npm --prefix dashboard run build
QWEN_CHECKPOINT=artifacts/qwen uv run python -m deploy.huggingface.app
```

Open `http://localhost:7860`. The Space adapter defaults to `/tmp/plastic-demo` for
its disposable store. `ARTIFACTS_ROOT`, `DASHBOARD_DIST`, and `PORT` override these
settings. `QWEN_CHECKPOINT` selects the downloaded pinned checkpoint (the Docker
image uses `/opt/qwen`). Use a fresh artifact store when switching an older demo
from the research text checkpoint to Qwen; incompatible sessions are refused. This adapter is for a public sandbox, not private multi-user hosting.

## Run the unrestricted development dashboard

```bash
uv run python -m deploy.huggingface.prepare --source . --artifacts-root artifacts
bash start.sh
```

This imports the published models without training. It preserves identical existing
model artifacts and refuses to overwrite changed ones. The ordinary local server
continues to support creating your own sessions and running experiments.

## Verify

```bash
uv run pytest tests deploy/huggingface/test_space.py
npm --prefix dashboard test
npm --prefix dashboard run build
```

The Space uses `Dockerfile` copied from this directory at publication time, port
7860, and Hugging Face's free `cpu-basic` hardware. No paid inference service or
training job is needed. Its build installs CPU PyTorch and Transformers 5.17.0, downloads the pinned
Huihui Qwen derivative, and compiles the React UI. The public text model is
`qwen3_5_0_8b_abliterated`; the old `text/` weights remain available for research reproduction.
