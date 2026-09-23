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
The model picker opens one shared session for each hosted model:

| Model | Session | Use and controls |
| --- | --- | --- |
| `qwen3_5_0_8b_abliterated` | `demo_text` | Default chat comparison. Observational: records and retains proposed context updates without automatic rollback. |
| `lm_wikitext_l4` | `demo_core` | Published 6.85M PlasticCore WikiText baseline for text continuation; not chat-tuned. Guarded: the transactional harness can commit, scale, project or roll back chunks. |

Qwen uses the pinned `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` checkpoint through
the native backend. No original-checkpoint thresholds are installed as calibrated
decisions for this derivative. PlasticCore uses the published `text/` bundle and
its saved calibration, canaries and Fisher reference. Its session permits learning
from prompts and generated tokens. A CUSUM alarm rolls back the current chunk
without latching future chunks read-only (`freeze_on_alarm=false`). Proposed signals
and accepted changes remain separate. These are exploratory controls, not evidence
of useful retention or protection. Saved calibration does not establish a
policy-level false-positive rate or adversarial robustness.

Both sessions are shared by all visitors. Prompts and outputs are public; storage
is disposable and resets when the Space restarts. Chat permits up to 1,024 prompt
characters and 128 generated tokens. One mutation
runs at a time. Sessions reaching position 4,096 need an explicit reset. Training,
calibration, red-team execution, Sleep execution, and session creation/forking/deletion
are unavailable in this deployment; use the local project for those workflows.
The Sleep tab still shows the published research observatory. The fine-tuned
TTT-MLP checkpoint is not yet published or selectable; its future integration is
tracked in [current research status](../../docs/research/current-status.md).

The served HTML marks public mode with `data-public-demo="true"` on its body.
The API supplies the public capabilities, model/session catalogs and matching
health counts. The dashboard uses them to show model selection, Chat, Signals and
the permitted session controls. Existing local artifacts are preserved. The
unrestricted local API and dashboard have no public-mode marker.

The published `text/` and `physics/` folders preserve the original research
checkpoints, tokenizers, saved calibration/canaries/Fisher references, and training
logs. Alongside the pinned Qwen checkpoint, the Space registers and exposes the
`text/` bundle at boot; the physics checkpoint
stays published as an internal adaptation benchmark and is not part of the public
playground. `prepare.py` verifies manifest checksums before registering a bundle and
refuses to overwrite changed files. Physics remains an internal benchmark and CLI
workflow, outside the playground.

## Run locally from the complete Hugging Face download

```bash
uv sync --extra dev --extra pretrained
uv run python -m deploy.huggingface.pretrained --download artifacts/qwen
npm --prefix dashboard ci
npm --prefix dashboard run build
PUBLIC_CHECKPOINT=artifacts/qwen uv run python -m deploy.huggingface.app
```

Open `http://localhost:7860`. The Space adapter defaults to `/tmp/plastic-demo` for
its disposable store. `ARTIFACTS_ROOT`, `DASHBOARD_DIST`, and `PORT` override these
settings. `PUBLIC_CHECKPOINT` selects the downloaded pinned checkpoint (the Docker
image uses `/opt/public-model`); `QWEN_CHECKPOINT` remains a fallback. The complete
download must also contain the published `text/` bundle and its manifest. Use a
fresh `ARTIFACTS_ROOT` if existing demo sessions belong to different models or use
different harness controls: startup refuses to silently replace them. This adapter
is for a public sandbox, not private multi-user hosting.

## Run the unrestricted development dashboard

```bash
uv run python -m deploy.huggingface.prepare --source . --artifacts-root artifacts
bash start.sh
```

This imports the published PlasticCore text model without training. It preserves identical existing
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
training job is needed. Its build installs PyTorch 2.14.0 and Transformers 5.17.0,
downloads the pinned Huihui Qwen derivative, copies the published PlasticCore
`text/` bundle, and compiles the React UI. The PyTorch wheel supports CUDA;
startup selects CUDA when available and otherwise CPU. The current Space runs on CPU.
