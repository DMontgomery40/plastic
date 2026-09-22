"""Pinned pretrained text model used by the public CPU playground."""
from __future__ import annotations

import argparse
from pathlib import Path

from plastic.harness.config import HarnessConfig
from plastic.store import ArtifactStore

MODEL_ID = 'qwen3_5_0_8b_abliterated'
REPO_ID = 'huihui-ai/Huihui-Qwen3.5-0.8B-abliterated'
REVISION = '4813135658fe51b2e535b8e77927d2f23909ce42'
CHECKPOINT_DIGEST = '594479f86b804ce6899eb2985a85ae6fc4b5ea6b404140d7d37f08a8b9c7a575'
TEXT_PARAMETERS = 752393024


def download_checkpoint(target: Path) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(REPO_ID, revision=REVISION, local_dir=str(target),
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', 'LICENSE', 'README.md'])


def prepare_pretrained_sessions(store: ArtifactStore, checkpoint: Path) -> None:
    from plastic.backends.qwen import _checkpoint_digest
    if _checkpoint_digest(str(checkpoint)) != CHECKPOINT_DIGEST:
        raise ValueError('The public Qwen checkpoint does not match the pinned release')
    # Refuse a silent backend switch in an existing session. The hosted store is ephemeral;
    # local users with an older demo store should choose a fresh ARTIFACTS_ROOT.
    for sid, mid in [('demo_text', MODEL_ID), ('demo_physics', 'phys_mps_3k')]:
        if store.session_exists(sid) and store.load_session_meta(sid)['model_id'] != mid:
            raise ValueError(f'{sid} belongs to another model; choose a fresh ARTIFACTS_ROOT')
    store.register_model(MODEL_ID, {
        'backend': 'qwen', 'domain': 'text', 'status': 'completed',
        'params': TEXT_PARAMETERS, 'source': REPO_ID, 'revision': REVISION,
        'checkpoint_dir': str(checkpoint.resolve()), 'checkpoint_digest': CHECKPOINT_DIGEST,
        'chunk': 8,
    })
    # Keep normal context processing for both prompt and generated tokens. Signals are visible,
    # but the unfinished Qwen retention policy must not freeze or reject ordinary conversation.
    native = HarnessConfig(log_only=True, learn_from_generation=True, freeze_on_alarm=False,
                           enable_projection=False, enable_budget=False)
    for sid, mid, domain, harness in [
        ('demo_text', MODEL_ID, 'text', native),
        ('demo_physics', 'phys_mps_3k', 'physics', HarnessConfig()),
    ]:
        if not store.session_exists(sid):
            store.create_session(sid, model_id=mid, domain=domain, harness_cfg=harness)
        elif sid == 'demo_text' and store.load_session_meta(sid)['harness'] != native.to_dict():
            raise ValueError('Existing demo_text has different controls; choose a fresh ARTIFACTS_ROOT')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path, required=True)
    download_checkpoint(parser.parse_args().download)
