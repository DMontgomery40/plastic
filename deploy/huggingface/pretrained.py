"""Pinned pretrained text model used by the public playground.

Two specs are known: the abliterated Qwen (the current public model) and the plastic TTT chat
checkpoint (the release that turns hosted sleep on). PUBLIC_MODEL selects one at build and boot time;
the chosen spec's id is what the public gate exposes. A TTT spec is only valid once its repo,
revision and digest are pinned to a released checkpoint.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from plastic.harness.config import HarnessConfig
from plastic.store import ArtifactStore


@dataclass(frozen=True)
class PublicModelSpec:
    kind: str            # 'qwen' | 'ttt'
    model_id: str
    repo_id: str
    revision: str
    checkpoint_digest: str
    params: int
    chunk: int
    chat_tuned: bool | None
    allow_patterns: tuple[str, ...]
    subfolder: str | None = None
    label: str = ''

    def digest_of(self, checkpoint: Path) -> str:
        if self.kind == 'qwen':
            from plastic.backends.qwen import _checkpoint_digest
        else:
            from plastic.backends.ttt_lm.backend import _checkpoint_digest
        return _checkpoint_digest(str(checkpoint))


QWEN = PublicModelSpec(
    kind='qwen', model_id='qwen3_5_0_8b_abliterated', repo_id='huihui-ai/Huihui-Qwen3.5-0.8B-abliterated',
    revision='4813135658fe51b2e535b8e77927d2f23909ce42',
    checkpoint_digest='594479f86b804ce6899eb2985a85ae6fc4b5ea6b404140d7d37f08a8b9c7a575',
    params=752393024, chunk=8, chat_tuned=None,
    allow_patterns=('*.json', '*.safetensors', '*.jinja', 'LICENSE', 'README.md'),
    label='Huihui Qwen3.5-0.8B · abliterated · observational mode',
)

# Filled in by the chat-checkpoint release: repo/subfolder, revision and digest of the published TTT-MLP
# 760M chat checkpoint. Until then this spec is not selectable.
TTT_CHAT = PublicModelSpec(
    kind='ttt', model_id='ttt_mlp_760m_chat_v1', repo_id='dmontgomery40/plastic', revision='', checkpoint_digest='',
    params=759000000, chunk=16, chat_tuned=True, subfolder='text-chat',
    allow_patterns=('text-chat/*.json', 'text-chat/*.safetensors'),
    label='plastic TTT-MLP 760M chat · fast weights · observational mode · sleep enabled',
)

SPECS = {'qwen': QWEN, 'ttt': TTT_CHAT}


def active_spec() -> PublicModelSpec:
    spec = SPECS[os.environ.get('PUBLIC_MODEL', 'qwen')]
    if not spec.revision or not spec.checkpoint_digest:
        raise ValueError(f'public model {spec.model_id} is not pinned to a released checkpoint yet')
    return spec


ACTIVE = SPECS[os.environ.get('PUBLIC_MODEL', 'qwen')]
MODEL_ID = ACTIVE.model_id
REPO_ID, REVISION, CHECKPOINT_DIGEST, TEXT_PARAMETERS = ACTIVE.repo_id, ACTIVE.revision, ACTIVE.checkpoint_digest, ACTIVE.params


def download_checkpoint(target: Path, spec: PublicModelSpec | None = None) -> None:
    from huggingface_hub import snapshot_download
    spec = spec or active_spec()
    snapshot_download(spec.repo_id, revision=spec.revision, local_dir=str(target), allow_patterns=list(spec.allow_patterns))


# Keep normal context processing for both prompt and generated tokens. Signals are visible, but the
# unfinished retention policy must not freeze or reject ordinary conversation.
NATIVE_HARNESS = HarnessConfig(log_only=True, learn_from_generation=True, freeze_on_alarm=False,
                               enable_projection=False, enable_budget=False)


def prepare_pretrained_sessions(store: ArtifactStore, checkpoint: Path, spec: PublicModelSpec | None = None) -> None:
    spec = spec or active_spec()
    folder = checkpoint / spec.subfolder if spec.subfolder and (checkpoint / spec.subfolder).is_dir() else checkpoint
    if spec.digest_of(folder) != spec.checkpoint_digest:
        raise ValueError(f'The public {spec.kind} checkpoint does not match the pinned release')
    # Refuse a silent backend switch in an existing session. The hosted store is ephemeral;
    # local users with an older demo store should choose a fresh ARTIFACTS_ROOT.
    for sid, mid in [('demo_text', spec.model_id)]:
        if store.session_exists(sid) and store.load_session_meta(sid)['model_id'] != mid:
            raise ValueError(f'{sid} belongs to another model; choose a fresh ARTIFACTS_ROOT')
    record = {
        'backend': spec.kind, 'domain': 'text', 'status': 'completed',
        'params': spec.params, 'source': spec.repo_id, 'revision': spec.revision,
        'checkpoint_dir': str(folder.resolve()), 'checkpoint_digest': spec.checkpoint_digest,
        'chunk': spec.chunk,
    }
    if spec.chat_tuned is not None:
        record['chat_tuned'] = spec.chat_tuned
    store.register_model(spec.model_id, record)
    native = NATIVE_HARNESS
    for sid, mid, domain, harness in [
        ('demo_text', spec.model_id, 'text', native),
    ]:
        if not store.session_exists(sid):
            store.create_session(sid, model_id=mid, domain=domain, harness_cfg=harness)
        elif sid == 'demo_text' and store.load_session_meta(sid)['harness'] != native.to_dict():
            raise ValueError('Existing demo_text has different controls; choose a fresh ARTIFACTS_ROOT')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path, required=True)
    download_checkpoint(parser.parse_args().download)
