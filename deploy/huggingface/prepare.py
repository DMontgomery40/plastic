"""Register published checkpoints in a local artifact store without retraining."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from plastic.store import ArtifactStore

PAIRS = (('text', 'lm_wikitext_l4'),)


def prepare_store(source: Path, target: Path, *, seed_sessions: bool = False):
    # Preflight every published file before copying or touching the registry.
    for sub, mid in PAIRS:
        folder = source / sub
        manifest = json.loads((folder / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            if Path(name).name != name:
                raise ValueError(f'invalid manifest filename: {name}')
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'checksum mismatch: {sub}/{name}')
            existing = target / 'models' / mid / name
            if existing.exists() and hashlib.sha256(existing.read_bytes()).hexdigest() != digest:
                raise ValueError(f'refusing to overwrite changed model artifact: {existing}')
    store = ArtifactStore(str(target))
    known = {r['model_id'] for r in store.list_models()}
    for sub, mid in PAIRS:
        folder = source / sub
        manifest = json.loads((folder / 'manifest.json').read_text())
        dest = Path(store.model_dir(mid))
        dest.mkdir(parents=True, exist_ok=True)
        for name in manifest['files']:
            if not (dest / name).exists():
                shutil.copy2(folder / name, dest / name)
        if mid not in known:
            cfg, model, checkpoint = store.load_checkpoint(mid)
            store.register_model(mid, {'domain': cfg.domain, 'status': 'completed',
                'params': model.num_params(), 'steps': checkpoint['step'],
                'tokens': checkpoint['extra'].get('tokens'),
                'eval': store.read_eval(mid), 'source': 'dmontgomery40/plastic'})
    if seed_sessions:
        from plastic.session.runner import Session
        for (_, mid), sid in zip(PAIRS, ('demo_text',)):
            if not store.session_exists(sid):
                Session.create(store, model_id=mid, session_id=sid, device='cpu')
    return store


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('.'), help='Download directory with text/')
    parser.add_argument('--artifacts-root', type=Path, default=Path('artifacts'))
    parser.add_argument('--seed-sessions', action='store_true', help='Create the public demo session')
    args = parser.parse_args()
    store = prepare_store(args.source, args.artifacts_root, seed_sessions=args.seed_sessions)
    for record in store.list_models():
        print(record['model_id'], record['domain'], record['params'])
