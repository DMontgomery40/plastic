"""Audit a saved text model's calibration/replay; writes only to --out.

Use PLASTIC_AUDIT_SOURCE to pin imports to an immutable checkout/export. This
reproduces the implementation, including any known calibration defects; it is
not a certificate that the chosen false-positive target holds in deployment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('PLASTIC_AUDIT_SOURCE', str(ROOT))).resolve()
sys.path.insert(0, str(SOURCE))

from plastic.data.text import TokenWindows
from plastic.harness.calibrate import Calibration, calibrate_from_runner, log_only
from plastic.harness.canary import CanarySuite
from plastic.harness.config import HarnessConfig
from plastic.harness.fisher import estimate_fisher_diag
from plastic.harness.transaction import TransactionRunner
from plastic.store import ArtifactStore


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordingRunner(TransactionRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_records = []

    def _transact(self):
        record = super()._transact()
        self.audit_records.append(record)
        return record


def summarize(records):
    counts = Counter(r['decision']['kind'] for r in records)
    reasons = Counter(reason.split('(')[0] for r in records for reason in r['decision']['reasons'])
    n = len(records)
    return {'chunks': n, 'decision_counts': dict(counts),
            'gated_fraction': sum(v for k, v in counts.items() if k != 'commit') / n,
            'direct_intervention_fraction': sum(v for k, v in counts.items() if k not in ('commit', 'readonly')) / n,
            'readonly_fraction': counts.get('readonly', 0) / n,
            'reasons': dict(reasons)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--model-id', default='lm_smoke_mps2')
    p.add_argument('--chunks', type=int, default=128)
    p.add_argument('--fisher-chunks', type=int, default=16)
    p.add_argument('--evaluation-chunks', type=int, default=None)
    p.add_argument('--fisher-before-calibration', action='store_true',
                   help='Match the revised CLI that attaches Fisher before reference collection.')
    p.add_argument('--session-init', choices=('reset', 'fresh'), default='reset',
                   help='Reuse/reset a runner (original audit) or instantiate each evaluation session.')
    args = p.parse_args()
    if args.chunks < 20 or args.chunks % 4:
        raise ValueError('Use at least 20 chunks, divisible by four.')
    evaluation_chunks = args.evaluation_chunks if args.evaluation_chunks is not None else args.chunks
    if evaluation_chunks < 4 or evaluation_chunks % 4 or evaluation_chunks > args.chunks:
        raise ValueError('Evaluation chunks must be a positive multiple of four, no larger than calibration chunks.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source_hashes = {str(f.relative_to(SOURCE)): sha(f) for f in sorted((SOURCE / 'plastic').rglob('*.py'))}
    store = ArtifactStore(str(ROOT / 'artifacts'))
    cfg, model, info = store.load_checkpoint(args.model_id, 'cpu')
    model.eval()
    checkpoint = Path(store.checkpoint_path(args.model_id))
    checkpoint_hash = sha(checkpoint)
    heldout = ROOT / 'artifacts/data/wikitext/validation.bin'
    data = np.memmap(heldout, dtype='<u2', mode='r')
    L = cfg.chunk
    start = 384
    width = args.chunks * L
    evaluation_width = evaluation_chunks * L
    if start + width + evaluation_width > len(data):
        raise ValueError('Not enough disjoint held-out tokens for calibration and fresh evaluation.')
    suite_path = Path(store.canary_path(args.model_id))
    suite = CanarySuite.load(str(suite_path)) if suite_path.exists() else CanarySuite.default_text(str(heldout), vocab_size=cfg.vocab_size)
    hcfg = HarnessConfig(target_fpr=.01)
    windows = TokenWindows(str(heldout), seq_len=L * 4)
    rng = torch.Generator().manual_seed(0)
    t0 = time.monotonic()
    fisher = estimate_fisher_diag(model, (windows.sample(4, rng) for _ in range(args.fisher_chunks)),
                                  chunk=L, n_chunks=args.fisher_chunks, device=torch.device('cpu'))
    print(json.dumps({'phase': 'fisher_complete', 'seconds': time.monotonic() - t0}), flush=True)

    def stream(offset):
        for j in range(args.chunks // 4):
            a = offset + j * L * 4
            yield data[a:a + L * 4].astype('int64').tolist()
            if (j + 1) % 8 == 0:
                print(json.dumps({'phase': 'calibration', 'chunks': (j + 1) * 4,
                                  'elapsed_seconds': time.monotonic() - t0}), flush=True)

    # Preserve the original reproduction by default; select the revised CLI's
    # pre-attached Fisher explicitly when auditing that implementation.
    initial_cal = Calibration(fisher=fisher) if args.fisher_before_calibration else None
    runner = RecordingRunner(model, cfg, log_only(hcfg), calibration=initial_cal, suite=suite)
    cal = calibrate_from_runner(runner, stream(start), n_chunks=args.chunks,
                                model_signature=store.model_signature(args.model_id),
                                target_fpr=.01, fisher=fisher, reset_every=4)
    cal.save(str(out))
    evaluations = {}
    all_records = {'calibration': runner.audit_records}
    for label, offset in [('same_data_replay', start), ('disjoint_fresh_sessions', start + width)]:
        r = TransactionRunner(model, cfg, hcfg, calibration=cal, suite=suite)
        records = []
        for j in range(evaluation_chunks // 4):
            if args.session_init == 'fresh':
                r = TransactionRunner(model, cfg, hcfg, calibration=cal, suite=suite)
            else:
                r.reset()
            a = offset + j * L * 4
            r.feed_tokens(data[a:a + L * 4].astype('int64').tolist())
            r.flush()
            records.extend(r.transactions)
            r.transactions.clear()
            if (j + 1) % 8 == 0:
                print(json.dumps({'phase': label, 'chunks': (j + 1) * 4,
                                  'elapsed_seconds': time.monotonic() - t0}), flush=True)
        evaluations[label] = summarize(records)
        all_records[label] = records
    result = {'source_root': str(SOURCE), 'source_sha256': source_hashes,
              'model_id': args.model_id, 'checkpoint_sha256': checkpoint_hash,
              'checkpoint_step': info.get('step'), 'config': cfg.to_dict(),
              'heldout_sha256': sha(heldout), 'device': 'cpu', 'torch_version': str(torch.__version__),
              'dtype': 'float32', 'target_fpr': .01, 'calibration_chunks': cal.n_chunks,
              'evaluation_chunks': evaluation_chunks, 'session_init': args.session_init,
              'fisher_before_calibration': args.fisher_before_calibration,
              'reset_every': 4, 'calibration_token_range': [start, start + width],
              'same_data_evaluation_token_range': [start, start + evaluation_width],
              'fresh_token_range': [start + width, start + width + evaluation_width],
              'fisher_chunks': args.fisher_chunks, 'thresholds': cal.thresholds,
              'achievable_fpr': getattr(cal, 'achievable_fpr', {}),
              'fisher_reference_present': bool(cal.reference.get('fisher_update')),
              'evaluations': evaluations, 'elapsed_seconds': time.monotonic() - t0,
              'source_unchanged': source_hashes == {str(f.relative_to(SOURCE)): sha(f) for f in sorted((SOURCE / 'plastic').rglob('*.py'))},
              'checkpoint_unchanged': sha(checkpoint) == checkpoint_hash}
    (out / 'records.json').write_text(json.dumps(all_records, indent=2) + '\n')
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
