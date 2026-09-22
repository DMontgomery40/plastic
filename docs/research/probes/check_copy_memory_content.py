"""Read-only model audit: reproduce the copy recipe and intervene on memory content.

Research probe, not a production test or proof of learning outside this toy task.
Writes its reusable checkpoint and JSON results only to the supplied output folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('PLASTIC_AUDIT_SOURCE', str(ROOT))).resolve()
sys.path.insert(0, str(SOURCE))
from plastic.config import ModelConfig
from plastic.model.lm import PlasticLM
from plastic.train.optim import build_optimizer


def snapshot() -> dict[str, str]:
    paths = [SOURCE / 'plastic/config.py', SOURCE / 'plastic/train/optim.py',
             *sorted((SOURCE / 'plastic/model').glob('*.py'))]
    return {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def repeat_batch(batch: int, generator=None) -> torch.Tensor:
    return torch.randint(0, 32, (batch, 8), generator=generator).repeat(1, 4)


def scores(logits: torch.Tensor, tokens: torch.Tensor, start: int) -> dict:
    # logits at global input t predicts global target t+1, no unknown final target.
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, 32),
                         tokens[:, start + 1:].reshape(-1), reduction='none')
    ce = ce.reshape(tokens.shape[0], -1)
    pos = torch.arange(start + 1, tokens.shape[1])
    per_rep = {str(int(rep) + 1): float(ce[:, pos // 8 == rep].mean())
               for rep in torch.unique(pos // 8)}
    return {'loss': float(ce.mean()), 'per_repetition_loss': per_rep,
            'per_example_loss': ce.mean(1).tolist()}


def paired_difference(left: dict, right: dict) -> dict:
    delta = torch.tensor(left['per_example_loss']) - torch.tensor(right['per_example_loss'])
    return {'mean': float(delta.mean()), 'standard_error': float(delta.std() / len(delta)**.5),
            'n_examples': len(delta)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    hashes = snapshot()
    head_file = SOURCE / '.audit-head'
    head = (head_file.read_text().strip() if head_file.exists() else
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    cfg = ModelConfig(d_model=64, n_heads=2, n_layers=2, chunk=16, vocab_size=32)
    model = PlasticLM(cfg)
    checkpoint = out / 'copy_seed0_steps150.pt'
    cached = False
    first = last = None
    t0 = time.monotonic()
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['source_sha256'] != hashes:
            raise RuntimeError('Cached checkpoint source differs: use a new output directory.')
        model.load_state_dict(saved['state_dict'])
        first, last = saved['first_loss'], saved['last_loss']
        cached = True
    else:
        opt = build_optimizer(model, lr_matrix=2e-2, lr_other=1e-2, use_muon=True)
        for step in range(150):
            loss = model.loss(repeat_batch(32))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            if first is None:
                first = float(loss.detach())
            last = float(loss.detach())
            if (step + 1) % 50 == 0:
                print(json.dumps({'step': step + 1, 'loss': last}), flush=True)
        torch.save({'state_dict': model.state_dict(), 'config': cfg.to_dict(),
                    'source_sha256': hashes, 'git_head': head,
                    'first_loss': first, 'last_loss': last}, checkpoint)
    train_seconds = time.monotonic() - t0
    model.eval()
    # Fresh examples, independent from the random stream consumed during training.
    tokens = repeat_batch(256, torch.Generator().manual_seed(260922))
    result = {'git_head': head, 'source_root': str(SOURCE), 'source_sha256': hashes, 'torch_version': str(torch.__version__),
              'device': 'cpu', 'dtype': 'float32', 'seed': 0, 'steps': 150,
              'cached_checkpoint': cached, 'training_or_load_seconds': train_seconds,
              'first_training_loss': first, 'last_training_loss': last,
              'evaluation_examples': len(tokens), 'boundaries': {}}
    with torch.no_grad():
        adaptive_logits, _, _ = model(tokens)
        disabled_logits, _, _ = model(tokens, beta_scale=0.)
        for boundary in (8, 16):
            _, original, _ = model(tokens[:, :boundary])
            cases = {'retained_memory_frozen': original.clone(),
                     'zero_memory_frozen': original.clone(),
                     'shuffled_memory_frozen': original.clone()}
            for layer in cases['zero_memory_frozen'].layers:
                layer.S.zero_()
            for layer in cases['shuffled_memory_frozen'].layers:
                layer.S = layer.S.roll(1, dims=0)
            output = {}
            for label, state in cases.items():
                for src, candidate in zip(original.layers, state.layers):
                    for name in ('h', 'conv_ssm', 'conv_mem', 'M'):
                        a, b = getattr(src, name), getattr(candidate, name)
                        assert (a is None and b is None) or torch.equal(a, b), name
                initial_s = [layer.S.clone() for layer in state.layers]
                logits, terminal, _ = model(tokens[:, boundary:], state, freeze=True)
                assert all(torch.equal(s, layer.S) for s, layer in zip(initial_s, terminal.layers))
                output[label] = scores(logits, tokens, boundary)
            output['uninterrupted_adaptation'] = scores(adaptive_logits[:, boundary:], tokens, boundary)
            output['writes_disabled_from_start'] = scores(disabled_logits[:, boundary:], tokens, boundary)
            output['zero_minus_retained'] = paired_difference(output['zero_memory_frozen'],
                                                             output['retained_memory_frozen'])
            output['shuffled_minus_retained'] = paired_difference(output['shuffled_memory_frozen'],
                                                                 output['retained_memory_frozen'])
            result['boundaries'][str(boundary)] = output
    result['source_unchanged_during_probe'] = snapshot() == hashes
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    summary = {**result, 'source_sha256': 'see results.json', 'boundaries': {
        boundary: {name: {k:v for k,v in metrics.items() if k != 'per_example_loss'}
                   for name,metrics in cases.items()}
        for boundary,cases in result['boundaries'].items()}}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
