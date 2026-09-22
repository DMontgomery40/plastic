"""Throughput check for the default text model: forward + backward + optimizer step.

Run: uv run python scripts/bench_block.py [--device mps|cpu|cuda] [--batch 8] [--seq 1024]
"""

from __future__ import annotations

import argparse
import time

import torch

from plastic.config import ModelConfig
from plastic.model.lm import PlasticLM
from plastic.train.optim import build_optimizer


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args()

    dev = pick_device(args.device)
    cfg = ModelConfig()
    lm = PlasticLM(cfg).to(dev)
    opt = build_optimizer(lm)
    print(f"device={dev} params={lm.num_params() / 1e6:.2f}M d={cfg.d_model} layers={cfg.n_layers} chunk={cfg.chunk}")
    toks = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=dev)

    def step() -> float:
        loss = lm.loss(toks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss)

    for _ in range(2):
        step()
    sync(dev)
    t0 = time.time()
    for _ in range(args.steps):
        loss = step()
    sync(dev)
    dt = (time.time() - t0) / args.steps
    tok = args.batch * args.seq
    print(f"fwd+bwd+step B={args.batch} T={args.seq}: {dt * 1000:.0f} ms -> {tok / dt / 1e3:.1f}K tok/s; loss {loss:.3f}")


if __name__ == "__main__":
    main()
