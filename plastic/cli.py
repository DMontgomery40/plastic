"""Command-line entry point: ``plastic <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from typing import Any

from plastic.config import ModelConfig


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--rule", choices=["delta", "chunk"], default=None)
    p.add_argument("--memory-input", choices=["ssm_out", "block_in"], default=None)
    p.add_argument("--conv-kernel", type=int, default=None)
    p.add_argument("--model-json", type=str, default=None, help="JSON object of ModelConfig overrides")


def _model_cfg_from_args(args: argparse.Namespace, *, domain: str, vocab_size: int | None) -> ModelConfig:
    d: dict[str, Any] = {"domain": domain}
    if vocab_size is not None:
        d["vocab_size"] = int(vocab_size)
    for arg, key in (("d_model", "d_model"), ("layers", "n_layers"), ("heads", "n_heads"), ("chunk", "chunk"),
                     ("rule", "rule"), ("memory_input", "memory_input"), ("conv_kernel", "conv_kernel")):
        v = getattr(args, arg, None)
        if v is not None:
            d[key] = v
    if args.model_json:
        d.update(json.loads(args.model_json))
    return ModelConfig.from_dict({**ModelConfig().to_dict(), **d})


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--artifacts-root", default="artifacts")
    p.add_argument("--model-id", default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr-matrix", type=float, default=2e-2)
    p.add_argument("--lr-other", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-muon", action="store_true")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")


def cmd_data_prepare(args: argparse.Namespace) -> int:
    from plastic.data.text import prepare_text_corpus

    meta = prepare_text_corpus(
        corpus=args.corpus,
        out_dir=args.out,
        vocab_size=args.vocab,
        tokenizer_lines=args.tokenizer_lines,
        max_train_tokens=args.max_train_tokens,
        cache_dir=args.cache_dir,
    )
    print(meta.to_json())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from plastic.train.loop import TrainConfig, train

    domain = args.domain
    vocab = None
    if domain == "text":
        from plastic.data.text import load_corpus_meta

        vocab = load_corpus_meta(args.data).vocab_size
    model_cfg = _model_cfg_from_args(args, domain=domain, vocab_size=vocab)
    cfg = TrainConfig(
        domain=domain,
        model=model_cfg,
        artifacts_root=args.artifacts_root,
        model_id=args.model_id,
        data_dir=getattr(args, "data", None),
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr_matrix=args.lr_matrix,
        lr_other=args.lr_other,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        use_muon=not args.no_muon,
        mqar_frac=getattr(args, "mqar_frac", 0.0),
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
        episodes_per_seq=getattr(args, "episodes_per_seq", 4),
        mu_range=(getattr(args, "mu_min", 0.02), getattr(args, "mu_max", 0.25)),
        nonlinear=getattr(args, "nonlinear", False),
        action_std=getattr(args, "action_std", 0.5),
    )
    model_id = train(cfg)
    print(model_id)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from plastic.store import ArtifactStore

    store = ArtifactStore(args.artifacts_root)
    for rec in store.list_models():
        ev = rec.get("eval") or {}
        print(
            f"{rec['model_id']:<28} {rec.get('domain', '?'):<8} {rec.get('status', '?'):<10} "
            f"params={rec.get('params', 0) / 1e6:.2f}M heldout={ev.get('heldout_loss', float('nan')):.4f} "
            f"memory_value={ev.get('memory_value', float('nan')):+.4f}"
        )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from scripts.bench_block import main as bench_main  # type: ignore[import-not-found]

    sys.argv = ["bench_block.py", "--device", args.device]
    bench_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plastic", description="plastic: a tiny test-time-training state-space model with a transactional safety harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    data = sub.add_parser("data", help="corpus preparation").add_subparsers(dest="data_cmd", required=True)
    prep = data.add_parser("prepare", help="download, tokenize, and encode a text corpus")
    prep.add_argument("--corpus", choices=["wikitext", "fineweb"], default="wikitext")
    prep.add_argument("--out", default="artifacts/data/wikitext")
    prep.add_argument("--vocab", type=int, default=8192)
    prep.add_argument("--tokenizer-lines", type=int, default=200_000)
    prep.add_argument("--max-train-tokens", type=int, default=None)
    prep.add_argument("--cache-dir", default=None)
    prep.set_defaults(fn=cmd_data_prepare)

    tr = sub.add_parser("train", help="train a model").add_subparsers(dest="domain", required=True)
    text = tr.add_parser("text")
    text.add_argument("--data", default="artifacts/data/wikitext")
    text.add_argument("--mqar-frac", type=float, default=0.2)
    _add_model_args(text)
    _add_train_args(text)
    text.set_defaults(fn=cmd_train)
    phys = tr.add_parser("physics")
    phys.add_argument("--episodes-per-seq", type=int, default=4)
    phys.add_argument("--mu-min", type=float, default=0.02)
    phys.add_argument("--mu-max", type=float, default=0.25)
    phys.add_argument("--nonlinear", action="store_true")
    phys.add_argument("--action-std", type=float, default=0.5)
    _add_model_args(phys)
    _add_train_args(phys)
    phys.set_defaults(fn=cmd_train, seq_len_default=512)

    models = sub.add_parser("models", help="list trained models")
    models.add_argument("--artifacts-root", default="artifacts")
    models.set_defaults(fn=cmd_models)

    bench = sub.add_parser("bench", help="throughput check")
    bench.add_argument("--device", default="auto")
    bench.set_defaults(fn=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
