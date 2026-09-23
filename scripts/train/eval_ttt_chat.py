"""Evaluate a chat-tuned TTT checkpoint honestly and cheaply.

Two measurements, both recorded with the checkpoint digest and data identity:
1. Held-out assistant-token NLL on SmolTalk test rows (never used in training), rendered with the
   backend's chat format; reported per subset with token counts (a likelihood, not a quality score).
2. Sampled answers to a fixed list of ordinary prompts through the SAME transaction path the playground
   uses (TransactionRunner + drive_chat_turn, log-only), so what is inspected is what the Space serves.
   Answers are saved verbatim for a human to read; nothing here scores them automatically.

Usage:
  python -m scripts.train.eval_ttt_chat --checkpoint <dir> --out <dir> [--device mps] [--test-rows 200]
"""

from __future__ import annotations

import argparse
import json
import os
import time

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Explain what a prime number is to a ten year old.",
    "Give me three ideas for a quick vegetarian dinner.",
    "Write a two-sentence thank-you note to a neighbor who watered my plants.",
    "What is the difference between weather and climate?",
    "I have eggs, spinach and cheese. What can I cook?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "How do I convert 30 degrees Celsius to Fahrenheit?",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--test-rows", type=int, default=200)
    ap.add_argument("--subsets", default="everyday-conversations,smol-magpie-ultra,openhermes-100k")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset

    from plastic.backends.ttt_lm.backend import CHAT_ASSISTANT, CHAT_USER, TTTBackend
    from plastic.config import ModelConfig
    from plastic.harness.config import HarnessConfig
    from plastic.harness.transaction import TransactionRunner
    from plastic.session.runner import _TTTTextIO, drive_chat_turn
    from plastic.backends.ttt_lm.backend import encode_conversation as encode_example

    os.makedirs(args.out, exist_ok=True)
    be = TTTBackend.load(args.checkpoint, device=args.device)
    tok = be.tokenizer
    t0 = time.time()

    # 1) held-out assistant-token NLL, teacher-forced, single pass per conversation
    nll: dict[str, dict[str, float]] = {}
    for name in [s for s in args.subsets.split(",") if s]:
        ds = load_dataset("HuggingFaceTB/smoltalk", name, split="test")
        rows = [r["messages"] for r in ds.select(range(min(args.test_rows, len(ds))))]
        tot, n = 0.0, 0
        for msgs in rows:
            ids, labels = encode_example(tok, msgs, CHAT_USER, CHAT_ASSISTANT)
            ids, labels = ids[:2048], labels[:2048]
            logits = be.logits_full(ids).float()
            tgt = torch.tensor(labels[1:], device=logits.device)
            mask = tgt != -100
            if int(mask.sum()) == 0:
                continue
            l = torch.nn.functional.cross_entropy(logits[:-1][mask], tgt[mask], reduction="sum")
            tot += float(l)
            n += int(mask.sum())
        nll[name] = {"assistant_nll_per_token": tot / max(n, 1), "assistant_tokens": n, "rows": len(rows)}
        print(f"[eval] {name}: assistant NLL {tot / max(n, 1):.3f} over {n} tokens ({len(rows)} rows)", flush=True)

    # 2) sampled answers through the playground's transaction path
    io = _TTTTextIO(be)
    runner = TransactionRunner(None, ModelConfig(domain="text", chunk=16),
                               HarnessConfig(log_only=True, learn_from_generation=True, enable_projection=False, enable_budget=False),
                               device=be.device, backend=be)
    samples = []
    for i, p in enumerate(PROMPTS):
        runner.reset()
        runner.transactions = []
        completion, out_ids, in_ids = drive_chat_turn(runner, io, p, max_new_tokens=args.max_new_tokens, temperature=0.7, top_k=40,
                                                      gen=torch.Generator().manual_seed(args.seed + i))
        sig = [r["signals"] for r in runner.transactions]
        samples.append({"prompt": p, "completion": completion, "n_in": len(in_ids), "n_out": len(out_ids),
                        "ended_with_eos": len(out_ids) < args.max_new_tokens,
                        "mean_surprise": sum(s["surprise_mean"] for s in sig if s["surprise_mean"] is not None) / max(1, len(sig)),
                        "chunks": len(sig)})
        print(f"[eval] Q: {p}\n       A: {completion!r}", flush=True)
    payload = {"checkpoint": os.path.abspath(args.checkpoint), "checkpoint_digest": be.checkpoint_digest, "device": args.device,
               "held_out_nll": nll, "samples": samples, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "chat_eval.json"), "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[eval] saved {os.path.join(args.out, 'chat_eval.json')} in {payload['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
