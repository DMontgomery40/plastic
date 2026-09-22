"""Build a conversational calibration corpus so the harness can be calibrated on the
distribution a chat demo actually sees, not on wikitext.

The harness flags out-of-distribution input by design. A wikitext-trained model has high
loss on short conversational prompts, so a harness calibrated on wikitext held-out text
flags *every* chat turn ("it's never not surprised"). Calibrating on conversational text
instead raises the thresholds to the chat baseline: normal turns commit, genuinely anomalous
turns (higher loss than ordinary chat) still trip the gate.

This writes an `artifacts/data/chat` corpus (train/validation .bin + tokenizer + meta) using
an existing model's tokenizer, sourced from the open `databricks/databricks-dolly-15k`
instruction dataset. Then:

    plastic calibrate <model_id> --data artifacts/data/chat --device mps

produces a chat-domain calibration.json to serve in the demo (keep the wikitext calibration
for the operating-point study — they are different operating points for different input
distributions).

Run: uv run python scripts/experiments/build_chat_calibration.py --tokenizer artifacts/data/wikitext/tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from plastic.tokenizer.bpe import Tokenizer

DOC_SEP = 2  # separator token between documents


def collect_conversational_docs(max_docs: int) -> tuple[list[str], str]:
    """Instruction/response text from dolly-15k; a small curated fallback if it can't load."""
    try:
        from datasets import load_dataset

        ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
        texts: list[str] = []
        for i, ex in enumerate(ds):
            if i >= max_docs:
                break
            parts = [ex.get("instruction", "").strip(), ex.get("context", "").strip(), ex.get("response", "").strip()]
            t = "\n".join(p for p in parts if p)
            if t:
                texts.append(t)
        if texts:
            return texts, "databricks/databricks-dolly-15k"
    except Exception as e:  # noqa: BLE001
        print(f"[chat] dataset load failed ({type(e).__name__}); using the curated fallback")
    import random

    seeds = [
        "Hello, how are you today?", "What is the capital of France?", "Can you write a short poem about the sea?",
        "Explain how a rainbow forms.", "What's a good recipe for banana bread?", "Summarize the plot of Romeo and Juliet.",
        "How do I center a div in CSS?", "Why is the sky blue?", "Give me three tips for better sleep.",
        "Translate 'good morning' into Spanish.", "What are the benefits of regular exercise?",
        "Tell me a fun fact about octopuses.", "What time zone is Tokyo in?", "Recommend a book about the ocean.",
    ]
    rng = random.Random(0)
    return [" ".join(rng.sample(seeds, k=rng.randint(2, 6))) for _ in range(2000)], "curated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="artifacts/data/wikitext/tokenizer.json")
    ap.add_argument("--out", default="artifacts/data/chat")
    ap.add_argument("--max-docs", type=int, default=4000)
    args = ap.parse_args()

    texts, source = collect_conversational_docs(args.max_docs)
    tok = Tokenizer.load(args.tokenizer)
    ids: list[int] = []
    for t in texts:
        ids.extend(tok.encode(t))
        ids.append(DOC_SEP)
    arr = np.array(ids[: (len(ids) // 2) * 2], dtype="<u2")
    n = len(arr)
    split = int(n * 0.9)
    os.makedirs(args.out, exist_ok=True)
    arr[:split].tofile(os.path.join(args.out, "train.bin"))
    arr[split:].tofile(os.path.join(args.out, "validation.bin"))
    tok.save(os.path.join(args.out, "tokenizer.json"))
    json.dump(
        {"corpus": "chat", "vocab_size": tok.vocab_size, "source": source,
         "splits": {"train": split, "validation": n - split, "test": 0}, "n_docs": len(texts)},
        open(os.path.join(args.out, "meta.json"), "w"),
    )
    print(f"[chat] {len(texts)} docs from {source} -> {n} tokens ({n - split} validation) at {args.out}")


if __name__ == "__main__":
    main()
