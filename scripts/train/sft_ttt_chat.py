"""Supervised chat fine-tuning of a TTT-MLP / TTT-Linear base checkpoint (Sun et al. 2024), full
parameter, so the playground's chat model IS the test-time-training architecture.

The rendering is the backend's (``plastic.backends.ttt_lm.backend.CHAT_USER/CHAT_ASSISTANT``), one user
turn then the assistant answer closed by EOS, so runtime prompts and training examples match exactly.
Multi-turn conversations are rendered as consecutive turns; loss is on assistant tokens only.

Data: SmolTalk subsets (HuggingFaceTB/smoltalk; the new subsets are Apache-2.0, the others carry
their source licenses), streamed and packed into fixed-length sequences. Recorded with the run: subset
names, dataset revision, row counts, token counts, seed, and every hyperparameter.

Usage (GPU):
  python -m scripts.train.sft_ttt_chat --checkpoint <dir> --out <dir> --subsets everyday-conversations,smol-magpie-ultra
      --seq-len 2048 --batch 8 --steps 2000 --lr 2e-5 --device cuda
A ``--measure N`` run does N optimizer steps and prints tokens/s so the budget is measured, not guessed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Any, Iterator

DEFAULT_SUBSETS = "everyday-conversations,smol-magpie-ultra,openhermes-100k,systemchats-30k,smol-constraints,smol-rewrite,smol-summarize"


def render_conversation(messages: list[dict[str, str]], user_tag: str, assistant_tag: str) -> list[tuple[str, bool]]:
    """(text, is_assistant) segments. A system message is folded into the first user turn."""
    system = ""
    out: list[tuple[str, bool]] = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            system = content.strip() + "\n\n"
            continue
        if role == "user":
            out.append((user_tag + system + content, False))
            system = ""
        elif role == "assistant":
            out.append((assistant_tag, False))
            out.append((content, True))
    return out


def encode_example(tok, messages: list[dict[str, str]], user_tag: str, assistant_tag: str) -> tuple[list[int], list[int]]:
    """Token ids and labels (-100 where the loss is masked). BOS first; EOS after every assistant answer."""
    ids: list[int] = [int(tok.bos_token_id)] if tok.bos_token_id is not None else []
    labels: list[int] = [-100] * len(ids)
    for text, is_assistant in render_conversation(messages, user_tag, assistant_tag):
        t = [int(x) for x in tok(text, add_special_tokens=False).input_ids]
        ids += t
        labels += t if is_assistant else [-100] * len(t)
        if is_assistant:
            ids.append(int(tok.eos_token_id))
            labels.append(int(tok.eos_token_id))
    return ids, labels


def pack(stream: Iterator[tuple[list[int], list[int]]], seq_len: int, pad_id: int) -> Iterator[tuple[list[int], list[int]]]:
    """Greedy packing of whole examples into ``seq_len`` windows; an example longer than the window is
    truncated. Pads the tail with pad_id and -100 labels."""
    buf_ids: list[int] = []
    buf_lab: list[int] = []
    for ids, labels in stream:
        ids, labels = ids[:seq_len], labels[:seq_len]
        if len(buf_ids) + len(ids) > seq_len:
            n = seq_len - len(buf_ids)
            yield buf_ids + [pad_id] * n, buf_lab + [-100] * n
            buf_ids, buf_lab = [], []
        buf_ids += ids
        buf_lab += labels
    if buf_ids:
        n = seq_len - len(buf_ids)
        yield buf_ids + [pad_id] * n, buf_lab + [-100] * n


def load_examples(subsets: list[str], seed: int, max_rows: int | None) -> tuple[list[list[dict[str, str]]], dict[str, Any]]:
    from datasets import load_dataset

    rows: list[list[dict[str, str]]] = []
    counts: dict[str, int] = {}
    for name in subsets:
        ds = load_dataset("HuggingFaceTB/smoltalk", name, split="train")
        msgs = [r["messages"] for r in ds]
        counts[name] = len(msgs)
        rows += msgs
    random.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    return rows, {"dataset": "HuggingFaceTB/smoltalk", "subsets": subsets, "rows_per_subset": counts, "rows_used": len(rows), "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsets", default=DEFAULT_SUBSETS)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "bf16-weights"],
                    help="bf16 = fp32 master weights with bf16 autocast (default); bf16-weights = pure bf16 (NaN-prone in the inner loop); fp32 = no autocast")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--measure", type=int, default=0, help="only time N optimizer steps and exit")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--grad-checkpoint-groups", type=int, default=0, help="scan checkpoint groups per layer (0 = off)")
    ap.add_argument("--layer-checkpoint", action="store_true", help="checkpoint every decoder layer (standard activation checkpointing)")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from plastic.backends.ttt_lm import modeling_ttt as M
    from plastic.backends.ttt_lm.backend import CHAT_ASSISTANT, CHAT_USER, _checkpoint_digest

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16-weights" else torch.float32
    autocast = args.dtype == "bf16"
    raw = json.loads(open(os.path.join(args.checkpoint, "config.json")).read())
    cfg = M.TTTConfig(**{k: v for k, v in raw.items() if k not in ("architectures", "auto_map", "transformers_version", "dtype", "model_type")})
    cfg.scan_checkpoint_group_size = int(args.grad_checkpoint_groups)
    model = M.TTTForCausalLM.from_pretrained(args.checkpoint, config=cfg, dtype=dtype).to(dev)
    if args.layer_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)

    subsets = [s for s in args.subsets.split(",") if s]
    if args.measure:
        rows, data_meta = [[{"role": "user", "content": "Say something about the number %d." % i},
                            {"role": "assistant", "content": "Here is a sentence about %d, written for a timing run only. " % i * 12}] for i in range(4096)], {"synthetic": True}
    else:
        rows, data_meta = load_examples(subsets, args.seed, args.max_rows)
    encoded = (encode_example(tok, m, CHAT_USER, CHAT_ASSISTANT) for m in rows)
    packed = list(pack(encoded, args.seq_len, pad_id))
    n_tokens = sum(sum(1 for l in lab if l != -100) for _, lab in packed)
    print(f"[sft] {len(rows)} conversations -> {len(packed)} packed sequences of {args.seq_len}; {n_tokens} supervised tokens", flush=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 and "embed" not in n else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    total = args.measure or args.steps

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    os.makedirs(args.out, exist_ok=True)
    meta = {"checkpoint": os.path.abspath(args.checkpoint), "checkpoint_digest": _checkpoint_digest(args.checkpoint),
            "data": data_meta, "packed_sequences": len(packed), "supervised_tokens": n_tokens, "args": vars(args),
            "render": {"user": CHAT_USER, "assistant": CHAT_ASSISTANT, "eos": int(tok.eos_token_id)}, "log": []}
    order = list(range(len(packed)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    t0 = time.time()
    seen = 0
    step = 0
    i = 0
    while step < total:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            batch = [packed[order[(i + k) % len(order)]] for k in range(args.batch)]
            i += args.batch
            x = torch.tensor([b[0] for b in batch], dtype=torch.long, device=dev)
            y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=autocast):
                out = model(x, use_cache=False)
            logits = out.logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}; dtype mode {args.dtype}")
            (loss / args.grad_accum).backward()
            loss_acc += float(loss) / args.grad_accum
            seen += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total or args.measure:
            el = time.time() - t0
            rec = {"step": step, "loss": round(loss_acc, 4), "lr": lr_at(step - 1), "tokens_seen": seen, "seconds": round(el, 1), "tok_per_s": round(seen / el, 1)}
            if dev.type == "cuda":
                rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
            meta["log"].append(rec)
            print("[sft]", json.dumps(rec), flush=True)
        if not args.measure and (step % args.save_every == 0 or step == total):
            model.save_pretrained(args.out, safe_serialization=True)
            tok.save_pretrained(args.out)
            with open(os.path.join(args.out, "sft_meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
    if args.measure:
        with open(os.path.join(args.out, "measure.json"), "w") as f:
            json.dump(meta, f, indent=1)
    print(f"[sft] done: {step} steps, {seen} tokens, {round(time.time() - t0)} s", flush=True)


if __name__ == "__main__":
    main()
