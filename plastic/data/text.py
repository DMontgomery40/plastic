"""Text corpora: download, tokenize to uint16 token files, sample windows.

Primary corpus: ``wikitext-103-raw-v1`` (``Salesforce/wikitext``). Optional:
``HuggingFaceFW/fineweb-edu`` ``sample-10BT`` streamed. Documents are joined
with ``<eos>``; training samples random ``seq_len + 1`` windows from the
train file; evaluation walks the held-out files sequentially.
"""

from __future__ import annotations

import json
import os
import time
from array import array
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, Literal

import numpy as np
import torch
from torch import Tensor

from plastic.tokenizer.bpe import Tokenizer

Corpus = Literal["wikitext", "fineweb"]


@dataclass
class CorpusMeta:
    corpus: str
    vocab_size: int
    splits: dict[str, int] = field(default_factory=dict)
    created_at_unix: int = 0
    tokenizer_lines: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def is_wikitext_title(line: str) -> bool:
    s = line.strip()
    return len(s) > 4 and s.startswith("= ") and s.endswith(" =") and not s.startswith("= =")


def split_wikitext_documents(lines: Iterable[str]) -> Iterator[str]:
    """Group raw wikitext lines into articles (a level-1 heading starts a new one)."""
    current: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        if is_wikitext_title(line):
            if current:
                yield "".join(current)
            current = [line.strip() + "\n"]
        else:
            current.append(line.strip() + "\n")
    if current:
        yield "".join(current)


def encode_documents_to_bin(
    tok: Tokenizer,
    docs: Iterable[str],
    path: str,
    *,
    max_tokens: int | None = None,
    batch_docs: int = 256,
) -> int:
    """Encode documents, separated by ``<eos>``, to a little-endian uint16 file. Returns token count."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    total = 0
    buf = array("H")
    pending: list[str] = []

    def _flush_pending(f) -> bool:
        nonlocal total, buf
        for ids in tok.encode_batch(pending):
            ids.append(tok.eos_id)
            if max_tokens is not None and total + len(ids) > max_tokens:
                ids = ids[: max(0, max_tokens - total)]
            buf.extend(ids)
            total += len(ids)
            if max_tokens is not None and total >= max_tokens:
                pending.clear()
                _write(f)
                return True
        pending.clear()
        if len(buf) > 8_000_000:
            _write(f)
        return False

    def _write(f) -> None:
        nonlocal buf
        if buf:
            if buf.itemsize != 2:
                raise RuntimeError("array('H') is not 16-bit on this platform")
            f.write(buf.tobytes() if _little_endian() else buf.byteswap().tobytes())
            buf = array("H")

    with open(path, "wb") as f:
        for doc in docs:
            pending.append(doc)
            if len(pending) >= batch_docs:
                if _flush_pending(f):
                    return total
        if pending:
            _flush_pending(f)
        _write(f)
    return total


def _little_endian() -> bool:
    import sys

    return sys.byteorder == "little"


class TokenWindows:
    """Random or sequential ``seq_len + 1`` windows over a uint16 token file."""

    def __init__(self, path: str, seq_len: int) -> None:
        self.path = path
        self.seq_len = int(seq_len)
        self.data = np.memmap(path, dtype="<u2", mode="r")
        if len(self.data) < self.seq_len + 2:
            raise ValueError(f"token file {path} has {len(self.data)} tokens, need > {self.seq_len + 1}")

    def __len__(self) -> int:
        return int(len(self.data))

    def sample(self, batch: int, rng: torch.Generator) -> Tensor:
        max_start = len(self.data) - (self.seq_len + 1)
        starts = torch.randint(0, max_start + 1, (batch,), generator=rng).tolist()
        rows = [np.asarray(self.data[s : s + self.seq_len + 1], dtype=np.int64) for s in starts]
        return torch.from_numpy(np.stack(rows))

    def sequential(self, batch: int) -> Iterator[Tensor]:
        L = self.seq_len
        n_windows = (len(self.data) - 1) // L
        rows: list[np.ndarray] = []
        for w in range(n_windows):
            s = w * L
            rows.append(np.asarray(self.data[s : s + L + 1], dtype=np.int64))
            if len(rows) == batch:
                yield torch.from_numpy(np.stack(rows))
                rows = []
        if rows:
            yield torch.from_numpy(np.stack(rows))


def _wikitext_documents(split: str, cache_dir: str | None) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)
    yield from split_wikitext_documents(row["text"] for row in ds)


def _fineweb_documents(max_docs: int | None, cache_dir: str | None) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True, cache_dir=cache_dir)
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            return
        yield str(row["text"])


def prepare_text_corpus(
    *,
    corpus: Corpus,
    out_dir: str,
    vocab_size: int = 8192,
    tokenizer_lines: int = 200_000,
    max_train_tokens: int | None = None,
    cache_dir: str | None = None,
    log=print,
) -> CorpusMeta:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    if corpus == "wikitext":
        splits = {"train": "train", "validation": "validation", "test": "test"}
        doc_iter = lambda split: _wikitext_documents(splits[split], cache_dir)  # noqa: E731
    elif corpus == "fineweb":
        n_docs = None if max_train_tokens is None else max(1000, max_train_tokens // 300)
        heldout_docs = 2000
        all_docs = list(_fineweb_documents((n_docs or 0) + heldout_docs if n_docs else None, cache_dir))
        train_docs, valid_docs, test_docs = all_docs[: -heldout_docs], all_docs[-heldout_docs:-heldout_docs // 2], all_docs[-heldout_docs // 2 :]
        parts = {"train": train_docs, "validation": valid_docs, "test": test_docs}
        doc_iter = lambda split: iter(parts[split])  # noqa: E731
    else:
        raise ValueError(f"unknown corpus {corpus!r}")

    log(f"[prepare] training tokenizer (vocab {vocab_size}) on {tokenizer_lines} lines of {corpus}")

    def _lines() -> Iterator[str]:
        n = 0
        for doc in doc_iter("train"):
            for line in doc.splitlines():
                if line.strip():
                    yield line
                    n += 1
                    if n >= tokenizer_lines:
                        return

    tok = Tokenizer.train(_lines(), vocab_size=vocab_size)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    log(f"[prepare] tokenizer ready ({time.time() - t0:.0f}s)")

    meta = CorpusMeta(corpus=corpus, vocab_size=tok.vocab_size, created_at_unix=int(time.time()), tokenizer_lines=tokenizer_lines)
    for split in ("train", "validation", "test"):
        path = os.path.join(out_dir, f"{split}.bin")
        limit = max_train_tokens if split == "train" else None
        n = encode_documents_to_bin(tok, doc_iter(split), path, max_tokens=limit)
        meta.splits[split] = n
        log(f"[prepare] {split}: {n:,} tokens -> {path} ({time.time() - t0:.0f}s)")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        f.write(meta.to_json())
    return meta


def load_corpus_meta(data_dir: str) -> CorpusMeta:
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    return CorpusMeta(**d)
