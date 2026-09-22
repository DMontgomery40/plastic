"""Byte-level BPE on top of the ``tokenizers`` library.

Special ids are fixed: ``<pad>=0``, ``<bos>=1``, ``<eos>=2``. The byte-level
pretokenizer preserves whitespace exactly, so ``decode(encode(text)) == text``.
"""

from __future__ import annotations

from typing import Iterable

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")


class Tokenizer:
    def __init__(self, hf: HFTokenizer) -> None:
        self._hf = hf
        self.pad_id = int(hf.token_to_id("<pad>"))
        self.bos_id = int(hf.token_to_id("<bos>"))
        self.eos_id = int(hf.token_to_id("<eos>"))

    @property
    def vocab_size(self) -> int:
        return int(self._hf.get_vocab_size())

    @classmethod
    def train(
        cls,
        lines: Iterable[str],
        *,
        vocab_size: int = 8192,
        max_lines: int | None = None,
    ) -> "Tokenizer":
        hf = HFTokenizer(models.BPE())
        hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        hf.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size),
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )

        def _iter() -> Iterable[str]:
            for i, line in enumerate(lines):
                if max_lines is not None and i >= max_lines:
                    return
                yield line

        hf.train_from_iterator(_iter(), trainer=trainer)
        return cls(hf)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self._hf.encode(text, add_special_tokens=False).ids)
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [list(e.ids) for e in self._hf.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        return self._hf.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    def save(self, path: str) -> None:
        self._hf.save(path)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        return cls(HFTokenizer.from_file(path))
