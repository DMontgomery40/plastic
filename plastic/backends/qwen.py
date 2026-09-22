"""Qwen3.5 as a pretrained text backend for the transaction harness.

Qwen3.5-0.8B is a post-trained assistant whose layers are mostly Gated DeltaNet — the same
recurrent-memory family as plastic's fast-weight memory — plus a few full-attention layers. That
makes it a natural fit for the harness: its recurrent state can be snapshotted, committed, rolled
back, frozen, and probed, exactly the operations the harness already performs on the native memory.

This module is the foundation: load the official checkpoint unchanged, run it chunk-by-chunk
through its native cache, and establish the two invariants any harness integration rests on —
**native-logit parity** (chunked-with-cache logits equal a single full-sequence pass, so the
harness never changes what an undisturbed model would output) and **snapshot/replay identity**
(a deep-copied cache reproduces continuation logits exactly and the original is not mutated).

Transformers (>= 5.17) is imported lazily; nothing here runs unless a QwenBackend is constructed.
The recurrent-state grad adapter (for canary gradients) is applied per cache instance via an
instance-bound method override — never a process-global monkeypatch — so a concurrent server's
sessions stay isolated.
"""

from __future__ import annotations

import copy
import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class QwenState:
    """A harness-facing wrapper over Qwen's native cache (recurrent + KV) at a position."""

    cache: Any  # transformers Cache (past_key_values)

    def clone(self) -> "QwenState":
        # deep copy is the verified snapshot (≈3ms at short prefix); it copies both the
        # gated-delta recurrent_states and the attention KV without aliasing the original.
        return QwenState(copy.deepcopy(self.cache))

    @property
    def position(self) -> int:
        return int(self.cache.get_seq_length()) if self.cache is not None else 0

    def recurrent_leaves(self) -> list[torch.Tensor]:
        leaves: list[torch.Tensor] = []
        for layer in getattr(self.cache, "layers", []):
            rs = getattr(layer, "recurrent_states", None)
            if rs:
                leaves.extend(rs.values())
        return leaves


class QwenBackend:
    """Loads the official Qwen3.5 checkpoint unchanged and drives it through its native cache."""

    def __init__(self, model, tokenizer, config) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.vocab_size = int(config.vocab_size)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, checkpoint_dir: str, *, dtype: torch.dtype = torch.float32, threads: int | None = 4) -> "QwenBackend":
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig  # lazy
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        if threads:
            torch.set_num_threads(int(threads))
        p = Path(checkpoint_dir)
        cfg = Qwen3_5TextConfig.from_dict(json.loads((p / "config.json").read_text())["text_config"])
        model, info = Qwen3_5ForCausalLM.from_pretrained(
            p, config=cfg, dtype=dtype, attn_implementation="eager",
            key_mapping={"^model.language_model.": "model."}, output_loading_info=True, local_files_only=True,
        )
        if any(info.values()):
            raise RuntimeError(f"Qwen load had non-empty diagnostics: {info}")
        model.eval()
        model.requires_grad_(False)
        tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
        return cls(model, tok, cfg)

    # ------------------------------------------------------------------ tokenization
    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()

    def encode_chat(self, user_message: str) -> list[int]:
        msgs = [{"role": "user", "content": user_message}]
        ids = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        return list(ids)

    def init_state(self) -> QwenState:
        return QwenState(cache=None)

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def process(self, ids: list[int], state: QwenState, *, freeze: bool = False) -> tuple[torch.Tensor, QwenState]:
        """Run one chunk of tokens through the model, advancing ``state``'s cache in place.

        ``freeze`` implements the harness's no-write semantics for the recurrent memory: the
        gated-delta ``recurrent_states`` are restored to their pre-chunk values afterward (memory
        unchanged), while the KV cache and position advance (activation state progresses). This
        matches "freeze disables both write and decay while the activation state advances".
        """
        x = torch.tensor([ids], dtype=torch.long)
        pre = _recurrent_snapshot(state.cache) if freeze else None
        out = self.model(x, past_key_values=state.cache, use_cache=True)
        state.cache = out.past_key_values
        if freeze and pre is not None:
            _recurrent_restore(state.cache, pre)
        return out.logits[0], state

    @torch.no_grad()
    def logits_full(self, ids: list[int]) -> torch.Tensor:
        """A single full-sequence pass (no incremental cache) — the parity reference."""
        x = torch.tensor([ids], dtype=torch.long)
        return self.model(x, use_cache=False).logits[0]

    # ------------------------------------------------------------------ canary gradient
    def recurrent_grad(self, state: QwenState, probe_ids: list[int], target_ids: list[int]) -> list[torch.Tensor]:
        """Gradient of a probe loss w.r.t. the recurrent state leaves, on a disposable cache copy.

        Uses ASTRA-046's grad-safe adapter: on the copy, the recurrent states become requires-grad
        leaves and the layer's ``update_recurrent_state`` is overridden (per instance, not globally)
        to *replace* the tensor instead of copying in place, so autograd reaches the leaves.
        """
        probe = copy.deepcopy(state.cache)
        leaves: list[torch.Tensor] = []
        for layer in getattr(probe, "layers", []):
            rs = getattr(layer, "recurrent_states", None)
            if not rs:
                continue
            layer.recurrent_states = {i: s.detach().clone().requires_grad_(True) for i, s in rs.items()}
            leaves.extend(layer.recurrent_states.values())

            def _update(self, s, state_idx=0, **kwargs):  # noqa: ANN001
                self.recurrent_states[state_idx] = s
                return s

            layer.update_recurrent_state = types.MethodType(_update, layer)
        x = torch.tensor([probe_ids], dtype=torch.long)
        y = self.model(x, past_key_values=probe, use_cache=True)
        loss = torch.nn.functional.cross_entropy(
            y.logits.reshape(-1, self.vocab_size), torch.tensor(target_ids, dtype=torch.long)
        )
        return list(torch.autograd.grad(loss, leaves))


def _recurrent_snapshot(cache) -> list[dict[int, torch.Tensor]] | None:
    if cache is None:
        return None
    return [
        {i: s.clone() for i, s in getattr(layer, "recurrent_states", {}).items()}
        for layer in getattr(cache, "layers", [])
    ]


def _recurrent_restore(cache, snap: list[dict[int, torch.Tensor]]) -> None:
    for layer, saved in zip(getattr(cache, "layers", []), snap):
        if getattr(layer, "recurrent_states", None):
            for i, s in saved.items():
                layer.recurrent_states[i] = s
