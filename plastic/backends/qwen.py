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
import threading
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# A genuine freeze zeroes the gated-delta kernel's write (beta) and log-decay (g) so the recurrent
# memory stays exactly constant while conv/attention/position advance — matching plastic's
# "no write, no decay, activation progresses". It is toggled by a THREAD-LOCAL flag and the kernel
# wrappers are installed once per process (idempotent), so concurrent sessions never race on a
# module-global monkeypatch (each thread's freeze flag is its own).
_tls = threading.local()
_KERNELS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")


def _install_frozen_kernels() -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native

    if getattr(native, "_plastic_freeze_installed", False):
        return
    for name in _KERNELS:
        original = getattr(native, name)

        def make(orig):
            def wrapped(query, key, value, g, beta, **kwargs):  # noqa: ANN001
                if getattr(_tls, "freeze", False):
                    g = torch.zeros_like(g)
                    beta = torch.zeros_like(beta)
                return orig(query, key, value, g=g, beta=beta, **kwargs)

            return wrapped

        setattr(native, name, make(original))
    native._plastic_freeze_installed = True


@contextmanager
def _frozen():
    prev = getattr(_tls, "freeze", False)
    _tls.freeze = True
    try:
        yield
    finally:
        _tls.freeze = prev


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

    def __init__(self, model, tokenizer, config, *, device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.vocab_size = int(config.vocab_size)
        _install_frozen_kernels()

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls, checkpoint_dir: str, *, device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32, threads: int | None = None,
    ) -> "QwenBackend":
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig  # lazy
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        if threads:  # explicit opt-in only; 1 CPU thread was fastest on the tested Mac, not 4
            torch.set_num_threads(int(threads))
        dev = torch.device(device)
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
        model.to(dev)
        tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
        return cls(model, tok, cfg, device=dev)

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

        ``freeze`` is a genuine no-write: the gated-delta kernel's write (β) and log-decay (g) are
        zeroed for the duration of the forward, so the recurrent memory stays exactly constant
        (each token reads the pre-chunk state; no intra-chunk writes) while the conv, attention KV,
        and position advance — plastic's "no write, no decay, activation progresses". This holds on
        an empty cache too (the kernel simply never writes). It is the real frozen replay the
        harness consumes, not a snapshot-restore of the final tensor.
        """
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        if freeze:
            with _frozen():
                out = self.model(x, past_key_values=state.cache, use_cache=True)
        else:
            out = self.model(x, past_key_values=state.cache, use_cache=True)
        state.cache = out.past_key_values
        return out.logits[0], state

    @torch.no_grad()
    def logits_full(self, ids: list[int]) -> torch.Tensor:
        """A single full-sequence pass (no incremental cache) — the parity reference."""
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        return self.model(x, use_cache=False).logits[0]

    # ------------------------------------------------------------------ canary gradient
    def recurrent_grad(self, state: QwenState, probe_ids: list[int], target_ids: list[int]) -> list[torch.Tensor]:
        """Gradient of a *frozen* probe loss w.r.t. the recurrent state leaves, on a disposable copy.

        The probe is scored with the kernel frozen (β=0, g=0), so it measures sensitivity of the
        canary loss to the committed recurrent state — the frozen-canary objective — rather than
        differentiating an adapting rollout that writes as it reads. ASTRA-046's grad-safe adapter
        makes the recurrent states requires-grad leaves and overrides ``update_recurrent_state``
        per instance (never globally) to replace the tensor instead of copying in place, so
        autograd reaches the leaves even though the frozen kernel would otherwise not update them.
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
        x = torch.tensor([probe_ids], dtype=torch.long, device=self.device)
        with _frozen():
            y = self.model(x, past_key_values=probe, use_cache=True)
        loss = torch.nn.functional.cross_entropy(
            y.logits.reshape(-1, self.vocab_size), torch.tensor(target_ids, dtype=torch.long, device=self.device)
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
