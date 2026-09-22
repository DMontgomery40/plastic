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
import hashlib
import json
import threading
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Bumped when the persisted cache layout changes, so a session saved under an older layout is
# rejected on load rather than silently mis-reconstructed.
_CACHE_SCHEMA = "qwen-cache-v1"
# Checkpoint files that define the model, tokenizer and chat template — hashed for the session
# identity so a session saved under a different checkpoint/tokenizer/template is refused on load.
_IDENTITY_FILES = (
    "config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "model.safetensors.index.json",
)

# A genuine freeze zeroes the gated-delta kernel's write (beta) and log-decay (g) so the recurrent
# memory stays exactly constant while conv/attention/position advance — matching plastic's
# "no write, no decay, activation progresses". It is toggled by a THREAD-LOCAL flag and the kernel
# wrappers are installed once per process (idempotent), so concurrent sessions never race on a
# module-global monkeypatch (each thread's freeze flag is its own).
_tls = threading.local()
_KERNELS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")


def _install_frozen_kernels() -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native

    if getattr(native, "_plastic_kernels_installed", False):
        return
    for name in _KERNELS:
        original = getattr(native, name)

        def make(orig):
            def wrapped(query, key, value, g, beta, **kwargs):  # noqa: ANN001
                # freeze wins: beta=0 AND g=0 is a genuine no-write, no-decay. Otherwise a beta
                # scale (leaving g/decay untouched) implements the harness scale-control contract —
                # beta_scale=0 writes nothing while decay still runs, distinct from freeze.
                if getattr(_tls, "freeze", False):
                    g = torch.zeros_like(g)
                    beta = torch.zeros_like(beta)
                else:
                    scale = getattr(_tls, "beta_scale", 1.0)
                    if scale != 1.0:
                        beta = beta * scale
                return orig(query, key, value, g=g, beta=beta, **kwargs)

            return wrapped

        setattr(native, name, make(original))
    native._plastic_kernels_installed = True


@contextmanager
def _frozen():
    prev = getattr(_tls, "freeze", False)
    _tls.freeze = True
    try:
        yield
    finally:
        _tls.freeze = prev


@contextmanager
def _scaled(scale: float):
    prev = getattr(_tls, "beta_scale", 1.0)
    _tls.beta_scale = scale
    try:
        yield
    finally:
        _tls.beta_scale = prev


def _checkpoint_digest(checkpoint_dir: str) -> str:
    """A content digest of the checkpoint's defining files (config, tokenizer, chat template, and the
    weight files) — computed ONCE at load, never per state operation. It binds a session's identity
    to the actual model/tokenizer content, so loading a session saved under a different checkpoint,
    a changed tokenizer, or an edited chat template is refused (ASTRA-066). The weight files are
    included so same-shaped different weights are distinguished; this reads them once (~seconds)."""
    p = Path(checkpoint_dir)
    weights = sorted(f.name for f in p.glob("*.safetensors"))
    h = hashlib.sha256()
    for name in list(_IDENTITY_FILES) + weights:
        f = p / name
        if not f.exists():
            h.update(f"MISSING:{name}\n".encode())
            continue
        fh = hashlib.sha256()
        with open(f, "rb") as fp:
            for block in iter(lambda: fp.read(1 << 20), b""):
                fh.update(block)
        h.update(f"{name}:{f.stat().st_size}:{fh.hexdigest()}\n".encode())
    return h.hexdigest()


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

    def __init__(self, model, tokenizer, config, *, device: torch.device, checkpoint_digest: str | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.vocab_size = int(config.vocab_size)
        self.checkpoint_digest = checkpoint_digest  # content digest for the session-identity check
        self._schema_cache: list[tuple] | None = None
        # Overlapping GPU work aborts the MPS runtime (Metal command-buffer assertion, exit 134,
        # ASTRA-053), and the thread-local freeze flag does not make concurrent forwards safe. All
        # model/cache/probe execution on this shared backend is serialized through one lock, with
        # an MPS sync before release; a per-session lock would not suffice (sessions share the
        # backend). CPU is unaffected but takes the same cheap uncontended path.
        self._lock = threading.Lock()
        _install_frozen_kernels()

    @contextmanager
    def _serialized(self):
        with self._lock:
            try:
                yield
            finally:
                if self.device.type == "mps":
                    torch.mps.synchronize()

    # ------------------------------------------------------------------ identity / signals
    def signal_names(self) -> tuple[str, ...]:
        """The reduced decision-signal set a Qwen session can actually produce: the chunk NLL and
        the recurrent-state change. The gated-delta kernel exposes neither the memory's own
        prediction error (``surprise_mean``) nor its write norm (``log_write_norm``) without
        per-session instrumentation, and there is no Fisher estimate on Qwen's state yet, so the
        runner carries those STAT_SIGNALS as ``None`` end-to-end rather than substituting zero or
        NaN. (Spec §2.)"""
        return ("chunk_loss", "log_delta_norm")

    def writes_for_source(self, source: str) -> bool:
        """Qwen's recurrent state *is* its language context, so generated tokens write to it and
        those writes persist and must be accounted — both ``user`` and ``model`` propose a write
        (the opposite of plastic, which freezes generation read-only). This is only the default
        eligibility; an explicit session read-only, a spent budget, or a requested freeze still take
        precedence in the runner — ``source`` alone cannot escape them. (Spec §3.)"""
        return True

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
        return cls(model, tok, cfg, device=dev, checkpoint_digest=_checkpoint_digest(checkpoint_dir))

    # ------------------------------------------------------------------ tokenization
    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()

    def encode_chat(self, user_message: str) -> list[int]:
        # apply_chat_template defaults to a BatchEncoding in Transformers 5.17, so list(...) would
        # yield the dict keys ("input_ids", ...) — return_dict=False gives native integer IDs
        # (equivalent to rendering the template then tokenizing). (ASTRA-054.)
        msgs = [{"role": "user", "content": user_message}]
        ids = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False
        )
        return [int(t) for t in ids]

    def init_state(self) -> QwenState:
        """A position-zero session state: an initialized cache at cursor 0 with zero recurrent
        memory and zero-history (left-padded) conv buffers — no consumed token, no KV. Forwarding
        from it is identical to an empty cache (verified), but it exposes correctly-shaped zero
        recurrent leaves so ``score_suite`` and ``canary_gradient`` are defined on the FIRST chunk,
        before any update — the harness must not skip its first projection or add a synthetic BOS
        (the tokenizer has none). Recipe verified in ASTRA-055's zero-initial-state probe.
        """
        from transformers import DynamicCache

        c = self.config
        with self._serialized():
            cache = DynamicCache(config=c)
            for layer in cache.layers:
                if not hasattr(layer, "recurrent_states"):
                    continue
                conv = torch.zeros(
                    1,
                    2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim,
                    c.linear_conv_kernel_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                memory = torch.zeros(
                    # the GDN kernels compute and return the recurrent state in float32 and
                    # DynamicCache preserves it there even when weights/conv/KV are bfloat16;
                    # allocating this at the model dtype would quantize it every update and diverge
                    # after the first chunk (ASTRA-057). Conv/KV stay at the model dtype.
                    1, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim,
                    device=self.device, dtype=torch.float32,
                )
                layer.lazy_initialization(conv_states=conv, recurrent_states=memory)
                layer.has_previous_state[0] = True
        return QwenState(cache)

    def position(self, state: QwenState) -> int:
        return state.position

    def clone(self, state: QwenState) -> QwenState:
        """Concurrency-safe snapshot: the cache deep copy runs under the backend lock (with an MPS
        sync), so it never races a forward on the shared device. The harness clones through here;
        QwenState.clone() is the equivalent for single-threaded use."""
        with self._serialized():
            return QwenState(copy.deepcopy(state.cache))

    def state_delta(self, a: QwenState, b: QwenState) -> list[torch.Tensor]:
        """Per-memory-unit change ``a − b`` over the 18 gated-delta recurrent tensors — the memory
        units the harness measures (update norm) and projects. KV and conv are activation/attention
        history, not memory units, so they are not part of the delta (matching plastic, whose delta
        is per-layer ``S`` only). The leaves are in a stable layer order on both states."""
        return [la - lb for la, lb in zip(a.recurrent_leaves(), b.recurrent_leaves())]

    def is_finite(self, state: QwenState) -> bool:
        """Every persisted tensor a commit or restore would carry must be finite: the recurrent
        memory and conv history on the linear-attention layers, and the attention KV on the
        full-attention layers. Missing/empty fields (e.g. KV before any token) are skipped."""
        # A genuinely uninitialized cache carries {0: None} slots (Transformers cache_utils), so guard
        # None — is_finite is a finiteness check, not a structure check, and must not raise on an
        # uninitialized state. Whether a *persisted* state is missing an expected initialized unit is
        # validated in load_state_dict, which rejects it rather than skipping it silently (ASTRA-063).
        for layer in getattr(state.cache, "layers", []):
            tensors: list[torch.Tensor] = []
            for attr in ("recurrent_states", "conv_states"):
                d = getattr(layer, attr, None)
                if d:
                    tensors.extend(v for v in d.values() if v is not None)
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if t is not None:
                    tensors.append(t)
            for t in tensors:
                if not bool(torch.isfinite(t).all()):
                    return False
        return True

    # ------------------------------------------------------------------ persistence
    def _identity(self) -> dict[str, Any]:
        """The compatibility signature stamped into a saved session and checked on load. It binds to
        the actual checkpoint/tokenizer/chat-template CONTENT (``checkpoint_digest``, hashed once at
        load) — not just dimensions — so same-shaped different weights or an edited chat template are
        refused, plus the Transformers runtime and the cache-layout schema so a format change is
        rejected rather than mis-reconstructed (ASTRA-066). The dimensions stay as a fast secondary
        signal. A backend built without a digest (never via ``load``) reports it as ``None``."""
        import transformers

        c = self.config
        return {
            "backend": "qwen",
            "checkpoint_digest": self.checkpoint_digest,
            "cache_schema": _CACHE_SCHEMA,
            "transformers_version": str(transformers.__version__),
            "vocab_size": int(c.vocab_size),
            "num_hidden_layers": int(getattr(c, "num_hidden_layers", 0)),
            "hidden_size": int(getattr(c, "hidden_size", 0)),
            "tokenizer_vocab": int(getattr(self.tokenizer, "vocab_size", 0)),
        }

    def _schema(self) -> list[tuple]:
        """The expected per-layer cache structure of THIS backend, derived once from a fresh
        ``init_state`` and cached: an ordered list of ``("linear", recurrent_shape, recurrent_dtype,
        conv_shape)`` and ``("dynamic",)`` descriptors. Loads are validated against this trusted
        reference — never against payload-supplied counts, which an attacker controls (ASTRA-066)."""
        if self._schema_cache is None:
            st = self.init_state()
            schema: list[tuple] = []
            for layer in st.cache.layers:
                rs = getattr(layer, "recurrent_states", None)
                if rs is not None and 0 in rs and rs[0] is not None:
                    r = rs[0]
                    c = getattr(layer, "conv_states", {}).get(0)
                    schema.append(("linear", tuple(r.shape), r.dtype, None if c is None else tuple(c.shape)))
                else:
                    schema.append(("dynamic",))
            self._schema_cache = schema
        return self._schema_cache

    def _validate_cache(self, cache) -> None:
        """Refuse a malformed persisted cache clearly, rather than letting a bad payload fail on the
        next forward (ASTRA-066). Checks the layer count and kinds against the trusted schema; each
        linear layer's recurrent unit is present, correctly shaped and typed, with a present conv;
        each dynamic layer carries no recurrent unit and consistent-length K/V; a position-zero cache
        has all-zero recurrent memory; and every stored tensor is finite."""
        schema = self._schema()
        layers = getattr(cache, "layers", None)
        if layers is None or len(layers) != len(schema):
            raise ValueError(f"Qwen state layer count {None if layers is None else len(layers)} != expected {len(schema)}")
        kv_lengths: list[int] = []
        for idx, (layer, spec) in enumerate(zip(layers, schema)):
            if spec[0] == "linear":
                rs = getattr(layer, "recurrent_states", None)
                if not rs or rs.get(0) is None:
                    raise ValueError(f"layer {idx}: missing an initialized recurrent unit")
                r = rs[0]
                if tuple(r.shape) != spec[1] or r.dtype != spec[2]:
                    raise ValueError(f"layer {idx}: recurrent shape/dtype {tuple(r.shape)}/{r.dtype} != expected {spec[1]}/{spec[2]}")
                cs = getattr(layer, "conv_states", None)
                if not cs or cs.get(0) is None:
                    raise ValueError(f"layer {idx}: missing conv state")
                if spec[3] is not None and tuple(cs[0].shape) != spec[3]:
                    raise ValueError(f"layer {idx}: conv shape {tuple(cs[0].shape)} != expected {spec[3]}")
            else:  # dynamic (full-attention) layer
                if getattr(layer, "recurrent_states", None) is not None:
                    raise ValueError(f"layer {idx}: expected a full-attention layer, found a recurrent one")
                k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
                if (k is None) != (v is None):
                    raise ValueError(f"layer {idx}: inconsistent attention K/V (one present, one absent)")
                if k is not None:
                    if k.shape[-2] != v.shape[-2]:
                        raise ValueError(f"layer {idx}: K/V length mismatch {k.shape[-2]} != {v.shape[-2]}")
                    kv_lengths.append(int(k.shape[-2]))
        if len(set(kv_lengths)) > 1:
            raise ValueError(f"inconsistent attention KV lengths across layers: {sorted(set(kv_lengths))}")
        # a cache with no KV is position-zero; its recurrent memory must be exactly zero (a warm state
        # with KV stripped would otherwise masquerade as position-zero)
        if not kv_lengths:
            for idx, layer in enumerate(layers):
                rs = getattr(layer, "recurrent_states", None)
                if rs and rs.get(0) is not None and int(torch.count_nonzero(rs[0])) != 0:
                    raise ValueError(f"layer {idx}: position-zero cache (no KV) with non-zero recurrent memory")
        if not self.is_finite(QwenState(cache)):
            raise ValueError("persisted Qwen state contains non-finite values")

    @staticmethod
    def _map_cache_tensors(cache, fn) -> None:
        """Apply ``fn`` in place to every stored tensor of ``cache`` (recurrent, conv, and attention
        KV), across both layer kinds — used to move a serialized cache CPU->device and back without
        reconstructing it (reconstruction via lazy_initialization loses the stored values)."""
        for layer in getattr(cache, "layers", []):
            for attr in ("recurrent_states", "conv_states"):
                d = getattr(layer, attr, None)
                if d:
                    for i, v in list(d.items()):
                        if v is not None:
                            d[i] = fn(v)
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if t is not None:
                    setattr(layer, attr, fn(t))

    def state_dict(self, state: QwenState) -> dict[str, Any]:
        """Full serialization: the complete native cache (recurrent memory, conv history, attention
        KV, positions and init flags) via a deep copy — the same faithful copy ``clone`` uses, so a
        load reproduces continuation logits exactly — with tensors moved to CPU for portable storage,
        plus the backend/checkpoint/tokenizer identity and a structural count for the load check."""
        with self._serialized():
            snap = copy.deepcopy(state.cache)
        self._map_cache_tensors(snap, lambda t: t.detach().cpu())
        n_recurrent = sum(
            1 for layer in getattr(snap, "layers", []) for v in (getattr(layer, "recurrent_states", None) or {}).values() if v is not None
        )
        return {
            "backend": "qwen",
            "identity": self._identity(),
            "num_layers": len(getattr(snap, "layers", [])),
            "recurrent_leaves": n_recurrent,
            "cache": snap,
        }

    def load_state_dict(self, data: dict[str, Any]) -> QwenState:
        """Reconstruct a session state, refusing an incompatible or malformed persisted state rather
        than loading it silently. The identity (checkpoint/tokenizer/template content + runtime +
        schema) must match this backend; the cache is validated against the trusted structural schema
        and for finiteness; and the validated cache is **deep-copied** before device mapping so the
        returned live state never aliases or mutates the saved payload — repeated loads and the fork
        helper (which reuses the committed payload for working) produce independent states (ASTRA-066)."""
        if not isinstance(data, dict) or data.get("backend") != "qwen":
            raise ValueError("not a Qwen backend state")
        if data.get("identity") != self._identity():
            raise ValueError(f"incompatible Qwen session: saved identity {data.get('identity')} != current {self._identity()}")
        cache = data.get("cache")
        if cache is None:
            raise ValueError("Qwen state has no cache")
        self._validate_cache(cache)
        with self._serialized():
            cache = copy.deepcopy(cache)  # independent live state; never alias/mutate the saved payload
            self._map_cache_tensors(cache, lambda t: t.to(self.device))
        return QwenState(cache)

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
        with self._serialized():
            if freeze:
                with _frozen():
                    out = self.model(x, past_key_values=state.cache, use_cache=True)
            else:
                out = self.model(x, past_key_values=state.cache, use_cache=True)
        state.cache = out.past_key_values
        return out.logits[0], state

    @torch.no_grad()
    def forward(self, chunk: list[int], state: QwenState, *, freeze: bool, beta_scale: float) -> tuple[torch.Tensor, QwenState, list[Any]]:
        """The Backend forward: advance ``chunk`` through the model, returning (logits, new_state,
        per-token signals). ``freeze`` is a genuine no-write (β=0 and g=0). ``beta_scale`` scales the
        write only (leaving decay g) via the per-call kernel wrapper; ``beta_scale=0`` writes nothing
        while decay still runs, distinct from ``freeze``. The signal list is **empty**: Qwen's kernel
        exposes no per-token memory signals (surprise/write-norm), so the runner carries those as
        ``None`` — the available signals (chunk NLL from the logits, recurrent-state change from
        ``state_delta``) are derived by the runner, not returned here."""
        x = torch.tensor([chunk], dtype=torch.long, device=self.device)
        with self._serialized():
            if freeze:
                with _frozen():
                    out = self.model(x, past_key_values=state.cache, use_cache=True)
            elif beta_scale != 1.0:
                with _scaled(beta_scale):
                    out = self.model(x, past_key_values=state.cache, use_cache=True)
            else:
                out = self.model(x, past_key_values=state.cache, use_cache=True)
        state.cache = out.past_key_values
        return out.logits[0], state, []

    @torch.no_grad()
    def logits_full(self, ids: list[int]) -> torch.Tensor:
        """A single full-sequence pass (no incremental cache) — the parity reference."""
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self._serialized():
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
        x = torch.tensor([probe_ids], dtype=torch.long, device=self.device)
        # the ENTIRE operation — including the cache deep copy — must be serialized: deep-copying
        # MPS cache tensors concurrently with a forward also aborts the runtime (ASTRA review of the
        # forward-only lock), so cache preparation goes inside the lock too.
        with self._serialized():
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
            with _frozen():
                y = self.model(x, past_key_values=probe, use_cache=True)
            loss = torch.nn.functional.cross_entropy(
                y.logits.reshape(-1, self.vocab_size), torch.tensor(target_ids, dtype=torch.long, device=self.device)
            )
            return list(torch.autograd.grad(loss, leaves))

    # ------------------------------------------------------------------ canaries (frozen)
    def _probe_nll(self, ids: list[int], state: QwenState) -> float:
        """Frozen continuation NLL of a probe from ``state`` — the probe is read, never learned
        (β=0, g=0), and it runs on a fresh clone so the measured session is untouched."""
        logits, _ = self.process(ids, self.clone(state), freeze=True)  # (len, vocab)
        tgt = torch.tensor(ids[1:], dtype=torch.long, device=self.device)
        return float(torch.nn.functional.cross_entropy(logits[:-1], tgt))

    def score_suite(self, state: QwenState, suite: Any) -> dict[str, float]:
        """Mean coherence and poison probe NLL from read-only clones of ``state`` (freeze=True), so
        the session state is never mutated — the same contract as plastic's ``score_suite``. An
        empty probe set, or one with no scorable (length ≥ 2) probe, scores ``nan``."""
        out: dict[str, float] = {}
        for name in ("coherence", "poison"):
            probes = getattr(suite, name, None) or []
            losses = [self._probe_nll([int(t) for t in p], state) for p in probes if len(p) >= 2]
            out[name] = float(sum(losses) / len(losses)) if losses else float("nan")
        return out

    def canary_gradient(self, state: Any, suite: Any) -> list[torch.Tensor]:
        """∂(coherence score)/∂(recurrent leaves) at ``state``, frozen — the projection direction.

        This must differentiate the SAME scalar ``score_suite`` reports: the mean per-probe NLL over
        the scorable coherence probes (equal weight per probe). So the per-probe frozen gradients are
        summed and **divided by the scorable-probe count** — a plain sum would be N× the reported
        score's derivative (ASTRA-064). Empty/one-token probes do not enter the count. Returns 18
        correctly-shaped zero tensors when there is no scorable coherence probe, so the harness never
        skips its first projection. Qwen weights each probe equally here and in ``score_suite``; this
        differs from plastic's per-token weighting on ragged probes — the chosen Qwen objective."""
        leaves0 = state.recurrent_leaves()
        total: list[torch.Tensor] | None = None
        n = 0
        for p in getattr(suite, "coherence", None) or []:
            ids = [int(t) for t in p]
            if len(ids) < 2:
                continue
            g = self.recurrent_grad(state, ids[:-1], ids[1:])
            total = g if total is None else [a + b for a, b in zip(total, g)]
            n += 1
        if total is None:
            return [torch.zeros_like(t) for t in leaves0]
        return [(t / n).detach() for t in total]

    @torch.no_grad()
    def apply_projected(self, working: QwenState, committed: QwenState, projected: list[torch.Tensor]) -> None:
        """Write a corrected per-unit delta onto ``working``'s recurrent memory for the project
        decision: each of the 18 gated-delta leaves becomes ``committed + projected[i]`` — an
        absolute overwrite from ``committed`` (matching plastic's ``layer.S = committed.S + d``, so a
        second apply with a rescaled delta never compounds onto an already-scaled state), replacing
        the tensor rather than writing in place so ``committed`` is never aliased. Conv, KV and
        position are left untouched. ``projected`` is consumed in ``recurrent_leaves`` order."""
        with self._serialized():
            it = iter(projected)
            for w_layer, c_layer in zip(working.cache.layers, committed.cache.layers):
                w_rs = getattr(w_layer, "recurrent_states", None)
                c_rs = getattr(c_layer, "recurrent_states", None)
                if not w_rs or not c_rs:
                    continue
                for k in w_rs:
                    d = next(it)
                    w_rs[k] = c_rs[k] + d.to(c_rs[k].device, c_rs[k].dtype)


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
