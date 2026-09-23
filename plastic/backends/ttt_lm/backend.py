"""TTT-MLP / TTT-Linear (Sun et al. 2024) as a plastic ``Backend``.

The fast weights of every layer (``W1, b1, W2, b2`` for the MLP learner) are trained online by the
model's own inner loop: one gradient step of the reconstruction loss per token, mini-batched 16 at a
time, with a learned per-token inner learning rate. That is the mechanism the harness was designed
around, so this backend produces the FULL signal set plastic does: chunk NLL, per-token inner loss
(surprise), effective inner step size (``beta``), exact per-token update norm (``write_norm``), the
committed fast-weight change (``delta_norm``), and frozen canary gradients w.r.t. the fast weights.
There is no decay in this learner, so ``alpha`` is reported as 1 and ``freeze`` and ``beta_scale=0``
coincide: both leave the fast weights (and their pending mini-batch gradients) exactly unchanged
while the short convolutions and the position advance.

Alignment: the reference implementation updates fast weights per 16-token mini-batch and accepts
either aligned blocks or single tokens inside a partial mini-batch. ``forward`` therefore feeds a chunk
as single tokens until the position is aligned, then whole blocks, then a remainder block; the
transaction chunk should be a multiple of 16 so most work runs in the parallel dual form.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from plastic.backends.ttt_lm import modeling_ttt as M
from plastic.model.memory import MemorySignals

_SCHEMA = "ttt-cache-v1"
_IDENTITY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")

# a minimal chat rendering for the SFT'd model; the base tokenizer has no chat template of its own
CHAT_USER, CHAT_ASSISTANT = "<|user|>\n", "\n<|assistant|>\n"


def _checkpoint_digest(checkpoint_dir: str) -> str:
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


class TTTState:
    """Harness-facing wrapper over a ``TTTCache``: fast weights, pending mini-batch gradients, conv
    states and the position."""

    def __init__(self, cache: M.TTTCache) -> None:
        self.cache = cache

    @property
    def position(self) -> int:
        return int(self.cache.seqlen_offset)

    def clone(self) -> "TTTState":
        return TTTState(copy.deepcopy(self.cache))

    def weight_leaves(self) -> list[Tensor]:
        """The fast-weight tensors in a stable (layer, name) order: the memory units the harness measures."""
        c = self.cache
        return [c.ttt_params_dict[f"{n}_states"][l] for l in sorted(c.ttt_params_dict[f"{c.ttt_param_names[0]}_states"])
                for n in c.ttt_param_names]

    def grad_leaves(self) -> list[Tensor]:
        c = self.cache
        return [c.ttt_params_dict[f"{n}_grad"][l] for l in sorted(c.ttt_params_dict[f"{c.ttt_param_names[0]}_grad"])
                for n in c.ttt_param_names]

    def conv_leaves(self) -> list[Tensor]:
        return [t for d in self.cache.conv_states_dic.values() for _, t in sorted(d.items())]


class TTTBackend:
    def __init__(self, model, tokenizer, config, *, device: torch.device, checkpoint_digest: str | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.vocab_size = int(config.vocab_size)
        self.checkpoint_digest = checkpoint_digest
        self.mini_batch = int(config.mini_batch_size)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ identity / signals
    def signal_names(self) -> tuple[str, ...]:
        # everything except the Fisher-weighted update, for which no Fisher over the fast weights exists yet
        return ("chunk_loss", "surprise_mean", "log_delta_norm", "log_write_norm")

    def writes_for_source(self, source: str) -> bool:
        # like Qwen's recurrent state, the fast weights are the model's only cross-mini-batch context:
        # generated tokens write to them natively and must be accounted
        return True

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, checkpoint_dir: str, *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> "TTTBackend":
        from transformers import AutoTokenizer

        p = Path(checkpoint_dir)
        raw = json.loads((p / "config.json").read_text())
        cfg = M.TTTConfig(**{k: v for k, v in raw.items() if k not in ("architectures", "auto_map", "transformers_version", "dtype", "model_type")})
        # build from the vendored class and load the safetensors directly: no remote-code prompt, no
        # auto-class type warning, and exactly the weights the digest covers
        from safetensors.torch import load_file

        model = M.TTTForCausalLM(cfg).to(dtype)
        state: dict[str, torch.Tensor] = {}
        for f in sorted(p.glob("*.safetensors")):
            state.update(load_file(str(f)))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if k != "lm_head.weight" or not cfg.tie_word_embeddings]
        if missing or unexpected:
            raise RuntimeError(f"TTT checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        model.eval()
        model.requires_grad_(False)
        dev = torch.device(device)
        model.to(dev)
        tok = AutoTokenizer.from_pretrained(str(p))
        return cls(model, tok, cfg, device=dev, checkpoint_digest=_checkpoint_digest(checkpoint_dir))

    # ------------------------------------------------------------------ tokenization
    def encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tokenizer(text, add_special_tokens=False).input_ids]

    def encode_chat(self, user_message: str, *, first_turn: bool = True) -> list[int]:
        """Render one user turn. The base model has no chat template; the SFT recipe in
        scripts/train/sft_ttt_chat.py uses exactly this rendering, so the two must move together."""
        text = CHAT_USER + user_message + CHAT_ASSISTANT
        ids = self.encode(text)
        if first_turn and self.tokenizer.bos_token_id is not None:
            ids = [int(self.tokenizer.bos_token_id)] + ids
        return ids

    # ------------------------------------------------------------------ state lifecycle
    def init_state(self) -> TTTState:
        with self._lock:
            return TTTState(M.TTTCache(self.model.model, 1))

    def position(self, state: TTTState) -> int:
        return state.position

    def clone(self, state: TTTState) -> TTTState:
        with self._lock:
            return state.clone()

    def state_delta(self, a: TTTState, b: TTTState) -> list[Tensor]:
        """Per-fast-weight change ``a − b`` over every layer's W1, b1, W2, b2 (the memory units)."""
        return [la - lb for la, lb in zip(a.weight_leaves(), b.weight_leaves())]

    def is_finite(self, state: TTTState) -> bool:
        for t in state.weight_leaves() + state.grad_leaves() + state.conv_leaves():
            if not bool(torch.isfinite(t).all()):
                return False
        return True

    def state_norms(self, state: TTTState) -> dict[str, Any]:
        per = [float(t.float().norm()) for t in state.weight_leaves()]
        return {"fast_weight_norm": per, "fast_weight_norm_total": float(sum(x * x for x in per) ** 0.5)}

    def _identity(self) -> dict[str, Any]:
        import transformers

        c = self.config
        return {"backend": "ttt", "checkpoint_digest": self.checkpoint_digest, "cache_schema": _SCHEMA,
                "transformers_version": str(transformers.__version__), "ttt_layer_type": str(c.ttt_layer_type),
                "num_hidden_layers": int(c.num_hidden_layers), "hidden_size": int(c.hidden_size),
                "mini_batch_size": int(c.mini_batch_size), "vocab_size": int(c.vocab_size)}

    def state_dict(self, state: TTTState) -> dict[str, Any]:
        with self._lock:
            c = copy.deepcopy(state.cache)
        params = {k: {l: t.detach().cpu() for l, t in d.items()} for k, d in c.ttt_params_dict.items()}
        convs = {k: {l: t.detach().cpu() for l, t in d.items()} for k, d in c.conv_states_dic.items()}
        return {"backend": "ttt", "identity": self._identity(), "seqlen_offset": int(c.seqlen_offset),
                "ttt_params": params, "conv_states": convs}

    def load_state_dict(self, data: dict[str, Any]) -> TTTState:
        if not isinstance(data, dict) or data.get("backend") != "ttt":
            raise ValueError("not a TTT backend state")
        if data.get("identity") != self._identity():
            raise ValueError(f"incompatible TTT session: saved identity {data.get('identity')} != current {self._identity()}")
        ref = self.init_state().cache
        cache = M.TTTCache.__new__(M.TTTCache)
        cache.seqlen_offset = int(data["seqlen_offset"])
        cache.mini_batch_size = ref.mini_batch_size
        cache.ttt_param_names = list(ref.ttt_param_names)
        cache.ttt_params_dict = M.defaultdict(dict)
        cache.conv_states_dic = M.defaultdict(dict)
        for k, d in ref.ttt_params_dict.items():
            saved = data["ttt_params"].get(k)
            if saved is None or set(saved) != set(d):
                raise ValueError(f"TTT state is missing fast-weight group {k}")
            for l, t in d.items():
                v = saved[l]
                if tuple(v.shape) != tuple(t.shape) or v.dtype != t.dtype:
                    raise ValueError(f"TTT state {k}[{l}] shape/dtype {tuple(v.shape)}/{v.dtype} != expected {tuple(t.shape)}/{t.dtype}")
                cache.ttt_params_dict[k][l] = v.clone().to(self.device)
        for k, d in ref.conv_states_dic.items():
            saved = data["conv_states"].get(k)
            if saved is None or set(saved) != set(d):
                raise ValueError(f"TTT state is missing conv group {k}")
            for l, t in d.items():
                v = saved[l]
                if tuple(v.shape) != tuple(t.shape):
                    raise ValueError(f"TTT conv state {k}[{l}] shape {tuple(v.shape)} != expected {tuple(t.shape)}")
                cache.conv_states_dic[k][l] = v.clone().to(self.device, t.dtype)
        st = TTTState(cache)
        if not self.is_finite(st):
            raise ValueError("persisted TTT state contains non-finite values")
        return st

    # ------------------------------------------------------------------ forward
    def _segments(self, n: int, offset: int) -> list[int]:
        """Feed lengths that respect the reference cache: singles until aligned, then whole mini-batches,
        then one remainder block (allowed at an aligned offset)."""
        out: list[int] = []
        while n > 0 and offset % self.mini_batch != 0:
            out.append(1)
            n -= 1
            offset += 1
        full = (n // self.mini_batch) * self.mini_batch
        if full:
            out.append(full)
            n -= full
        if n:
            out.append(n)
        return out

    @torch.no_grad()
    def forward(self, chunk: list[int], state: TTTState, *, freeze: bool, beta_scale: float) -> tuple[Tensor, TTTState, list[MemorySignals]]:
        scale = 0.0 if freeze else float(beta_scale)
        x = torch.tensor([list(map(int, chunk))], dtype=torch.long, device=self.device)
        logits: list[Tensor] = []
        collected: list[dict[str, Any]] = []
        with self._lock:
            prev_scale, prev_col = getattr(M._tls, "eta_scale", 1.0), getattr(M._tls, "collect", None)
            M._tls.eta_scale, M._tls.collect = scale, collected
            try:
                i = 0
                for n in self._segments(x.shape[1], state.position):
                    out = self.model(x[:, i:i + n], cache_params=state.cache, use_cache=True)
                    logits.append(out.logits[0])
                    i += n
            finally:
                M._tls.eta_scale, M._tls.collect = prev_scale, prev_col
            if self.device.type == "mps":
                torch.mps.synchronize()
        return torch.cat(logits, dim=0), state, self._pack_signals(collected, x.shape[1])

    def _pack_signals(self, collected: list[dict[str, Any]], n_tokens: int) -> list[MemorySignals]:
        """One MemorySignals per layer, each (B=1, H, T): the collector emits one entry per (layer,
        mini-batch step) in forward order, so concatenating a layer's entries along T rebuilds the chunk."""
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for e in collected:
            by_layer.setdefault(int(e["layer"]), []).append(e)
        out: list[MemorySignals] = []
        for layer in sorted(by_layer):
            err = torch.cat([e["err"] for e in by_layer[layer]], dim=-1)
            eta = torch.cat([e["eta"] for e in by_layer[layer]], dim=-1)
            wn = torch.cat([e["write_norm"] for e in by_layer[layer]], dim=-1)
            if err.shape[-1] != n_tokens:
                raise RuntimeError(f"signal length {err.shape[-1]} != chunk length {n_tokens} on layer {layer}")
            out.append(MemorySignals(err=err.detach(), beta=eta.detach(), alpha=torch.ones_like(err), write_norm=wn.detach()))
        return out

    @torch.no_grad()
    def process(self, ids: list[int], state: TTTState, *, freeze: bool = False) -> tuple[Tensor, TTTState]:
        logits, state, _ = self.forward(ids, state, freeze=freeze, beta_scale=1.0)
        return logits, state

    @torch.no_grad()
    def logits_full(self, ids: list[int]) -> Tensor:
        """A single pass with a fresh cache, fed in the same alignment-respecting segments (the reference
        code's dual and primal forms agree only up to float error, so parity is checked at a tolerance)."""
        return self.process(ids, self.init_state())[0]

    # ------------------------------------------------------------------ canaries (frozen)
    def _probe_nll(self, ids: list[int], state: TTTState) -> float:
        logits, _ = self.process(ids, self.clone(state), freeze=True)
        tgt = torch.tensor(ids[1:], dtype=torch.long, device=self.device)
        return float(torch.nn.functional.cross_entropy(logits[:-1], tgt))

    def score_suite(self, state: TTTState, suite: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in ("coherence", "poison"):
            probes = getattr(suite, name, None) or []
            losses = [self._probe_nll([int(t) for t in p], state) for p in probes if len(p) >= 2]
            out[name] = float(sum(losses) / len(losses)) if losses else float("nan")
        return out

    def fast_weight_grad(self, state: TTTState, probe_ids: list[int], target_ids: list[int]) -> list[Tensor]:
        """∂(frozen probe CE)/∂(fast weights) on a disposable copy: the probe is read with the inner
        step disabled, states are grad leaves, and the cache's in-place update is suppressed."""
        with self._lock:
            probe = copy.deepcopy(state.cache)
            probe.no_update = True
            leaves: list[Tensor] = []
            names = probe.ttt_param_names
            for l in sorted(probe.ttt_params_dict[f"{names[0]}_states"]):
                for n in names:
                    t = probe.ttt_params_dict[f"{n}_states"][l].detach().clone().requires_grad_(True)
                    probe.ttt_params_dict[f"{n}_states"][l] = t
                    leaves.append(t)
            prev_scale = getattr(M._tls, "eta_scale", 1.0)
            M._tls.eta_scale = 0.0
            try:
                x = torch.tensor([list(map(int, probe_ids))], dtype=torch.long, device=self.device)
                logits = []
                i = 0
                with torch.enable_grad():
                    for n in self._segments(x.shape[1], probe.seqlen_offset):
                        out = self.model(x[:, i:i + n], cache_params=probe, use_cache=True)
                        logits.append(out.logits[0])
                        i += n
                    loss = torch.nn.functional.cross_entropy(torch.cat(logits, 0), torch.tensor(target_ids, dtype=torch.long, device=self.device))
                    grads = torch.autograd.grad(loss, leaves, allow_unused=True)
            finally:
                M._tls.eta_scale = prev_scale
        return [torch.zeros_like(l) if g is None else g.detach() for l, g in zip(leaves, grads)]

    def canary_gradient(self, state: TTTState, suite: Any) -> list[Tensor]:
        total: list[Tensor] | None = None
        n = 0
        for p in getattr(suite, "coherence", None) or []:
            ids = [int(t) for t in p]
            if len(ids) < 2:
                continue
            g = self.fast_weight_grad(state, ids[:-1], ids[1:])
            total = g if total is None else [a + b for a, b in zip(total, g)]
            n += 1
        if total is None:
            return [torch.zeros_like(t) for t in state.weight_leaves()]
        return [(t / n).detach() for t in total]

    @torch.no_grad()
    def apply_projected(self, working: TTTState, committed: TTTState, projected: list[Tensor]) -> None:
        """Overwrite working's fast weights with committed + projected delta, leaf by leaf, in ``weight_leaves`` order."""
        it = iter(projected)
        c = working.cache
        for l in sorted(c.ttt_params_dict[f"{c.ttt_param_names[0]}_states"]):
            for n in c.ttt_param_names:
                d = next(it)
                base = committed.cache.ttt_params_dict[f"{n}_states"][l]
                c.ttt_params_dict[f"{n}_states"][l] = base + d.to(base.device, base.dtype)
