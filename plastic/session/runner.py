"""Session: a model, its calibration and canaries, a transaction runner, and persistence.

Text sessions chat (prompt tokens are learned through transactions; generated
tokens are read-only unless configured otherwise). Physics sessions run
episodes and report the three-way comparison on the same trajectory: base
(zero state, frozen), frozen (session state, no learning), adaptive (session
state, learning through the harness).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from plastic.config import ModelConfig
from plastic.data.physics import PhysicsEnv
from plastic.harness.calibrate import Calibration
from plastic.harness.canary import CanarySuite
from plastic.harness.config import HarnessConfig
from plastic.harness.transaction import TransactionRunner
from plastic.store import ArtifactStore
from plastic.tokenizer.bpe import Tokenizer


class _QwenTextIO:
    """A tokenizer-shaped adapter over a QwenBackend, so ``Session.chat`` drives a pretrained
    backend through the same ``encode(add_bos=...) / eos_id / decode`` interface the native BPE
    tokenizer exposes. ``encode`` renders the native chat template (add_bos is moot — the template
    carries the special tokens); ``decode`` and ``eos_id`` come from the Qwen tokenizer."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.eos_id = int(backend.tokenizer.eos_token_id)
        # the native assistant turn terminator + separator (chat_template: content + '<|im_end|>\n').
        # Fed into state after generation so the next user turn follows a correctly-closed assistant
        # turn; kept out of the displayed completion. eos IS <|im_end|> for Qwen.
        self.assistant_close_ids = list(backend.tokenizer.encode("<|im_end|>\n", add_special_tokens=False))

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        return self._backend.encode_chat(text)

    def decode(self, ids: list[int]) -> str:
        return self._backend.tokenizer.decode(ids)


class _TTTTextIO:
    """The same tokenizer-shaped adapter for a TTTBackend. The chat rendering lives in the backend
    (``encode_chat``) so the SFT recipe and the runtime cannot drift apart; a turn is closed by EOS."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self.eos_id = int(backend.tokenizer.eos_token_id)
        self.assistant_close_ids: list[int] = [self.eos_id]

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        return self._backend.encode_chat(text, first_turn=add_bos)

    def decode(self, ids: list[int]) -> str:
        return self._backend.tokenizer.decode(ids)


@dataclass
class ChatResult:
    prompt: str
    completion: str
    transactions: list[dict[str, Any]] = field(default_factory=list)
    n_tokens_in: int = 0
    n_tokens_out: int = 0
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "transactions": self.transactions,
            "n_tokens_in": self.n_tokens_in,
            "n_tokens_out": self.n_tokens_out,
            "summary": self.summary,
        }


@dataclass
class EpisodeResult:
    mu: float
    steps: int
    per_step: list[dict[str, float]] = field(default_factory=list)
    transactions: list[dict[str, Any]] = field(default_factory=list)
    means: dict[str, float] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu": self.mu,
            "steps": self.steps,
            "per_step": self.per_step,
            "transactions": self.transactions,
            "means": self.means,
            "summary": self.summary,
        }


def _sample(logits: Tensor, *, temperature: float, top_k: int, gen: torch.Generator | None) -> int:
    logits = logits.float() / max(1e-6, float(temperature))
    if top_k > 0 and top_k < logits.numel():
        vals, _ = torch.topk(logits, int(top_k))
        logits = logits.masked_fill(logits < vals[-1], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if gen is not None:
        return int(torch.multinomial(probs.cpu(), 1, generator=gen))
    return int(torch.multinomial(probs, 1))


def _counts(transactions: list[dict[str, Any]]) -> dict[str, int]:
    out = {"commits": 0, "rollbacks": 0, "scales": 0, "projects": 0, "readonly": 0}
    key = {"commit": "commits", "rollback": "rollbacks", "scale": "scales", "project": "projects", "readonly": "readonly"}
    for t in transactions:
        out[key[t["decision"]["kind"]]] += 1
    return out


def drive_chat_turn(runner, tok: Any, prompt: str, *, max_new_tokens: int, temperature: float, top_k: int, gen=None) -> tuple[str, list[int], list[int]]:
    """Drive one chat turn through ``runner``: encode the prompt (user source), partial-flush so the
    prompt transacts before generation, sample the model token-by-token (model source), close the
    assistant turn once, then final-flush. Returns (completion, generated_ids, prompt_ids). Both
    ``Session.chat`` and the Qwen calibrator use this so they exercise the IDENTICAL transaction
    protocol — the prompt-vs-generation/closure source partition and chunk boundaries are the same."""
    ids = tok.encode(prompt, add_bos=(runner.pos == 0 and not runner.pending))
    runner.feed_tokens(ids, source="user")
    runner.flush()
    logits = runner._last_logits  # the flush may have recomputed the chunk
    out_ids: list[int] = []
    for _ in range(int(max_new_tokens)):
        if logits is None:
            break
        nxt = _sample(logits, temperature=temperature, top_k=top_k, gen=gen)
        if nxt == tok.eos_id:
            break
        out_ids.append(nxt)
        logits = runner.feed_tokens([nxt], source="model")
    # close the assistant turn once (EOS / length cap / zero generation), as governed model-source
    # writes not shown to the user (native separator; empty for the plastic BPE path)
    close_ids = list(getattr(tok, "assistant_close_ids", []) or [])
    if close_ids:
        runner.feed_tokens(close_ids, source="model")
    runner.flush()
    return tok.decode(out_ids), out_ids, ids


def _verify_calibration(cal: Any, model_identity: str) -> tuple[Any, str]:
    """Install a persisted calibration only if it was built for THIS model's actual content. A
    mismatched or unsigned calibration is discarded — its thresholds would misgate a different model
    (ASTRA-078) — and the reason is returned for observability. Never silently trusts an unsigned or
    mismatched signature."""
    if cal is None:
        return None, "absent"
    if not cal.model_signature:
        return None, "rejected_unsigned"
    if cal.model_signature != model_identity:
        return None, "rejected_signature_mismatch"
    return cal, "installed"


class Session:
    def __init__(self, store: ArtifactStore, session_id: str, *, device: torch.device | str = "cpu") -> None:
        self.store = store
        self.session_id = session_id
        self.device = torch.device(device)
        store.verify_session_model(session_id)
        self.meta = store.load_session_meta(session_id)
        self.model_id = str(self.meta["model_id"])
        self.hcfg = HarnessConfig.from_dict(self.meta["harness"])
        record = store.load_model_record(self.model_id)
        self.backend_kind = str(record.get("backend", "plastic"))
        model_dir = store.model_dir(self.model_id)
        loaded_cal = Calibration.load(model_dir) if Calibration.exists(model_dir) else None
        canary_path = store.canary_path(self.model_id)
        self.suite = CanarySuite.load(canary_path) if os.path.exists(canary_path) else None
        if self.backend_kind == "qwen":
            # a pretrained backend: load Qwen and its native tokenizer, and drive the runner through it
            from plastic.backends.qwen import QwenBackend

            self.backend = QwenBackend.load(record["checkpoint_dir"], device=self.device)
            self.cfg = ModelConfig(domain="text", chunk=int(record.get("chunk", 8)))
            self.model = None
            self.tokenizer: Any = _QwenTextIO(self.backend)
            # bind to the ACTUAL loaded checkpoint content, not the (possibly stale) registry digest,
            # so a calibration built for a different checkpoint is refused even if the registry agrees
            model_identity = f"qwen:{self.backend.checkpoint_digest}"
        elif self.backend_kind == "ttt":
            # a pretrained TTT-MLP/TTT-Linear model (Sun et al. 2024): genuine inner-loop fast weights, full signals
            from plastic.backends.ttt_lm.backend import TTTBackend

            self.backend = TTTBackend.load(record["checkpoint_dir"], device=self.device)
            self.cfg = ModelConfig(domain="text", chunk=int(record.get("chunk", 16)))
            self.model = None
            self.tokenizer = _TTTTextIO(self.backend)
            model_identity = f"ttt:{self.backend.checkpoint_digest}"
        else:
            self.cfg, self.model, _ = store.load_checkpoint(self.model_id, self.device)
            self.backend = None
            self.tokenizer = Tokenizer.load(store.tokenizer_path(self.model_id)) if self.cfg.domain == "text" else None
            model_identity = store.model_signature(self.model_id)  # hashes the actual checkpoint content
        # install a persisted calibration only if it was built for THIS model's actual content
        self.calibration, self.calibration_status = _verify_calibration(loaded_cal, model_identity)
        self.runner = TransactionRunner(
            self.model, self.cfg, self.hcfg, calibration=self.calibration, suite=self.suite,
            device=self.device, backend=self.backend,
        )
        state = store.load_runner_state(session_id)
        if state:
            self.runner.load_state_dict(state)

    @classmethod
    def open(cls, store: ArtifactStore, session_id: str, *, device: torch.device | str = "cpu") -> "Session":
        return cls(store, session_id, device=device)

    @classmethod
    def create(
        cls,
        store: ArtifactStore,
        *,
        model_id: str,
        harness_cfg: HarnessConfig | None = None,
        session_id: str | None = None,
        device: torch.device | str = "cpu",
        extra: dict[str, Any] | None = None,
    ) -> "Session":
        record = store.load_model_record(model_id)
        # a pretrained-backend model has no local plastic config; its domain comes from the record
        domain = str(record.get("domain")) if record.get("backend", "plastic") != "plastic" else store.load_config(model_id).domain
        sid = session_id or store.new_session_id("chat" if domain == "text" else "phys")
        store.create_session(sid, model_id=model_id, domain=domain, harness_cfg=harness_cfg or HarnessConfig(), extra=extra)
        return cls(store, sid, device=device)

    # ------------------------------------------------------------------ persistence
    def _persist(self, transactions: list[dict[str, Any]]) -> dict[str, Any]:
        for rec in transactions:
            self.store.append_transaction(self.session_id, rec)
        self.meta = self.store.save_runner_state(
            self.session_id, self.runner.state_dict(), summary=self.runner.summary(), counts=_counts(transactions)
        )
        return self.meta

    def summary(self) -> dict[str, Any]:
        s = self.runner.summary()
        s.update({
            "session_id": self.session_id,
            "model_id": self.model_id,
            "domain": self.cfg.domain,
            # surface the backend and its honest state so a client shows them plainly: which backend
            # drives the session, whether a persisted calibration was installed/rejected/absent, and
            # exactly which decision signals this backend can produce (a pretrained backend gates on a
            # reduced set; the rest are carried as None).
            "backend": self.backend_kind,
            "calibration": self.calibration_status,
            "signals_available": list(self.runner.backend.signal_names()),
            # the EFFECTIVE generation-write policy for this session: model-source (generated) tokens
            # are write-eligible when the backend writes that source (Qwen's recurrent state does) OR
            # learn_from_generation overrides it. This is the configured capability; the read-only latch
            # (surfaced separately) suppresses all writes regardless. Write-eligible is not the same as
            # retained learning: each chunk still transacts and can be rolled back or scaled.
            "writes_generation": bool(self.runner.backend.writes_for_source("model")) or bool(self.hcfg.learn_from_generation),
        })
        return s

    # ------------------------------------------------------------------ text
    def chat(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        seed: int | None = None,
    ) -> ChatResult:
        if self.cfg.domain != "text" or self.tokenizer is None:
            raise ValueError("chat requires a text session")
        self.runner.transactions = []
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        completion, out_ids, ids = drive_chat_turn(
            self.runner, self.tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, gen=gen
        )
        transactions = list(self.runner.transactions)
        self._persist(transactions)
        # the transaction-index range is the turn's provenance key: positions restart after a reset, indices do not
        idx = [int(r["index"]) for r in transactions if "index" in r]
        self.store.append_trace(
            self.session_id,
            {"t_unix": int(time.time()), "kind": "chat", "prompt": prompt, "completion": completion,
             "pos_end": self.runner.pos, "n_transactions": len(transactions),
             "tx_start": (min(idx) if idx else None), "tx_end": (max(idx) + 1 if idx else None)},
        )
        return ChatResult(prompt, completion, transactions, len(ids), len(out_ids), self.summary())

    # ------------------------------------------------------------------ physics
    def physics_episode(
        self,
        *,
        steps: int,
        mu: float,
        seed: int = 0,
        actions: Tensor | None = None,
        nonlinear: bool = False,
        action_std: float = 0.5,
    ) -> EpisodeResult:
        if self.cfg.domain != "physics":
            raise ValueError("physics_episode requires a physics session")
        g = torch.Generator().manual_seed(int(seed))
        acts = actions if actions is not None else torch.randn(int(steps), 2, generator=g) * float(action_std)
        env = PhysicsEnv(float(mu), nonlinear=nonlinear)
        obs = env.reset()
        rows, targets = [], []
        for t in range(int(steps)):
            rows.append(torch.cat([obs, acts[t], torch.tensor([1.0 if t == 0 else 0.0])]))
            nxt = env.step(acts[t])
            targets.append(nxt - obs)
            obs = nxt
        rows_t = torch.stack(rows)
        targets_t = torch.stack(targets)
        x = rows_t.unsqueeze(0).to(self.device)
        y = targets_t.unsqueeze(0).to(self.device)
        with torch.no_grad():
            base_pred, _, _ = self.model(x, self.model.init_state(1, self.device), freeze=True)
            frozen_pred, _, _ = self.model(x, self.runner.committed.clone(), freeze=True)
        self.runner.transactions = []
        adaptive_pred = self.runner.feed_physics(rows_t, targets_t)
        self.runner.flush()
        transactions = list(self.runner.transactions)
        per_step = []
        for t in range(int(steps)):
            per_step.append(
                {
                    "t": t,
                    "base_mse": float(F.mse_loss(base_pred[0, t].cpu(), targets_t[t])),
                    "frozen_mse": float(F.mse_loss(frozen_pred[0, t].cpu(), targets_t[t])),
                    "adaptive_mse": float(F.mse_loss(adaptive_pred[t].cpu(), targets_t[t])),
                }
            )
        means = {k: float(sum(p[k] for p in per_step) / len(per_step)) for k in ("base_mse", "frozen_mse", "adaptive_mse")}
        self._persist(transactions)
        self.store.append_trace(
            self.session_id,
            {"t_unix": int(time.time()), "kind": "episode", "mu": float(mu), "steps": int(steps), "seed": int(seed),
             "nonlinear": bool(nonlinear), "means": means, "pos_end": self.runner.pos, "n_transactions": len(transactions)},
        )
        return EpisodeResult(float(mu), int(steps), per_step, transactions, means, self.summary())

    # ------------------------------------------------------------------ control
    def reset(self) -> dict[str, Any]:
        self.runner.reset()
        return self._persist([])

    def resume(self) -> dict[str, Any]:
        self.runner.resume()
        return self._persist([])

    def fork(self, child_session_id: str | None = None) -> str:
        self._persist([])
        child = child_session_id or self.store.new_session_id(f"{self.session_id}_fork")
        self.store.fork_session(self.session_id, child)
        return child
