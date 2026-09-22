"""Session: a model, its calibration and canaries, a transaction runner, and persistence.

Text sessions chat (prompt tokens are learned through transactions; generated
tokens are read-only unless configured otherwise). Physics sessions run
episodes and report the three-way comparison on the same trajectory: base
(zero state, frozen), frozen (session state, no learning), adaptive (session
state, learning through the harness).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from plastic.data.physics import PhysicsEnv
from plastic.harness.calibrate import Calibration
from plastic.harness.canary import CanarySuite
from plastic.harness.config import HarnessConfig
from plastic.harness.transaction import TransactionRunner
from plastic.store import ArtifactStore
from plastic.tokenizer.bpe import Tokenizer


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


class Session:
    def __init__(self, store: ArtifactStore, session_id: str, *, device: torch.device | str = "cpu") -> None:
        self.store = store
        self.session_id = session_id
        self.device = torch.device(device)
        store.verify_session_model(session_id)
        self.meta = store.load_session_meta(session_id)
        self.model_id = str(self.meta["model_id"])
        self.cfg, self.model, _ = store.load_checkpoint(self.model_id, self.device)
        self.hcfg = HarnessConfig.from_dict(self.meta["harness"])
        model_dir = store.model_dir(self.model_id)
        self.calibration = Calibration.load(model_dir) if Calibration.exists(model_dir) else None
        canary_path = store.canary_path(self.model_id)
        self.suite = CanarySuite.load(canary_path) if __import__("os").path.exists(canary_path) else None
        self.tokenizer = Tokenizer.load(store.tokenizer_path(self.model_id)) if self.cfg.domain == "text" else None
        self.runner = TransactionRunner(
            self.model, self.cfg, self.hcfg, calibration=self.calibration, suite=self.suite, device=self.device
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
        cfg = store.load_config(model_id)
        sid = session_id or store.new_session_id("chat" if cfg.domain == "text" else "phys")
        store.create_session(sid, model_id=model_id, domain=cfg.domain, harness_cfg=harness_cfg or HarnessConfig(), extra=extra)
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
        s.update({"session_id": self.session_id, "model_id": self.model_id, "domain": self.cfg.domain})
        return s

    # ------------------------------------------------------------------ text
    def chat(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.9,
        top_k: int = 50,
        seed: int | None = None,
    ) -> ChatResult:
        if self.cfg.domain != "text" or self.tokenizer is None:
            raise ValueError("chat requires a text session")
        tok = self.tokenizer
        ids = tok.encode(prompt, add_bos=(self.runner.pos == 0 and not self.runner.pending))
        self.runner.transactions = []
        logits = self.runner.feed_tokens(ids, source="user")
        # generation is a control switch (generated tokens are read-only by default); it happens
        # only at a chunk boundary, so the prompt's partial chunk is transacted first
        self.runner.flush()
        logits = self.runner._last_logits  # the flush may have recomputed the chunk
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        out_ids: list[int] = []
        for _ in range(int(max_new_tokens)):
            if logits is None:
                break
            nxt = _sample(logits, temperature=temperature, top_k=top_k, gen=gen)
            if nxt == tok.eos_id:
                break
            out_ids.append(nxt)
            logits = self.runner.feed_tokens([nxt], source="model")
        self.runner.flush()
        transactions = list(self.runner.transactions)
        completion = tok.decode(out_ids)
        self._persist(transactions)
        self.store.append_trace(
            self.session_id,
            {"t_unix": int(time.time()), "kind": "chat", "prompt": prompt, "completion": completion,
             "pos_end": self.runner.pos, "n_transactions": len(transactions)},
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
