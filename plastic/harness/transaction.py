"""The chunk transaction runner.

Inputs advance the ``working`` state; at every chunk boundary the runner measures
the chunk (loss, surprise, β, update norm, Fisher, canaries, alignment, robust z,
CUSUM, budget), asks the policy for a decision, and applies it:

- commit:   committed := working
- rollback: working := committed, the chunk is reprocessed frozen (read, not learned), committed := working
- scale:    working := committed, the chunk is reprocessed with β scaled, committed := working
- project:  the multi-layer S delta is projected against the coherence-canary gradient;
            if too much of it is removed the decision falls back to rollback
- readonly: as rollback, for a session whose budget is exhausted or whose alarm is latched

Outputs already produced inside a chunk are not retracted. Everything the
runner decides is logged with every signal that informed it.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from plastic.config import ModelConfig
from plastic.harness.calibrate import Calibration
from plastic.harness.canary import CanarySuite, canary_gradient, score_suite
from plastic.harness.config import HarnessConfig
from plastic.harness.fisher import fisher_norm
from plastic.harness.policy import Decision, decide
from plastic.harness.projection import project_delta
from plastic.harness.signals import (
    STAT_SIGNALS,
    ChunkSignals,
    compression_ratio,
    compute_z,
    cosine,
    delta_norms,
    summarize_memory_signals,
)
from plastic.harness.stats import Cusum, SignalHistory
from plastic.model.memory import MemorySignals
from plastic.model.state import SessionState

Source = Literal["user", "model"]


class TransactionRunner:
    def __init__(
        self,
        model,
        model_cfg: ModelConfig,
        harness_cfg: HarnessConfig,
        *,
        calibration: Calibration | None = None,
        suite: CanarySuite | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.cfg = model_cfg
        self.hcfg = harness_cfg
        self.calibration = calibration
        self.suite = suite
        self.device = torch.device(device)
        self.L = int(model_cfg.chunk)
        self.domain = model_cfg.domain
        self.fisher = calibration.fisher if calibration is not None else None

        self.committed: SessionState = model.init_state(1, self.device)
        self.working: SessionState = self.committed.clone()
        self.anchor: SessionState = self.committed.clone()
        self.pending: list[Any] = []
        self.pending_targets: list[Tensor] = []
        self.pending_sources: list[str] = []
        self.pending_loss: list[float | None] = []
        self.pending_signals: list[list[MemorySignals]] = []
        self._last_logits: Tensor | None = None
        self.history: dict[str, SignalHistory] = {n: SignalHistory(harness_cfg.history_window) for n in STAT_SIGNALS}
        self.cusum = Cusum(harness_cfg.cusum_k, harness_cfg.cusum_h)
        self.budget_used = 0.0
        self.read_only = False
        self.read_only_reason: str | None = None
        self.n_transactions = 0
        self.transactions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ feeding
    @property
    def pos(self) -> int:
        return int(self.working.pos)

    def _forward_segment(self, freeze: bool, beta_scale: float, items: list[Any]) -> tuple[Tensor, list[MemorySignals]]:
        if self.domain == "text":
            x = torch.tensor([items], dtype=torch.long, device=self.device)
        else:
            x = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in items]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out, self.working, signals = self.model(x, self.working, mode="chunk", freeze=freeze, beta_scale=beta_scale)
        return out[0], [s.detach() for s in signals]

    def _token_freeze(self, source: str) -> bool:
        return self.read_only or (source == "model" and not self.hcfg.learn_from_generation)

    def feed_tokens(self, ids: list[int], *, source: Source = "user") -> Tensor | None:
        if self.domain != "text":
            raise ValueError("feed_tokens is for text sessions")
        i = 0
        while i < len(ids):
            take = min(self.L - len(self.pending), len(ids) - i)
            seg = [int(t) for t in ids[i : i + take]]
            logits, signals = self._forward_segment(self._token_freeze(source), 1.0, seg)
            self._absorb_text(seg, source, logits, signals)
            i += take
            if len(self.pending) == self.L:
                self._transact()
        return self._last_logits

    def _absorb_text(self, seg: list[int], source: str, logits: Tensor, signals: list[MemorySignals]) -> None:
        # NLL of token t comes from the logits that preceded it
        prev = self._last_logits
        for j, tok in enumerate(seg):
            nll = None if prev is None else float(F.cross_entropy(prev.unsqueeze(0), torch.tensor([tok], device=prev.device)))
            self.pending.append(tok)
            self.pending_sources.append(source)
            self.pending_loss.append(nll)
            prev = logits[j]
        self._last_logits = logits[-1].detach()
        self.pending_signals.append(signals)

    def feed_physics(self, rows: Tensor, targets: Tensor) -> Tensor:
        if self.domain != "physics":
            raise ValueError("feed_physics is for physics sessions")
        preds: list[Tensor] = []
        i = 0
        while i < rows.shape[0]:
            take = min(self.L - len(self.pending), rows.shape[0] - i)
            seg = [rows[i + j].detach().cpu() for j in range(take)]
            pred, signals = self._forward_segment(self.read_only, 1.0, seg)
            for j in range(take):
                tgt = targets[i + j].detach().cpu()
                self.pending.append(seg[j])
                self.pending_targets.append(tgt)
                self.pending_sources.append("user")
                self.pending_loss.append(float(F.mse_loss(pred[j].cpu(), tgt)))
            self.pending_signals.append(signals)
            preds.append(pred.detach())
            i += take
            if len(self.pending) == self.L:
                self._transact()
        return torch.cat(preds, dim=0)

    def flush(self) -> dict[str, Any] | None:
        """Transact a partial chunk (end of a turn)."""
        if not self.pending:
            return None
        return self._transact()

    # ------------------------------------------------------------------ transaction
    def _reprocess(self, *, freeze: bool, beta_scale: float) -> None:
        """Rebuild ``working`` from ``committed`` by replaying the pending inputs."""
        self.working = self.committed.clone()
        new_signals: list[list[MemorySignals]] = []
        items = self.pending
        sources = self.pending_sources
        start = 0
        while start < len(items):
            # group consecutive items that share the same freeze flag
            end = start
            fz = freeze or self._token_freeze(sources[start])
            while end < len(items) and (freeze or self._token_freeze(sources[end])) == fz:
                end += 1
            out, signals = self._forward_segment(fz, beta_scale, items[start:end])
            new_signals.append(signals)
            if self.domain == "text":
                self._last_logits = out[-1].detach()
            start = end
        self.pending_signals = new_signals

    def _measure(self) -> tuple[ChunkSignals, list[Tensor], list[Tensor] | None]:
        n = len(self.pending)
        losses = [x for x in self.pending_loss if x is not None]
        chunk_loss = float(sum(losses) / len(losses)) if losses else float("nan")
        mem = summarize_memory_signals([s for group in self.pending_signals for s in group])
        deltas = self.working.s_delta(self.committed)
        dnorm, per_layer = delta_norms(deltas)
        fisher_update = fisher_norm(deltas, self.fisher) if self.fisher is not None else None
        fisher_drift = fisher_norm(self.working.s_delta(self.anchor), self.fisher) if self.fisher is not None else None
        before = after = {}
        g: list[Tensor] | None = None
        alignment = None
        if self.suite is not None:
            before = score_suite(self.model, self.committed, self.suite, device=self.device)
            after = score_suite(self.model, self.working, self.suite, device=self.device)
            if self.hcfg.enable_projection:
                g = canary_gradient(self.model, self.committed, self.suite, device=self.device)
                alignment = cosine(deltas, g)
        sig = ChunkSignals(
            pos_start=self.pos - n,
            pos_end=self.pos,
            n_tokens=n,
            chunk_loss=chunk_loss,
            surprise_mean=mem["surprise_mean"],
            surprise_max=mem["surprise_max"],
            beta_mean=mem["beta_mean"],
            alpha_mean=mem["alpha_mean"],
            write_norm_sum=mem["write_norm_sum"],
            delta_norm=dnorm,
            delta_norm_per_layer=per_layer,
            fisher_update=fisher_update,
            fisher_drift=fisher_drift,
            canary_coherence_before=before.get("coherence"),
            canary_coherence_after=after.get("coherence"),
            canary_poison_before=before.get("poison"),
            canary_poison_after=after.get("poison"),
            canary_delta_coherence=(after["coherence"] - before["coherence"]) if before else None,
            canary_delta_poison=(after["poison"] - before["poison"]) if before else None,
            canary_alignment=alignment,
            compression_ratio=compression_ratio(self.pending) if self.domain == "text" else None,
            budget_used=self.budget_used,
            budget_remaining=(None if self.hcfg.budget_session is None else self.hcfg.budget_session - self.budget_used),
        )
        if self.read_only:
            # frozen chunks have zero writes by construction: they carry no evidence
            sig.z = {name: None for name in STAT_SIGNALS}
            sig.cusum_alarm = False
            return sig, deltas, g
        sig.z = compute_z(sig, reference=(self.calibration.reference if self.calibration else None), history=self.history)
        z_ld = sig.z.get("log_delta_norm")
        sig.cusum_alarm = self.cusum.update(z_ld) if (self.hcfg.enable_stats and z_ld is not None) else False
        return sig, deltas, g

    def _transact(self) -> dict[str, Any]:
        t0 = time.time()
        sig, deltas, g = self._measure()
        thresholds = self.calibration.thresholds if self.calibration is not None else None
        decision = decide(sig, self.hcfg, thresholds, read_only=self.read_only)
        applied = self._apply(decision, sig, deltas, g)
        if self.hcfg.enable_budget and self.hcfg.budget_session is not None and self.budget_used > self.hcfg.budget_session:
            self.read_only = True
            self.read_only_reason = f"budget_session({self.budget_used:.4g}>{self.hcfg.budget_session:.4g})"
        if sig.cusum_alarm and self.hcfg.freeze_on_alarm and not self.hcfg.log_only:
            self.read_only = True
            self.read_only_reason = "cusum_alarm"
        if applied.kind in ("commit", "project"):
            for name in STAT_SIGNALS:
                v = sig.value(name)
                if v is not None and v == v:
                    self.history[name].append(float(v))
        record = {
            "index": self.n_transactions,
            "t_unix": int(time.time()),
            "pos_start": sig.pos_start,
            "pos_end": sig.pos_end,
            "decision": applied.to_dict(),
            "requested": decision.to_dict(),
            "signals": sig.to_dict(),
            "read_only": self.read_only,
            "read_only_reason": self.read_only_reason,
            "seconds": time.time() - t0,
        }
        self.transactions.append(record)
        self.n_transactions += 1
        self.pending, self.pending_targets, self.pending_sources = [], [], []
        self.pending_loss, self.pending_signals = [], []
        return record

    def _apply(self, decision: Decision, sig: ChunkSignals, deltas: list[Tensor], g: list[Tensor] | None) -> Decision:
        kind = decision.kind
        if kind == "commit":
            self.committed = self.working.clone()
            self.budget_used += sig.delta_norm
            return decision
        if kind in ("rollback", "readonly"):
            self._reprocess(freeze=True, beta_scale=1.0)
            self.committed = self.working.clone()
            return decision
        if kind == "scale":
            self._reprocess(freeze=False, beta_scale=decision.scale)
            self.budget_used += delta_norms(self.working.s_delta(self.committed))[0]
            self.committed = self.working.clone()
            return decision
        if kind == "project":
            if g is None:
                return self._apply(Decision("rollback", decision.reasons + ["project_unavailable"]), sig, deltas, None)
            projected, pstats = project_delta(deltas, g, eps_dot=self.hcfg.project_eps_dot, eps_cos=self.hcfg.project_eps_cos)
            if pstats.removed_ratio > self.hcfg.project_max_removed:
                return self._apply(
                    Decision("rollback", decision.reasons + [f"project_removed({pstats.removed_ratio:.3f}>{self.hcfg.project_max_removed})"]),
                    sig, deltas, None,
                )
            for layer, base, d in zip(self.working.layers, self.committed.layers, projected):
                layer.S = base.S + d.to(base.S.device)
            self.budget_used += delta_norms(projected)[0]
            self.committed = self.working.clone()
            return Decision("project", decision.reasons + [f"removed_ratio({pstats.removed_ratio:.3f})", f"dot({pstats.dot_before:+.4g}->{pstats.dot_after:+.4g})"])
        raise ValueError(f"unknown decision {kind!r}")

    # ------------------------------------------------------------------ control
    def resume(self) -> None:
        """Lift a latched read-only state (after a verification pass by the caller)."""
        self.read_only = False
        self.read_only_reason = None

    def reset(self) -> None:
        self.committed = self.model.init_state(1, self.device)
        self.working = self.committed.clone()
        self.anchor = self.committed.clone()
        self.pending, self.pending_targets, self.pending_sources = [], [], []
        self.pending_loss, self.pending_signals = [], []
        self._last_logits = None
        self.history = {n: SignalHistory(self.hcfg.history_window) for n in STAT_SIGNALS}
        self.cusum = Cusum(self.hcfg.cusum_k, self.hcfg.cusum_h)
        self.budget_used = 0.0
        self.read_only = False
        self.read_only_reason = None

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "committed": self.committed.state_dict(),
            "working": self.working.state_dict(),
            "anchor": self.anchor.state_dict(),
            "pending": [p.clone() if torch.is_tensor(p) else int(p) for p in self.pending],
            "pending_targets": [t.clone() for t in self.pending_targets],
            "pending_sources": list(self.pending_sources),
            "pending_loss": list(self.pending_loss),
            "pending_signals": [
                [{"err": s.err.cpu(), "beta": s.beta.cpu(), "alpha": s.alpha.cpu(), "write_norm": s.write_norm.cpu()} for s in group]
                for group in self.pending_signals
            ],
            "last_logits": None if self._last_logits is None else self._last_logits.detach().cpu(),
            "history": {k: v.values() for k, v in self.history.items()},
            "cusum": self.cusum.state(),
            "budget_used": self.budget_used,
            "read_only": self.read_only,
            "read_only_reason": self.read_only_reason,
            "n_transactions": self.n_transactions,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.committed = SessionState.from_state_dict(d["committed"]).to(self.device)
        self.working = SessionState.from_state_dict(d["working"]).to(self.device)
        self.anchor = SessionState.from_state_dict(d["anchor"]).to(self.device)
        self.pending = [p.clone() if torch.is_tensor(p) else int(p) for p in d.get("pending", [])]
        self.pending_targets = [t.clone() for t in d.get("pending_targets", [])]
        self.pending_sources = list(d.get("pending_sources", []))
        self.pending_loss = list(d.get("pending_loss", []))
        self.pending_signals = [
            [MemorySignals(err=s["err"].to(self.device), beta=s["beta"].to(self.device), alpha=s["alpha"].to(self.device), write_norm=s["write_norm"].to(self.device)) for s in group]
            for group in d.get("pending_signals", [])
        ]
        ll = d.get("last_logits")
        self._last_logits = None if ll is None else ll.to(self.device)
        self.history = {k: SignalHistory(self.hcfg.history_window, v) for k, v in d.get("history", {}).items()}
        for n in STAT_SIGNALS:
            self.history.setdefault(n, SignalHistory(self.hcfg.history_window))
        self.cusum = Cusum.from_state(d.get("cusum", {}))
        self.budget_used = float(d.get("budget_used", 0.0))
        self.read_only = bool(d.get("read_only", False))
        self.read_only_reason = d.get("read_only_reason")
        self.n_transactions = int(d.get("n_transactions", 0))

    def summary(self) -> dict[str, Any]:
        return {
            "pos": self.pos,
            "pending": len(self.pending),
            "budget_used": self.budget_used,
            "budget_session": self.hcfg.budget_session,
            "read_only": self.read_only,
            "read_only_reason": self.read_only_reason,
            "n_transactions": self.n_transactions,
            "cusum": self.cusum.state(),
            "state_norms": self.committed.norms(),
            "drift_from_anchor": delta_norms(self.committed.s_delta(self.anchor))[0],
        }
