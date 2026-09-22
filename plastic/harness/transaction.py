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

import math
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
        self.pending_eligible = False  # any pending token processed with learning enabled
        self._last_logits: Tensor | None = None
        self.history: dict[str, SignalHistory] = {n: SignalHistory(harness_cfg.history_window) for n in STAT_SIGNALS}
        self.cusum = Cusum(harness_cfg.cusum_k, self._effective_cusum_h())
        self.budget_used = 0.0
        self._exhausted = False
        self.read_only = False
        self.read_only_reason: str | None = None
        self.n_transactions = 0
        self.transactions: list[dict[str, Any]] = []

    def _effective_cusum_h(self) -> float:
        """The calibrated CUSUM threshold when one exists, else the configured default."""
        if self.calibration is not None and "cusum_h" in self.calibration.thresholds:
            return float(self.calibration.thresholds["cusum_h"])
        return float(self.hcfg.cusum_h)

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
            fz = self._token_freeze(source)
            logits, signals = self._forward_segment(fz, 1.0, seg)
            self._absorb_text(seg, source, logits, signals)
            self.pending_eligible = self.pending_eligible or not fz
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
            self.pending_eligible = self.pending_eligible or not self.read_only
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
        if self.read_only or not self.pending_eligible:
            # a chunk with no token permitted to learn has zero writes by construction:
            # it carries no evidence about the update stream
            sig.z = {name: None for name in STAT_SIGNALS}
            sig.cusum_alarm = False
            return sig, deltas, g
        sig.z = compute_z(sig, reference=(self.calibration.reference if self.calibration else None), history=self.history)
        z_ld = sig.z.get("log_delta_norm")
        sig.cusum_alarm = self.cusum.update(z_ld) if (self.hcfg.enable_stats and z_ld is not None) else False
        return sig, deltas, g

    def _accepted_metrics(self, pre_committed: SessionState, budget_before: float, sig: ChunkSignals) -> dict[str, Any]:
        """Metrics of what was actually committed, as opposed to the proposed update in ``signals``.

        The ``signals`` fields are measured on the provisional ``working`` state before the
        decision; these are measured on the final ``committed`` state after it, so a rollback
        reports an accepted delta of zero and a scale reports the delta it actually kept.
        """
        accepted_delta = delta_norms(self.committed.s_delta(pre_committed))[0]
        acc: dict[str, Any] = {
            "delta_norm": accepted_delta,
            "budget_charge": self.budget_used - budget_before,
            "budget_used": self.budget_used,
            "budget_remaining": (None if self.hcfg.budget_session is None else self.hcfg.budget_session - self.budget_used),
        }
        if self.suite is not None:
            after = score_suite(self.model, self.committed, self.suite, device=self.device)
            acc["canary_coherence_after"] = after["coherence"]
            acc["canary_poison_after"] = after["poison"]
            acc["canary_delta_coherence"] = (
                None if sig.canary_coherence_before is None else after["coherence"] - sig.canary_coherence_before
            )
            acc["canary_delta_poison"] = (
                None if sig.canary_poison_before is None else after["poison"] - sig.canary_poison_before
            )
        return acc

    def _transact(self) -> dict[str, Any]:
        t0 = time.time()
        pre_committed = self.committed.clone()
        budget_before = self.budget_used
        sig, deltas, g = self._measure()
        thresholds = self.calibration.thresholds if self.calibration is not None else None
        decision = decide(sig, self.hcfg, thresholds, read_only=self.read_only)
        applied = self._apply(decision, sig, deltas, g)
        accepted = self._accepted_metrics(pre_committed, budget_before, sig)
        if (
            self.hcfg.enable_budget
            and not self.hcfg.log_only
            and self.hcfg.budget_session is not None
            and (self.budget_used >= self.hcfg.budget_session * (1.0 - 1e-3) or self._exhausted)
        ):
            # the remaining budget is effectively exhausted: later chunks could only be
            # scaled into noise, so the session becomes read-only until resumed
            self.read_only = True
            self.read_only_reason = f"budget_session({self.budget_used:.4g}>={self.hcfg.budget_session:.4g})"
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
            "signals": sig.to_dict(),  # PROPOSED: measured on the provisional state before the decision
            "accepted": accepted,      # ACCEPTED: measured on the committed state after the decision
            "read_only": self.read_only,
            "read_only_reason": self.read_only_reason,
            "seconds": time.time() - t0,
        }
        self.transactions.append(record)
        self.n_transactions += 1
        self.pending, self.pending_targets, self.pending_sources = [], [], []
        self.pending_loss, self.pending_signals = [], []
        self.pending_eligible = False
        return record

    # ------------------------------------------------------------------ acceptance
    def _cap(self) -> float | None:
        """Hard cap on the complete state delta of this chunk: min(chunk cap, remaining session budget)."""
        if not self.hcfg.enable_budget:
            return None
        caps: list[float] = []
        if self.hcfg.budget_chunk is not None:
            caps.append(float(self.hcfg.budget_chunk))
        if self.hcfg.budget_session is not None:
            caps.append(max(0.0, float(self.hcfg.budget_session) - self.budget_used))
        return min(caps) if caps else None

    def _cap_is_session_remaining(self, cap: float) -> bool:
        """True when the remaining session budget (not the per-chunk cap) is what bound the chunk."""
        if self.hcfg.budget_session is None:
            return False
        remaining = max(0.0, float(self.hcfg.budget_session) - self.budget_used)
        if self.hcfg.budget_chunk is not None and self.hcfg.budget_chunk < remaining:
            return False
        return abs(cap - remaining) <= 1e-12 * max(1.0, remaining)

    def _candidate_delta_norm(self) -> float:
        return delta_norms(self.working.s_delta(self.committed))[0]

    @staticmethod
    def _finite(state: SessionState) -> bool:
        """Every tensor that would be committed or restored must be finite."""
        for l in state.layers:
            tensors = [l.h, l.S, l.M, l.conv_ssm, l.conv_mem]
            if l.chunk is not None:
                tensors += [l.chunk.A, l.chunk.Bv, l.chunk.alpha_sum]
            for t in tensors:
                if t is not None and not bool(torch.isfinite(t).all()):
                    return False
        return True

    def _accept(self, kind: str, reasons: list[str], scale: float) -> Decision:
        self.budget_used += self._candidate_delta_norm()
        self.committed = self.working.clone()
        return Decision(kind, reasons, scale)  # type: ignore[arg-type]

    def _reject_frozen(self, kind: str, reasons: list[str]) -> Decision:
        """Refuse the chunk as training signal: recompute it frozen and commit that."""
        self._reprocess(freeze=True, beta_scale=1.0)
        if not self._finite(self.working):
            # even reading the chunk produced non-finite state: keep the last good state and stop learning
            self.working = self.committed.clone()
            self.read_only = True
            self.read_only_reason = "nonfinite_frozen_recompute"
            return Decision("rollback", reasons + ["nonfinite_frozen_recompute"])
        self.committed = self.working.clone()
        return Decision(kind, reasons)  # type: ignore[arg-type]

    def _apply(self, decision: Decision, sig: ChunkSignals, deltas: list[Tensor], g: list[Tensor] | None) -> Decision:
        reasons = list(decision.reasons)
        kind = decision.kind
        if kind in ("rollback", "readonly"):
            return self._reject_frozen(kind, reasons)
        if not self._finite(self.working) or not math.isfinite(sig.delta_norm):
            return self._reject_frozen("rollback", reasons + ["nonfinite_candidate"])
        if self.hcfg.log_only:
            # observation only: the reference trajectory must be the ungated one
            return self._accept("commit", reasons, 1.0)
        cap = self._cap()

        if kind == "project":
            if g is None:
                return self._reject_frozen("rollback", reasons + ["project_unavailable"])
            projected, pstats = project_delta(deltas, g, eps_dot=self.hcfg.project_eps_dot, eps_cos=self.hcfg.project_eps_cos)
            if pstats.removed_ratio > self.hcfg.project_max_removed:
                return self._reject_frozen(
                    "rollback", reasons + [f"project_removed({pstats.removed_ratio:.3f}>{self.hcfg.project_max_removed})"]
                )
            scale_p = 1.0
            norm_p = delta_norms(projected)[0]
            if cap is not None and norm_p > cap:
                if cap <= 0.0:
                    self._exhausted = True
                    return self._reject_frozen("rollback", reasons + ["budget_exhausted"])
                scale_p = cap / norm_p  # a shrunk delta keeps the half-space constraint (eps >= 0)
                reasons.append(f"budget_scaled_projection(scale={scale_p:.3g})")
            for layer, base, d in zip(self.working.layers, self.committed.layers, projected):
                layer.S = base.S + scale_p * d.to(base.S.device)
            if not self._finite(self.working):
                return self._reject_frozen("rollback", reasons + ["nonfinite_candidate"])
            if cap is not None:
                # recheck the representable stored difference, not the ideal delta
                actual = self._candidate_delta_norm()
                if actual > cap * (1.0 + 1e-6) and actual > 0.0:
                    scale_p *= (cap / actual) * 0.999
                    for layer, base, d in zip(self.working.layers, self.committed.layers, projected):
                        layer.S = base.S + scale_p * d.to(base.S.device)
                    actual = self._candidate_delta_norm()
                    reasons.append(f"budget_recheck(scale={scale_p:.3g},delta={actual:.4g})")
                if actual > cap * (1.0 + 1e-6):
                    return self._reject_frozen("rollback", reasons + [f"budget_unrepresentable(delta={actual:.4g}>cap={cap:.4g})"])
            reasons += [f"removed_ratio({pstats.removed_ratio:.3f})", f"dot({pstats.dot_before:+.4g}->{pstats.dot_after:+.4g})"]
            return self._accept("project", reasons, scale_p)

        # commit or scale: the policy's scale is a suggestion; the cap is checked on the actual candidate
        scale = float(decision.scale) if kind == "scale" else 1.0
        if kind == "scale":
            self._reprocess(freeze=False, beta_scale=scale)
        actual = self._candidate_delta_norm()
        tries = 0
        while cap is not None and actual > cap * (1.0 + 1e-6) and tries < 3:
            if actual <= 0.0 or cap <= 0.0:
                break
            scale = scale * (cap / actual) * 0.95
            self._reprocess(freeze=False, beta_scale=scale)
            actual = self._candidate_delta_norm()
            tries += 1
            reasons.append(f"budget_retry(scale={scale:.3g},delta={actual:.4g})")
        if cap is not None and actual > cap * (1.0 + 1e-6):
            reasons.append(f"budget_unsatisfiable(delta={actual:.4g}>cap={cap:.4g})")
            if self._cap_is_session_remaining(cap):
                self._exhausted = True
            return self._reject_frozen("rollback", reasons)
        if not self._finite(self.working):
            return self._reject_frozen("rollback", reasons + ["nonfinite_candidate"])
        return self._accept("scale" if scale != 1.0 else "commit", reasons, scale)

    # ------------------------------------------------------------------ control
    def resume(self) -> None:
        """Lift a latched read-only state (after a verification pass by the caller)."""
        self.read_only = False
        self.read_only_reason = None
        self._exhausted = False

    def reset(self) -> None:
        self.committed = self.model.init_state(1, self.device)
        self.working = self.committed.clone()
        self.anchor = self.committed.clone()
        self.pending, self.pending_targets, self.pending_sources = [], [], []
        self.pending_loss, self.pending_signals = [], []
        self.pending_eligible = False
        self._last_logits = None
        self.history = {n: SignalHistory(self.hcfg.history_window) for n in STAT_SIGNALS}
        self.cusum = Cusum(self.hcfg.cusum_k, self._effective_cusum_h())
        self.budget_used = 0.0
        self._exhausted = False
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
            "pending_eligible": bool(self.pending_eligible),
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
        self.pending_eligible = bool(d.get("pending_eligible", False))
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


def fork_state_dict(d: dict[str, Any]) -> dict[str, Any]:
    """The runner state a fork starts from: the parent's committed state, no pending inputs,
    the harness history and controls carried over, and a fresh transaction count."""
    committed = d.get("committed", {})
    out = {
        "committed": committed,
        "working": committed,
        "anchor": d.get("anchor", committed),
        "pending": [],
        "pending_targets": [],
        "pending_sources": [],
        "pending_loss": [],
        "pending_signals": [],
        "pending_eligible": False,
        "last_logits": None,
        "history": d.get("history", {}),
        "cusum": d.get("cusum", {}),
        "budget_used": float(d.get("budget_used", 0.0)),
        "read_only": bool(d.get("read_only", False)),
        "read_only_reason": d.get("read_only_reason"),
        "n_transactions": 0,
    }
    return out
