"""Calibration: per-signal benign reference windows and thresholds, Fisher diagonal, canary baselines.

Produced once per model by running a benign stream through the transaction
runner in ``log_only`` mode. Thresholds are empirical upper quantiles at the
target false-positive rate, split across the decision signals by a union bound
so the total benign gating rate stays near ``target_fpr``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import Tensor

from plastic.harness.config import HarnessConfig
from plastic.harness.signals import STAT_SIGNALS
from plastic.harness.stats import quantile_threshold

ROLLBACK_DECISION_SIGNALS: tuple[str, ...] = STAT_SIGNALS + ("canary_delta_coherence",)


@dataclass
class Calibration:
    model_signature: str = ""
    n_chunks: int = 0
    reference: dict[str, list[float]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    canary_baseline: dict[str, float] = field(default_factory=dict)
    target_fpr: float = 0.01
    created_at_unix: int = 0
    fisher: list[Tensor] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_signature": self.model_signature,
            "n_chunks": self.n_chunks,
            "reference": {k: [float(x) for x in v] for k, v in self.reference.items()},
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            "canary_baseline": {k: float(v) for k, v in self.canary_baseline.items()},
            "target_fpr": self.target_fpr,
            "created_at_unix": self.created_at_unix,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Calibration":
        return cls(
            model_signature=str(d.get("model_signature", "")),
            n_chunks=int(d.get("n_chunks", 0)),
            reference={k: list(v) for k, v in d.get("reference", {}).items()},
            thresholds={k: float(v) for k, v in d.get("thresholds", {}).items()},
            canary_baseline={k: float(v) for k, v in d.get("canary_baseline", {}).items()},
            target_fpr=float(d.get("target_fpr", 0.01)),
            created_at_unix=int(d.get("created_at_unix", 0)),
        )

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        if self.fisher is not None:
            torch.save([t.detach().cpu() for t in self.fisher], os.path.join(model_dir, "fisher.pt"))

    @classmethod
    def load(cls, model_dir: str) -> "Calibration":
        with open(os.path.join(model_dir, "calibration.json"), "r", encoding="utf-8") as f:
            cal = cls.from_dict(json.load(f))
        fp = os.path.join(model_dir, "fisher.pt")
        if os.path.exists(fp):
            cal.fisher = torch.load(fp, map_location="cpu")
        return cal

    @staticmethod
    def exists(model_dir: str) -> bool:
        return os.path.exists(os.path.join(model_dir, "calibration.json"))


def thresholds_from_records(records: list[dict[str, Any]], *, target_fpr: float) -> dict[str, float]:
    """Upper quantile per decision signal, with the false-positive budget split by union bound."""
    present = [name for name in ROLLBACK_DECISION_SIGNALS if sum(1 for r in records if r.get(name) is not None) >= 20]
    if not present:
        return {}
    per_signal_fpr = max(1e-4, float(target_fpr) / len(present))
    out: dict[str, float] = {}
    for name in present:
        vals = [float(r[name]) for r in records if r.get(name) is not None]
        out[name] = quantile_threshold(vals, per_signal_fpr)
    return out


def calibrate_from_runner(
    runner,
    stream: Iterable[Any],
    *,
    n_chunks: int,
    model_signature: str,
    target_fpr: float = 0.01,
    fisher: list[Tensor] | None = None,
    reference_cap: int = 512,
    reset_every: int | None = 32,
) -> Calibration:
    """Run ``stream`` items through ``runner`` (which must be in log-only mode) and build a calibration.

    ``stream`` yields token-id lists (text) or ``(inputs (T, 7), targets (T, 4))`` tuples (physics).
    The runner is reset every ``reset_every`` chunks so the reference distribution covers
    fresh sessions as well as mature ones (a session that has just started has a
    different surprise and write profile from one that has absorbed context).
    """
    if not runner.hcfg.log_only:
        raise ValueError("calibration requires a runner in log_only mode")
    records: list[dict[str, Any]] = []
    since_reset = 0
    for item in stream:
        if isinstance(item, tuple):
            runner.feed_physics(item[0], item[1])
        else:
            runner.feed_tokens(list(item), source="user")
        while runner.transactions:
            rec = runner.transactions.pop(0)
            records.append(rec["signals"])
            since_reset += 1
        if reset_every is not None and since_reset >= reset_every:
            runner.flush()
            while runner.transactions:
                records.append(runner.transactions.pop(0)["signals"])
            runner.reset()
            since_reset = 0
        if len(records) >= n_chunks:
            break
    runner.flush()
    while runner.transactions:
        records.append(runner.transactions.pop(0)["signals"])
    if not records:
        raise ValueError("no chunks observed during calibration")
    reference = {
        name: [float(r[name]) for r in records if r.get(name) is not None][-reference_cap:]
        for name in ROLLBACK_DECISION_SIGNALS
    }
    reference = {k: v for k, v in reference.items() if v}
    baseline: dict[str, float] = {}
    for key in ("canary_coherence_before", "canary_poison_before"):
        vals = [float(r[key]) for r in records if r.get(key) is not None]
        if vals:
            baseline[key.replace("_before", "")] = sum(vals) / len(vals)
    return Calibration(
        model_signature=model_signature,
        n_chunks=len(records),
        reference=reference,
        thresholds=thresholds_from_records(records, target_fpr=target_fpr),
        canary_baseline=baseline,
        target_fpr=float(target_fpr),
        created_at_unix=int(time.time()),
        fisher=fisher,
    )


def log_only(cfg: HarnessConfig) -> HarnessConfig:
    return HarnessConfig.from_dict({**cfg.to_dict(), "log_only": True})


# ---------------------------------------------------------------------- model-level entry point
def calibrate_model(
    store,
    model_id: str,
    *,
    data_dir: str | None = None,
    n_chunks: int = 256,
    fisher_chunks: int = 64,
    target_fpr: float = 0.01,
    harness_cfg: HarnessConfig | None = None,
    device: torch.device | str = "cpu",
    seed: int = 0,
    log=print,
) -> Calibration:
    """Build the canary suite (if absent), estimate the Fisher diagonal, and calibrate thresholds
    for ``model_id`` on a benign stream: held-out windows for text (``data_dir/validation.bin``),
    generated episodes for physics. Writes ``canary.json``, ``calibration.json``, ``fisher.pt``.
    """
    from plastic.data.physics import physics_batch
    from plastic.data.text import TokenWindows
    from plastic.harness.canary import CanarySuite
    from plastic.harness.fisher import estimate_fisher_diag
    from plastic.harness.transaction import TransactionRunner

    device = torch.device(device)
    cfg, model, _ = store.load_checkpoint(model_id, device)
    model_dir = store.model_dir(model_id)
    hcfg = log_only(harness_cfg or HarnessConfig(target_fpr=target_fpr))
    L = cfg.chunk
    canary_path = store.canary_path(model_id)
    g = torch.Generator().manual_seed(int(seed))

    if cfg.domain == "text":
        if not data_dir:
            raise ValueError("text calibration needs data_dir with validation.bin")
        heldout_path = os.path.join(data_dir, "validation.bin")
        if os.path.exists(canary_path):
            suite = CanarySuite.load(canary_path)
        else:
            suite = CanarySuite.default_text(heldout_path, vocab_size=cfg.vocab_size, seed=seed)
            suite.save(canary_path)
        windows = TokenWindows(heldout_path, seq_len=L * 4)
        canary_tokens = 3 * 128  # skip the region the coherence canaries were cut from

        def stream():
            # sessions of 4 chunks each, starting after the canary region, non-overlapping
            start = canary_tokens
            while True:
                if start + L * 4 + 1 > len(windows):
                    start = canary_tokens
                seq = torch.from_numpy(windows.data[start : start + L * 4].astype("int64")).tolist()
                start += L * 4
                yield seq

        def fisher_seqs():
            for _ in range(fisher_chunks):
                yield windows.sample(4, g)

        log(f"[calibrate] {model_id}: text, {n_chunks} chunks from {heldout_path}")
        fisher = estimate_fisher_diag(model, fisher_seqs(), chunk=L, n_chunks=fisher_chunks, device=device)
    else:
        if os.path.exists(canary_path):
            suite = CanarySuite.load(canary_path)
        else:
            suite = CanarySuite.default_physics(seed=seed, steps=L)
            suite.save(canary_path)

        def stream():
            while True:
                b = physics_batch(1, seq_len=L * 4, episodes_per_seq=2, mu_range=(0.02, 0.25), nonlinear=False, action_std=0.5, rng=g)
                yield (b.inputs[0], b.target_delta[0])

        def fisher_seqs():
            for _ in range(fisher_chunks):
                b = physics_batch(4, seq_len=L * 4, episodes_per_seq=2, mu_range=(0.02, 0.25), nonlinear=False, action_std=0.5, rng=g)
                yield (b.inputs, b.target_delta)

        log(f"[calibrate] {model_id}: physics, {n_chunks} chunks of generated episodes")
        fisher = estimate_fisher_diag(model, fisher_seqs(), chunk=L, n_chunks=fisher_chunks, device=device)

    runner = TransactionRunner(model, cfg, hcfg, calibration=None, suite=suite, device=device)
    cal = calibrate_from_runner(
        runner, stream(), n_chunks=n_chunks, model_signature=store.model_signature(model_id),
        target_fpr=target_fpr, fisher=fisher, reset_every=4,
    )
    cal.save(model_dir)
    store.register_model(model_id, {"calibrated_at_unix": cal.created_at_unix, "calibration_chunks": cal.n_chunks})
    log(f"[calibrate] thresholds: " + ", ".join(f"{k}={v:.4g}" for k, v in cal.thresholds.items()))
    return cal
