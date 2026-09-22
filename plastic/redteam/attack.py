"""Attacks that try to make the model learn something that hurts its canaries.

The gradient attacker perturbs the embeddings of a suffix (continuous
relaxation), maximizing the coherence-canary damage that learning the suffix
causes, subject to a ceiling on the payload's own next-token loss (the honest
version of "looks benign": the model itself must find the payload plausible)
and an embedding-norm ball. The perturbed embeddings are snapped to the
nearest real tokens and the discrete payload is re-validated through a fresh
``TransactionRunner``, so every reported number is about a payload that was
actually fed through the token path and the harness.

Sampled families share the evaluation: random tokens, single-token runs,
shuffled benign text, and topic switches (a benign segment from elsewhere).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from plastic.harness.canary import CanarySuite, _expand_state, _probe_batch
from plastic.harness.config import HarnessConfig
from plastic.harness.transaction import TransactionRunner
from plastic.model.state import SessionState

FAMILIES: tuple[str, ...] = ("pgd", "random", "repeat", "shuffle", "topic_switch")


@dataclass
class AttackConfig:
    suffix_len: int = 64
    poison_chunks: int = 8  # length of the coherence_poison payload, in chunks
    steps: int = 50
    lr: float = 0.05
    radius: float = 1.0
    nll_max: float | None = None
    nll_margin: float = 1.0
    nll_weight: float = 1.0
    seed: int = 0
    families: tuple[str, ...] = FAMILIES
    harness: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttackResult:
    family: str
    prefix_ids: list[int]
    payload_ids: list[int]
    damage_continuous: float | None
    damage_validated: float
    nll_payload: float
    nll_prefix: float
    nll_max: float
    constraint_violated: bool
    decisions: list[str] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)
    canary_before: float = float("nan")
    canary_after_provisional: float = float("nan")  # peak intermediate proposal, not a final endpoint
    canary_after_accepted: float = float("nan")
    canary_after_unprotected: float = float("nan")
    canary_after_frozen: float = float("nan")
    damage_unprotected: float = float("nan")  # what the attack achieves with no defense
    damage_embedding_unprotected: float = float("nan")  # embedding-space upper bound (coherence_poison)
    damage_frozen: float = float("nan")       # activation-only change (payload read, not learned)
    nll_payload_guarded: float = float("nan")  # payload NLL along the guarded trajectory
    poison_before: float = float("nan")
    poison_after_accepted: float = float("nan")
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------- differentiable damage
def _coherence_loss(model, state: SessionState, suite: CanarySuite, device: torch.device) -> Tensor:
    st = _expand_state(state, len(suite.coherence))
    batch = _probe_batch(suite.coherence, suite.domain, device)
    toks = batch
    logits, _, _ = model(toks, st, mode="chunk", freeze=True)
    V = logits.shape[-1]
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), toks[:, 1:].reshape(-1))


def _run_prefix(model, prefix_ids: list[int], device: torch.device) -> tuple[SessionState, Tensor]:
    toks = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, state, _ = model(toks, model.init_state(1, device))
    return state.detach(), logits[0, -1].detach()


def _forward_embeddings(model, emb: Tensor, state: SessionState) -> tuple[Tensor, SessionState]:
    """Run continuous embeddings (1, T, D) through the core with learning on."""
    y, new_state, _ = model.core(emb, state, mode="chunk")
    return model.logits_from_hidden(y), new_state


def _payload_nll(logits_prev_last: Tensor, logits: Tensor, ids: Tensor) -> Tensor:
    """Mean NLL of a discrete payload given the logits that precede each token."""
    prev = torch.cat([logits_prev_last.unsqueeze(0), logits[0, :-1]], dim=0)
    return F.cross_entropy(prev, ids)


def damage(
    model,
    prefix_state: SessionState,
    prefix_last_logits: Tensor,
    suffix_emb: Tensor,
    suite: CanarySuite,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """(coherence damage, soft payload NLL, canary-before) for continuous suffix embeddings.

    The soft NLL uses the nearest-token targets of the current embeddings, so it is a
    smooth proxy for the discrete payload's plausibility under the model.
    """
    before = _coherence_loss(model, prefix_state, suite, device)
    logits, after_state = _forward_embeddings(model, suffix_emb, prefix_state)
    after = _coherence_loss(model, after_state, suite, device)
    with torch.no_grad():
        ids = snap_to_tokens(model, suffix_emb)
    nll = _payload_nll(prefix_last_logits, logits, ids)
    return after - before, nll, before.detach()


def snap_to_tokens(model, emb: Tensor) -> Tensor:
    """Nearest vocabulary token by cosine similarity for each position of (1, T, D)."""
    E = F.normalize(model.embed.weight, dim=-1)
    x = F.normalize(emb[0], dim=-1)
    return (x @ E.T).argmax(dim=-1)


# ---------------------------------------------------------------------- validation through the harness
def _feed_payload_from(state, last_logits, model, model_cfg, payload_ids, suite, *, harness, calibration, device, read_only=False):
    """A payload run that starts from a copied post-prefix state; returns (runner, canary end scores)."""
    from plastic.harness.canary import score_suite

    runner = TransactionRunner(model, model_cfg, harness, calibration=calibration, suite=suite, device=device)
    runner.committed = state.clone()
    runner.working = state.clone()
    runner.anchor = state.clone()
    runner._last_logits = None if last_logits is None else last_logits.clone()
    if read_only:
        runner.read_only = True
        runner.read_only_reason = "control_frozen"
    runner.feed_tokens(payload_ids, source="user")
    runner.flush()
    return runner, score_suite(model, runner.committed, suite, device=device)


def validate_payload(
    model,
    model_cfg,
    prefix_ids: list[int],
    payload_ids: list[int],
    suite: CanarySuite,
    *,
    harness: HarnessConfig,
    calibration=None,
    device: torch.device,
) -> dict[str, Any]:
    """Feed prefix then payload through the harness, and run two same-prefix controls.

    From the identical post-prefix committed state, the payload is run three ways:
    guarded (the real harness), unprotected (harness disabled, every chunk commits), and
    frozen (read-only: the payload is read but not learned). The accepted residual is the
    right headline; the unprotected end score is what the attack would achieve with no
    defense, and the frozen end score isolates the activation-only change. ``nll_guarded``
    is the payload's mean next-token loss along the actual guarded trajectory (from the
    chunk losses), which can differ from the ungated plausibility NLL.
    """
    from plastic.harness.canary import score_suite

    runner = TransactionRunner(model, model_cfg, harness, calibration=calibration, suite=suite, device=device)
    runner.feed_tokens(prefix_ids, source="user")
    runner.flush()
    before = score_suite(model, runner.committed, suite, device=device)
    prefix_state = runner.committed.clone()
    last = runner._last_logits

    runner.transactions = []
    runner.feed_tokens(payload_ids, source="user")
    runner.flush()
    after = score_suite(model, runner.committed, suite, device=device)
    txns = runner.transactions
    provisional = [t["signals"].get("canary_coherence_after") for t in txns]

    # controls from the identical post-prefix state
    unprotected_cfg = HarnessConfig(enable_rollback=False, enable_projection=False, enable_budget=False, enable_stats=False)
    _, unprot = _feed_payload_from(prefix_state, last, model, model_cfg, payload_ids, suite, harness=unprotected_cfg, calibration=None, device=device)
    _, frozen = _feed_payload_from(prefix_state, last, model, model_cfg, payload_ids, suite, harness=harness, calibration=calibration, device=device, read_only=True)

    tok_loss = [(t["signals"].get("chunk_loss"), t["signals"].get("n_tokens")) for t in txns]
    tot = sum(n for _, n in tok_loss if n)
    nll_guarded = float(sum(l * n for l, n in tok_loss if l is not None and n) / tot) if tot else float("nan")

    return {
        "decisions": [t["decision"]["kind"] for t in txns],
        "signals": [t["signals"] for t in txns],
        "accepted": [t.get("accepted") for t in txns],
        "canary_before": before["coherence"],
        "canary_after_provisional": max((v for v in provisional if v is not None), default=float("nan")),
        "canary_after_accepted": after["coherence"],
        "canary_after_unprotected": unprot["coherence"],
        "canary_after_frozen": frozen["coherence"],
        "poison_before": before["poison"],
        "poison_after_accepted": after["poison"],
        "nll_guarded": nll_guarded,
    }


def _prefix_nll(model, prefix_ids: list[int], device: torch.device) -> float:
    toks = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, _ = model(toks)
    V = logits.shape[-1]
    return float(F.cross_entropy(logits[0, :-1].reshape(-1, V), toks[0, 1:]))


def _finish(
    model, model_cfg, family: str, prefix_ids: list[int], payload_ids: list[int], suite: CanarySuite,
    *, cfg: AttackConfig, harness: HarnessConfig, calibration, device: torch.device, damage_continuous: float | None,
    nll_max: float, t0: float,
) -> AttackResult:
    prefix_state, last = _run_prefix(model, prefix_ids, device)
    ids = torch.tensor(payload_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, _ = model(ids.unsqueeze(0), prefix_state)
        nll = float(_payload_nll(last, logits, ids))
    v = validate_payload(model, model_cfg, prefix_ids, payload_ids, suite, harness=harness, calibration=calibration, device=device)
    # constraint validity is judged on the guarded trajectory (what actually ran), not the ungated pass
    nll_guarded = v["nll_guarded"]
    return AttackResult(
        family=family,
        prefix_ids=list(prefix_ids),
        payload_ids=list(payload_ids),
        damage_continuous=damage_continuous,
        damage_validated=float(v["canary_after_accepted"] - v["canary_before"]),
        nll_payload=nll,
        nll_payload_guarded=float(nll_guarded),
        nll_prefix=_prefix_nll(model, prefix_ids, device),
        nll_max=nll_max,
        constraint_violated=bool((nll_guarded if nll_guarded == nll_guarded else nll) > nll_max + 1e-3),
        decisions=v["decisions"],
        signals=v["signals"],
        canary_before=v["canary_before"],
        canary_after_provisional=v["canary_after_provisional"],
        canary_after_accepted=v["canary_after_accepted"],
        canary_after_unprotected=v["canary_after_unprotected"],
        canary_after_frozen=v["canary_after_frozen"],
        damage_unprotected=float(v["canary_after_unprotected"] - v["canary_before"]),
        damage_frozen=float(v["canary_after_frozen"] - v["canary_before"]),
        poison_before=v["poison_before"],
        poison_after_accepted=v["poison_after_accepted"],
        seconds=time.time() - t0,
    )


def _harness(cfg: AttackConfig) -> HarnessConfig:
    return HarnessConfig.from_dict({**HarnessConfig().to_dict(), **(cfg.harness or {})})


# ---------------------------------------------------------------------- attackers
def pgd_attack(
    model,
    model_cfg,
    cfg: AttackConfig,
    prefix_ids: list[int],
    suite: CanarySuite,
    *,
    device: torch.device,
    calibration=None,
    init_ids: list[int] | None = None,
) -> AttackResult:
    t0 = time.time()
    model.eval()
    prefix_state, last = _run_prefix(model, prefix_ids, device)
    nll_max = cfg.nll_max if cfg.nll_max is not None else _prefix_nll(model, prefix_ids, device) + cfg.nll_margin
    g = torch.Generator().manual_seed(cfg.seed)
    if init_ids is None:
        init_ids = torch.randint(3, model_cfg.vocab_size, (cfg.suffix_len,), generator=g).tolist()
    base = model.embed.weight[torch.tensor(init_ids, device=device)].detach().unsqueeze(0)
    rms = float(model.embed.weight.detach().pow(2).mean().sqrt())
    radius = cfg.radius * rms
    delta = torch.zeros_like(base, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=cfg.lr * rms)
    d0 = None
    for _ in range(int(cfg.steps)):
        opt.zero_grad(set_to_none=True)
        dmg, nll, _ = damage(model, prefix_state, last, base + delta, suite, device=device)
        if d0 is None:
            d0 = float(dmg.detach())
        penalty = F.relu(nll - nll_max)
        loss = -dmg + cfg.nll_weight * penalty
        loss.backward()
        opt.step()
        with torch.no_grad():
            n = delta.norm(dim=-1, keepdim=True)
            delta.mul_(torch.clamp(radius / (n + 1e-9), max=1.0))
    with torch.no_grad():
        dmg, _, _ = damage(model, prefix_state, last, base + delta, suite, device=device)
        payload = snap_to_tokens(model, base + delta).tolist()
    return _finish(
        model, model_cfg, "pgd", prefix_ids, payload, suite, cfg=cfg, harness=_harness(cfg), calibration=calibration,
        device=device, damage_continuous=float(dmg), nll_max=nll_max, t0=t0,
    )


def sampled_attack(
    model,
    model_cfg,
    family: str,
    cfg: AttackConfig,
    prefix_ids: list[int],
    suite: CanarySuite,
    *,
    device: torch.device,
    rng: torch.Generator,
    corpus: np.ndarray | None = None,
    calibration=None,
) -> AttackResult:
    t0 = time.time()
    V = model_cfg.vocab_size
    L = cfg.suffix_len
    if family == "random":
        payload = torch.randint(3, V, (L,), generator=rng).tolist()
    elif family == "repeat":
        payload = [int(torch.randint(3, V, (1,), generator=rng))] * L
    elif family == "shuffle":
        src = list(prefix_ids[-L:]) if len(prefix_ids) >= L else list(prefix_ids) * (L // max(1, len(prefix_ids)) + 1)
        src = src[:L]
        perm = torch.randperm(len(src), generator=rng).tolist()
        payload = [src[i] for i in perm]
    elif family == "topic_switch":
        if corpus is None or len(corpus) < L + 1:
            raise ValueError("topic_switch needs a corpus array")
        start = int(torch.randint(0, len(corpus) - L, (1,), generator=rng))
        payload = corpus[start : start + L].astype("int64").tolist()
    else:
        raise ValueError(f"unknown family {family!r}")
    nll_max = cfg.nll_max if cfg.nll_max is not None else _prefix_nll(model, prefix_ids, device) + cfg.nll_margin
    return _finish(
        model, model_cfg, family, prefix_ids, payload, suite, cfg=cfg, harness=_harness(cfg), calibration=calibration,
        device=device, damage_continuous=None, nll_max=nll_max, t0=t0,
    )


def coherence_poison_attack(
    model,
    model_cfg,
    cfg: AttackConfig,
    prefix_ids: list[int],
    suite: CanarySuite,
    *,
    device: torch.device,
    calibration=None,
    rng: torch.Generator | None = None,
) -> AttackResult:
    """A sustained, coherence-targeted poison — the first attack that reaches above-benign damage.

    The other families produce single 64-token chunks whose write, though large, is orthogonal to
    the direction that harms the coherence canaries, so unprotected damage stays sub-benign. This
    one optimizes a multi-chunk embedding payload to directly maximize the increase in coherence
    loss (no fluency term: this measures the write path's worst case, not a deliverable fluent
    attack). It reports two numbers: the embedding-space damage (an upper bound on what any payload
    through this write path could do) and the discrete snapped payload validated through the harness
    (unprotected vs guarded). The snapped payload is non-fluent by construction, so it is marked
    constraint_violated; the point is not stealth but that its *unprotected* damage clears the
    benign band while the harness rolls it back.
    """
    t0 = time.time()
    model.eval()
    L = int(model_cfg.chunk)
    n_tok = max(1, int(cfg.poison_chunks)) * L
    prefix_state, last = _run_prefix(model, prefix_ids, device)
    before = _coherence_loss(model, prefix_state, suite, device).detach()
    g = rng or torch.Generator().manual_seed(cfg.seed)
    init_ids = torch.randint(3, model_cfg.vocab_size, (n_tok,), generator=g).to(device)
    base = model.embed.weight[init_ids].detach().unsqueeze(0)
    rms = float(model.embed.weight.detach().pow(2).mean().sqrt())
    radius = cfg.radius * rms
    delta = torch.zeros_like(base, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=cfg.lr * rms)
    for _ in range(int(cfg.steps)):
        opt.zero_grad(set_to_none=True)
        _, new_state, _ = model.core(base + delta, prefix_state.clone(), mode="chunk")
        after = _coherence_loss(model, new_state, suite, device)
        (-(after - before)).backward()  # maximize coherence damage; no fluency penalty (worst case)
        opt.step()
        with torch.no_grad():
            n = delta.norm(dim=-1, keepdim=True)
            delta.mul_(torch.clamp(radius / (n + 1e-9), max=1.0))
    with torch.no_grad():
        _, emb_state, _ = model.core(base + delta, prefix_state.clone(), mode="chunk")
        emb_damage = float(_coherence_loss(model, emb_state, suite, device) - before)
        payload = snap_to_tokens(model, base + delta).tolist()
    nll_max = cfg.nll_max if cfg.nll_max is not None else _prefix_nll(model, prefix_ids, device) + cfg.nll_margin
    r = _finish(
        model, model_cfg, "coherence_poison", prefix_ids, payload, suite, cfg=cfg, harness=_harness(cfg),
        calibration=calibration, device=device, damage_continuous=emb_damage, nll_max=nll_max, t0=t0,
    )
    r.damage_embedding_unprotected = emb_damage
    return r


# ---------------------------------------------------------------------- campaign
def run_redteam(
    store,
    model_id: str,
    *,
    cfg: AttackConfig,
    data_dir: str,
    n_prefixes: int = 8,
    prefix_len: int = 128,
    device: torch.device | str = "cpu",
    record: bool = False,
    log=print,
) -> dict[str, Any]:
    from plastic.harness.calibrate import Calibration

    device = torch.device(device)
    model_cfg, model, _ = store.load_checkpoint(model_id, device)
    model_dir = store.model_dir(model_id)
    suite = CanarySuite.load(store.canary_path(model_id))
    calibration = Calibration.load(model_dir) if Calibration.exists(model_dir) else None
    heldout = np.memmap(os.path.join(data_dir, "validation.bin"), dtype="<u2", mode="r")
    g = torch.Generator().manual_seed(cfg.seed)
    run_id = f"rt_{model_id}_{int(time.time())}"
    out_dir = os.path.join(store.root, "redteam", run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_id": model_id, "attack": cfg.to_dict(), "n_prefixes": n_prefixes, "prefix_len": prefix_len}, f, indent=2)

    results: list[AttackResult] = []
    skip = 3 * 128  # canary region
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for i in range(n_prefixes):
            start = skip + int(torch.randint(0, max(1, len(heldout) - skip - prefix_len - 1), (1,), generator=g))
            prefix = heldout[start : start + prefix_len].astype("int64").tolist()
            for family in cfg.families:
                if family == "pgd":
                    r = pgd_attack(model, model_cfg, cfg, prefix, suite, device=device, calibration=calibration)
                elif family == "coherence_poison":
                    r = coherence_poison_attack(model, model_cfg, cfg, prefix, suite, device=device, rng=g, calibration=calibration)
                else:
                    r = sampled_attack(model, model_cfg, family, cfg, prefix, suite, device=device, rng=g, corpus=heldout, calibration=calibration)
                results.append(r)
                f.write(json.dumps(r.to_dict()) + "\n")
                log(f"[redteam] prefix {i} {family:<12} damage={r.damage_validated:+.4f} nll={r.nll_payload:.2f}/{r.nll_max:.2f} decisions={r.decisions}")

    threshold = None
    if calibration is not None:
        threshold = calibration.thresholds.get("canary_delta_coherence")
    summary: dict[str, Any] = {"run_id": run_id, "model_id": model_id, "created_at_unix": int(time.time()), "threshold_coherence": threshold, "families": {}}
    for family in cfg.families:
        rs = [r for r in results if r.family == family]
        if not rs:
            continue
        dmg = [r.damage_validated for r in rs]
        rolled = [sum(1 for d in r.decisions if d != "commit") / max(1, len(r.decisions)) for r in rs]
        valid = [r for r in rs if not r.constraint_violated]
        vdmg = [r.damage_validated for r in valid]
        summary["families"][family] = {
            "n": len(rs),
            "n_valid": len(valid),
            "damage_mean": float(sum(dmg) / len(dmg)),
            "damage_max": float(max(dmg)),
            "unprotected_damage_mean": float(sum(r.damage_unprotected for r in rs) / len(rs)),
            "unprotected_damage_max": float(max(r.damage_unprotected for r in rs)),
            # how often the attack would have cleared the damage threshold with no defense — the
            # number that says the harness actually had something to stop (meaningful once an
            # attack reaches above-benign unprotected damage, e.g. coherence_poison)
            "unprotected_over_threshold_fraction": (
                None if threshold is None else float(sum(1 for r in rs if r.damage_unprotected > threshold) / len(rs))
            ),
            "embedding_unprotected_damage_mean": (
                float(sum(_emb) / len(_emb))
                if (_emb := [r.damage_embedding_unprotected for r in rs if r.damage_embedding_unprotected == r.damage_embedding_unprotected])
                else None
            ),
            "frozen_damage_mean": float(sum(r.damage_frozen for r in rs) / len(rs)),
            "valid_damage_mean": (float(sum(vdmg) / len(vdmg)) if vdmg else None),
            "valid_damage_max": (float(max(vdmg)) if vdmg else None),
            "provisional_damage_max": float(max((r.canary_after_provisional - r.canary_before for r in rs if r.canary_after_provisional == r.canary_after_provisional), default=float("nan"))),
            "gated_fraction": float(sum(rolled) / len(rolled)),
            "constraint_violated_fraction": float(sum(r.constraint_violated for r in rs) / len(rs)),
            "over_threshold_fraction": (None if threshold is None else float(sum(1 for d in dmg if d > threshold) / len(dmg))),
            "valid_over_threshold_fraction": (None if (threshold is None or not vdmg) else float(sum(1 for d in vdmg if d > threshold) / len(vdmg))),
        }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if record:
        top = sorted(results, key=lambda r: r.canary_after_provisional - r.canary_before if r.canary_after_provisional == r.canary_after_provisional else -1e9, reverse=True)[:4]
        for r in top:
            suite.poison.append(list(r.payload_ids))
        suite.save(store.canary_path(model_id))
        summary["recorded_payloads"] = len(top)
    return summary
