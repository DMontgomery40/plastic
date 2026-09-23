"""The learning contract: what counts as a durable, transferable change in competence.

A ``Learner`` exposes its lasting (slow) state through ``snapshot_slow``/``restore_slow``,
consumes an experience stream through ``consume`` (the lasting update, which may refuse),
and reports per-step MSE from a fresh state through ``step_mse``. ``run_contract`` measures,
on fixed seeds so before and after see identical inputs:

1. transfer: MSE on held-out mechanism combinations under held-out intervention policies,
   with and without fast adaptation;
2. speed: the adapting error as a fraction of the no-adaptation error, by step within the
   episode;
3. forgetting: MSE on the training distribution (backward transfer with MSE);
4. correction: transfer after a poisoned stream (harm) and after a corrective clean stream;
5. revert: restoring the pre-stream slow state must reproduce the before measurements.

Measurement rows hold exactly ONE episode each, so fast state fitted to one world is never
carried into another inside a measurement, and every scored episode must be long enough for
at least one fast-update boundary to fall inside it (a learner declares its ``update_period``;
the contract refuses a spec that cannot measure adaptation). The experience stream, by
contrast, packs many episodes into one row: that is what a stream of experience is.

Every measurement is taken after the activation state and fast parameters are cleared (a
fresh state per call) and with the stream removed, so relearning from context cannot count
as having learned. The stream is checked against the split (training combinations and
training policies only), the learner is told the split through ``bind_split`` if it has
development data of its own, and measurement worlds are checked to be disjoint from stream
worlds. Acceptance is a pair of rates, accepted-good and refused-bad; an empty side is
``None``, never zero.

Spec: docs/superpowers/specs/2026-09-23-mechanism-testbed-and-contract.md
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from plastic.data.mechanisms import (
    HELDOUT_POLICIES,
    TRAIN_POLICIES,
    MechanismBatch,
    World,
    mechanism_batch,
    poison_stream,
    split_combinations,
)

CONTRACT_VERSION = "2026-09-23.2"


class Learner(Protocol):
    def snapshot_slow(self) -> Any: ...

    def restore_slow(self, snapshot: Any) -> None: ...

    def consume(self, stream: MechanismBatch) -> dict[str, Any]: ...

    def step_mse(self, batch: MechanismBatch, *, adapt: bool) -> Tensor: ...

    # optional: the number of steps between fast-update boundaries (1 for per-token rules);
    # optional: bind_split(train_combos, train_policies) for learners with development data.


@dataclass
class ContractSpec:
    k: int = 2
    n_heldout: int = 5
    split_seed: int = 0
    seq_len: int = 64  # one scored episode per measurement row
    eval_batch: int = 8
    stream_episodes: int = 16  # the experience stream packs this many episodes into one row
    probe_steps: int | None = None  # None: the whole episode
    poison_bias: float = 0.5
    revert_tolerance: float = 1e-6
    heldout_policies: tuple[str, ...] = HELDOUT_POLICIES
    train_policies: tuple[str, ...] = TRAIN_POLICIES

    def __post_init__(self) -> None:
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if self.probe_steps is not None and not (1 <= self.probe_steps <= self.seq_len):
            raise ValueError("probe_steps must be in [1, seq_len]")
        if self.stream_episodes < 1:
            raise ValueError("stream_episodes must be at least 1")

    @property
    def episode_len(self) -> int:
        return self.seq_len


_OFFSETS = {"heldout": 100, "speed": 200, "train": 300, "stream_clean": 400, "stream_correct": 500}


def _gen(seed: int, tag: str, index: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) * 10_000 + _OFFSETS[tag] + index)


def _mean(t: Tensor) -> float:
    return float(t.float().mean())


def world_key(w: World) -> str:
    """A stable identity for a sampled world: its combination and rounded parameters."""
    parts = []
    for name in sorted(w.active):
        v = w.params[name]
        vals = [float(x) for x in torch.as_tensor(v).flatten()]
        parts.append((name, tuple(round(x, 6) for x in vals)))
    return json.dumps(parts)


def split_id(split: tuple[list, list]) -> str:
    train, heldout = split
    return hashlib.sha256(json.dumps({"train": train, "heldout": heldout}, sort_keys=True).encode()).hexdigest()[:16]


def check_stream_in_split(stream: MechanismBatch, train: list[tuple[str, ...]], train_policies: tuple[str, ...]) -> None:
    """Every world in an experience stream must be a training combination, and its policy a
    training policy. Refuses otherwise, so held-out pairs can never leak into a lasting update."""
    allowed = {tuple(c) for c in train}
    for row in stream.worlds:
        for w in row:
            if w.combination() not in allowed:
                raise ValueError(f"stream world {w.combination()} is not a training combination")
    if stream.policy not in train_policies:
        raise ValueError(f"stream policy {stream.policy!r} is not a training policy {train_policies}")


def _one_episode_batch(spec: ContractSpec, combos: list[tuple[str, ...]], policy: str, rng: torch.Generator) -> MechanismBatch:
    return mechanism_batch(spec.eval_batch, seq_len=spec.seq_len, episodes_per_seq=1, combos=combos, policy=policy, rng=rng)


def measure(learner: Learner, spec: ContractSpec, split: tuple[list, list], seed: int) -> dict[str, Any]:
    """One set of measurements from fixed seeds. Calling twice on an unchanged learner returns
    identical numbers. Every row is one episode from a fresh state."""
    train, heldout = split
    out: dict[str, Any] = {"transfer": {}, "speed": {}, "forgetting": {}, "tokens": 0, "worlds": []}
    for i, policy in enumerate(spec.heldout_policies):
        b = _one_episode_batch(spec, heldout, policy, _gen(seed, "heldout", i))
        adapt = learner.step_mse(b, adapt=True)
        frozen = learner.step_mse(b, adapt=False)
        out["transfer"][policy] = {"adapt": _mean(adapt), "no_adapt": _mean(frozen), "elements": int(b.target_delta.numel())}
        out["tokens"] += 2 * b.tokens
        out["worlds"] += [world_key(w) for row in b.worlds for w in row]
    # adaptation speed: at each within-episode step, the adapting error as a fraction of the
    # no-adaptation error on the same inputs and matched activation state. 1.0 means the fast
    # path has removed nothing yet; the step where it first drops below 0.5 is the half-life.
    b = _one_episode_batch(spec, heldout, "gaussian", _gen(seed, "speed"))
    mse = learner.step_mse(b, adapt=True).float()
    mse0 = learner.step_mse(b, adapt=False).float()
    out["tokens"] += 2 * b.tokens
    out["worlds"] += [world_key(w) for row in b.worlds for w in row]
    by_pos = mse.mean(dim=0)
    by_pos0 = mse0.mean(dim=0)
    n = spec.probe_steps or spec.episode_len
    curve = [float(a) / float(z) if float(z) > 0 else 1.0 for a, z in zip(by_pos[:n], by_pos0[:n])]
    steps_to_half = next((i + 1 for i, v in enumerate(curve) if v < 0.5), n)
    out["speed"] = {"curve": curve, "area": sum(curve) / len(curve), "steps_to_half": steps_to_half, "episodes": int(mse.shape[0])}
    fb = _one_episode_batch(spec, train, spec.train_policies[0], _gen(seed, "train"))
    out["forgetting"] = {"adapt": _mean(learner.step_mse(fb, adapt=True)), "no_adapt": _mean(learner.step_mse(fb, adapt=False)), "elements": int(fb.target_delta.numel())}
    out["tokens"] += 2 * fb.tokens
    out["worlds"] += [world_key(w) for row in fb.worlds for w in row]
    return out


def _leaves(d: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_leaves(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_leaves(v, f"{prefix}{i}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix.rstrip(".")] = float(d)
    return out


def _max_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    la, lb = _leaves(a), _leaves(b)
    keys = set(la) & set(lb)
    keys.discard("tokens")
    return max((abs(la[k] - lb[k]) for k in keys), default=0.0)


def make_stream(spec: ContractSpec, split: tuple[list, list], seed: int, *, tag: str) -> MechanismBatch:
    train, _ = split
    return mechanism_batch(
        1, seq_len=spec.stream_episodes * spec.episode_len, episodes_per_seq=spec.stream_episodes,
        combos=train, policy=spec.train_policies[0], rng=_gen(seed, tag),
    )


def acceptance_rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Accepted-good and refused-bad rates over decisions with a recorded outcome. A side with
    no decisions is ``None``: a missing aggregate is not zero."""
    good = [r for r in records if r.get("beneficial") and r.get("accepted") is not None]
    bad = [r for r in records if not r.get("beneficial") and r.get("accepted") is not None]
    return {
        "accepted_good": (sum(1 for r in good if r["accepted"]) / len(good)) if good else None,
        "refused_bad": (sum(1 for r in bad if not r["accepted"]) / len(bad)) if bad else None,
        "n_good": len(good),
        "n_bad": len(bad),
    }


def adaptation_window(spec: ContractSpec, update_period: int | None) -> dict[str, Any]:
    """How many fast-update boundaries fall strictly inside one scored episode. Refuses a spec
    under which adaptation could not be measured at all."""
    if update_period is None:
        return {"update_period": None, "boundaries_per_episode": None, "checked": False}
    if update_period < 1:
        raise ValueError("update_period must be a positive number of steps")
    boundaries = (spec.episode_len - 1) // update_period
    if boundaries < 1:
        raise ValueError(
            f"no fast-update boundary can fall inside a scored episode: episode length {spec.episode_len} "
            f"with update period {update_period}; lengthen seq_len or shorten the learner's period"
        )
    return {"update_period": update_period, "boundaries_per_episode": boundaries, "checked": True}


def run_contract(learner: Learner, spec: ContractSpec, *, seed: int = 0) -> dict[str, Any]:
    t0 = time.time()
    split = split_combinations(k=spec.k, n_heldout=spec.n_heldout, seed=spec.split_seed)
    train, heldout = split
    window = adaptation_window(spec, getattr(learner, "update_period", lambda: None)())
    bind = getattr(learner, "bind_split", None)
    if bind is not None:
        bind(list(train), tuple(spec.train_policies))

    before = measure(learner, spec, split, seed)
    snapshot = learner.snapshot_slow()

    clean = make_stream(spec, split, seed, tag="stream_clean")
    corrective = make_stream(spec, split, seed, tag="stream_correct")
    for s in (clean, corrective):
        check_stream_in_split(s, train, spec.train_policies)
    stream_worlds = {world_key(w) for s in (clean, corrective) for row in s.worlds for w in row}
    overlap = stream_worlds & set(before["worlds"])
    if overlap:
        raise ValueError(f"{len(overlap)} measurement worlds also appear in the experience stream")

    rec_clean = learner.consume(clean)
    after = measure(learner, spec, split, seed)

    learner.restore_slow(snapshot)
    reverted = measure(learner, spec, split, seed)
    gap = _max_gap(reverted, before)

    poisoned = poison_stream(clean, bias=spec.poison_bias)
    rec_poison = learner.consume(poisoned)
    after_poison = measure(learner, spec, split, seed)
    rec_correct = learner.consume(corrective)
    after_correction = measure(learner, spec, split, seed)
    learner.restore_slow(snapshot)

    def _avg_transfer(m: dict[str, Any]) -> float:
        rows = m["transfer"].values()
        return sum(r["adapt"] for r in rows) / len(m["transfer"])

    transfer = {
        policy: {
            "before": before["transfer"][policy],
            "after": after["transfer"][policy],
            "delta_mse": after["transfer"][policy]["adapt"] - before["transfer"][policy]["adapt"],
            "delta_mse_no_adapt": after["transfer"][policy]["no_adapt"] - before["transfer"][policy]["no_adapt"],
            "elements": before["transfer"][policy]["elements"],
        }
        for policy in before["transfer"]
    }
    clean_after = _avg_transfer(after)
    harm = _avg_transfer(after_poison) - clean_after
    residual = _avg_transfer(after_correction) - clean_after
    records = [
        {"beneficial": True, "accepted": rec_clean.get("accepted"), "stream": "clean"},
        {"beneficial": False, "accepted": rec_poison.get("accepted"), "stream": "poisoned"},
        {"beneficial": True, "accepted": rec_correct.get("accepted"), "stream": "corrective"},
    ]
    parameter_count = getattr(learner, "parameter_count", lambda: None)()
    tokens_consumed = 3 * clean.tokens
    plain = before["tokens"] + after["tokens"] + reverted["tokens"] + after_poison["tokens"] + after_correction["tokens"]
    tokens_measured = plain + int(getattr(learner, "context_tokens_measured", 0))
    for m in (before, after, reverted, after_poison, after_correction):
        m.pop("worlds", None)
    return {
        "contract_version": CONTRACT_VERSION,
        "spec": asdict(spec),
        "split": {"id": split_id(split), "train": [list(c) for c in train], "heldout": [list(c) for c in heldout], "bound_to_learner": bind is not None},
        "adaptation_window": window,
        "seed": seed,
        "stream": {"episodes": spec.stream_episodes, "tokens": clean.tokens, "policy": clean.policy, "worlds_disjoint_from_measurement": True},
        "transfer": transfer,
        "speed": {"before": before["speed"], "after": after["speed"]},
        "forgetting": {
            "before": before["forgetting"],
            "after": after["forgetting"],
            "delta_mse": after["forgetting"]["adapt"] - before["forgetting"]["adapt"],
            "elements": before["forgetting"]["elements"],
        },
        "correction": {
            "before_mse": _avg_transfer(before),
            "after_clean_mse": clean_after,
            "after_poison_mse": _avg_transfer(after_poison),
            "after_correction_mse": _avg_transfer(after_correction),
            # poison minus the clean arm: damage plus the clean gain the poisoned learner forwent
            "harm": harm,
            # poison minus its own (reverted, pre-stream) start: the damage alone
            "harm_vs_before": _avg_transfer(after_poison) - _avg_transfer(before),
            "residual": residual,
            "residual_vs_before": _avg_transfer(after_correction) - _avg_transfer(before),
            "poison_bias": spec.poison_bias,
        },
        "revert": {"gap": gap, "tolerance": spec.revert_tolerance, "ok": gap <= spec.revert_tolerance},
        "decisions": records,
        "acceptance": acceptance_rates(records),
        "compute": {
            "parameters": parameter_count,
            "tokens_consumed": tokens_consumed,
            "tokens_measured": tokens_measured,
            "tokens_measured_without_context": plain,
            "wall_clock_s": time.time() - t0,
        },
    }


# ------------------------------------------------------------------ baselines on PlasticDynamics


class DynamicsLearner:
    """``PlasticDynamics`` under the contract, in three baseline modes.

    ``frozen``: no lasting update (with ``adapt=True`` this is the fast-weights-only baseline).
    ``continued``: Adam steps on the stream's loss (the continued-training baseline; the
    optimizer is named in the consume record).
    ``in_context``: the stream is prepended at measurement time (everything in context); its
    extra tokens are counted in ``context_tokens_measured``.

    The delta rule writes at every token, so its update period is 1 and its no-adaptation
    control is ``beta_scale=0`` (writes disabled, decay active), which is a different
    intervention from freezing and keeps that label.
    """

    MODES = ("frozen", "continued", "in_context")

    def __init__(self, model: Any, *, mode: str = "frozen", lr: float = 1e-3, steps: int = 10, device: torch.device | None = None) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.model = model
        self.mode = mode
        self.lr = float(lr)
        self.steps = int(steps)
        self.device = device or next(model.parameters()).device
        self.context: MechanismBatch | None = None
        self.context_tokens_measured = 0

    def update_period(self) -> int:
        return 1

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def snapshot_slow(self) -> Any:
        return {"params": copy.deepcopy(self.model.state_dict()), "context": self.context}

    def restore_slow(self, snapshot: Any) -> None:
        self.model.load_state_dict(snapshot["params"])
        self.context = snapshot["context"]

    def consume(self, stream: MechanismBatch) -> dict[str, Any]:
        if self.mode == "frozen":
            return {"accepted": None, "mode": self.mode}
        if self.mode == "in_context":
            if self.context is None:
                self.context = stream
            else:
                self.context = MechanismBatch(
                    inputs=torch.cat([self.context.inputs, stream.inputs], dim=1),
                    target_delta=torch.cat([self.context.target_delta, stream.target_delta], dim=1),
                    worlds=[a + b for a, b in zip(self.context.worlds, stream.worlds)],
                    reset_flag=torch.cat([self.context.reset_flag, stream.reset_flag], dim=1),
                    policy=stream.policy,
                    poisoned=self.context.poisoned or stream.poisoned,
                )
            return {"accepted": True, "mode": self.mode, "context_tokens": self.context.tokens}
        x = stream.inputs.to(self.device)
        y = stream.target_delta.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        losses: list[float] = []
        self.model.train()
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            loss = self.model.loss(x, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        self.model.eval()
        return {"accepted": True, "mode": self.mode, "optimizer": "adam", "lr": self.lr, "loss_first": losses[0], "loss_last": losses[-1], "steps": self.steps}

    @torch.no_grad()
    def step_mse(self, batch: MechanismBatch, *, adapt: bool) -> Tensor:
        x = batch.inputs.to(self.device)
        y = batch.target_delta.to(self.device)
        prefix = 0
        if self.mode == "in_context" and self.context is not None:
            ctx = self.context.inputs.to(self.device).expand(x.shape[0], -1, -1)
            prefix = int(ctx.shape[1])
            x = torch.cat([ctx, x], dim=1)
            self.context_tokens_measured += int(x.shape[0] * prefix)
        self.model.eval()
        # adapt=False disables memory writes only (beta_scale=0); freeze would also stop decay,
        # which is a different intervention (AGENTS.md contract 2)
        pred, _, _ = self.model(x, mode="chunk", beta_scale=1.0 if adapt else 0.0)
        pred = pred[:, prefix:]
        return (pred - y).pow(2).mean(-1).cpu()
