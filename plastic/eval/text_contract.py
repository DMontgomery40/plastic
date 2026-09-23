"""The learning contract on a text stream: the T1 five measures with MSE replaced by exact match and per-token NLL.

A ``TextLearner`` exposes its lasting state through ``snapshot_slow``/``restore_slow``, consumes a ``RuleBatch``
stream through ``consume`` (the lasting update, which may refuse) and scores a batch of episodes from a fresh
state through ``score`` with the fast path adapting or frozen. ``run_text_contract`` measures, on fixed seeds so
before and after see identical inputs:

1. transfer: exact and nll on held-out compositions with unseen word lists, adapt on and off;
2. speed: within an episode on held-out compositions, the adapting nll at situation i as a fraction of the
   writes-disabled nll on the same inputs;
3. forgetting: exact and nll on training compositions, and held-out chat NLL when a corpus is given;
4. correction: transfer after a poisoned stream (harm) and after a corrective clean stream (residual);
5. revert: restoring the pre-stream slow state must reproduce the before measurements.

Report shape and version tag follow plastic/eval/contract.py so both contracts read on one page.
Spec: docs/superpowers/specs/2026-09-23-text-rule-contract.md
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from plastic.data.rules import RuleBatch, poison_batch, rule_batch, split_pairs
from plastic.eval.contract import CONTRACT_VERSION, _max_gap, acceptance_rates


class TextLearner(Protocol):
    def snapshot_slow(self) -> Any: ...

    def restore_slow(self, snapshot: Any) -> None: ...

    def consume(self, stream: RuleBatch) -> dict[str, Any]: ...

    def score(self, batch: RuleBatch, *, adapt: bool) -> list[list[dict[str, float]]]:
        """Per episode, per situation: {"exact": 0/1, "nll": mean nll per answer token, "tokens": n}."""
        ...


@dataclass
class TextContractSpec:
    n_heldout: int = 18                    # 12 training pairs / 18 held-out (the T4 sources memo)
    split_seed: int = 0
    situations_per_episode: int = 8         # worked examples per lesson; the in-context curve is read at 2/4/8
    eval_episodes_per_composition: int = 2
    stream_episodes: int = 18               # one lesson per training composition (6 singles + 12 pairs)
    probe_situations: int = 8
    poison_operator: str = "#R"
    revert_tolerance: float = 1e-6
    n_words: tuple[int, int] = (4, 5)


_OFFSETS = {"heldout": 100, "speed": 200, "train": 300, "stream_clean": 400, "stream_correct": 500}


def _seed(seed: int, tag: str, index: int = 0) -> int:
    return int(seed) * 10_000 + _OFFSETS[tag] + index


def _summarize(scores: list[list[dict[str, float]]]) -> dict[str, float | int]:
    flat = [s for ep in scores for s in ep]
    n = len(flat)
    return {"exact": (sum(s["exact"] for s in flat) / n) if n else float("nan"),
            "nll": (sum(s["nll"] for s in flat) / n) if n else float("nan"),
            "situations": n, "answer_tokens": int(sum(s.get("tokens", 0) for s in flat))}


def measure(learner: TextLearner, spec: TextContractSpec, split: tuple[list, list], seed: int, *, chat_nll: Any = None) -> dict[str, Any]:
    """One set of measurements from fixed seeds; identical numbers on an unchanged learner."""
    train, heldout = split
    out: dict[str, Any] = {"transfer": {}, "speed": {}, "forgetting": {}, "situations": 0}
    hb = rule_batch(heldout, episodes=spec.eval_episodes_per_composition * len(heldout), n_situations=spec.situations_per_episode,
                    seed=_seed(seed, "heldout"), split_tag="heldout")
    adapt = learner.score(hb, adapt=True)
    frozen = learner.score(hb, adapt=False)
    per_comp: dict[str, dict[str, Any]] = {}
    for ep, a, f in zip(hb.episodes, adapt, frozen):
        key = " ".join(ep.ops)
        per_comp.setdefault(key, {"adapt": [], "no_adapt": []})
        per_comp[key]["adapt"].append(a)
        per_comp[key]["no_adapt"].append(f)
    out["transfer"] = {k: {"adapt": _summarize(v["adapt"]), "no_adapt": _summarize(v["no_adapt"])} for k, v in per_comp.items()}
    out["transfer_mean"] = {"adapt": _summarize(adapt), "no_adapt": _summarize(frozen)}
    out["situations"] += 2 * hb.situations
    # speed: the adapting nll at each situation of a held-out episode as a fraction of the writes-disabled nll
    sb = rule_batch(heldout, episodes=len(heldout), n_situations=spec.probe_situations, seed=_seed(seed, "speed"), split_tag="heldout")
    sa, sf = learner.score(sb, adapt=True), learner.score(sb, adapt=False)
    out["situations"] += 2 * sb.situations
    curve = []
    for i in range(spec.probe_situations):
        a = sum(ep[i]["nll"] for ep in sa) / len(sa)
        z = sum(ep[i]["nll"] for ep in sf) / len(sf)
        curve.append(a / z if z > 0 else 1.0)
    out["speed"] = {"curve": curve, "area": sum(curve) / len(curve), "situations_to_half": next((i + 1 for i, v in enumerate(curve) if v < 0.5), spec.probe_situations),
                    "episodes": len(sb.episodes)}
    tb = rule_batch(train, episodes=len(train), n_situations=spec.situations_per_episode, seed=_seed(seed, "train"), split_tag="train")
    out["forgetting"] = {"adapt": _summarize(learner.score(tb, adapt=True)), "no_adapt": _summarize(learner.score(tb, adapt=False))}
    out["situations"] += 2 * tb.situations
    if chat_nll is not None:
        out["forgetting"]["chat_nll"] = chat_nll()
    return out


def make_stream(spec: TextContractSpec, split: tuple[list, list], seed: int, *, tag: str) -> RuleBatch:
    train, _ = split
    return rule_batch(train, episodes=spec.stream_episodes, n_situations=spec.situations_per_episode, seed=_seed(seed, tag), split_tag="train")


def run_text_contract(learner: TextLearner, spec: TextContractSpec, *, seed: int = 0, chat_nll: Any = None) -> dict[str, Any]:
    t0 = time.time()
    split = split_pairs(n_heldout=spec.n_heldout, seed=spec.split_seed)
    train, heldout = split

    before = measure(learner, spec, split, seed, chat_nll=chat_nll)
    snapshot = learner.snapshot_slow()

    clean = make_stream(spec, split, seed, tag="stream_clean")
    rec_clean = learner.consume(clean)
    after = measure(learner, spec, split, seed, chat_nll=chat_nll)

    learner.restore_slow(snapshot)
    reverted = measure(learner, spec, split, seed, chat_nll=chat_nll)
    gap = _max_gap({k: v for k, v in reverted.items() if k != "situations"}, {k: v for k, v in before.items() if k != "situations"})

    poisoned = poison_batch(clean, operator=spec.poison_operator)
    rec_poison = learner.consume(poisoned)
    after_poison = measure(learner, spec, split, seed, chat_nll=chat_nll)
    corrective = make_stream(spec, split, seed, tag="stream_correct")
    rec_correct = learner.consume(corrective)
    after_correction = measure(learner, spec, split, seed, chat_nll=chat_nll)
    learner.restore_slow(snapshot)

    def tm(m: dict[str, Any], key: str) -> float:
        return float(m["transfer_mean"]["adapt"][key])

    records = [
        {"beneficial": True, "accepted": rec_clean.get("accepted"), "stream": "clean"},
        {"beneficial": False, "accepted": rec_poison.get("accepted"), "stream": "poisoned"},
        {"beneficial": True, "accepted": rec_correct.get("accepted"), "stream": "corrective"},
    ]
    parameter_count = getattr(learner, "parameter_count", lambda: None)()
    return {
        "contract_version": CONTRACT_VERSION,
        "contract": "text_rules",
        "spec": asdict(spec),
        "split": {"train": [list(c) for c in train], "heldout": [list(c) for c in heldout]},
        "seed": seed,
        "stream": {"episodes": spec.stream_episodes, "situations": clean.situations, "poison_operator": spec.poison_operator},
        "transfer": {
            "before": before["transfer_mean"], "after": after["transfer_mean"],
            "delta_exact": tm(after, "exact") - tm(before, "exact"),
            "delta_nll": tm(after, "nll") - tm(before, "nll"),
            "delta_nll_no_adapt": float(after["transfer_mean"]["no_adapt"]["nll"]) - float(before["transfer_mean"]["no_adapt"]["nll"]),
            "per_composition_before": before["transfer"], "per_composition_after": after["transfer"],
        },
        "speed": {"before": before["speed"], "after": after["speed"]},
        "forgetting": {"before": before["forgetting"], "after": after["forgetting"],
                       "delta_nll": float(after["forgetting"]["adapt"]["nll"]) - float(before["forgetting"]["adapt"]["nll"])},
        "correction": {
            "after_clean_nll": tm(after, "nll"), "after_poison_nll": tm(after_poison, "nll"), "after_correction_nll": tm(after_correction, "nll"),
            "after_clean_exact": tm(after, "exact"), "after_poison_exact": tm(after_poison, "exact"), "after_correction_exact": tm(after_correction, "exact"),
            "harm_nll": tm(after_poison, "nll") - tm(after, "nll"),
            "residual_nll": tm(after_correction, "nll") - tm(after, "nll"),
        },
        "revert": {"gap": gap, "tolerance": spec.revert_tolerance, "ok": gap <= spec.revert_tolerance},
        "decisions": records,
        "consume_records": {"clean": rec_clean, "poisoned": rec_poison, "corrective": rec_correct},
        "acceptance": acceptance_rates(records),
        "compute": {
            "parameters": parameter_count,
            "situations_consumed": 3 * clean.situations,
            "situations_measured": before["situations"] + after["situations"] + reverted["situations"] + after_poison["situations"] + after_correction["situations"],
            "context_situations_measured": int(getattr(learner, "context_situations_measured", 0)),
            "wall_clock_s": time.time() - t0,
        },
    }
