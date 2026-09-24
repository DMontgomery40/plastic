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

The poisoned stream is consumed twice: from the pre-stream snapshot (the T1 shape) and, when ``sequential_poison`` is set,
on top of the accepted clean lessons without a revert between, the arm on which a verifier can compare the proposal with
what it has already accepted. Both decisions count in the acceptance pair. The poisoned stream is the clean stream with
one operator's answers changed, so the sequential arm shares every other episode with a second clean pass; with
``sequential_clean`` the clean stream is also consumed a second time from the same post-clean state, the matched control
without which a refusal on the sequential arm cannot be attributed to the poison (OPUS-LEAD-001). The control is recorded
beside the decisions, not counted in the acceptance pair.

Every measurement keeps per-item records (per episode, per situation) and, when the learner offers ``choice``, a
first-situation ranking of every composition's output on fresh inputs for training and held-out compositions alike, so
the revert check is prediction-level and a changed decision can be traced to the items that moved.

Report shape and version tag follow plastic/eval/contract.py so both contracts read on one page.
Spec: docs/superpowers/specs/2026-09-23-text-rule-contract.md
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from plastic.data.rules import RuleBatch, apply, permute_names, poison_batch, rule_batch, shuffle_answers, split_pairs
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
    min_reversed_heldout: int = 6           # held-out pairs that reverse a training pair (split_pairs); part of the split's identity
    split_seed: int = 0
    situations_per_episode: int = 8         # worked examples per lesson; the in-context curve is read at 2/4/8
    eval_episodes_per_composition: int = 2
    stream_episodes: int = 18               # one lesson per training composition (6 singles + 12 pairs)
    probe_situations: int = 8
    poison_operator: str = "#R"
    revert_tolerance: float = 1e-6
    n_words: tuple[int, int] = (4, 5)
    rule_set: str = "transform"             # "transform" (the six word operators) or "decorate" (fixed tokens around a copied list)
    sequential_poison: bool = True          # also consume the poisoned stream ON TOP of the accepted clean one (no revert between)
    stated_rules: bool = True               # False: no definitions in any preface (streams, verification, measurement); rules come from examples only
    poison_kind: str = "consistent"         # "consistent": false definition stated with false answers; "inconsistent": true definitions stated, false answers
    sequential_clean: bool = True           # the matched control for the sequential poison: the clean stream consumed again from the post-clean state
    choice_episodes_per_composition: int = 2  # first-situation choice items per composition (0 disables; used only when the learner offers ``choice``)
    choice_at: tuple[str, ...] = ("before", "after", "after_clean_again", "after_poison_sequential", "after_poison", "after_format", "after_names")  # measurements that carry the choice score
    name_permuted_control: bool = False     # the content null: the clean lessons with each name's answers computed by another composition of its arity


_OFFSETS = {"heldout": 100, "speed": 200, "train": 300, "stream_clean": 400, "stream_correct": 500, "choice": 600}


def _seed(seed: int, tag: str, index: int = 0) -> int:
    return int(seed) * 10_000 + _OFFSETS[tag] + index


def _summarize(scores: list[list[dict[str, float]]]) -> dict[str, Any]:
    """Means over every situation, plus the per-situation-index curve: the first situation of an episode has no worked
    example before it, so a fast learner cannot answer it from the episode; what the slow parameters carry shows there
    and in ``no_adapt``, what the fast path does shows from the second situation on. Averaging over the whole episode
    would blend the two (the objective/measurement prior named in FABLE-202)."""
    flat = [s for ep in scores for s in ep]
    n = len(flat)
    n_pos = max((len(ep) for ep in scores), default=0)
    by_situation = []
    for i in range(n_pos):
        col = [ep[i] for ep in scores if len(ep) > i]
        by_situation.append({"exact": sum(c["exact"] for c in col) / len(col), "nll": sum(c["nll"] for c in col) / len(col), "n": len(col)})
    later = [s for ep in scores for s in ep[1:]]
    return {"exact": (sum(s["exact"] for s in flat) / n) if n else float("nan"),
            "nll": (sum(s["nll"] for s in flat) / n) if n else float("nan"),
            "exact_first": by_situation[0]["exact"] if by_situation else float("nan"),
            "exact_after_first": (sum(s["exact"] for s in later) / len(later)) if later else float("nan"),
            "nll_after_first": (sum(s["nll"] for s in later) / len(later)) if later else float("nan"),
            "by_situation": by_situation,
            "situations": n, "answer_tokens": int(sum(s.get("tokens", 0) for s in flat))}


def _items(batch: RuleBatch, scores: list[list[dict[str, float]]]) -> list[dict[str, Any]]:
    """Per episode: the composition and its per-situation exact and nll, in batch order (fixed seeds make the items the
    same inputs before and after)."""
    return [{"ops": " ".join(ep.ops), "exact": [x["exact"] for x in sc], "nll": [x["nll"] for x in sc]} for ep, sc in zip(batch.episodes, scores)]


def choice_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy (the correct output ranks first), mean margin over the best other candidate, mean chance level, and the
    same per composition; the items stay attached."""
    def agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        return {"accuracy": sum(r["choice"] for r in rows) / n, "margin_mean": sum(r["margin"] for r in rows) / n,
                "correct_logp_mean": sum(r["correct_logp"] for r in rows) / n, "chance_mean": sum(1.0 / r["candidates"] for r in rows) / n, "n": n}
    by: dict[str, list[dict[str, Any]]] = {}
    for r in items:
        by.setdefault(r["ops"], []).append(r)
    return agg(items) | {"by_composition": {k: agg(v) for k, v in by.items()}, "items": items}


def measure(learner: TextLearner, spec: TextContractSpec, split: tuple[list, list], seed: int, *, chat_nll: Any = None, with_choice: bool = True) -> dict[str, Any]:
    """One set of measurements from fixed seeds; identical numbers on an unchanged learner. The choice score costs a
    forward pass per candidate output, so the contract computes it only at the measurements named in ``spec.choice_at``."""
    train, heldout = split
    out: dict[str, Any] = {"transfer": {}, "speed": {}, "forgetting": {}, "situations": 0}
    hb = rule_batch(heldout, episodes=spec.eval_episodes_per_composition * len(heldout), n_situations=spec.situations_per_episode,
                    seed=_seed(seed, "heldout"), split_tag="heldout", rule_set=spec.rule_set, n_words=spec.n_words, stated=spec.stated_rules)
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
    out["items"] = {"heldout": _items(hb, adapt)}
    out["situations"] += 2 * hb.situations
    # speed: the adapting nll at each situation of a held-out episode as a fraction of the writes-disabled nll
    sb = rule_batch(heldout, episodes=len(heldout), n_situations=spec.probe_situations, seed=_seed(seed, "speed"), split_tag="heldout", rule_set=spec.rule_set, n_words=spec.n_words,
                    stated=spec.stated_rules)
    sa, sf = learner.score(sb, adapt=True), learner.score(sb, adapt=False)
    out["situations"] += 2 * sb.situations
    curve = []
    for i in range(spec.probe_situations):
        a = sum(ep[i]["nll"] for ep in sa) / len(sa)
        z = sum(ep[i]["nll"] for ep in sf) / len(sf)
        curve.append(a / z if z > 0 else 1.0)
    out["speed"] = {"curve": curve, "area": sum(curve) / len(curve), "situations_to_half": next((i + 1 for i, v in enumerate(curve) if v < 0.5), spec.probe_situations),
                    "episodes": len(sb.episodes)}
    tb = rule_batch(train, episodes=len(train), n_situations=spec.situations_per_episode, seed=_seed(seed, "train"), split_tag="train", rule_set=spec.rule_set, n_words=spec.n_words,
                    stated=spec.stated_rules)
    tb_adapt = learner.score(tb, adapt=True)
    out["forgetting"] = {"adapt": _summarize(tb_adapt), "no_adapt": _summarize(learner.score(tb, adapt=False))}
    out["items"]["train"] = _items(tb, tb_adapt)
    out["situations"] += 2 * tb.situations
    choose = getattr(learner, "choice", None)
    n_choice = spec.choice_episodes_per_composition
    if choose is not None and n_choice > 0 and with_choice:
        common = {"n_situations": 1, "rule_set": spec.rule_set, "n_words": spec.n_words, "stated": spec.stated_rules}
        cb_h = rule_batch(heldout, episodes=n_choice * len(heldout), seed=_seed(seed, "choice"), split_tag="heldout", **common)
        cb_t = rule_batch(train, episodes=n_choice * len(train), seed=_seed(seed, "choice", 1), split_tag="train", **common)
        out["choice"] = {"heldout": choice_summary(choose(cb_h)), "train": choice_summary(choose(cb_t))}
        out["situations"] += cb_h.situations + cb_t.situations
    if chat_nll is not None:
        out["forgetting"]["chat_nll"] = chat_nll()
    return out


def make_stream(spec: TextContractSpec, split: tuple[list, list], seed: int, *, tag: str) -> RuleBatch:
    train, _ = split
    return rule_batch(train, episodes=spec.stream_episodes, n_situations=spec.situations_per_episode, seed=_seed(seed, tag), split_tag="train", rule_set=spec.rule_set, n_words=spec.n_words,
                      stated=spec.stated_rules)


def paired_margins(a: dict[str, Any] | None, b: dict[str, Any] | None, groups: dict[str, set[str]]) -> dict[str, Any] | None:
    """Per-item change in the choice margin from measurement ``a`` to ``b`` (same inputs, same order, so items pair by
    index), summarized per group of compositions: mean change, items improved / worsened, and n. Counts of first-rank
    hits on a dozen items are too coarse to read; the paired margin is the per-item reading."""
    if not a or not b:
        return None
    out: dict[str, Any] = {}
    for part in ("train", "heldout"):
        ia, ib = a.get(part, {}).get("items"), b.get(part, {}).get("items")
        if not ia or not ib:
            continue
        if [x["ops"] for x in ia] != [x["ops"] for x in ib]:
            raise ValueError("choice items do not pair: the measurements saw different material")
        for name, members in groups.items():
            if not name.startswith(part):
                continue
            d = [y["margin"] - x["margin"] for x, y in zip(ia, ib) if x["ops"] in members]
            if d:
                out[name] = {"mean_change": sum(d) / len(d), "improved": sum(1 for v in d if v > 0), "worsened": sum(1 for v in d if v < 0), "n": len(d)}
    return out


def output_duplicates(train: list[tuple[str, ...]], heldout: list[tuple[str, ...]], rule_set: str) -> dict[str, str]:
    """Held-out compositions whose output equals a training composition's on every input, mapped to that composition.
    On the decoration set a prefix word and a suffix word commute (``#P #Q`` = ``#Q #P``, ``#H #Q`` = ``#Q #H``), and the
    reversed-pair constraint puts such pairs in the held-out set, so part of "held-out transfer" repeats trained answers
    (OPUS-LEAD-002). Decorations never inspect the words, so one probe list decides it; the transforming set is checked on
    three lists of different lengths."""
    probes = [["apple", "pear", "plum", "fig"], ["river", "stone", "cloud"], ["hammer", "needle", "basket", "candle", "mirror"]]
    def sig(c: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(" ".join(apply(c, w, rule_set=rule_set)) for w in probes)
    trained = {sig(c): " ".join(c) for c in train}
    return {" ".join(h): trained[sig(h)] for h in heldout if sig(h) in trained}


def contract_split(spec: TextContractSpec) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """The one split the contract uses; callers that hand the learner its held-in material must take it from here."""
    return split_pairs(n_heldout=spec.n_heldout, seed=spec.split_seed, rule_set=spec.rule_set, min_reversed_heldout=spec.min_reversed_heldout)


def run_text_contract(learner: TextLearner, spec: TextContractSpec, *, seed: int = 0, chat_nll: Any = None) -> dict[str, Any]:
    t0 = time.time()
    split = contract_split(spec)
    train, heldout = split
    dups = output_duplicates(train, heldout, spec.rule_set)
    given = getattr(learner, "train_compositions", None)
    if given is not None and [tuple(c) for c in given] != [tuple(c) for c in train]:
        raise ValueError("the learner's held-in material (train_compositions) is not the contract's training split; build it with contract_split(spec)")

    before = measure(learner, spec, split, seed, chat_nll=chat_nll, with_choice="before" in spec.choice_at)
    snapshot = learner.snapshot_slow()

    clean = make_stream(spec, split, seed, tag="stream_clean")
    rec_clean = learner.consume(clean)
    after = measure(learner, spec, split, seed, chat_nll=chat_nll, with_choice="after" in spec.choice_at)

    # the matched control for the sequential poison: the same clean stream again from the same post-clean state
    rec_clean_again: dict[str, Any] | None = None
    after_clean_again: dict[str, Any] | None = None
    if spec.sequential_clean:
        post_clean = learner.snapshot_slow()
        rec_clean_again = learner.consume(clean)
        after_clean_again = measure(learner, spec, split, seed, chat_nll=chat_nll, with_choice="after_clean_again" in spec.choice_at)
        learner.restore_slow(post_clean)

    # the sequential arm: the poison arrives after the clean lessons were accepted, so a verifier that checks the
    # proposal against what it already accepted has something to check it against (from the snapshot the operator
    # has no prior, and a consistent false rule is then indistinguishable from a true one)
    poisoned = poison_batch(clean, operator=spec.poison_operator, kind=spec.poison_kind)
    rec_poison_seq: dict[str, Any] | None = None
    after_seq: dict[str, Any] | None = None
    if spec.sequential_poison:
        rec_poison_seq = learner.consume(poisoned)
        after_seq = measure(learner, spec, split, seed, chat_nll=chat_nll, with_choice="after_poison_sequential" in spec.choice_at)

    learner.restore_slow(snapshot)
    reverted = measure(learner, spec, split, seed, chat_nll=chat_nll, with_choice="reverted" in spec.choice_at)
    gap = _max_gap({k: v for k, v in reverted.items() if k != "situations"}, {k: v for k, v in before.items() if k != "situations"})

    rec_poison = learner.consume(poisoned)
    after_poison = measure(learner, spec, split, seed, chat_nll=chat_nll, with_choice="after_poison" in spec.choice_at)
    corrective = make_stream(spec, split, seed, tag="stream_correct")
    rec_correct = learner.consume(corrective)
    after_correction = measure(learner, spec, split, seed, chat_nll=chat_nll, with_choice="after_correction" in spec.choice_at)
    learner.restore_slow(snapshot)

    # the format-only control (the sources memo's third falsifier): the clean lessons with shuffled answers
    shuffled = shuffle_answers(clean, seed=seed)
    rec_format = learner.consume(shuffled)
    after_format = measure(learner, spec, split, seed, chat_nll=chat_nll, with_choice="after_format" in spec.choice_at)
    learner.restore_slow(snapshot)

    # the content null: names consistently paired with the wrong decoration, copying and answer shapes kept
    rec_names: dict[str, Any] | None = None
    after_names: dict[str, Any] | None = None
    permuted: RuleBatch | None = None
    if spec.name_permuted_control:
        permuted = permute_names(clean, seed=seed)
        rec_names = learner.consume(permuted)
        after_names = measure(learner, spec, split, seed, chat_nll=chat_nll, with_choice="after_names" in spec.choice_at)
        learner.restore_slow(snapshot)

    def tm(m: dict[str, Any], key: str) -> float:
        return float(m["transfer_mean"]["adapt"][key])

    def ch(m: dict[str, Any] | None, part: str) -> float | None:
        return None if m is None or "choice" not in m else float(m["choice"][part]["accuracy"])

    def ch_delta(a: dict[str, Any] | None, b: dict[str, Any] | None, part: str) -> float | None:
        x, y = ch(a, part), ch(b, part)
        return None if x is None or y is None else y - x

    def first_item_changes(rec: dict[str, Any] | None) -> list[dict[str, Any]] | None:
        """Verification items whose first-situation exact changed across a verified consume."""
        v = (rec or {}).get("verify")
        if not v or "items_before" not in v:
            return None
        return [{"ops": b["ops"], "before": b["exact_first"], "after": a["exact_first"]}
                for b, a in zip(v["items_before"], v["items_after"]) if b["exact_first"] != a["exact_first"]]

    records = [
        {"beneficial": True, "accepted": rec_clean.get("accepted"), "stream": "clean"},
        {"beneficial": False, "accepted": rec_poison.get("accepted"), "stream": "poisoned"},
        {"beneficial": True, "accepted": rec_correct.get("accepted"), "stream": "corrective"},
        {"beneficial": False, "accepted": rec_format.get("accepted"), "stream": "format_only"},
    ]
    if rec_names is not None:
        records.append({"beneficial": False, "accepted": rec_names.get("accepted"), "stream": "name_permuted"})
    if rec_poison_seq is not None:
        records.insert(2, {"beneficial": False, "accepted": rec_poison_seq.get("accepted"), "stream": "poisoned_sequential"})
    sequential = None
    if after_seq is not None:
        sequential = {"accepted": rec_poison_seq.get("accepted"), "after_clean_exact": tm(after, "exact"), "after_clean_then_poison_exact": tm(after_seq, "exact"),
                      "after_clean_nll": tm(after, "nll"), "after_clean_then_poison_nll": tm(after_seq, "nll"),
                      "harm_nll": tm(after_seq, "nll") - tm(after, "nll"), "harm_exact": tm(after, "exact") - tm(after_seq, "exact"),
                      "choice_heldout_delta": ch_delta(after, after_seq, "heldout"), "choice_train_delta": ch_delta(after, after_seq, "train"),
                      "verify_first_changes": first_item_changes(rec_poison_seq),
                      "note": "the poison consumed on top of the accepted clean lessons, no revert between; harm is relative to the clean state"}
    sequential_control = None
    if after_clean_again is not None:
        sequential_control = {"accepted": rec_clean_again.get("accepted"), "after_clean_exact": tm(after, "exact"), "after_clean_again_exact": tm(after_clean_again, "exact"),
                              "delta_nll": tm(after_clean_again, "nll") - tm(after, "nll"), "delta_exact": tm(after_clean_again, "exact") - tm(after, "exact"),
                              "choice_heldout_delta": ch_delta(after, after_clean_again, "heldout"), "choice_train_delta": ch_delta(after, after_clean_again, "train"),
                              "verify_first_changes": first_item_changes(rec_clean_again),
                              "note": "the clean stream consumed a second time from the post-clean state: the matched control for the sequential poison, "
                                      "which shares every episode not involving the poisoned operator; not counted in the acceptance pair"}
    choice = None
    if spec.choice_at and any("choice" in m for m in (before, after)):
        def cs(m: dict[str, Any] | None, keep_items: bool) -> dict[str, Any] | None:
            if m is None or "choice" not in m:
                return None
            return {part: (v if keep_items else {k: x for k, x in v.items() if k != "items"}) for part, v in m["choice"].items()}
        choice = {"before": cs(before, True), "after": cs(after, True), "after_clean_again": cs(after_clean_again, True), "after_poison_sequential": cs(after_seq, True),
                  "reverted": cs(reverted, False), "after_poison": cs(after_poison, True), "after_correction": cs(after_correction, False), "after_format": cs(after_format, True),
                  "after_names": cs(after_names, True),
                  "at": list(spec.choice_at),
                  "note": "first situation only (no worked example before it): the correct output's rank among every composition's distinct output on the same input; "
                          "measured only at the measurements named in 'at' (None elsewhere)"}
    parameter_count = getattr(learner, "parameter_count", lambda: None)()
    if choice is not None:
        novel = {" ".join(h) for h in heldout if " ".join(h) not in dups}
        groups = {"train": {" ".join(c) for c in train}, "heldout_novel": novel, "heldout_duplicate": set(dups)}
        choice["paired"] = {"after_vs_before": paired_margins(choice["before"], choice["after"], groups),
                            "clean_again_vs_after": paired_margins(choice["after"], choice["after_clean_again"], groups),
                            "poison_sequential_vs_after": paired_margins(choice["after"], choice["after_poison_sequential"], groups),
                            "poison_vs_before": paired_margins(choice["before"], choice["after_poison"], groups),
                            "format_vs_before": paired_margins(choice["before"], choice["after_format"], groups),
                            "names_vs_before": paired_margins(choice["before"], choice["after_names"], groups)}
        choice["heldout_novel_accuracy"] = {k: (None if not v or "heldout" not in v else
                                                (lambda rows: sum(r["choice"] for r in rows) / len(rows) if rows else None)(
                                                    [r for r in v["heldout"].get("items", []) if r["ops"] in novel]))
                                            for k, v in choice.items() if k in ("before", "after", "after_clean_again", "after_poison_sequential")}
    return {
        "contract_version": CONTRACT_VERSION,
        "contract": "text_rules",
        "spec": asdict(spec),
        "split": {"train": [list(c) for c in train], "heldout": [list(c) for c in heldout], "heldout_output_duplicates": dups,
                  "heldout_novel": [" ".join(h) for h in heldout if " ".join(h) not in dups]},
        "seed": seed,
        "stream": {"episodes": spec.stream_episodes, "situations": clean.situations, "poison_operator": spec.poison_operator, "poison_kind": spec.poison_kind,
                   "rule_set": spec.rule_set, "stated_rules": spec.stated_rules},
        "transfer": {
            "before": before["transfer_mean"], "after": after["transfer_mean"],
            "delta_exact": tm(after, "exact") - tm(before, "exact"),
            "delta_nll": tm(after, "nll") - tm(before, "nll"),
            "delta_nll_no_adapt": float(after["transfer_mean"]["no_adapt"]["nll"]) - float(before["transfer_mean"]["no_adapt"]["nll"]),
            "delta_exact_first": float(after["transfer_mean"]["adapt"]["exact_first"]) - float(before["transfer_mean"]["adapt"]["exact_first"]),
            "delta_exact_after_first": float(after["transfer_mean"]["adapt"]["exact_after_first"]) - float(before["transfer_mean"]["adapt"]["exact_after_first"]),
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
        "format_only": {
            "after_exact": tm(after_format, "exact"), "after_nll": tm(after_format, "nll"),
            "gain_exact": tm(after_format, "exact") - tm(before, "exact"), "gain_nll": tm(after_format, "nll") - tm(before, "nll"),
            "true_stream_gain_exact": tm(after, "exact") - tm(before, "exact"), "true_stream_gain_nll": tm(after, "nll") - tm(before, "nll"),
            "note": "a lasting update that gains as much from shuffled answers as from the true lessons learned format, not rules",
        },
        "sequential_poison": sequential,
        "sequential_control": sequential_control,
        "name_permuted": None if after_names is None else {
            "after_exact": tm(after_names, "exact"), "after_nll": tm(after_names, "nll"),
            "gain_exact": tm(after_names, "exact") - tm(before, "exact"), "gain_nll": tm(after_names, "nll") - tm(before, "nll"),
            "true_stream_gain_exact": tm(after, "exact") - tm(before, "exact"), "true_stream_gain_nll": tm(after, "nll") - tm(before, "nll"),
            "map": [[" ".join(k), " ".join(v)] for k, v in (permuted.name_map or ())],
            "note": "the content null: each name's answers follow another composition of its arity; a first-situation choice gain as large as the true stream's is not name-to-decoration content"},
        "choice": choice,
        "items": {"before": before["items"], "after": after["items"]},
        "revert": {"gap": gap, "tolerance": spec.revert_tolerance, "ok": gap <= spec.revert_tolerance},
        "decisions": records,
        "verifier": getattr(learner, "verifier_version", None),
        "consume_records": {"clean": rec_clean, "clean_again": rec_clean_again, "poisoned": rec_poison, "poisoned_sequential": rec_poison_seq, "corrective": rec_correct, "format_only": rec_format,
                            "name_permuted": rec_names},
        "acceptance": acceptance_rates(records),
        "compute": {
            "parameters": parameter_count,
            "situations_consumed": (4 + (after_seq is not None) + (after_clean_again is not None) + (after_names is not None)) * clean.situations,
            "situations_measured": before["situations"] + after["situations"] + reverted["situations"] + after_poison["situations"] + after_correction["situations"] + after_format["situations"]
                                   + (after_seq["situations"] if after_seq is not None else 0) + (after_clean_again["situations"] if after_clean_again is not None else 0)
                                   + (after_names["situations"] if after_names is not None else 0),
            "context_situations_measured": int(getattr(learner, "context_situations_measured", 0)),
            "wall_clock_s": time.time() - t0,
        },
    }
