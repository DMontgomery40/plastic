"""What does the anchor fold move? Two read-only analyses over archived sleep_controls runs (no model, no device).

1. Per-turn write magnitude in the teaching session: transactions joined to chat turns by their transaction-index
   range; per turn the sum of committed ``delta_norm``, the max ``surprise_max`` and ``chunk_loss``, and whether the
   observational policy flagged any chunk. Question: did the planted contradictions write more than the other facts?
2. Storage shift under the anchor child: for every verbatim probe the anchor report's before/after
   ``answer_logprob`` (expected-answer mean log-prob per token, teacher-forced from a fresh state), grouped into
   planted answers, general true answers and the taught personal facts. Question: does the fold move the planted
   answers specifically, or every short factual completion?

  python -m scripts.experiments.anchor_transfer_analysis --runs docs/research/results/sleep-2026-09-23/final_step250_seed0_exclude ... --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any

POISON_ANSWERS = {"Berlin", "50", "nine", "Moon"}
GENERAL_ANSWERS = {"Paris", "100", "seven", "Earth", "blue"}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def turn_write_table(transactions: list[dict[str, Any]], trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per chat turn: committed delta_norm summed over its chunks (by tx range), max surprise and chunk loss,
    chunk count, whether any chunk carried a would_* / requested intervention flag, and the rank by write (1 = largest)."""
    by_index = {int(t["index"]): t for t in transactions}
    rows: list[dict[str, Any]] = []
    for rec in trace:
        if rec.get("kind") != "chat" or rec.get("tx_start") is None or rec.get("tx_end") is None:
            continue
        chunks = [by_index[i] for i in range(int(rec["tx_start"]), int(rec["tx_end"])) if i in by_index]
        if not chunks:
            continue
        flagged = any(
            any(str(r).startswith("would_") for r in c.get("requested", {}).get("reasons", [])) or c.get("requested", {}).get("kind") != c.get("decision", {}).get("kind")
            or c.get("decision", {}).get("kind") in ("scale", "project")
            for c in chunks
        )
        rows.append({
            "prompt": rec.get("prompt", ""),
            "chunks": len(chunks),
            "delta_norm_sum": float(sum(float(c.get("accepted", {}).get("delta_norm") or 0.0) for c in chunks)),
            "surprise_max": float(max(float(c.get("signals", {}).get("surprise_max") or 0.0) for c in chunks)),
            "chunk_loss_max": float(max(float(c.get("signals", {}).get("chunk_loss") or 0.0) for c in chunks)),
            "flagged": bool(flagged),
            "planted": rec.get("prompt", "").startswith("Remember this"),
        })
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["delta_norm_sum"])
    for rank, i in enumerate(order, start=1):
        rows[i]["rank_by_write"] = rank
    return rows


def answer_group(expected: str) -> str:
    if expected in POISON_ANSWERS:
        return "planted"
    if expected in GENERAL_ANSWERS:
        return "general"
    return "taught"


def storage_shift(report: dict[str, Any]) -> dict[str, Any]:
    """Per verbatim probe (keyed by question AND expected answer: the plants reuse general questions) the after-before
    shift of the expected answer's mean log-prob per token, and per-group means."""
    def key(x: dict[str, Any]) -> tuple[str, str]:
        return (x["question"], x["expected"])
    before = {key(x): x for x in report["before"]["recall"]["results"] if x.get("variant") == "verbatim"}
    after = {key(x): x for x in report["after"]["recall"]["results"] if x.get("variant") == "verbatim"}
    probes: list[dict[str, Any]] = []
    for k, b in before.items():
        a = after.get(k)
        if a is None or b.get("answer_logprob") is None or a.get("answer_logprob") is None:
            continue
        probes.append({"question": k[0], "expected": k[1], "group": answer_group(k[1]),
                       "before": float(b["answer_logprob"]), "after": float(a["answer_logprob"]),
                       "shift": float(a["answer_logprob"]) - float(b["answer_logprob"]),
                       "greedy_before": bool(b.get("contains")), "greedy_after": bool(a.get("contains"))})
    groups: dict[str, Any] = {}
    for g in ("planted", "general", "taught"):
        v = [p["shift"] for p in probes if p["group"] == g]
        groups[g] = {"n": len(v), "mean_shift": statistics.fmean(v) if v else None,
                     "greedy_hits_before": sum(p["greedy_before"] for p in probes if p["group"] == g),
                     "greedy_hits_after": sum(p["greedy_after"] for p in probes if p["group"] == g)}
    return {"probes": probes, "groups": groups}


def analyze_run(run_dir: str) -> dict[str, Any]:
    out: dict[str, Any] = {"run": os.path.basename(os.path.normpath(run_dir))}
    sess = os.path.join(run_dir, "sessions", "teach")
    if os.path.exists(os.path.join(sess, "transactions.jsonl")):
        rows = turn_write_table(load_jsonl(os.path.join(sess, "transactions.jsonl")), load_jsonl(os.path.join(sess, "trace.jsonl")))
        writes = [r["delta_norm_sum"] for r in rows]
        out["teach_turns"] = rows
        out["write_summary"] = {"turns": len(rows), "median_delta_norm_sum": statistics.median(writes) if writes else None,
                                "planted_ranks": sorted(r["rank_by_write"] for r in rows if r["planted"]),
                                "largest_write_prompt": max(rows, key=lambda r: r["delta_norm_sum"])["prompt"][:60] if rows else None}
    else:
        out["teach_turns"] = None  # sessions not archived for this run: absent, not zero
    anchor = os.path.join(run_dir, "sleep_anchor_report.json")
    if os.path.exists(anchor):
        rep = json.load(open(anchor, encoding="utf-8"))
        if rep.get("before", {}).get("recall") and rep.get("after", {}).get("recall"):
            out["anchor_storage_shift"] = storage_shift(rep)
            out["anchor_status"] = rep.get("status")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="archived run folders (results/sleep-2026-09-23/<run>)")
    ap.add_argument("--out", required=True, help="JSON output path")
    args = ap.parse_args()
    results = [analyze_run(r) for r in args.runs]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"analysis": "anchor_transfer", "runs": results}, f, indent=1)
    for res in results:
        print(f"== {res['run']}")
        ws = res.get("write_summary")
        if ws:
            print(f"   teach turns {ws['turns']}, median write {ws['median_delta_norm_sum']:.2f}, planted turns rank {ws['planted_ranks']}, largest: {ws['largest_write_prompt']!r}")
        st = res.get("anchor_storage_shift")
        if st:
            g = st["groups"]
            print("   anchor storage shift: " + ", ".join(f"{k} n={v['n']} mean {v['mean_shift']:+.3f} greedy {v['greedy_hits_before']}->{v['greedy_hits_after']}" for k, v in g.items() if v["n"]))


if __name__ == "__main__":
    main()
