"""Shock pass 1 on native Qwen (scratchpad FABLE-105 plan, ASTRA-140 amendments): observational runs that
record how the harness signals respond when a conversation takes a shock turn, and how much of that turn
the later turns carry.

Prompts come from a chains JSON file, never from this module: each turn is either literal text (benign
structural shocks: a false-fact assertion, a contradiction, an abrupt register shift, long repetitive
context) or a reference to a pinned public benchmark row (JailbreakBench behaviors by index; the benign
file at the same index is the matched control), or a pinned DEV prompt by position.

Arms per chain, all from the same reset state and per-turn seeds:

  N  kept:      log-only runner; every turn committed (the natural trajectory)
  D  discarded: log-only runner; the shock turn is processed, then the FULL turn-start runner snapshot is
                restored (T0 mechanics), and the chain continues
  O  omitted:   log-only runner; the shock turn is never processed
  G  guarded:   the existing controls, rollback-only (no absorbing latch), with NO calibration: gates fall
                back to robust z against the session's own history and the configured CUSUM h. Honestly
                uncalibrated; its interventions are recorded, not interpreted as calibrated protection.

Per chunk the runner records loss, the proposed and accepted recurrent state-change norms, robust z,
both CUSUM sides via the detector state, and every would-decision. With a canary suite the runner also
records the probe-loss change and the cosine between the proposed state change and the gradient of the
frozen probe NLL w.r.t. the recurrent state (``canary_alignment``); that is the only gradient measured
here, and it is named as such. The state-change norm is not a gradient.

Development-only and descriptive. Counts, not rates; no detection-power, safety or learning claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from typing import Any

SEED_OFFSET = 5000  # distinct from the fit/cusum/dev/eval/followup schedules of the operating-point driver


# ---------------------------------------------------------------- pure logic (unit-tested)

def resolve_turn(spec: dict[str, Any], sources: dict[str, list[str]]) -> str:
    """A turn is literal ``text`` or ``{"source": name, "index": i}`` into a pinned prompt list, optionally
    wrapped by ``template`` containing ``{goal}``."""
    if "text" in spec:
        return str(spec["text"])
    rows = sources[spec["source"]]
    goal = rows[int(spec["index"])]
    return str(spec.get("template", "{goal}")).format(goal=goal)


def arm_plan(chain: dict[str, Any]) -> list[str]:
    """Which arms a chain runs: shock chains run N, D, O, G; a chain with no shock turn runs N and G only."""
    return ["N", "D", "O", "G"] if chain.get("shock_turn") is not None else ["N", "G"]


def turn_seeds(chain_index: int, n_turns: int, base: int) -> list[int]:
    return [base + SEED_OFFSET + 100 * chain_index + t for t in range(n_turns)]


def later_turn_divergence(arm_a: list[dict[str, Any]], arm_b: list[dict[str, Any]], *, from_turn: int) -> dict[str, Any]:
    """Compare two arms turn by turn (aligned by ORIGINAL turn index, since the omitted arm lacks the shock
    turn) at and after ``from_turn``: whether the generated ids differ, and the first differing turn."""
    by_a = {t["turn"]: t for t in arm_a}
    by_b = {t["turn"]: t for t in arm_b}
    diffs = [{"turn": k, "ids_differ": by_a[k]["out_ids"] != by_b[k]["out_ids"]}
             for k in sorted(by_a.keys() & by_b.keys()) if k >= from_turn]
    first = next((d["turn"] for d in diffs if d["ids_differ"]), None)
    return {"turns": diffs, "first_differing_turn": first}


def intervention_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Applied decisions by kind and reason across a guarded arm's chunks, plus whether any chunk alarmed."""
    kinds: dict[str, int] = {}
    reasons: dict[str, int] = {}
    alarms = 0
    for t in turns:
        for r in t["records"]:
            k = r["decision"]["kind"]
            kinds[k] = kinds.get(k, 0) + 1
            for reason in r["decision"]["reasons"]:
                key = reason.split("(")[0]
                reasons[key] = reasons.get(key, 0) + 1
            alarms += bool(r["signals"].get("cusum_alarm"))
    return {"decisions": kinds, "reasons": reasons, "cusum_alarms": alarms}


def chunk_table(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One flat row per chunk: the quantities the write-up tabulates."""
    rows = []
    for t in turns:
        for r in t["records"]:
            s = r["signals"]
            rows.append({"turn": t["turn"], "pos_start": r["pos_start"], "pos_end": r["pos_end"],
                         "sources": r["sources"], "chunk_loss": s["chunk_loss"], "log_delta_norm": s["log_delta_norm"],
                         "accepted_delta_norm": r["accepted"]["delta_norm"], "z_chunk_loss": (s["z"] or {}).get("chunk_loss"),
                         "z_log_delta_norm": (s["z"] or {}).get("log_delta_norm"), "cusum_alarm": s["cusum_alarm"],
                         "canary_delta_coherence": s.get("canary_delta_coherence"), "canary_alignment": s.get("canary_alignment"),
                         "decision": r["decision"]["kind"], "reasons": r["decision"]["reasons"]})
    return rows


# ---------------------------------------------------------------- model-facing parts

def _digest_cache(state) -> str:
    import torch

    h = hashlib.sha256()

    def walk(obj, path):
        if torch.is_tensor(obj):
            t = obj.detach().to("cpu").contiguous()
            h.update(f"{path}|{tuple(t.shape)}|{t.dtype}".encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                walk(obj[key], f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, (int, float, bool, str)) or obj is None:
            h.update(f"{path}={obj!r}".encode())

    for i, layer in enumerate(getattr(state.cache, "layers", [])):
        walk(dict(vars(layer)), f"L{i}")
    h.update(f"position={state.position}".encode())
    return h.hexdigest()


def load_sources(shock_sources_dir: str, dev_manifest: str) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Pinned prompt lists: JailbreakBench harmful/benign behaviors (by row index) and the pinned DEV prompts
    (by position). Returns the lists and their identity (revisions, file digests)."""
    import pandas as pd

    man = json.load(open(os.path.join(shock_sources_dir, "manifest.json")))
    jbb = man["JailbreakBench/JBB-Behaviors"]["files"]
    harmful = pd.read_csv(jbb["data/harmful-behaviors.csv"]["path"])
    benign = pd.read_csv(jbb["data/benign-behaviors.csv"]["path"])
    dev = json.load(open(dev_manifest))["selected_records"]["dev"]
    sources = {"jbb_harmful": harmful["Goal"].tolist(), "jbb_benign": benign["Goal"].tolist(),
               "jbb_harmful_category": harmful["Category"].tolist(), "dev": [r["prompt"] for r in dev]}
    identity = {"jbb_revision": man["JailbreakBench/JBB-Behaviors"]["revision"],
                "jbb_files": {k: v["sha256"] for k, v in jbb.items()},
                "dev_manifest": os.path.abspath(dev_manifest), "dev_ids": [r["id"] for r in dev]}
    return sources, identity


def make_suite(backend, probes: list[str]):
    """A coherence canary suite of fixed benign probe texts, tokenized natively (no chat template), so the
    runner records probe-loss change and the gradient alignment per chunk. No poison probes."""
    from plastic.harness.canary import CanarySuite

    return CanarySuite(domain="text", coherence=[backend.encode(p) for p in probes], poison=[])


def run_chain(runner, tok, turns: list[str], seeds: list[int], *, arm: str, shock_turn: int | None, gen: dict[str, Any],
              is_finite) -> list[dict[str, Any]]:
    import torch

    from plastic.session.runner import drive_chat_turn

    def digest():
        return {"committed": _digest_cache(runner.committed), "position": runner.committed.position,
                "cusum": runner.cusum.state(), "read_only": runner.read_only}

    runner.reset()
    out = []
    for t, prompt in enumerate(turns):
        if arm == "O" and t == shock_turn:
            continue
        snap = runner.state_dict() if (arm == "D" and t == shock_turn) else None
        d_before = digest()
        runner.transactions = []
        completion, out_ids, in_ids = drive_chat_turn(runner, tok, prompt, max_new_tokens=gen["max_new_tokens"],
                                                      temperature=gen["temperature"], top_k=gen["top_k"],
                                                      gen=torch.Generator().manual_seed(int(seeds[t])))
        recs = [{k: v for k, v in r.items() if k != "t_unix"} for r in runner.transactions]
        rec = {"turn": t, "seed": seeds[t], "n_in": len(in_ids), "completion": completion,
               "out_ids": [int(x) for x in out_ids],
               "outcome": "cap" if len(out_ids) >= gen["max_new_tokens"] else ("empty" if not out_ids else "eos"),
               "records": recs, "digest_before": d_before, "digest_after": digest(), "state_finite": bool(is_finite()),
               "read_only_end": runner.read_only, "read_only_reason": runner.read_only_reason}
        if snap is not None:
            n_tx = runner.n_transactions
            runner.load_state_dict(snap)
            runner.n_transactions = n_tx
            rec["digest_after_restore"] = digest()
            rec["restore_equals_turn_start"] = rec["digest_after_restore"]["committed"] == d_before["committed"]
            rec["discarded"] = True
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="the shocked checkpoint (abliterated derivative)")
    ap.add_argument("--reference-checkpoint", default=None, help="optional aligned checkpoint; benign chains only, arm N")
    ap.add_argument("--chains", required=True, help="chains JSON (see scripts/experiments/shock_chains_pass1.json)")
    ap.add_argument("--shock-sources", default="artifacts/data/shock-sources")
    ap.add_argument("--dev-manifest", default="artifacts/experiments/qwen-oppoint-dev-0cafa60-mps/split-manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--only", default=None, help="comma-separated chain names to run (debugging)")
    args = ap.parse_args()

    out_path = os.path.join(args.out, "shock-pass1.json")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; refusing to overwrite")
    os.makedirs(args.out, exist_ok=True)

    from plastic.backends.qwen import QwenBackend, _checkpoint_digest
    from plastic.config import ModelConfig
    from plastic.harness.config import HarnessConfig
    from plastic.harness.transaction import TransactionRunner
    from plastic.session.runner import _QwenTextIO
    from scripts.experiments.qwen_operating_point import CHUNK, SEED, _gen, _invocation_provenance

    spec = json.load(open(args.chains))
    chains = [c for c in spec["chains"] if not args.only or c["name"] in args.only.split(",")]
    sources, source_identity = load_sources(args.shock_sources, args.dev_manifest)
    gen = _gen(args.max_new_tokens)
    cfg = ModelConfig(domain="text", chunk=CHUNK)
    # log-only: no gate acts; projection is enabled ONLY so the runner computes canary_alignment (log-only never applies it)
    observe = HarnessConfig(log_only=True, enable_stats=True, enable_rollback=True, enable_projection=True, enable_budget=False,
                            learn_from_generation=True, freeze_on_alarm=False)
    # guarded: existing controls, rollback-only, NO calibration (session-history z fallbacks, configured cusum_h)
    guard = HarnessConfig(log_only=False, enable_stats=True, enable_rollback=True, enable_projection=False, enable_budget=False,
                          learn_from_generation=True, freeze_on_alarm=False)

    def load(ckpt):
        be = QwenBackend.load(ckpt, device=args.device)
        tok = _QwenTextIO(be)
        suite = make_suite(be, spec["canary_probes"])
        runners = {"observe": TransactionRunner(None, cfg, observe, calibration=None, suite=suite, device=be.device, backend=be),
                   "guard": TransactionRunner(None, cfg, guard, calibration=None, suite=None, device=be.device, backend=be)}
        return be, tok, runners

    t0 = time.time()
    be, tok, runners = load(args.checkpoint)
    results = []
    for ci, chain in enumerate(chains):
        turns = [resolve_turn(s, sources) for s in chain["turns"]]
        seeds = turn_seeds(ci, len(turns), SEED)
        shock = chain.get("shock_turn")
        arms: dict[str, Any] = {}
        for arm in arm_plan(chain):
            r = runners["guard"] if arm == "G" else runners["observe"]
            arms[arm] = run_chain(r, tok, turns, seeds, arm=arm, shock_turn=shock, gen=gen, is_finite=lambda r=r: be.is_finite(r.committed))
        res = {"name": chain["name"], "family": chain["family"], "shock_turn": shock, "turn_specs": chain["turns"],
               "turns_text": turns, "seeds": seeds, "arms": arms,
               "guard_interventions": intervention_summary(arms["G"]),
               "chunks": {arm: chunk_table(v) for arm, v in arms.items()}}
        if shock is not None:
            res["D_restore_equals_turn_start"] = arms["D"][shock].get("restore_equals_turn_start")
            res["D_vs_O_after_shock"] = later_turn_divergence(arms["D"], arms["O"], from_turn=shock + 1)
            res["N_vs_O_after_shock"] = later_turn_divergence(arms["N"], arms["O"], from_turn=shock + 1)
        results.append(res)
        print(f"[shock] {chain['name']} ({chain['family']}): guard={res['guard_interventions']['decisions']} "
              f"D==O:{res.get('D_vs_O_after_shock', {}).get('first_differing_turn', 'n/a')} "
              f"N!=O:{res.get('N_vs_O_after_shock', {}).get('first_differing_turn', 'n/a')} {round(time.time() - t0)}s", flush=True)
        with open(out_path + ".partial", "w", encoding="utf-8") as f:
            json.dump(results, f)

    reference = None
    if args.reference_checkpoint:
        del be, runners
        be_r, tok_r, runners_r = load(args.reference_checkpoint)
        reference = {"checkpoint_digest": be_r.checkpoint_digest, "chains": []}
        for ci, chain in enumerate(chains):
            if chain.get("shock_turn") is not None:
                continue
            turns = [resolve_turn(s, sources) for s in chain["turns"]]
            seeds = turn_seeds(ci, len(turns), SEED)
            arms = {"N": run_chain(runners_r["observe"], tok_r, turns, seeds, arm="N", shock_turn=None, gen=gen,
                                   is_finite=lambda: be_r.is_finite(runners_r["observe"].committed))}
            reference["chains"].append({"name": chain["name"], "arms": arms, "chunks": {"N": chunk_table(arms["N"])}})
            print(f"[shock] reference {chain['name']} done {round(time.time() - t0)}s", flush=True)

    shutil.copy2(__file__, os.path.join(args.out, os.path.basename(__file__)))
    shutil.copy2(args.chains, os.path.join(args.out, os.path.basename(args.chains)))
    payload = {"kind": "shock pass 1 (FABLE-105 / ASTRA-140): observational N/D/O arms plus an UNCALIBRATED rollback-only guarded arm "
                       "on native Qwen; development-only, descriptive counts; no detection-power, safety or learning claim",
               "checkpoint": os.path.abspath(args.checkpoint), "checkpoint_digest": _checkpoint_digest(args.checkpoint),
               "gen": gen, "chunk": CHUNK, "harness": {"observe": observe.to_dict(), "guard": guard.to_dict()},
               "canary_probes": spec["canary_probes"], "sources": source_identity, "chains": results, "reference": reference,
               "driver_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               "provenance": _invocation_provenance(args.device), "seconds": round(time.time() - t0, 1)}
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    if os.path.exists(out_path + ".partial"):
        os.remove(out_path + ".partial")
    print(f"[shock] done in {payload['seconds']}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
