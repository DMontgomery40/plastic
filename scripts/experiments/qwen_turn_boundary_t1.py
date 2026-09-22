"""T1 of docs/research/2026-09-22-qwen-turn-boundary-retention.md, under the protocol predeclared in
scratchpad FABLE-097 (read it first; any change needs a new declared entry).

The turn-boundary retention CANDIDATE for native Qwen, applied by this driver on top of an always log-only
TransactionRunner (no gate acts inside a turn): at each turn end the driver scores the turn's chunks and
either COMMITS the post-turn state or DISCARDS the turn by restoring the full turn-start runner snapshot
(model cache + detector). Calibrated per TURN on the FIT split (16 chains x 4 turns), then evaluated on the
DEV split (4 chains x 4 turns) against a log-only arm and the current chunk-level guarded semantics.

Development-only and descriptive: 16 DEV turns are counts, not rates; low benign discards cannot show
detection power (no positive control); a discard does not protect the current turn's already-emitted
output; retention is context carry, not learning utility. The locked eval split and follow-up fixtures are
never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any

SESSION_TURNS = 4
TURN_ALPHA = 1.0 / 60.0        # per gate; three gates -> ~5% nominal per-turn discard budget (union bound)
CUSUM_K = 0.5
H_GRID = [3.0 + 0.5 * i for i in range(35)]  # 3.0 .. 20.0, no h_min floor (FABLE-097 #5)


# ---------------------------------------------------------------- pure boundary logic (unit-tested)

def scored_chunks(records: list[dict[str, Any]], *, session_first_turn: bool) -> list[tuple[float, float]]:
    """(chunk_loss, log_delta_norm) of the chunks a gate may score. Startup treatment (FABLE-097 #3): in the
    session's FIRST turn, chunk 0 -- the only chunk written into an empty state -- is recorded but never
    scored; every other chunk, including later turns' first chunks, is scored."""
    recs = records[1:] if session_first_turn else records
    return [(float(r["signals"]["chunk_loss"]), float(r["signals"]["log_delta_norm"])) for r in recs]


def turn_maxima(scored: list[tuple[float, float]]) -> tuple[float | None, float | None]:
    if not scored:
        return None, None
    return max(s[0] for s in scored), max(s[1] for s in scored)


def cusum_turn(zs: list[float], state: tuple[float, float], *, k: float, h: float) -> tuple[bool, tuple[float, float]]:
    """Run the two-sided CUSUM (lower side kept) over a turn's scored z values from ``state`` = (s_hi, s_lo).
    Every z is applied (no short-circuit on the first alarm); returns (alarmed during the turn, end state)."""
    from plastic.harness.stats import Cusum

    c = Cusum(k, h)
    c.s_hi, c.s_lo = float(state[0]), float(state[1])
    alarms = [c.update(float(z)) for z in zs]
    return any(alarms), (c.s_hi, c.s_lo)


def fit_h(chains_z: list[list[list[float]]], *, k: float, max_rate: float, grid: list[float]) -> tuple[float | None, float | None]:
    """Smallest h on ``grid`` whose per-turn alarm fraction over the fit chains, under lifecycle L (detector
    carried across committed turns, RESTORED to its turn-start value after an alarming = discarded turn,
    reset per chain), is <= ``max_rate``. (None, None) if no grid value meets it."""
    for h in grid:
        n_turns = n_alarm = 0
        for chain in chains_z:
            state = (0.0, 0.0)
            for zs in chain:
                alarm, end = cusum_turn(zs, state, k=k, h=h)
                n_turns += 1
                if alarm:
                    n_alarm += 1          # discarded under L: detector state restored, not advanced
                else:
                    state = end
        if n_turns and n_alarm / n_turns <= max_rate:
            return h, n_alarm / n_turns
    return None, None


def decide(g1: float | None, g2: float | None, alarm: bool, th: dict[str, float]) -> tuple[bool, list[str]]:
    """The declared boundary rule: DISCARD iff G1 > tau1 or G2 > tau2 or the CUSUM alarmed in the turn.
    A turn with no scored chunk cannot fire G1/G2 (declared)."""
    reasons = []
    if g1 is not None and g1 > th["chunk_loss"]:
        reasons.append(f"G1_chunk_loss({g1:.4g}>{th['chunk_loss']:.4g})")
    if g2 is not None and g2 > th["log_delta_norm"]:
        reasons.append(f"G2_log_delta_norm({g2:.4g}>{th['log_delta_norm']:.4g})")
    if alarm:
        reasons.append("G3_cusum_alarm")
    return bool(reasons), reasons


def candidate_chain(runner, turns: list[tuple[int, str, int]], *, run_turn, digest, is_finite, th: dict[str, float],
                    ref: list[float], h: float, k: float = CUSUM_K) -> list[dict[str, Any]]:
    """One session under lifecycle L. ``turns`` = [(prompt id, prompt, seed)]. The runner is log-only; this
    function applies the boundary rule at each turn end. A discard restores the full turn-start runner state
    (model + the driver's detector) while audit identities stay monotonic (``n_transactions`` is set back to
    its pre-restore value). A nonfinite committed state is restored and ends the chain as an execution
    failure. Returns per-turn records with digests (not booleans alone)."""
    from plastic.harness.stats import robust_z

    runner.reset()
    det = (0.0, 0.0)
    out = []
    for t, (pid, prompt, seed) in enumerate(turns):
        snap, det_before = runner.state_dict(), det
        d_before = digest()
        turn = run_turn(prompt, seed)
        d_after = digest()
        scored = scored_chunks(turn["records"], session_first_turn=(t == 0))
        g1, g2 = turn_maxima(scored)
        zs = [robust_z(s[1], ref) for s in scored]
        alarm, det_end = cusum_turn(zs, det, k=k, h=h)
        discard, reasons = decide(g1, g2, alarm, th)
        failure = not is_finite()
        if failure:
            discard, reasons = True, reasons + ["execution_failure:nonfinite_state"]
        rec = {"turn": t, "id": pid, "seed": seed, **{k2: turn[k2] for k2 in ("completion", "out_ids", "n_in", "outcome")},
               "records": turn["records"], "G1": g1, "G2": g2, "G3_alarm": alarm, "cusum_z": zs,
               "detector_before": list(det_before), "detector_end": list(det_end),
               "decision": "discard" if discard else "commit", "reasons": reasons,
               "digest_before": d_before, "digest_after_turn": d_after}
        if discard:
            n_tx = runner.n_transactions
            runner.load_state_dict(snap)
            runner.n_transactions = n_tx              # audit identities never rewind or get reused
            rec["digest_after_restore"] = digest()
            rec["restore_equals_turn_start"] = rec["digest_after_restore"] == d_before
            rec["n_transactions_after_restore"] = runner.n_transactions
            det = det_before                          # detector restored with the model (FABLE-097 #2)
        else:
            det = det_end
        out.append(rec)
        if failure:
            rec["chain_ended"] = "execution_failure"
            break
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mps-run", required=True, help="completed MPS dev-diagnostic out dir (pinned split + calibration)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import glob

    import torch

    from plastic.backends.qwen import QwenBackend, _checkpoint_digest
    from plastic.config import ModelConfig
    from plastic.harness.calibrate import Calibration, conformal_threshold, log_only
    from plastic.harness.config import HarnessConfig
    from plastic.harness.stats import robust_z
    from plastic.harness.transaction import TransactionRunner
    from plastic.session.runner import _QwenTextIO, drive_chat_turn
    from scripts.experiments.qwen_operating_point import CHUNK, DEV_SEED_OFFSET, SEED, TARGET_FPR, _gen, _invocation_provenance

    out_path = os.path.join(args.out, "t1.json")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; refusing to overwrite")
    manifest = json.load(open(os.path.join(args.mps_run, "split-manifest.json")))
    fit, dev = manifest["selected_records"]["fit"], manifest["selected_records"]["dev"]
    assert len(fit) == 64 and len(dev) == 16
    old_cal = Calibration.load(glob.glob(os.path.join(args.mps_run, "store", "models", "*"))[0])
    be = QwenBackend.load(args.checkpoint, device=args.device)
    assert _checkpoint_digest(args.checkpoint) == manifest["checkpoint_digest"]
    tok = _QwenTextIO(be)
    gen = _gen(64)
    # EXACTLY the recorded dev run's guarded config ("current semantics" arm); the candidate/log-only arms run
    # log_only(guarded), where no gate -- projection/canary/Fisher/budget/scale included -- can act (and Qwen
    # produces no canary/projection/Fisher signals at all)
    guarded = HarnessConfig(target_fpr=TARGET_FPR, enable_stats=True, enable_rollback=True, log_only=False,
                            learn_from_generation=True, freeze_on_alarm=True, alarm_cooldown=0)
    cfg = ModelConfig(domain="text", chunk=CHUNK)

    def make_runner(hcfg):
        return TransactionRunner(None, cfg, hcfg, calibration=old_cal, device=be.device, backend=be)

    def turn_fn(runner):
        def run_turn(prompt, seed):
            runner.transactions = []
            completion, out_ids, in_ids = drive_chat_turn(runner, tok, prompt, max_new_tokens=gen["max_new_tokens"],
                                                          temperature=gen["temperature"], top_k=gen["top_k"],
                                                          gen=torch.Generator().manual_seed(int(seed)))
            recs = [{k: v for k, v in t.items() if k != "t_unix"} for t in runner.transactions]
            outcome = "cap" if len(out_ids) >= gen["max_new_tokens"] else ("empty" if not out_ids else "eos")
            return {"completion": completion, "out_ids": [int(x) for x in out_ids], "n_in": len(in_ids),
                    "outcome": outcome, "records": recs}
        return run_turn

    def digest_fn(runner):
        return lambda: {"committed": _digest_cache(runner.committed), "working": _digest_cache(runner.working),
                        "anchor": _digest_cache(runner.anchor), "position": runner.committed.position}

    def chains_of(records, seed_of):
        return [[(r["id"], r["prompt"], seed_of(i)) for i, r in list(enumerate(records))[c * SESSION_TURNS:(c + 1) * SESSION_TURNS]]
                for c in range(len(records) // SESSION_TURNS)]

    t0 = time.time()
    lo = make_runner(log_only(guarded))
    run_lo = turn_fn(lo)

    # ---- FIT: 16 chains x 4 turns, log-only, lifecycle L without model discards (declared approximation)
    fit_chains, fit_turns = chains_of(fit, lambda i: SEED + i), []
    for chain in fit_chains:
        lo.reset()
        fit_turns.append([run_lo(p, s) | {"id": pid, "seed": s} for pid, p, s in chain])
    scored = [[scored_chunks(t["records"], session_first_turn=(j == 0)) for j, t in enumerate(ch)] for ch in fit_turns]
    maxima = [turn_maxima(s) for ch in scored for s in ch]
    g1s = [m[0] for m in maxima if m[0] is not None]
    g2s = [m[1] for m in maxima if m[1] is not None]
    tau1, ach1 = conformal_threshold(g1s, TURN_ALPHA)
    tau2, ach2 = conformal_threshold(g2s, TURN_ALPHA)
    ref = [s[1] for ch in scored for turn in ch for s in turn]
    chains_z = [[[robust_z(s[1], ref) for s in turn] for turn in ch] for ch in scored]
    h, h_rate = fit_h(chains_z, k=CUSUM_K, max_rate=1.0 / 64.0, grid=H_GRID)
    th = {"chunk_loss": tau1, "log_delta_norm": tau2}
    fit_summary = {"n_turns": len(maxima), "tau1_chunk_loss": tau1, "tau1_achievable": ach1, "tau2_log_delta_norm": tau2,
                   "tau2_achievable": ach2, "cusum_h": h, "cusum_fit_turn_alarm_rate": h_rate, "cusum_reference_n": len(ref),
                   "fit_turn_maxima": maxima, "seconds": round(time.time() - t0, 1)}
    print("[t1] fit:", json.dumps({k: v for k, v in fit_summary.items() if k != "fit_turn_maxima"}), flush=True)
    if h is None:
        raise SystemExit("no CUSUM h on the declared grid meets the declared fit rate; report and stop (FABLE-097 #5)")

    # ---- EVAL on DEV: 4 chains x 4 turns, three arms
    dev_chains = chains_of(dev, lambda i: SEED + DEV_SEED_OFFSET + i)
    arms: dict[str, list] = {"log_only": [], "current": [], "candidate": []}
    cur = make_runner(guarded)
    run_cur = turn_fn(cur)
    for chain in dev_chains:
        lo.reset()
        arms["log_only"].append([run_lo(p, s) | {"id": pid, "seed": s} for pid, p, s in chain])
        cur.reset()
        arms["current"].append([run_cur(p, s) | {"id": pid, "seed": s, "read_only_end": cur.read_only} for pid, p, s in chain])
        arms["candidate"].append(candidate_chain(lo, chain, run_turn=run_lo, digest=digest_fn(lo),
                                                 is_finite=lambda: be.is_finite(lo.committed), th=th, ref=ref, h=h))
        print(f"[t1] chain {len(arms['candidate'])}: candidate decisions "
              f"{[(r['decision'], r['reasons']) for r in arms['candidate'][-1]]}", flush=True)

    def outcomes(turns):
        flat = [t for ch in turns for t in ch]
        return {o: sum(t["outcome"] == o for t in flat) for o in ("empty", "cap", "eos")}

    cand = [t for ch in arms["candidate"] for t in ch]
    discards = [t for t in cand if t["decision"] == "discard"]
    summary = {
        "outcomes": {arm: outcomes(v) for arm, v in arms.items()},
        "candidate_turns": len(cand), "candidate_discards": len(discards),
        "candidate_discard_reasons": [(t["id"], t["turn"], t["reasons"]) for t in discards],
        "restores_equal_turn_start": sum(bool(t.get("restore_equals_turn_start")) for t in discards),
        "audit_monotonic": all(
            all(b["records"][0]["index"] > a["records"][-1]["index"] for a, b in zip(ch, ch[1:]) if a["records"] and b["records"])
            for ch in arms["candidate"]),
        "current_read_only_end": sum(t["read_only_end"] for ch in arms["current"] for t in ch),
        "candidate_turns_identical_to_log_only_until_first_discard": None,
        "seconds": round(time.time() - t0, 1),
    }
    ident = 0
    for c_ch, l_ch in zip(arms["candidate"], arms["log_only"]):
        for c_t, l_t in zip(c_ch, l_ch):
            if c_t["out_ids"] != l_t["out_ids"]:
                break
            ident += 1
            if c_t["decision"] == "discard":
                break
    summary["candidate_turns_identical_to_log_only_until_first_discard"] = ident
    os.makedirs(args.out, exist_ok=True)
    payload = {"kind": "T1 (FABLE-097 protocol): turn-boundary retention candidate on DEV chains; development-only, "
                       "descriptive counts; not a policy FPR, detection-power, learning-utility or safety result",
               "declaration": "SHARED_SCRATCHPAD FABLE-097", "fit": fit_summary, "thresholds": th, "cusum_h": h,
               "summary": summary, "arms": arms, "fit_turns": fit_turns,
               "provenance": _invocation_provenance(args.device), "checkpoint_digest": manifest["checkpoint_digest"],
               "mps_run": os.path.abspath(args.mps_run)}
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(out_path + ".tmp", out_path)
    print("[t1] summary:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
