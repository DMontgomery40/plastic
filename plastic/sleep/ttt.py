"""Sleep for the TTT backend: consolidate harness-ACCEPTED fast-weight learning into slow weights.

During a session the TTT layers' fast weights (W1, b1, W2, b2 per layer) learn from the text and the
harness decides per chunk whether that learning is kept. When the session ends the fast weights are
gone: the next session restarts from the checkpoint's learned initial fast weights W0. Sleep is the
offline step that turns what was accepted into a durable change to slow weights, under a locality
gate, and registers the result as a child model. See docs/research/2026-09-23-sleep-consolidation.md
for the design, the prior work it is measured against, and what counts as "the model improved".

Provenance: only chat turns whose every chunk was committed, scaled or projected are used. A turn with
a rolled-back or read-only chunk is excluded and counted. Observational (log-only) sessions commit
everything; the report says so per session.

Methods (the options a person picks in the playground or with ``plastic sleep``):
- ``replay``  fine-tune the target parameters with next-token cross-entropy on the accepted turns
              mixed with a replay sample of the SFT corpus (locality);
- ``distill`` teacher = this model with a source session's committed fast weights loaded and frozen;
              student = the model from a reset state; KL(teacher || student) on the session's own text,
              plus cross-entropy on replay text. The mechanism-native option: the prior absorbs what
              the fast learner knew, not only the raw text;
- ``anchor``  gradient-free: W0 += lambda * mean_s(W_s - W0) over the sessions' final effective fast
              weights. The control that tells whether session fast weights transfer at all.

Targets: ``w0`` = the TTT layers' initial fast weights (about 3.7 % of the 760M model), ``all`` = every
parameter. The gate is locality: held-out assistant NLL (mean and median per token) may not rise beyond
tolerance and the canaries, when a suite exists, may not move against the model. Recall is measured
and reported before and after; it is the result, not the gate, so a negative result is recorded.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F

from plastic.sleep.recall import RecallProbe, RecallReport, run_probes
from plastic.store import ArtifactStore

ACCEPTED_KINDS = frozenset({"commit", "scale", "project"})
FAST_WEIGHT_NAMES = ("W1", "b1", "W2", "b2")
Method = Literal["replay", "distill", "anchor"]
Target = Literal["w0", "all"]


@dataclass
class SleepConfig:
    method: Method = "replay"
    target: Target = "w0"
    steps: int = 40
    lr: float = 1e-4
    batch_size: int = 2
    seq_len: int = 512
    replay_ratio: float = 0.5          # share of each batch drawn from the SFT replay corpus
    replay_rows: int = 64              # conversations sampled from the replay corpus
    heldout_rows: int = 24             # conversations for the locality measurement
    anchor_lambda: float = 0.5
    distill_temperature: float = 1.0
    tolerance_nll: float = 0.05        # allowed rise in mean held-out assistant NLL (nats per token)
    tolerance_canary: dict[str, float] = field(default_factory=lambda: {"coherence": 0.1, "poison": 0.1})
    # behavioral locality: the share of distinct probe replies may not fall below this when it was higher before.
    # A run that lowers held-out NLL while every reply becomes the same sentence has collapsed, not learned.
    tolerance_collapse: float = 0.5
    recall_max_new_tokens: int = 48
    seed: int = 0
    device: str = "cpu"
    scan_checkpoint_groups: int = 4
    replay_subset: str = "everyday-conversations"
    # "accepted" (the product rule) or "all": consume every turn including rolled-back ones. "all" exists only
    # so an experiment can measure what the provenance rule buys; the API and UI never offer it.
    provenance: Literal["accepted", "all"] = "accepted"

    def validate(self) -> None:
        if self.method not in ("replay", "distill", "anchor"):
            raise ValueError(f"unknown sleep method {self.method!r}")
        if self.target not in ("w0", "all"):
            raise ValueError(f"unknown sleep target {self.target!r}")
        if self.steps < 1 or self.batch_size < 1 or self.seq_len < 32:
            raise ValueError("steps and batch_size must be >= 1 and seq_len >= 32")
        if not 0.0 <= self.replay_ratio <= 1.0:
            raise ValueError("replay_ratio must be within [0, 1]")
        if not 0.0 <= self.anchor_lambda <= 1.0:
            raise ValueError("anchor_lambda must be within [0, 1]")
        if self.lr <= 0 or self.distill_temperature <= 0:
            raise ValueError("lr and distill_temperature must be positive")
        if self.provenance not in ("accepted", "all"):
            raise ValueError(f"unknown provenance rule {self.provenance!r}")


@dataclass
class TurnRecord:
    session_id: str
    prompt: str
    completion: str
    n_chunks: int
    n_tokens: int
    accepted: bool
    reason: str  # accepted | rolled_back | read_only | no_chunks | missing_chunks | ambiguous_provenance | empty_completion


@dataclass
class SessionHarvest:
    session_id: str
    log_only: bool
    turns: list[TurnRecord]
    has_committed_state: bool

    @property
    def accepted_turns(self) -> list[TurnRecord]:
        return [t for t in self.turns if t.accepted]


def harvest_sessions(store: ArtifactStore, model_id: str, session_ids: list[str] | None = None) -> list[SessionHarvest]:
    """Group each session's transactions under its chat turns by position and keep the turns whose every
    chunk was accepted. This is the only place sleep decides what it is allowed to learn from."""
    metas = [m for m in store.list_sessions() if m.get("model_id") == model_id]
    if session_ids is not None:
        wanted = set(session_ids)
        metas = [m for m in metas if m["session_id"] in wanted]
        missing = wanted - {m["session_id"] for m in metas}
        if missing:
            raise ValueError(f"sessions not found for model {model_id}: {sorted(missing)}")
    out: list[SessionHarvest] = []
    for meta in metas:
        sid = meta["session_id"]
        trace = [t for t in store.read_trace(sid) if t.get("kind") == "chat"]
        txs = store.read_transactions(sid)
        by_index = {int(r["index"]): r for r in txs if "index" in r}
        # legacy traces (no tx_start/tx_end) can only be grouped by position, and only when positions never
        # restart: a reset mid-session makes two turns share an interval, so those turns are marked ambiguous
        legacy_ok = all(int(a.get("pos_end", 0)) < int(b.get("pos_end", 0)) for a, b in zip(trace, trace[1:])) and \
            all(int(a.get("pos_start", 0)) < int(b.get("pos_start", 0)) or int(a.get("pos_end", 0)) <= int(b.get("pos_start", 0))
                for a, b in zip(txs, txs[1:]))
        turns: list[TurnRecord] = []
        start = 0
        for t in trace:
            prompt, completion = str(t.get("prompt", "")), str(t.get("completion", ""))
            if t.get("tx_start") is not None and t.get("tx_end") is not None:
                chunks = [by_index[i] for i in range(int(t["tx_start"]), int(t["tx_end"])) if i in by_index]
                covered = len(chunks) == int(t["tx_end"]) - int(t["tx_start"])
            elif legacy_ok:
                end = int(t.get("pos_end", 0))
                chunks = [r for r in txs if int(r.get("pos_start", 0)) >= start and int(r.get("pos_end", 0)) <= end]
                start = end
                covered = True
            else:
                chunks, covered = [], False
            n_tokens = sum(int((r.get("signals") or {}).get("n_tokens", 0)) for r in chunks)
            if not chunks and not covered:
                reason, ok = ("ambiguous_provenance" if t.get("tx_start") is None else "missing_chunks"), False
            elif not chunks:
                reason, ok = "no_chunks", False
            elif not covered:
                reason, ok = "missing_chunks", False
            elif any(bool(r.get("read_only")) or (r.get("decision") or {}).get("kind") == "readonly" for r in chunks):
                reason, ok = "read_only", False
            elif any((r.get("decision") or {}).get("kind") not in ACCEPTED_KINDS for r in chunks):
                reason, ok = "rolled_back", False
            elif not completion.strip():
                reason, ok = "empty_completion", False
            else:
                reason, ok = "accepted", True
            turns.append(TurnRecord(sid, prompt, completion, len(chunks), n_tokens, ok, reason))
        harness = store.load_session_meta(sid).get("harness") or {}
        out.append(SessionHarvest(sid, bool(harness.get("log_only", False)), turns, bool(store.load_runner_state(sid).get("committed"))))
    return out


def harvest_summary(harvests: list[SessionHarvest]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    excluded_tokens = 0
    accepted_tokens = 0
    for h in harvests:
        for t in h.turns:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
            if t.accepted:
                accepted_tokens += t.n_tokens
            else:
                excluded_tokens += t.n_tokens
    return {
        "sessions": [{"session_id": h.session_id, "log_only": h.log_only, "turns": len(h.turns),
                      "accepted_turns": len(h.accepted_turns), "has_committed_state": h.has_committed_state} for h in harvests],
        "turns_by_reason": reasons,
        "accepted_tokens": accepted_tokens,
        "excluded_tokens": excluded_tokens,
    }


# ---------------------------------------------------------------------------------------------------- data


def pack_examples(examples: list[tuple[list[int], list[int]]], seq_len: int, pad_id: int) -> list[tuple[list[int], list[int]]]:
    """Greedy packing of whole examples into ``seq_len`` windows; a longer example is truncated."""
    out: list[tuple[list[int], list[int]]] = []
    buf_ids: list[int] = []
    buf_lab: list[int] = []
    for ids, labels in examples:
        ids, labels = ids[:seq_len], labels[:seq_len]
        if buf_ids and len(buf_ids) + len(ids) > seq_len:
            n = seq_len - len(buf_ids)
            out.append((buf_ids + [pad_id] * n, buf_lab + [-100] * n))
            buf_ids, buf_lab = [], []
        buf_ids += ids
        buf_lab += labels
    if buf_ids:
        n = seq_len - len(buf_ids)
        out.append((buf_ids + [pad_id] * n, buf_lab + [-100] * n))
    return out


def load_replay_conversations(subset: str, split: str, n: int, seed: int, log: Callable[[str], None]) -> list[list[dict[str, str]]]:
    """A sample of SmolTalk conversations for replay and the held-out locality measurement. Returns an
    empty list, and says so, when the dataset is unavailable (no network, no cache)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceTB/smoltalk", subset, split=split)
    except Exception as e:  # noqa: BLE001 - any failure means "no replay", which the report records
        log(f"[sleep] replay corpus unavailable ({type(e).__name__}: {e}); continuing without replay")
        return []
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [ds[i]["messages"] for i in idx[:n]]


# ---------------------------------------------------------------------------------------------------- model


def fast_weight_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    """The TTT layers' initial fast weights W0: the ``w0`` target set."""
    out = []
    for name, p in model.named_parameters():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "seq_modeling_block" and parts[-1] in FAST_WEIGHT_NAMES:
            out.append((name, p))
    if not out:
        raise RuntimeError("no TTT fast-weight parameters found on the model")
    return out


def select_target(model, target: Target) -> list[torch.nn.Parameter]:
    model.requires_grad_(False)
    if target == "all":
        params = [p for _, p in model.named_parameters()]
    else:
        params = [p for _, p in fast_weight_parameters(model)]
    for p in params:
        p.requires_grad_(True)
    return params


def anchor_update(model, session_leaves: list[list[torch.Tensor]], lam: float) -> dict[str, float]:
    """``W0 += lam * mean_s(W_s - W0)`` per fast-weight parameter, in the backend's leaf order (layer-major,
    then W1, b1, W2, b2). ``session_leaves[s][i]`` is session s's final effective fast weight i with a
    leading batch dimension of 1. Returns the per-parameter relative update size for the report."""
    params = [p for _, p in fast_weight_parameters(model)]
    if not session_leaves:
        raise ValueError("anchor needs at least one session with a committed state")
    for leaves in session_leaves:
        if len(leaves) != len(params):
            raise ValueError(f"session state has {len(leaves)} fast-weight leaves; the model has {len(params)}")
    rel: dict[str, float] = {}
    with torch.no_grad():
        for i, (name, p) in enumerate(fast_weight_parameters(model)):
            stack = torch.stack([leaves[i].reshape(p.shape).to(p.device, p.dtype) for leaves in session_leaves])
            delta = lam * (stack.mean(0) - p)
            rel[name] = float(delta.norm() / (p.norm() + 1e-12))
            p.add_(delta)
    return rel


# ---------------------------------------------------------------------------------------------------- measurement


def heldout_nll(backend, conversations: list[list[dict[str, str]]], seq_len: int) -> dict[str, float | int]:
    """Assistant-token NLL from a reset state, teacher-forced: mean and median per token, so a few
    long tails cannot hide a shift (Dennis et al. 2026 found the median tracks accuracy)."""
    from plastic.backends.ttt_lm.backend import encode_conversation

    per_token: list[float] = []
    for msgs in conversations:
        ids, labels = encode_conversation(backend.tokenizer, msgs)
        ids, labels = ids[:seq_len], labels[:seq_len]
        if len(ids) < 2:
            continue
        logits = backend.logits_full(ids).float()
        tgt = torch.tensor(labels[1:], device=logits.device)
        mask = tgt != -100
        if int(mask.sum()) == 0:
            continue
        nll = F.cross_entropy(logits[:-1][mask], tgt[mask], reduction="none")
        per_token.extend(float(v) for v in nll.cpu())
    if not per_token:
        return {"mean": float("nan"), "median": float("nan"), "tokens": 0}
    return {"mean": statistics.fmean(per_token), "median": statistics.median(per_token), "tokens": len(per_token)}


def fresh_session_answer(backend, *, max_new_tokens: int) -> Callable[[str], str]:
    """Greedy chat from a reset state through the transaction path (log-only): the behavioral probe."""
    from plastic.config import ModelConfig
    from plastic.harness.config import HarnessConfig
    from plastic.harness.transaction import TransactionRunner
    from plastic.session.runner import _TTTTextIO, drive_chat_turn

    io = _TTTTextIO(backend)
    runner = TransactionRunner(None, ModelConfig(domain="text", chunk=int(backend.mini_batch)),
                               HarnessConfig(log_only=True, learn_from_generation=True, enable_projection=False, enable_budget=False),
                               device=backend.device, backend=backend)

    def answer(question: str) -> str:
        runner.reset()
        runner.transactions = []
        completion, _, _ = drive_chat_turn(runner, io, question, max_new_tokens=max_new_tokens, temperature=1e-3, top_k=1,
                                           gen=torch.Generator().manual_seed(0))
        return completion

    return answer


def _canary_scores(store: ArtifactStore, backend, model_id: str) -> dict[str, float] | None:
    from plastic.harness.canary import CanarySuite

    path = store.canary_path(model_id)
    if not os.path.exists(path):
        return None
    suite = CanarySuite.load(path)
    return backend.score_suite(backend.init_state(), suite)


# ---------------------------------------------------------------------------------------------------- the job


def _log_to(path: str, log: Callable[[str], None]) -> Callable[[str], None]:
    def both(msg: str) -> None:
        log(msg)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return both


def sleep_ttt(
    store: ArtifactStore,
    model_id: str,
    cfg: SleepConfig,
    *,
    session_ids: list[str] | None = None,
    probes: list[RecallProbe] | None = None,
    run_dir: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run one sleep transaction for a TTT model record. Returns the report; registers a child model only
    when the locality gate passes. Never mutates the parent's files or its sessions."""
    from plastic.backends.ttt_lm.backend import TTTBackend, _checkpoint_digest, encode_conversation

    cfg.validate()
    record = store.load_model_record(model_id)
    if record.get("backend") != "ttt":
        raise ValueError(f"sleep_ttt needs a ttt model record; {model_id} is {record.get('backend', 'plastic')!r}")
    ckpt = record.get("checkpoint_dir")
    if not ckpt or not os.path.isdir(ckpt):
        raise ValueError(f"model {model_id} has no checkpoint directory")
    run_id = f"sleep_{int(time.time())}_{cfg.method}_{cfg.target}"
    run_dir = run_dir or os.path.join(store.root, "sleep", run_id)
    os.makedirs(run_dir, exist_ok=True)
    log = _log_to(os.path.join(run_dir, "log.txt"), log)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    t0 = time.time()
    report: dict[str, Any] = {"run_id": run_id, "parent_model_id": model_id, "config": asdict(cfg), "status": "running",
                              "created_at_unix": int(t0)}

    def save_report() -> None:
        with open(os.path.join(run_dir, "sleep_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)

    # 1) provenance: what sleep may learn from
    harvests = harvest_sessions(store, model_id, session_ids)
    report["harvest"] = harvest_summary(harvests)
    if cfg.provenance == "all":  # experiment-only control: rolled-back and read-only turns are consumed too
        accepted = [t for h in harvests for t in h.turns if t.completion.strip() and t.n_chunks > 0]
        report["harvest"]["provenance"] = "all (control: rolled-back turns included)"
    else:
        accepted = [t for h in harvests for t in h.accepted_turns]
        report["harvest"]["provenance"] = "accepted"
    log(f"[sleep] {model_id}: {len(harvests)} sessions, {len(accepted)} accepted turns, "
        f"{report['harvest']['accepted_tokens']} accepted tokens, {report['harvest']['excluded_tokens']} excluded")
    if not accepted:
        report["status"] = "rejected"
        report["reason"] = "no accepted turns to consolidate"
        save_report()
        return report

    # 2) the model to change (a separate copy from any live session's model) and the measurements before
    be = TTTBackend.load(ckpt, device=cfg.device, scan_checkpoint_groups=cfg.scan_checkpoint_groups)
    model, tok = be.model, be.tokenizer
    heldout = load_replay_conversations(cfg.replay_subset, "test", cfg.heldout_rows, cfg.seed + 1, log)
    replay = load_replay_conversations(cfg.replay_subset, "train", cfg.replay_rows, cfg.seed, log) if cfg.replay_ratio > 0 else []
    answer = fresh_session_answer(be, max_new_tokens=cfg.recall_max_new_tokens)
    before = {
        "heldout_nll": heldout_nll(be, heldout, cfg.seq_len) if heldout else None,
        "canary": _canary_scores(store, be, model_id),
        "recall": run_probes(probes, answer).to_dict() if probes else None,
    }
    report["before"] = before
    log(f"[sleep] before: heldout NLL {_fmt(before['heldout_nll'])}; recall {_fmt_recall(before['recall'])}")
    save_report()

    # 3) consolidate
    losses: list[float] = []
    if cfg.method == "anchor":
        leaves = []
        for h in harvests:
            if not h.accepted_turns:
                continue
            st = store.load_runner_state(h.session_id).get("committed")
            if st is None:
                continue
            try:
                state = be.load_state_dict(st)
            except ValueError as e:  # saved against another checkpoint: not this model's fast weights
                report.setdefault("skipped_states", []).append({"session_id": h.session_id, "reason": str(e)})
                log(f"[sleep] skipping {h.session_id}: {e}")
                continue
            leaves.append([t.detach().clone() for t in state.effective_leaves(be._c15)])
        if not leaves:
            report["status"] = "rejected"
            report["reason"] = "anchor needs committed session states"
            save_report()
            return report
        report["anchor_relative_update"] = anchor_update(model, leaves, cfg.anchor_lambda)
        log(f"[sleep] anchor: {len(leaves)} session states folded with lambda {cfg.anchor_lambda}")
    else:
        pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        session_examples = [encode_conversation(tok, [{"role": "user", "content": t.prompt}, {"role": "assistant", "content": t.completion}])
                            for t in accepted]
        session_packed = pack_examples(session_examples, cfg.seq_len, pad_id)
        replay_packed = pack_examples([encode_conversation(tok, m) for m in replay], cfg.seq_len, pad_id) if replay else []
        report["packed"] = {"session": len(session_packed), "replay": len(replay_packed)}
        teachers = None
        if cfg.method == "distill":
            teachers = _teacher_states(store, be, harvests, report, log)
            if not teachers:
                report["status"] = "rejected"
                report["reason"] = "distill needs committed session states"
                save_report()
                return report
        params = select_target(model, cfg.target)
        opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
        n_replay = int(round(cfg.batch_size * cfg.replay_ratio)) if replay_packed else 0
        n_session = max(1, cfg.batch_size - n_replay)
        dev = be.device
        model.eval()  # no dropout in this model; eval keeps the forward identical to inference
        for step in range(1, cfg.steps + 1):
            lr = cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / max(1, cfg.steps))))
            for g in opt.param_groups:
                g["lr"] = lr
            sess = [session_packed[rng.randrange(len(session_packed))] for _ in range(n_session)]
            rep = [replay_packed[rng.randrange(len(replay_packed))] for _ in range(n_replay)]
            loss = torch.zeros((), device=dev)
            with torch.enable_grad():
                if cfg.method == "replay":
                    batch = sess + rep
                    x = torch.tensor([b[0] for b in batch], dtype=torch.long, device=dev)
                    y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=dev)
                    logits = model(x, use_cache=False).logits[:, :-1].float()
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100)
                else:
                    x = torch.tensor([b[0] for b in sess], dtype=torch.long, device=dev)
                    with torch.no_grad():
                        t_logits = torch.stack([_teacher_logits(be, teachers[rng.randrange(len(teachers))], b[0]) for b in sess])
                    s_logits = model(x, use_cache=False).logits.float()
                    T = cfg.distill_temperature
                    pad_mask = (x != pad_id).float()
                    kl = F.kl_div(F.log_softmax(s_logits / T, -1), F.log_softmax(t_logits / T, -1), log_target=True, reduction="none").sum(-1)
                    loss = (kl * pad_mask).sum() / pad_mask.sum().clamp_min(1.0) * (T * T)
                    if rep:
                        xr = torch.tensor([b[0] for b in rep], dtype=torch.long, device=dev)
                        yr = torch.tensor([b[1] for b in rep], dtype=torch.long, device=dev)
                        lr_logits = model(xr, use_cache=False).logits[:, :-1].float()
                        loss = loss + F.cross_entropy(lr_logits.reshape(-1, lr_logits.shape[-1]), yr[:, 1:].reshape(-1), ignore_index=-100)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite sleep loss at step {step}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            losses.append(float(loss.detach()))
            if step == 1 or step % 5 == 0 or step == cfg.steps:
                log(f"[sleep] step {step}/{cfg.steps} loss {losses[-1]:.4f} lr {lr:.2e}")
                report["losses"] = losses
                save_report()
        model.requires_grad_(False)
        if dev.type == "mps":
            torch.mps.synchronize()
    report["losses"] = losses

    # 4) measurements after, and the gate
    after = {
        "heldout_nll": heldout_nll(be, heldout, cfg.seq_len) if heldout else None,
        "canary": _canary_scores(store, be, model_id),
        "recall": run_probes(probes, answer).to_dict() if probes else None,
    }
    report["after"] = after
    log(f"[sleep] after:  heldout NLL {_fmt(after['heldout_nll'])}; recall {_fmt_recall(after['recall'])}")
    gate = gate_from_measurements(before, after, tolerance_nll=cfg.tolerance_nll, tolerance_canary=cfg.tolerance_canary,
                                  tolerance_collapse=cfg.tolerance_collapse)
    report["gate"] = gate
    if before["recall"] and after["recall"]:
        report["recall_gain"] = {"recalled": after["recall"]["recalled"] - before["recall"]["recalled"],
                                 "recalled_paraphrase": after["recall"]["recalled_paraphrase"] - before["recall"]["recalled_paraphrase"],
                                 "n_probes": after["recall"]["n_probes"]}

    if gate["measured"] and not gate["passed"]:
        report["status"] = "rejected"
        report["reason"] = "locality gate failed"
        report["seconds"] = round(time.time() - t0, 1)
        save_report()
        log(f"[sleep] rejected: {[c for c in gate['checks'] if not c['passed']]}")
        return report

    # 5) the child model: a new checkpoint directory, lineage in the record, no inherited calibration
    child = store.new_model_id("sleep")
    child_dir = store.model_dir(child)
    child_ckpt = os.path.join(child_dir, "checkpoint")
    os.makedirs(child_ckpt, exist_ok=True)
    model.save_pretrained(child_ckpt, safe_serialization=True)
    tok.save_pretrained(child_ckpt)
    for name in ("generation_config.json",):
        src = os.path.join(ckpt, name)
        if os.path.exists(src) and not os.path.exists(os.path.join(child_ckpt, name)):
            shutil.copy2(src, child_ckpt)
    report["status"] = "accepted" if gate["measured"] else "accepted_unmeasured"
    report["model_id"] = child
    report["seconds"] = round(time.time() - t0, 1)
    save_report()
    shutil.copy2(os.path.join(run_dir, "sleep_report.json"), os.path.join(child_dir, "sleep_report.json"))
    inherited = {k: record[k] for k in ("domain", "chunk", "chat_tuned", "params", "source") if k in record}
    store.register_model(child, {
        **inherited,
        "backend": "ttt", "status": "completed", "type": "sleep",
        "parent_model_id": model_id, "checkpoint_dir": os.path.abspath(child_ckpt),
        "checkpoint_digest": _checkpoint_digest(child_ckpt),
        "sleep": {k: report[k] for k in ("run_id", "config", "harvest", "gate", "recall_gain", "seconds") if k in report},
    })
    log(f"[sleep] {report['status']} -> {child} ({report['seconds']} s)")
    return report


def gate_from_measurements(before: dict[str, Any], after: dict[str, Any], *, tolerance_nll: float, tolerance_canary: dict[str, float],
                           tolerance_collapse: float = 0.5) -> dict[str, Any]:
    """The locality gate. ``measured`` says whether any check could run; ``passed`` is None when nothing was
    measured, so an unmeasured run is never reported as verified. A nonfinite measurement fails its check.
    With recall probes present, the share of distinct replies after sleep may not drop below
    ``tolerance_collapse`` when it was at or above it before: a perplexity drop with collapsed replies is a
    failure (observed on the step-100 checkpoint, 40 steps on all parameters: NLL fell, 6 of 15 answers
    became the same sentence)."""
    gate: dict[str, Any] = {"passed": None, "measured": False, "checks": []}

    def check(name: str, value: float, limit: float, ok: bool) -> None:
        finite = isinstance(value, (int, float)) and math.isfinite(value)
        gate["checks"].append({"name": name, "value": float(value) if finite else None, "limit": limit, "passed": bool(ok and finite)})

    if before.get("heldout_nll") and after.get("heldout_nll"):
        rise = after["heldout_nll"]["mean"] - before["heldout_nll"]["mean"]
        check("heldout_nll_mean_rise", rise, tolerance_nll, rise <= tolerance_nll)
    if before.get("canary") and after.get("canary"):
        d_coh = after["canary"]["coherence"] - before["canary"]["coherence"]
        d_poi = after["canary"]["poison"] - before["canary"]["poison"]
        lim_c, lim_p = tolerance_canary.get("coherence", 0.1), tolerance_canary.get("poison", 0.1)
        check("canary_coherence_rise", d_coh, lim_c, d_coh <= lim_c)
        check("canary_poison_drop", d_poi, lim_p, d_poi >= -lim_p)
    rb, ra = (before.get("recall") or {}).get("distinct_ratio"), (after.get("recall") or {}).get("distinct_ratio")
    if rb is not None and ra is not None:
        ok = (ra >= tolerance_collapse) or (ra >= rb)  # a run may not push distinct replies below the floor unless they already were
        check("reply_distinct_ratio", ra, tolerance_collapse, ok)
    gate["measured"] = bool(gate["checks"])
    gate["passed"] = all(c["passed"] for c in gate["checks"]) if gate["checks"] else None
    if not gate["measured"]:
        gate["note"] = "no locality measurement was available (no replay corpus and no canary suite): this run is exploratory, not verified"
    return gate


def _teacher_states(store: ArtifactStore, be, harvests: list[SessionHarvest], report: dict[str, Any], log: Callable[[str], None]) -> list[Any]:
    """Committed states, one per source session with accepted turns; the teacher reads each through a
    folded cache with the inner step disabled, the same fixed function the canaries use. A state saved
    against another checkpoint is skipped and recorded."""
    out = []
    for h in harvests:
        if not h.accepted_turns:
            continue
        st = store.load_runner_state(h.session_id).get("committed")
        if st is None:
            continue
        try:
            out.append(be.load_state_dict(st))
        except ValueError as e:
            report.setdefault("skipped_states", []).append({"session_id": h.session_id, "reason": str(e)})
            log(f"[sleep] skipping {h.session_id}: {e}")
    return out


@torch.no_grad()
def _teacher_logits(be, state, ids: list[int]) -> torch.Tensor:
    from plastic.backends.ttt_lm.backend import TTTState

    probe = be._folded_probe_cache(state)
    logits, _ = be.process(ids, TTTState(probe), freeze=True)
    return logits.float()


def _fmt(d: dict[str, float | int] | None) -> str:
    if not d or not isinstance(d.get("mean"), float) or math.isnan(d["mean"]):
        return "n/a"
    return f"mean {d['mean']:.3f} median {d['median']:.3f} over {d['tokens']} tokens"


def _fmt_recall(d: dict[str, Any] | None) -> str:
    if not d:
        return "n/a"
    s = f"{d['recalled']}/{d['n_probes']}"
    if d.get("n_paraphrase"):
        s += f" (paraphrase {d['recalled_paraphrase']}/{d['n_paraphrase']})"
    return s
