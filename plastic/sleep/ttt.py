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
Method = Literal["replay", "distill", "anchor", "dream"]
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
    # behavioral locality: after sleep, no single reply may be given to more than this share of the probes (unless it
    # already was before). A run that lowers held-out NLL while many questions get the same sentence has collapsed.
    tolerance_collapse: float = 0.25
    # which tokens of a SESSION turn carry the loss: "all" (the user's statements are the content to consolidate;
    # default) or "assistant" (SFT-style, which can only teach the model to reproduce its own replies).
    session_loss: Literal["all", "assistant"] = "all"
    # weight of the user's (prompt) tokens relative to the assistant's when session_loss="all"; 1.0 = equal.
    # Prompt-loss weights near 0.2 were the optimum for short completions in arXiv 2401.13586.
    prompt_loss_weight: float = 1.0
    recall_max_new_tokens: int = 48
    seed: int = 0
    device: str = "cpu"
    scan_checkpoint_groups: int = 4
    replay_subset: str = "everyday-conversations"
    # "accepted" (the product rule) or "all": consume every turn including rolled-back ones. "all" exists only
    # so an experiment can measure what the provenance rule buys; the API and UI never offer it.
    provenance: Literal["accepted", "all"] = "accepted"
    # accepted turns the policy flagged (scaled/projected, or would-have-intervened in observational mode): "exclude"
    # from sleep (default; online acceptance is necessary, not sufficient), "downweight" their rows by flagged_weight,
    # or "include" them like any other accepted turn. Importance by surprise is highest for exactly this content.
    flagged_policy: Literal["exclude", "downweight", "include"] = "exclude"
    flagged_weight: float = 0.25

    # dream: the session's fast weights (teacher) generate study items; only dreams the teacher finds at least
    # ``dream_min_gain`` nats/token more likely than the reset model are kept (they carry session information)
    dream_per_prompt: int = 2
    dream_min_gain: float = 0.2
    dream_max_keep: int = 24
    dream_max_new_tokens: int = 48
    dream_temperature: float = 0.7
    # "uniform": every reply token weighs the same in the dream KL; "gain": weighted by the token's teacher(+turn)-minus-
    # student log-ratio; "fw_gain": weighted by the fast-weights-only log-ratio (the research-specific part, OPUS-004 1a).
    # Weights are floored and average 1, so a memory is consolidated where its information lives.
    dream_token_weighting: Literal["uniform", "gain", "fw_gain"] = "uniform"

    def validate(self) -> None:
        if self.method not in ("replay", "distill", "anchor", "dream"):
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
        if self.session_loss not in ("all", "assistant"):
            raise ValueError(f"unknown session_loss {self.session_loss!r}")
        if self.flagged_policy not in ("exclude", "downweight", "include"):
            raise ValueError(f"unknown flagged_policy {self.flagged_policy!r}")
        if not 0.0 <= self.flagged_weight <= 1.0:
            raise ValueError("flagged_weight must be within [0, 1]")
        if not 0.0 <= self.prompt_loss_weight <= 1.0:
            raise ValueError("prompt_loss_weight must be within [0, 1]")
        if self.dream_temperature <= 0:
            raise ValueError("dream_temperature must be positive")
        if self.dream_token_weighting not in ("uniform", "gain", "fw_gain"):
            raise ValueError(f"unknown dream_token_weighting {self.dream_token_weighting!r}")
        if self.dream_per_prompt < 1 or self.dream_max_keep < 1 or self.dream_max_new_tokens < 4:
            raise ValueError("dream_per_prompt and dream_max_keep must be >= 1 and dream_max_new_tokens >= 4")


@dataclass
class TurnRecord:
    session_id: str
    prompt: str
    completion: str
    n_chunks: int
    n_tokens: int
    accepted: bool
    reason: str  # accepted | rolled_back | read_only | no_chunks | missing_chunks | ambiguous_provenance | empty_completion
    # accepted online, but the policy flagged a chunk of this turn: it was scaled or projected, or in observational mode
    # it would have been rolled back / scaled / projected. Online acceptance is necessary, not sufficient, for sleep.
    flagged: bool = False


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
            turns.append(TurnRecord(sid, prompt, completion, len(chunks), n_tokens, ok, reason, flagged=any(chunk_flagged(r) for r in chunks)))
        harness = store.load_session_meta(sid).get("harness") or {}
        out.append(SessionHarvest(sid, bool(harness.get("log_only", False)), turns, bool(store.load_runner_state(sid).get("committed"))))
    return out


def chunk_flagged(rec: dict[str, Any]) -> bool:
    """A chunk the policy did not pass cleanly: applied scale/project, or a requested decision other than commit, or any
    would_* reason recorded in observational mode. Its turn stays accepted online; sleep treats it separately."""
    decision = (rec.get("decision") or {}).get("kind")
    requested = rec.get("requested") or {}
    if decision in ("scale", "project") or requested.get("kind") in ("rollback", "scale", "project"):
        return True
    return any(str(r).startswith("would_") for r in requested.get("reasons") or [])


def harvest_summary(harvests: list[SessionHarvest]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    excluded_tokens = 0
    accepted_tokens = 0
    flagged_turns = 0
    for h in harvests:
        for t in h.turns:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
            flagged_turns += int(t.accepted and t.flagged)
            if t.accepted:
                accepted_tokens += t.n_tokens
            else:
                excluded_tokens += t.n_tokens
    return {
        "sessions": [{"session_id": h.session_id, "log_only": h.log_only, "turns": len(h.turns),
                      "accepted_turns": len(h.accepted_turns), "has_committed_state": h.has_committed_state} for h in harvests],
        "turns_by_reason": reasons,
        "accepted_turns_flagged": flagged_turns,
        "accepted_tokens": accepted_tokens,
        "excluded_tokens": excluded_tokens,
    }


# ---------------------------------------------------------------------------------------------------- data


def batch_mix(batch_size: int, replay_ratio: float, have_replay: bool) -> tuple[int, int]:
    """(session rows, replay rows) for one batch. The batch size is preserved exactly and at least one session
    row is always present (sleep without session rows learns nothing from the session); the replay share is
    rounded down to what the batch can realize, and the caller records the realized ratio."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not have_replay or replay_ratio <= 0:
        return batch_size, 0
    n_replay = min(batch_size - 1, int(round(batch_size * replay_ratio)))
    return batch_size - n_replay, n_replay


def session_labels(ids: list[int], labels: list[int], session_loss: str) -> list[int]:
    """Labels for one accepted turn. ``"all"``: every token after BOS is supervised, so the user's statement is
    learned, not only the assistant's reply; ``"assistant"``: the SFT labels (user tokens masked), which on the
    step-100 checkpoint only taught the model to repeat its own replies and collapsed."""
    if session_loss == "all" and ids:
        return [-100] + ids[1:]  # BOS has no target; everything else is supervised
    return labels


def session_weights(sft_labels: list[int], labels: list[int], prompt_loss_weight: float) -> list[float]:
    """Per-token loss weights: 0 where there is no target, ``prompt_loss_weight`` on tokens the SFT labels
    masked (the user's turn and tags), 1 on the assistant's tokens."""
    return [0.0 if l == -100 else (prompt_loss_weight if sft == -100 else 1.0) for sft, l in zip(sft_labels, labels)]


def session_example(tok, prompt: str, completion: str, session_loss: str, prompt_loss_weight: float = 1.0) -> tuple[list[int], list[int], list[float]]:
    from plastic.backends.ttt_lm.backend import encode_conversation

    ids, sft = encode_conversation(tok, [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}])
    labels = session_labels(ids, sft, session_loss)
    return ids, labels, session_weights(sft, labels, prompt_loss_weight)


def pack_examples(examples: list[tuple], seq_len: int, pad_id: int) -> list[tuple]:
    """Greedy packing of whole examples into ``seq_len`` windows; a longer example is truncated. Examples are
    (ids, labels) or (ids, labels, weights); padding gets label -100 and weight 0."""
    out: list[tuple] = []
    buf: list[list] = []
    def flush():
        if not buf or not buf[0]:
            return
        n = seq_len - len(buf[0])
        pads = [[pad_id] * n, [-100] * n, [0.0] * n]
        out.append(tuple(b + pads[i] for i, b in enumerate(buf)))
    for ex in examples:
        parts = [list(x)[:seq_len] for x in ex]
        if len(parts) == 2:
            parts.append([0.0 if l == -100 else 1.0 for l in parts[1]])
        if buf and buf[0] and len(buf[0]) + len(parts[0]) > seq_len:
            flush()
            buf = []
        if not buf:
            buf = [[], [], []]
        for i in range(3):
            buf[i] += parts[i]
    flush()
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


def fresh_session_answer_logprob(backend) -> Callable[[str, str], float]:
    """Mean log-probability per token of ``expected`` after the rendered user turn, from a fresh state (storage;
    compare with greedy recall for access). Rendering matches training/runtime (render_user_turn)."""
    def logprob(question: str, expected: str) -> float:
        prompt_ids = backend.encode_chat(question, first_turn=True)
        ans_ids = backend.encode(expected)
        if not ans_ids:
            return float("nan")
        ids = prompt_ids + ans_ids
        logits = backend.logits_full(ids).float()
        lp = torch.log_softmax(logits[len(prompt_ids) - 1:len(ids) - 1], dim=-1)
        tgt = torch.tensor(ans_ids, dtype=torch.long, device=lp.device)
        return float(lp.gather(1, tgt[:, None]).mean())

    return logprob


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
                              "created_at_unix": int(t0), "code_commit": _git_head()}

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
    n_flagged = sum(1 for t in accepted if t.flagged)
    if cfg.flagged_policy == "exclude" and n_flagged:
        accepted = [t for t in accepted if not t.flagged]
        report["harvest"]["flagged_excluded"] = n_flagged
        log(f"[sleep] {n_flagged} accepted turn(s) the policy flagged are excluded from sleep (flagged_policy=exclude)")
    report["harvest"]["flagged_policy"] = cfg.flagged_policy
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
    answer_lp = fresh_session_answer_logprob(be)
    before = {
        "heldout_nll": heldout_nll(be, heldout, cfg.seq_len) if heldout else None,
        "canary": _canary_scores(store, be, model_id),
        "recall": run_probes(probes, answer, answer_lp).to_dict() if probes else None,
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
        session_examples = []
        for t in accepted:
            ids_, labels_, weights_ = session_example(tok, t.prompt, t.completion, cfg.session_loss, cfg.prompt_loss_weight)
            if t.flagged and cfg.flagged_policy == "downweight":
                weights_ = [w * cfg.flagged_weight for w in weights_]
            session_examples.append((ids_, labels_, weights_))
        session_packed = pack_examples(session_examples, cfg.seq_len, pad_id)
        replay_packed = pack_examples([encode_conversation(tok, m) for m in replay], cfg.seq_len, pad_id) if replay else []
        report["packed"] = {"session": len(session_packed), "replay": len(replay_packed)}
        teachers: list[Any] = []
        teachers = None
        dream_rows: list[Any] = []  # kept Dream objects (student and teacher renderings) for the dream method
        teacher_be = be
        if cfg.method in ("distill", "dream"):
            teachers = _teacher_states(store, be, harvests, report, log)
            teacher_be = frozen_teacher_backend(be, cfg.target)
            if not teachers:
                report["status"] = "rejected"
                report["reason"] = f"{cfg.method} needs committed session states"
                save_report()
                return report
        if cfg.method == "dream":
            from plastic.sleep.dream import DreamReport, generate_dreams, select_dreams

            dream_report = DreamReport()
            candidates = []
            turns_by_session = {h.session_id: [tr.prompt for tr in h.accepted_turns] for h in harvests}
            for ti, (sid, state) in enumerate(teachers):
                ds = generate_dreams(teacher_be, state, session_id=sid, turns=turns_by_session.get(sid, []), per_prompt=cfg.dream_per_prompt,
                                     max_new_tokens=cfg.dream_max_new_tokens, temperature=cfg.dream_temperature, seed=cfg.seed + ti, log=log)
                for d in ds:
                    d.teacher_index = ti
                candidates.extend(ds)
            dream_report.generated = len(candidates)
            kept = select_dreams(candidates, min_gain=cfg.dream_min_gain, max_keep=cfg.dream_max_keep, report=dream_report)
            report["dreams"] = dream_report.to_dict()
            log(f"[sleep] dreams: {dream_report.generated} generated, {len(kept)} kept (gain >= {cfg.dream_min_gain}), "
                f"{dream_report.degenerate} degenerate, {dream_report.duplicate} duplicate, {dream_report.low_gain} low gain")
            if not kept:
                report["status"] = "rejected"
                report["reason"] = "no dream carried session information above the gain threshold"
                save_report()
                return report
            dream_rows, too_long = split_by_length(kept, cfg.seq_len)
            for d in too_long:  # accounted like every other rejection; the kept list holds only what trains
                dream_report.rejected_examples.append({"text": d.text[:120], "reason": f"too long for seq_len {cfg.seq_len} "
                                                                                         f"(student {len(d.ids)}, teacher {len(d.teacher_ids)} tokens)"})
            dream_report.kept = dream_rows
            report["dreams"] = dream_report.to_dict()
            report["dreams"]["too_long"] = len(too_long)
            if not dream_rows:
                report["status"] = "rejected"
                report["reason"] = f"every kept dream exceeded seq_len {cfg.seq_len}"
                save_report()
                return report
            session_packed = [(d.ids, d.labels, [1.0] * len(d.ids)) for d in dream_rows]  # rows are not packed: one dream per row
            report["packed"]["session"] = len(session_packed)
        params = select_target(model, cfg.target)
        opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
        n_session, n_replay = batch_mix(cfg.batch_size, cfg.replay_ratio, bool(replay_packed))
        report["batch"] = {"session_rows": n_session, "replay_rows": n_replay, "requested_replay_ratio": cfg.replay_ratio,
                           "realized_replay_ratio": n_replay / (n_session + n_replay),
                           # distill adds a session KL term and a replay CE term with unit weights; the ratio changes row
                           # sampling only, never the relative weight of those two terms
                           "loss_terms": {"replay": "weighted cross_entropy over all rows", "distill": "session_kl + replay_ce, unit weights",
                                          "dream": "reply_kl(frozen teacher) + replay_ce, unit weights"}.get(cfg.method, "n/a"),
                           "teacher_weights": ("frozen copy" if (cfg.method in ("distill", "dream") and cfg.target == "all") else
                                               "live model (only W0 changes, overridden by the session state)" if cfg.method in ("distill", "dream") else "n/a")}
        if abs(report["batch"]["realized_replay_ratio"] - cfg.replay_ratio) > 1e-9:
            log(f"[sleep] replay ratio {cfg.replay_ratio} is not realizable at batch {cfg.batch_size}: using {n_session} session + {n_replay} replay rows "
                f"(realized {report['batch']['realized_replay_ratio']:.3f})")
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
                    w = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=dev)[:, 1:]
                    logits = model(x, use_cache=False).logits[:, :-1].float()
                    per_tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100, reduction="none").view(w.shape)
                    loss = (per_tok * w).sum() / w.sum().clamp_min(1.0)
                elif cfg.method == "dream":
                    from plastic.sleep.dream import reply_slices

                    rows = [dream_rows[rng.randrange(len(dream_rows))] for _ in range(n_session)]
                    L = max(len(d.ids) for d in rows)
                    x = torch.tensor([d.ids + [pad_id] * (L - len(d.ids)) for d in rows], dtype=torch.long, device=dev)
                    T = cfg.distill_temperature
                    R = max(d.reply_len for d in rows)
                    with torch.no_grad():
                        # teacher: its own rendering (the dream continues the session), read from the frozen teacher backend
                        t_slices = []
                        for d in rows:
                            ts, _ = reply_slices(d.teacher_prefix, d.student_prefix, d.reply_len)
                            t_slices.append(_teacher_logits(teacher_be, teachers[d.teacher_index][1], d.teacher_ids)[ts])
                    s_full = model(x, use_cache=False).logits.float()
                    from plastic.sleep.dream import token_gain_weights

                    s_slices, masks = [], []
                    for i, d in enumerate(rows):
                        _, ss = reply_slices(d.teacher_prefix, d.student_prefix, d.reply_len)
                        s_slices.append(s_full[i, ss])
                        gains = {"gain": d.token_gain, "fw_gain": d.token_fw_gain}.get(cfg.dream_token_weighting)
                        w = token_gain_weights(gains) if (gains is not None and len(gains) == d.reply_len) else [1.0] * d.reply_len
                        masks.append(torch.tensor(w + [0.0] * (R - d.reply_len), device=dev))
                    t_logits = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, R - t.shape[0])) for t in t_slices])
                    s_logits = torch.stack([torch.nn.functional.pad(t, (0, 0, 0, R - t.shape[0])) for t in s_slices])
                    reply_mask = torch.stack(masks)
                    kl = F.kl_div(F.log_softmax(s_logits / T, -1), F.log_softmax(t_logits / T, -1), log_target=True, reduction="none").sum(-1)
                    loss = (kl * reply_mask).sum() / reply_mask.sum().clamp_min(1.0) * (T * T)
                else:
                    x = torch.tensor([b[0] for b in sess], dtype=torch.long, device=dev)
                    with torch.no_grad():
                        t_logits = torch.stack([_teacher_logits(teacher_be, teachers[rng.randrange(len(teachers))][1], b[0]) for b in sess])
                    s_logits = model(x, use_cache=False).logits.float()
                    T = cfg.distill_temperature
                    pad_mask = (x != pad_id).float()
                    kl = F.kl_div(F.log_softmax(s_logits / T, -1), F.log_softmax(t_logits / T, -1), log_target=True, reduction="none").sum(-1)
                    loss = (kl * pad_mask).sum() / pad_mask.sum().clamp_min(1.0) * (T * T)
                if cfg.method in ("distill", "dream") and rep:
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
        "recall": run_probes(probes, answer, answer_lp).to_dict() if probes else None,
    }
    report["after"] = after
    log(f"[sleep] after:  heldout NLL {_fmt(after['heldout_nll'])}; recall {_fmt_recall(after['recall'])}")
    gate = gate_from_measurements(before, after, tolerance_nll=cfg.tolerance_nll, tolerance_canary=cfg.tolerance_canary,
                                  tolerance_collapse=cfg.tolerance_collapse)
    report["gate"] = gate
    if before["recall"] and after["recall"]:
        lb, la = before["recall"].get("mean_answer_logprob"), after["recall"].get("mean_answer_logprob")
        report["recall_gain"] = {"recalled": after["recall"]["recalled"] - before["recall"]["recalled"],
                                 "recalled_paraphrase": after["recall"]["recalled_paraphrase"] - before["recall"]["recalled_paraphrase"],
                                 "n_probes": after["recall"]["n_probes"],
                                 # storage: how much more likely the expected answers became, independent of greedy access
                                 "answer_logprob_lift": (la - lb) if (la is not None and lb is not None) else None}

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


def _git_head() -> str:
    """Source commit at run time, best effort ('unknown' outside a git checkout); '+dirty' when the tree had changes."""
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def gate_from_measurements(before: dict[str, Any], after: dict[str, Any], *, tolerance_nll: float, tolerance_canary: dict[str, float],
                           tolerance_collapse: float = 0.25) -> dict[str, Any]:
    """The locality gate. ``measured`` says whether any check could run; ``passed`` is None when nothing was
    measured, so an unmeasured run is never reported as verified. A nonfinite measurement fails its check.
    With recall probes present, no single reply may cover more than ``tolerance_collapse`` of the probes after
    sleep unless it already did before: a perplexity drop with collapsed replies is a failure. Observed on the
    step-100 checkpoint, 40 steps on all parameters: NLL fell while one sentence answered 14 of 30 probes
    (share 0.47; 0.03 before). The real reply lists are a test fixture."""
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
    sb, sa = (before.get("recall") or {}).get("max_cluster_share"), (after.get("recall") or {}).get("max_cluster_share")
    if sb is not None and sa is not None:
        ok = (sa <= tolerance_collapse) or (sa <= sb)  # one reply for many questions is collapse, unless it already was so
        check("reply_cluster_share", sa, tolerance_collapse, ok)
    gate["measured"] = bool(gate["checks"])
    gate["passed"] = all(c["passed"] for c in gate["checks"]) if gate["checks"] else None
    if not gate["measured"]:
        gate["note"] = "no locality measurement was available (no replay corpus and no canary suite): this run is exploratory, not verified"
    return gate


def split_by_length(dreams: list[Any], seq_len: int) -> tuple[list[Any], list[Any]]:
    """Dreams whose student AND teacher renderings fit ``seq_len`` train; the rest are rejected and accounted.
    The teacher rendering quotes the turn, so it is usually the longer one."""
    fit = [d for d in dreams if len(d.ids) <= seq_len and len(d.teacher_ids) <= seq_len]
    rest = [d for d in dreams if not (len(d.ids) <= seq_len and len(d.teacher_ids) <= seq_len)]
    return fit, rest


def _teacher_states(store: ArtifactStore, be, harvests: list[SessionHarvest], report: dict[str, Any], log: Callable[[str], None]) -> list[tuple[str, Any]]:
    """(session_id, committed state) for each source session with accepted turns and a loadable state; the
    teacher reads each through a folded cache with the inner step disabled, the same fixed function the
    canaries use. A state saved against another checkpoint is skipped and recorded; identity stays bound to
    the state that loaded, so a skipped session never relabels a later one."""
    out: list[tuple[str, Any]] = []
    for h in harvests:
        if not h.accepted_turns:
            continue
        st = store.load_runner_state(h.session_id).get("committed")
        if st is None:
            continue
        try:
            out.append((h.session_id, be.load_state_dict(st)))
        except ValueError as e:
            report.setdefault("skipped_states", []).append({"session_id": h.session_id, "reason": str(e)})
            log(f"[sleep] skipping {h.session_id}: {e}")
    return out


def frozen_teacher_backend(be, target: str, factory: Callable[..., Any] | None = None):
    """The backend whose slow weights the teacher reads. With target "w0" the student only changes the initial
    fast weights, which a loaded session state overrides, so the live model serves; with target "all" every
    student step would move the teacher too (ASTRA-175), so the teacher gets its own frozen copy of the model
    wrapped in a new backend (``factory`` defaults to TTTBackend)."""
    if target == "w0":
        return be
    if factory is None:
        from plastic.backends.ttt_lm.backend import TTTBackend as factory  # noqa: N813 - the class is the factory

    model_copy = copy.deepcopy(be.model)
    model_copy.requires_grad_(False)
    model_copy.eval()
    return factory(model_copy, be.tokenizer, be.config, device=be.device, checkpoint_digest=be.checkpoint_digest)


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
    if d.get("mean_answer_logprob") is not None:
        s += f"; answer logprob {d['mean_answer_logprob']:.3f}"
    return s
