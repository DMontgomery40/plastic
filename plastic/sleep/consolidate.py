"""Consolidate what sessions learned into the slow weights, carefully.

Harvest the chat turns the harness accepted (prompt and completion) from sessions
of a model, mix them with a sample of the core corpus, train only the blocks (embeddings and head
frozen) for a few hundred small steps, and accept the candidate only if the
coherence canary does not rise and the poison canary does not fall beyond
tolerance, both scored from zero state. An accepted candidate is registered as
a child model; a rejected one leaves nothing behind but the manifest.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Any

import torch
import torch.nn.functional as F

from plastic.data.text import TokenWindows
from plastic.harness.calibrate import Calibration
from plastic.harness.canary import CanarySuite, score_suite
from plastic.store import ArtifactStore
from plastic.tokenizer.bpe import Tokenizer
from plastic.train.optim import build_optimizer


def harvest_traces(store: ArtifactStore, model_id: str, sessions: list[str] | None = None, *, flagged_policy: str = "exclude",
                   summary: dict[str, Any] | None = None) -> list[str]:
    """The chat turns native sleep may learn from, under the provenance rule the TTT path uses
    (``harvest_sessions`` + ``select_sleep_turns``): a turn counts only when every one of its chunks was accepted by
    the harness and none was read-only; accepted turns the policy flagged (scaled, projected, or would-have-intervened
    in observational mode) are dropped unless ``flagged_policy="include"``. Text the harness rolled back is never
    returned: an online refusal must also be a refusal as offline training material (external review of 6cf4457,
    finding 1). A completion is returned only when it was learning material online: PlasticCore does not learn its own
    generation unless ``learn_from_generation``, so by default a turn contributes its accepted prompt alone.
    ``summary`` receives the per-reason turn counts, the selected-turn count and how many completions were dropped."""
    from plastic.sleep.ttt import SleepConfig, harvest_sessions, harvest_summary, select_sleep_turns

    if flagged_policy not in ("exclude", "include"):
        raise ValueError(f"flagged_policy must be exclude or include for native sleep, not {flagged_policy!r}")
    harvests = harvest_sessions(store, model_id, sessions)
    info = harvest_summary(harvests)
    turns, _ = select_sleep_turns(harvests, SleepConfig(flagged_policy=flagged_policy), info)
    info["completions_not_learned"] = sum(1 for t in turns if not t.completion_learned)
    if summary is not None:
        summary.update(info)
    return [t.prompt.rstrip() + "\n" + (t.completion.rstrip() + "\n" if t.completion_learned else "") for t in turns]


def consolidate(
    store: ArtifactStore,
    model_id: str,
    *,
    sessions: list[str] | None = None,
    core_data_dir: str | None = None,
    steps: int = 200,
    lr: float = 1e-4,
    core_ratio: float = 0.8,
    seq_len: int = 256,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
    tolerance: dict[str, float] | None = None,
    seed: int = 0,
    flagged_policy: str = "exclude",
    log=print,
) -> dict[str, Any]:
    device = torch.device(device)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    cfg, model, info = store.load_checkpoint(model_id, device)
    if cfg.domain != "text":
        raise ValueError("sleep consolidation is defined for text models")
    tok = Tokenizer.load(store.tokenizer_path(model_id))
    model_dir = store.model_dir(model_id)
    suite = CanarySuite.load(store.canary_path(model_id)) if os.path.exists(store.canary_path(model_id)) else None
    if suite is None:
        if not core_data_dir:
            raise ValueError("no canary suite for the model and no core_data_dir to build one")
        suite = CanarySuite.default_text(os.path.join(core_data_dir, "validation.bin"), vocab_size=cfg.vocab_size, seed=seed)
        suite.save(store.canary_path(model_id))

    harvest: dict[str, Any] = {}
    memories = harvest_traces(store, model_id, sessions, flagged_policy=flagged_policy, summary=harvest)
    if not memories:
        raise ValueError(f"no accepted chat turns to consolidate for {model_id} (turns by reason: {harvest.get('turns_by_reason')})")
    mem_ids: list[int] = []
    for t in memories:
        mem_ids.extend(tok.encode(t, add_bos=True, add_eos=True))
    if len(mem_ids) < seq_len + 2:
        raise ValueError(f"sleep corpus too small: {len(mem_ids)} memory tokens (need > {seq_len + 1})")
    core: TokenWindows | None = None
    if core_data_dir and os.path.exists(os.path.join(core_data_dir, "train.bin")):
        core = TokenWindows(os.path.join(core_data_dir, "train.bin"), seq_len)
    mem_tensor = torch.tensor(mem_ids, dtype=torch.long)

    def sample_mem(n: int) -> torch.Tensor:
        max_start = len(mem_tensor) - (seq_len + 1)
        starts = torch.randint(0, max_start + 1, (n,), generator=g)
        return torch.stack([mem_tensor[s : s + seq_len + 1] for s in starts.tolist()])

    tol = {"coherence": 0.1, "poison": 0.1}
    if Calibration.exists(model_dir):
        cal = Calibration.load(model_dir)
        if "canary_delta_coherence" in cal.thresholds:
            tol["coherence"] = float(cal.thresholds["canary_delta_coherence"])
    tol.update(tolerance or {})

    zero = model.init_state(1, device)
    before = score_suite(model, zero, suite, device=device)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.core.parameters():
        p.requires_grad_(True)
    opt = build_optimizer(model, lr_matrix=lr * 10, lr_other=lr, use_muon=True, weight_decay=0.0)
    model.train()
    t0 = time.time()
    losses: list[float] = []
    for step in range(1, int(steps) + 1):
        n_core = int(round(batch_size * core_ratio)) if core is not None else 0
        parts = []
        if n_core > 0 and core is not None:
            parts.append(core.sample(n_core, g))
        if batch_size - n_core > 0:
            parts.append(sample_mem(batch_size - n_core))
        toks = torch.cat(parts).to(device)
        logits, _, _ = model(toks)
        V = logits.shape[-1]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), toks[:, 1:].reshape(-1), ignore_index=0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 25 == 0 or step == steps:
            log(f"[sleep] step {step}/{steps} loss {losses[-1]:.4f}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    after = score_suite(model, zero, suite, device=device)
    d_coh = after["coherence"] - before["coherence"]
    d_poi = after["poison"] - before["poison"]
    accepted = (d_coh <= tol["coherence"]) and (d_poi >= -tol["poison"])
    manifest: dict[str, Any] = {
        "base_model_id": model_id,
        "sessions": sessions,
        "memories": len(memories),
        "memory_tokens": len(mem_ids),
        "harvest": harvest,
        "steps": int(steps),
        "lr": float(lr),
        "core_ratio": float(core_ratio) if core is not None else 0.0,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "canary_before": before,
        "canary_after": after,
        "delta_coherence": d_coh,
        "delta_poison": d_poi,
        "tolerance": tol,
        "accepted": bool(accepted),
        "seconds": time.time() - t0,
        "created_at_unix": int(time.time()),
    }
    if not accepted:
        log(f"[sleep] rejected: coherence {d_coh:+.4f} (max {tol['coherence']:.4f}), poison {d_poi:+.4f} (min {-tol['poison']:.4f})")
        return manifest
    child = store.new_model_id("sleep")
    os.makedirs(store.model_dir(child), exist_ok=True)
    shutil.copyfile(store.tokenizer_path(model_id), store.tokenizer_path(child))
    if os.path.exists(store.canary_path(model_id)):
        shutil.copyfile(store.canary_path(model_id), store.canary_path(child))
    store.save_checkpoint(child, cfg, model, step=int(info.get("step", 0)) + int(steps), extra={"sleep": manifest})
    store.register_model(
        child,
        {
            "domain": "text",
            "status": "completed",
            "type": "sleep_consolidation",
            "parent_model_id": model_id,
            "params": int(model.num_params()),
            "sleep": manifest,
        },
    )
    manifest["model_id"] = child
    log(f"[sleep] accepted -> {child}: coherence {d_coh:+.4f}, poison {d_poi:+.4f}")
    return manifest
