"""Outer-loop training for both domains.

The fast state starts at zero for every sequence and is produced by the
forward pass; the loss is backpropagated through every inner update. Every
evaluation records the three checkpoint numbers from the spec: held-out loss
with the memory, held-out loss with writes disabled (their difference is the
value of the memory), and recall accuracy, plus the histogram of learned β.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from plastic.config import ModelConfig
from plastic.data.mqar import PAD_ID, mqar_accuracy, mqar_batch
from plastic.data.physics import physics_batch
from plastic.data.text import TokenWindows
from plastic.model.lm import PlasticDynamics, PlasticLM, build_model
from plastic.model.memory import MemorySignals
from plastic.store import ArtifactStore
from plastic.train.optim import build_optimizer
from plastic.train.schedule import lr_scale


@dataclass
class TrainConfig:
    domain: str = "text"
    model: ModelConfig = field(default_factory=ModelConfig)
    artifacts_root: str = "artifacts"
    model_id: str | None = None
    data_dir: str | None = None

    steps: int = 1000
    batch_size: int = 16
    seq_len: int = 1024
    lr_matrix: float = 2e-2
    lr_other: float = 1e-3
    weight_decay: float = 0.1
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    use_muon: bool = True

    mqar_frac: float = 0.2
    mqar_pairs: tuple[int, ...] = (4, 8, 16)

    eval_every: int = 200
    eval_batches: int = 8
    save_every: int = 200
    log_every: int = 10
    seed: int = 0
    device: str = "auto"

    episodes_per_seq: int = 4
    mu_range: tuple[float, float] = (0.02, 0.25)
    nonlinear: bool = False
    action_std: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"] = self.model.to_dict()
        return d


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def beta_histogram(signals: list[MemorySignals], bins: int = 20) -> dict[str, Any]:
    betas = torch.cat([s.beta.detach().flatten().float().cpu() for s in signals])
    alphas = torch.cat([s.alpha.detach().flatten().float().cpu() for s in signals])
    counts = torch.histc(betas, bins=bins, min=0.0, max=1.0)
    edges = torch.linspace(0.0, 1.0, bins + 1)
    return {
        "edges": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in counts],
        "beta_mean": float(betas.mean()),
        "beta_std": float(betas.std()),
        "alpha_mean": float(alphas.mean()),
    }


def text_loss(model: PlasticLM, toks: torch.Tensor, *, beta_scale: float = 1.0) -> tuple[torch.Tensor, list[MemorySignals]]:
    logits, _, signals = model(toks, beta_scale=beta_scale)
    V = logits.shape[-1]
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), toks[:, 1:].reshape(-1), ignore_index=PAD_ID)
    return loss, signals


def text_nll_sum(model: PlasticLM, toks: torch.Tensor, *, beta_scale: float = 1.0) -> tuple[torch.Tensor, int, list[MemorySignals]]:
    """Summed next-token NLL over non-pad targets and the number of such targets."""
    logits, _, signals = model(toks, beta_scale=beta_scale)
    V = logits.shape[-1]
    target = toks[:, 1:].reshape(-1)
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, V), target, ignore_index=PAD_ID, reduction="sum")
    return nll, int((target != PAD_ID).sum()), signals


@torch.no_grad()
def evaluate_text(model: PlasticLM, cfg: TrainConfig, heldout: TokenWindows, device: torch.device) -> dict[str, Any]:
    model.eval()
    nll_sum, nll0_sum, count, signals_all = 0.0, 0.0, 0, []
    for i, batch in enumerate(heldout.sequential(cfg.batch_size)):
        if i >= cfg.eval_batches:
            break
        toks = batch.to(device)
        nll, n, signals = text_nll_sum(model, toks)
        nll0, _, _ = text_nll_sum(model, toks, beta_scale=0.0)
        nll_sum += float(nll)
        nll0_sum += float(nll0)
        count += n
        signals_all.extend(signals)
    V = cfg.model.vocab_size
    g = torch.Generator().manual_seed(cfg.seed + 1)
    mqar: dict[str, float] = {}
    for pairs in cfg.mqar_pairs:
        need = 1 + 4 * pairs
        if need > cfg.seq_len:
            continue
        toks, mask = mqar_batch(
            cfg.batch_size,
            n_pairs=pairs,
            seq_len=min(cfg.seq_len, need + 8),
            vocab_size=V,
            key_range=(3, V // 2),
            value_range=(V // 2, V),
            rng=g,
        )
        toks, mask = toks.to(device), mask.to(device)
        logits, _, _ = model(toks)
        mqar[str(pairs)] = mqar_accuracy(logits, toks, mask)
    heldout_loss = nll_sum / max(1, count)
    heldout_loss0 = nll0_sum / max(1, count)
    model.train()
    return {
        "heldout_loss": heldout_loss,
        "heldout_loss_beta0": heldout_loss0,
        "heldout_tokens": count,
        "memory_value": heldout_loss0 - heldout_loss,
        "mqar_accuracy": mqar,
        "beta_hist": beta_histogram(signals_all) if signals_all else None,
    }


@torch.no_grad()
def evaluate_physics(model: PlasticDynamics, cfg: TrainConfig, device: torch.device) -> dict[str, Any]:
    model.eval()
    g = torch.Generator().manual_seed(cfg.seed + 7)
    se_sum, se0_sum, count, signals_all = 0.0, 0.0, 0, []
    for _ in range(cfg.eval_batches):
        b = physics_batch(
            cfg.batch_size,
            seq_len=cfg.seq_len,
            episodes_per_seq=cfg.episodes_per_seq,
            mu_range=cfg.mu_range,
            nonlinear=cfg.nonlinear,
            action_std=cfg.action_std,
            rng=g,
        )
        x, y = b.inputs.to(device), b.target_delta.to(device)
        pred, _, signals = model(x)
        pred0, _, _ = model(x, beta_scale=0.0)
        se_sum += float((pred - y).pow(2).sum())
        se0_sum += float((pred0 - y).pow(2).sum())
        count += int(y.numel())
        signals_all.extend(signals)
    model.train()
    heldout = se_sum / max(1, count)
    heldout0 = se0_sum / max(1, count)
    return {
        "heldout_loss": heldout,
        "heldout_loss_beta0": heldout0,
        "heldout_elements": count,
        "memory_value": heldout0 - heldout,
        "beta_hist": beta_histogram(signals_all),
    }


def _print_flush(msg: str) -> None:
    print(msg, flush=True)


def train(cfg: TrainConfig, *, log: Callable[[str], None] = _print_flush) -> str:
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    store = ArtifactStore(cfg.artifacts_root)
    store.ensure()
    model_id = cfg.model_id or store.new_model_id("lm" if cfg.domain == "text" else "phys")
    os.makedirs(store.model_dir(model_id), exist_ok=True)

    model_cfg = cfg.model
    if model_cfg.domain != cfg.domain:
        model_cfg = ModelConfig.from_dict({**model_cfg.to_dict(), "domain": cfg.domain})

    heldout: TokenWindows | None = None
    train_windows: TokenWindows | None = None
    if cfg.domain == "text":
        if not cfg.data_dir:
            raise ValueError("text training requires data_dir")
        tok_src = os.path.join(cfg.data_dir, "tokenizer.json")
        shutil.copyfile(tok_src, store.tokenizer_path(model_id))
        train_windows = TokenWindows(os.path.join(cfg.data_dir, "train.bin"), cfg.seq_len)
        heldout = TokenWindows(os.path.join(cfg.data_dir, "validation.bin"), cfg.seq_len)

    model = build_model(model_cfg).to(device)
    opt = build_optimizer(
        model,
        lr_matrix=cfg.lr_matrix,
        lr_other=cfg.lr_other,
        weight_decay=cfg.weight_decay,
        use_muon=cfg.use_muon,
    )

    record = {
        "model_id": model_id,
        "domain": cfg.domain,
        "status": "running",
        "train_config": cfg.to_dict(),
        "params": int(model.num_params()),
        "device": str(device),
        "created_at_unix": int(time.time()),
    }
    store.register_model(model_id, record)
    store.save_checkpoint(model_id, model_cfg, model, step=0)
    log(f"[train] {model_id}: {cfg.domain} model {model.num_params() / 1e6:.2f}M params on {device}")

    def do_eval(step: int) -> dict[str, Any]:
        if cfg.domain == "text":
            assert heldout is not None
            ev = evaluate_text(model, cfg, heldout, device)  # type: ignore[arg-type]
        else:
            ev = evaluate_physics(model, cfg, device)  # type: ignore[arg-type]
        ev["step"] = int(step)
        ev["evaluated_at_unix"] = int(time.time())
        store.write_eval(model_id, ev)
        store.append_log(model_id, {"event": "eval", "step": int(step), **{k: v for k, v in ev.items() if k != "beta_hist"}})
        return ev

    model.train()
    t0 = time.time()
    tokens_seen = 0
    last_log_t = t0
    last_log_tokens = 0
    try:
        for step in range(1, cfg.steps + 1):
            scale = lr_scale(step, warmup=cfg.warmup_steps, total=cfg.steps, min_ratio=cfg.min_lr_ratio)
            opt.set_lr_scale(scale)

            if cfg.domain == "text":
                assert train_windows is not None
                V = model_cfg.vocab_size
                use_mqar = cfg.mqar_frac > 0 and float(torch.rand(1, generator=g)) < cfg.mqar_frac
                if use_mqar:
                    pairs = cfg.mqar_pairs[int(torch.randint(0, len(cfg.mqar_pairs), (1,), generator=g))]
                    toks, _ = mqar_batch(
                        cfg.batch_size,
                        n_pairs=pairs,
                        seq_len=cfg.seq_len,
                        vocab_size=V,
                        key_range=(3, V // 2),
                        value_range=(V // 2, V),
                        rng=g,
                    )
                else:
                    toks = train_windows.sample(cfg.batch_size, g)
                toks = toks.to(device)
                loss, _ = text_loss(model, toks)
                n_tok = int(toks.numel())
            else:
                b = physics_batch(
                    cfg.batch_size,
                    seq_len=cfg.seq_len,
                    episodes_per_seq=cfg.episodes_per_seq,
                    mu_range=cfg.mu_range,
                    nonlinear=cfg.nonlinear,
                    action_std=cfg.action_std,
                    rng=g,
                )
                loss = model.loss(b.inputs.to(device), b.target_delta.to(device))
                n_tok = int(b.inputs.shape[0] * b.inputs.shape[1])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            opt.step()
            tokens_seen += n_tok

            if step % cfg.log_every == 0 or step == 1:
                _sync(device)
                now = time.time()
                tps = (tokens_seen - last_log_tokens) / max(1e-6, now - last_log_t)
                last_log_t, last_log_tokens = now, tokens_seen
                rec = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "grad_norm": grad_norm,
                    "lr_scale": scale,
                    "tokens": tokens_seen,
                    "seconds": now - t0,
                    "tok_per_s": tps,
                }
                store.append_log(model_id, rec)
                log(f"[train] step {step}/{cfg.steps} loss {rec['loss']:.4f} grad {grad_norm:.3f} lr x{scale:.3f} {tps / 1e3:.1f}K tok/s")

            if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step != cfg.steps:
                ev = do_eval(step)
                log(f"[eval] step {step} heldout {ev['heldout_loss']:.4f} beta0 {ev['heldout_loss_beta0']:.4f} memory_value {ev['memory_value']:+.4f}")
            if cfg.save_every > 0 and step % cfg.save_every == 0 and step != cfg.steps:
                store.save_checkpoint(model_id, model_cfg, model, step=step, extra={"tokens": tokens_seen})
    except Exception as e:  # noqa: BLE001
        store.register_model(model_id, {"status": "failed", "error": f"{type(e).__name__}: {e}", "completed_at_unix": int(time.time())})
        raise

    ev = do_eval(cfg.steps)
    store.save_checkpoint(model_id, model_cfg, model, step=cfg.steps, extra={"tokens": tokens_seen})
    store.register_model(
        model_id,
        {
            "status": "completed",
            "steps": cfg.steps,
            "tokens": tokens_seen,
            "seconds": time.time() - t0,
            "eval": {k: v for k, v in ev.items() if k != "beta_hist"},
            "completed_at_unix": int(time.time()),
        },
    )
    log(f"[train] done {model_id}: heldout {ev['heldout_loss']:.4f} memory_value {ev['memory_value']:+.4f} ({time.time() - t0:.0f}s)")
    return model_id
