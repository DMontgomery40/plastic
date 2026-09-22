"""Adversarial write-gate training: the outer loop also minimizes canary damage under attack.

Every ``adv_every`` steps a short embedding-space attack is optimized against the
current slow weights (treated as constant), then the damage the attack causes
is recomputed with gradients flowing to the slow weights. Minimizing it pushes
the learned write rate β down on adversarial tokens and moves the key/value
projections toward writes that do not disturb the canaries, while the ordinary
language-modeling and recall losses keep the memory useful on benign text.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from plastic.harness.canary import CanarySuite
from plastic.redteam.attack import _coherence_loss, _forward_embeddings, snap_to_tokens


def adversarial_damage(
    model,
    prefix_ids: Tensor,
    suite: CanarySuite,
    *,
    steps: int,
    suffix_len: int,
    radius: float,
    lr: float,
    device: torch.device,
    rng: torch.Generator,
) -> tuple[Tensor, dict[str, float]]:
    """Attack the current model on ``prefix_ids`` (1, T) and return a differentiable damage.

    Returns (damage, info) where ``damage`` depends on the slow weights and ``info`` holds
    the continuous damage reached by the attacker and the attack's payload ids.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, prefix_state, _ = model(prefix_ids)
    prefix_state = prefix_state.detach()
    V = model.cfg.vocab_size
    init = torch.randint(3, V, (int(suffix_len),), generator=rng).to(device)
    base = model.embed.weight[init].detach().unsqueeze(0)
    rms = float(model.embed.weight.detach().pow(2).mean().sqrt())
    delta = torch.zeros_like(base, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr * rms)
    # inner attack: slow weights constant
    params = [p for p in model.parameters() if p.requires_grad]
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        for _ in range(int(steps)):
            opt.zero_grad(set_to_none=True)
            before = _coherence_loss(model, prefix_state, suite, device)
            _, after_state = _forward_embeddings(model, base + delta, prefix_state)
            after = _coherence_loss(model, after_state, suite, device)
            (-(after - before)).backward()
            opt.step()
            with torch.no_grad():
                n = delta.norm(dim=-1, keepdim=True)
                delta.mul_(torch.clamp(radius * rms / (n + 1e-9), max=1.0))
    finally:
        for p, f in zip(params, flags):
            p.requires_grad_(f)
    # outer: the attack is constant, the damage depends on the slow weights
    adv_emb = (base + delta).detach()
    with torch.no_grad():
        payload = snap_to_tokens(model, adv_emb).tolist()
        continuous = float(after - before)
    if was_training:
        model.train()
    before = _coherence_loss(model, prefix_state, suite, device)
    _, after_state = _forward_embeddings(model, adv_emb, prefix_state)
    after = _coherence_loss(model, after_state, suite, device)
    return after - before, {"attack_damage_continuous": continuous, "payload": payload}


@torch.no_grad()
def beta_auroc(model, benign_ids: Tensor, attack_ids: Tensor, *, device: torch.device) -> float:
    """AUROC of the learned write rate β as a detector: lower β on attack tokens scores higher."""
    _, _, sig_b = model(benign_ids.to(device))
    _, _, sig_a = model(attack_ids.to(device))
    b = torch.cat([s.beta.flatten() for s in sig_b]).cpu()
    a = torch.cat([s.beta.flatten() for s in sig_a]).cpu()
    # AUROC of the score (-beta) for "attack" as the positive class
    scores = torch.cat([-b, -a])
    labels = torch.cat([torch.zeros_like(b), torch.ones_like(a)])
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
