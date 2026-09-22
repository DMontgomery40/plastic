import time

import torch
import torch.nn.functional as F

from plastic.config import ModelConfig
from plastic.model.lm import PlasticDynamics, PlasticLM
from plastic.train.optim import build_optimizer, split_parameters


def _copy_batch(B: int, V: int, seg: int, reps: int) -> torch.Tensor:
    """A random segment repeated ``reps`` times: every repetition after the first is
    predictable only by recalling the previous one, which a delta-rule memory can do
    as an induction mechanism (write prev->current, read current->next)."""
    return torch.randint(0, V, (B, seg)).repeat(1, reps)


def _repeat_loss(lm: PlasticLM, toks: torch.Tensor, seg: int, *, beta_scale: float) -> float:
    with torch.no_grad():
        logits, _, _ = lm(toks, beta_scale=beta_scale)
        V = logits.shape[-1]
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), toks[:, 1:].reshape(-1), reduction="none")
        ce = ce.view(toks.shape[0], -1)
        return float(ce[:, seg:].mean())


def test_split_parameters_covers_everything_once():
    lm = PlasticLM(ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=16, vocab_size=50))
    matrix, other = split_parameters(lm)
    ids = [id(p) for p in matrix + other]
    assert len(ids) == len(set(ids))
    assert sum(p.numel() for p in matrix + other) == lm.num_params()
    assert all(p.ndim == 2 for p in matrix)
    assert not any(id(lm.embed.weight) == id(p) for p in matrix)


def test_lm_learns_to_use_its_memory():
    """Training on repeated segments must (a) reduce the loss and (b) make the model rely on
    the fast weights: disabling writes at evaluation must make the repeat loss much worse."""
    torch.manual_seed(0)
    V, seg, reps = 32, 8, 4
    cfg = ModelConfig(d_model=64, n_heads=2, n_layers=2, chunk=16, vocab_size=V)
    lm = PlasticLM(cfg)
    opt = build_optimizer(lm, lr_matrix=2e-2, lr_other=1e-2, use_muon=True)
    first = None
    t0 = time.time()
    for _ in range(150):
        toks = _copy_batch(32, V, seg, reps)
        loss = lm.loss(toks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()
        if first is None:
            first = float(loss.detach())
    last = float(loss.detach())
    assert last < 0.9 * first, (first, last)

    toks = _copy_batch(64, V, seg, reps)
    with_memory = _repeat_loss(lm, toks, seg, beta_scale=1.0)
    without_memory = _repeat_loss(lm, toks, seg, beta_scale=0.0)
    ln_v = float(torch.log(torch.tensor(float(V))))
    assert with_memory < ln_v - 0.3, (with_memory, ln_v)
    assert with_memory < without_memory - 0.5, (with_memory, without_memory, time.time() - t0)


def test_optimizer_state_roundtrip():
    lm = PlasticLM(ModelConfig(d_model=32, n_heads=2, n_layers=1, chunk=16, vocab_size=50))
    opt = build_optimizer(lm)
    lm.loss(torch.randint(0, 50, (2, 32))).backward()
    opt.step()
    sd = opt.state_dict()
    opt2 = build_optimizer(lm)
    opt2.load_state_dict(sd)
    opt2.set_lr_scale(0.5)
    assert all(abs(g["lr"] - 0.5 * g["base_lr"]) < 1e-12 for g in opt2.param_groups)


def test_dynamics_smoke():
    torch.manual_seed(0)
    cfg = ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=2, chunk=16)
    m = PlasticDynamics(cfg)
    opt = build_optimizer(m, use_muon=False, lr_other=3e-3)
    losses = []
    for _ in range(40):
        x = torch.randn(8, 48, 7)
        target = x[..., :4] * 0.5 + 0.1
        loss = m.loss(x, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.5 * losses[0], losses
