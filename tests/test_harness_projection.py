import pytest
import torch

from plastic.config import ModelConfig
from plastic.data.text import encode_documents_to_bin
from plastic.harness.fisher import estimate_fisher_diag, fisher_norm
from plastic.harness.projection import project_delta
from plastic.model.lm import PlasticLM
from plastic.tokenizer.bpe import Tokenizer


def test_projection_enforces_half_space():
    torch.manual_seed(0)
    g = [torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)]
    d_aligned = [x.clone() * 0.5 for x in g]  # strongly aligned with the gradient
    out, st = project_delta(d_aligned, g, eps_dot=0.0, eps_cos=0.0)
    assert st.applied and st.dot_before > 0
    assert st.dot_after <= st.allowed + 1e-4
    assert 0.0 <= st.removed_ratio <= 1.0
    d_opposed = [-x for x in g]
    out2, st2 = project_delta(d_opposed, g)
    assert not st2.applied and all(torch.equal(a, b) for a, b in zip(out2, d_opposed))


def test_projection_slack():
    torch.manual_seed(1)
    g = [torch.randn(1, 1, 4, 4)]
    d = [g[0] * 0.1]
    out, st = project_delta(d, g, eps_cos=1.5)  # slack above cos = 1: never projects
    assert not st.applied
    out, st = project_delta(d, g, eps_cos=0.0)
    assert st.applied and abs(st.dot_after) < 1e-5


def test_fisher_diag_shapes_and_norm(tmp_path):
    docs = ["alpha beta gamma delta epsilon " * 40, "one two three four five six " * 40]
    tok = Tokenizer.train(docs, vocab_size=300)
    path = str(tmp_path / "train.bin")
    encode_documents_to_bin(tok, docs, path)
    import numpy as np

    data = torch.from_numpy(np.fromfile(path, dtype="<u2").astype("int64"))
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=8, vocab_size=tok.vocab_size)
    lm = PlasticLM(cfg)
    seqs = [data[:65].unsqueeze(0), data[65:130].unsqueeze(0)]
    fisher = estimate_fisher_diag(lm, seqs, chunk=8, n_chunks=6, device=torch.device("cpu"))
    assert len(fisher) == 2 and fisher[0].shape == (1, 2, 16, 16)
    assert all((f >= 0).all() for f in fisher) and any(float(f.sum()) > 0 for f in fisher)
    zeros = [torch.zeros(1, 2, 16, 16) for _ in range(2)]
    assert fisher_norm(zeros, fisher) == 0.0
    ones = [torch.ones(1, 2, 16, 16) for _ in range(2)]
    assert fisher_norm(ones, fisher) > 0


def test_projection_is_invariant_to_gradient_scale():
    d = [torch.tensor([[[[1000.0, 2000.0]]]])]
    for scale in (1.0, 1e-6, 1e-9, 1e3):
        g = [torch.tensor([[[[scale, 0.0]]]])]
        out, st = project_delta(d, g, eps_dot=0.0, eps_cos=0.0)
        assert st.applied
        assert torch.allclose(out[0], torch.tensor([[[[0.0, 2000.0]]]]), atol=1e-3), (scale, out)
        assert abs(st.dot_after) <= 1e-6 * scale + 1e-9


def _fisher_fixture(tmp_path):
    docs = ["alpha beta gamma delta epsilon " * 40, "one two three four five six " * 40]
    tok = Tokenizer.train(docs, vocab_size=300)
    path = str(tmp_path / "train.bin")
    encode_documents_to_bin(tok, docs, path)
    import numpy as np

    data = torch.from_numpy(np.fromfile(path, dtype="<u2").astype("int64"))
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=8, vocab_size=tok.vocab_size)
    return PlasticLM(cfg), data


def test_fisher_is_invariant_to_batch_replication(tmp_path):
    lm, data = _fisher_fixture(tmp_path)
    seq = data[:65].unsqueeze(0)
    sums = []
    for B in (1, 2, 4):
        f = estimate_fisher_diag(lm, [seq.repeat(B, 1)], chunk=8, n_chunks=4, device=torch.device("cpu"))
        sums.append(sum(float(x.sum()) for x in f))
    assert abs(sums[0] - sums[1]) < 1e-5 * max(1.0, sums[0]) and abs(sums[0] - sums[2]) < 1e-5 * max(1.0, sums[0]), sums


def test_fisher_traverses_the_models_own_trajectory(tmp_path):
    """The carried state after k chunks equals the model's state after k*chunk tokens."""
    from plastic.harness.fisher import chunk_loss, state_with_grad_S

    lm, data = _fisher_fixture(tmp_path)
    toks = data[:33].unsqueeze(0)  # 32 inputs, last token is a target only
    x, y = toks[:, :-1], toks[:, 1:]
    state = lm.init_state(1)
    for start in range(0, 32, 8):
        st, leaves = state_with_grad_S(state)
        _, state = chunk_loss(lm, (x[:, start : start + 8], y[:, start : start + 8]), st)
        state = state.detach()
    _, ref, _ = lm(x)
    for a, b in zip(state.layers, ref.layers):
        assert torch.allclose(a.S, b.S, atol=1e-5) and torch.allclose(a.h, b.h, atol=1e-5)
    assert state.pos == 32


@pytest.mark.parametrize("T", [5, 8, 9, 17])
def test_fisher_physics_uses_every_transition(T):
    from plastic.model.lm import PlasticDynamics

    cfg = ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=1, chunk=8)
    m = PlasticDynamics(cfg)
    x = torch.randn(2, T, 7)
    y = torch.randn(2, T, 4)
    f = estimate_fisher_diag(m, [(x, y)], chunk=8, n_chunks=100, device=torch.device("cpu"))
    assert f[0].shape == (1, 2, 16, 16) and torch.isfinite(f[0]).all()
