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
