import torch

from plastic.config import ModelConfig
from plastic.model.state import SessionState


def test_config_roundtrip():
    cfg = ModelConfig(domain="physics", n_layers=3)
    s = cfg.to_json()
    back = ModelConfig.from_json(s)
    assert back == cfg
    assert back.input_dim == 7 and back.head_dim == 64
    assert cfg.signature_material() == ModelConfig.from_json(s).signature_material()


def test_config_rejects_bad_head_split():
    try:
        ModelConfig(d_model=30, n_heads=4)
    except ValueError:
        return
    raise AssertionError("expected ValueError for d_model not divisible by n_heads")


def test_state_zeros_clone_roundtrip():
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2)
    st = SessionState.zeros(cfg, batch=3)
    assert len(st.layers) == 2
    assert st.layers[0].S.shape == (3, 2, 16, 16)
    assert st.layers[0].h.shape == (3, 32)
    assert st.layers[0].M is None
    assert st.layers[0].conv_ssm is not None and st.layers[0].conv_ssm.shape == (3, cfg.conv_kernel - 1, 32)
    assert st.layers[0].conv_mem is not None and st.layers[0].conv_mem.shape == (3, cfg.conv_kernel - 1, 32)
    st.layers[0].S += 1.0
    c = st.clone()
    c.layers[0].S += 1.0
    assert float(st.layers[0].S[0, 0, 0, 0]) == 1.0
    d = st.state_dict()
    back = SessionState.from_state_dict(d)
    assert back.pos == st.pos and torch.equal(back.layers[0].S, st.layers[0].S)
    assert back.layers[0].conv_ssm is not None and torch.equal(back.layers[0].conv_ssm, st.layers[0].conv_ssm)
    deltas = c.s_delta(st)
    assert torch.allclose(deltas[0], torch.ones_like(deltas[0]))
    norms = st.norms()
    assert set(norms.keys()) >= {"s_norm", "h_norm", "s_norm_total", "h_norm_total"}


def test_state_chunk_rule_allocates_momentum():
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, rule="chunk")
    st = SessionState.zeros(cfg, batch=1)
    assert st.layers[0].M is not None and st.layers[0].M.shape == (1, 2, 16, 16)
    assert st.layers[0].chunk is not None and st.layers[0].chunk.A.shape == (1, 2, 16, 16)
    st.layers[0].chunk.count = 5
    st.layers[0].chunk.Bv += 2.0
    back = SessionState.from_state_dict(st.state_dict())
    assert back.layers[0].M is not None and back.layers[0].chunk is not None
    assert back.layers[0].chunk.count == 5 and torch.equal(back.layers[0].chunk.Bv, st.layers[0].chunk.Bv)
    c = st.clone()
    c.layers[0].chunk.A += 1.0
    assert float(st.layers[0].chunk.A.abs().sum()) == 0.0


def test_state_without_conv_buffers():
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, conv_kernel=1)
    st = SessionState.zeros(cfg, batch=2)
    assert st.layers[0].conv_ssm is None and st.layers[0].conv_mem is None
    back = SessionState.from_state_dict(st.state_dict())
    assert back.layers[0].conv_ssm is None
