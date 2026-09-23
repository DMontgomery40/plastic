"""CoordinateBlock (docs/research/2026-09-21-plastic-coordinate-recurrence.md): identities,
boundary semantics, causality, meta-gradient, amplitude bound and falsifier switches.

Equivalence and gradient checks run in float64 so inner steps cannot amplify rounding into
a loose tolerance; one float32 case keeps the repository's usual 1e-4 tolerance."""

from __future__ import annotations

import pytest
import torch

from plastic.data.physics import physics_batch
from plastic.model.coordinate import (
    THETA_KEY,
    W_KEYS,
    CoordinateConfig,
    CoordinateDynamics,
    CoordinateState,
    decode,
    encode,
    transport,
    uniform_bound,
)

F64 = torch.float64


def _cfg(**kw) -> CoordinateConfig:
    base = dict(d_model=32, n_heads=2, n_layers=2, chunk=8)
    base.update(kw)
    return CoordinateConfig(**base)


def _model(dtype=F64, seed: int = 0, **kw) -> CoordinateDynamics:
    torch.manual_seed(seed)
    return CoordinateDynamics(_cfg(**kw)).to(dtype)


def _stream(B: int, T: int, dtype=F64, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 7, generator=g).to(dtype), torch.randn(B, T, 4, generator=g).to(dtype)


def _random_W(B: int, cfg: CoordinateConfig, scale: float, seed: int, dtype=F64) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    H, k, h = cfg.n_heads, cfg.head_dim // 2, cfg.coupling_hidden
    shapes = {
        "f_in": (H, k, h), "f_in_b": (H, h), "f_out": (H, h, k), "f_out_b": (H, k),
        "g_in": (H, k, h), "g_in_b": (H, h), "g_out": (H, h, k), "g_out_b": (H, k),
    }
    out = {key: (torch.randn(B, *s, generator=g) * scale).to(dtype) for key, s in shapes.items()}
    out[THETA_KEY] = (torch.randn(B, cfg.d_model, generator=g) * scale).to(dtype)
    return out


def _nonzero_state(m: CoordinateDynamics, B: int, seed: int = 7) -> CoordinateState:
    g = torch.Generator().manual_seed(seed)
    st = m.init_state(B).detach()
    dtype = st.r[0].dtype
    st.r = [torch.randn(t.shape, generator=g).to(dtype) * 0.8 for t in st.r]
    st.omega = [{k: v + 0.3 * torch.randn(v.shape, generator=g).to(dtype) for k, v in om.items()} for om in st.omega]
    return st


def _run(m, x, tg, st, parts, **kw):
    outs, chunks, a = [], [], 0
    for n in parts:
        p, st, rep = m(x[:, a : a + n], st, target_delta=tg[:, a : a + n], **kw)
        outs.append(p)
        chunks.extend(rep.chunks)
        a += n
    return torch.cat(outs, 1), st, chunks


def _assert_states_close(s1: CoordinateState, s2: CoordinateState, atol: float) -> None:
    assert s1.pos == s2.pos
    for a, b in zip(s1.r, s2.r):
        assert torch.allclose(a, b, atol=atol, rtol=0)
    for oa, ob in zip(s1.omega, s2.omega):
        assert oa.keys() == ob.keys()
        for k in oa:
            assert torch.allclose(oa[k], ob[k], atol=atol, rtol=0), k
    assert (s1.pending is None) == (s2.pending is None)
    if s1.pending is not None:
        assert torch.allclose(s1.pending.x, s2.pending.x, atol=atol, rtol=0)
        assert torch.equal(s1.pending.mask, s2.pending.mask)
        for a, b in zip(s1.pending.r_start, s2.pending.r_start):
            assert torch.allclose(a, b, atol=atol, rtol=0)


def test_config_validation_and_constants_are_not_parameters():
    assert _cfg(meta_gradient=False).meta_gradient == "none"
    assert _cfg(meta_gradient=True).meta_gradient == "full"
    bad = [
        dict(rho=1.0), dict(rho=0.0), dict(epsilon=0.0), dict(d_model=30), dict(n_heads=32),
        dict(eta_init=20.0), dict(meta_gradient="second"), dict(commit_rule="replay"),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            _cfg(**kw)
    m = _model()
    names = [n for n, _ in m.named_parameters()]
    assert not any("rho" in n or "epsilon" in n for n in names)
    assert CoordinateConfig.from_dict(_cfg().to_dict()) == _cfg()


@pytest.mark.parametrize("scale", [1.0, 50.0, 1e3])
def test_coordinate_displacement_and_decay_are_bounded_for_any_weights(scale):
    cfg = _cfg()
    eps, H = cfg.epsilon, cfg.n_heads
    W = _random_W(3, cfg, scale, seed=int(scale))
    g = torch.Generator().manual_seed(3)
    r = torch.randn(3, 5, cfg.d_model, generator=g, dtype=F64) * 10
    z = torch.randn(3, 5, cfg.d_model, generator=g, dtype=F64) * 10
    assert float((encode(W, r, eps=eps, n_heads=H) - r).abs().max()) <= eps * (1 + 1e-12)
    assert float((decode(W, z, eps=eps, n_heads=H) - z).abs().max()) <= eps * (1 + 1e-12)
    assert float((encode(W, r[:, 0], eps=eps, n_heads=H) - r[:, 0]).abs().max()) <= eps * (1 + 1e-12)
    assert torch.allclose(decode(W, encode(W, r, eps=eps, n_heads=H), eps=eps, n_heads=H), r, atol=1e-10, rtol=0)
    blk = _model().core.blocks[0]
    u = torch.randn(3, 5, cfg.d_model, generator=g, dtype=F64) * scale
    a = blk.decay(W, u).detach()
    assert float(a.max()) <= cfg.rho and float(a.min()) >= 0.0


def test_transport_identities_preserve_the_canonical_state():
    cfg = _cfg()
    kw = dict(eps=cfg.epsilon, n_heads=cfg.n_heads)
    Wa, Wb, Wc = (_random_W(2, cfg, 2.0, seed=s) for s in (1, 2, 3))
    z = torch.randn(2, cfg.d_model, dtype=F64, generator=torch.Generator().manual_seed(4))
    zab = transport(Wa, Wb, z, **kw)
    assert torch.allclose(transport(Wb, Wc, zab, **kw), transport(Wa, Wc, z, **kw), atol=1e-12, rtol=0)
    assert torch.allclose(transport(Wb, Wa, zab, **kw), z, atol=1e-12, rtol=0)
    assert torch.allclose(decode(Wb, zab, **kw), decode(Wa, z, **kw), atol=1e-12, rtol=0)
    assert float((zab - z).abs().max()) > 1e-3  # the charts genuinely differ


@pytest.mark.parametrize(
    "mode,parts",
    [
        ("recurrent", [37]),
        ("chunk", [1] * 37),
        ("chunk", [5, 11, 21]),
        ("chunk", [8, 8, 8, 13]),
        ("recurrent", [3, 13, 16, 5]),
        ("chunk", [16, 21]),
    ],
)
def test_chunk_and_recurrent_paths_agree_for_any_partition(mode, parts):
    m = _model()
    x, tg = _stream(2, 37)
    st0 = _nonzero_state(m, 2)
    ref_p, ref_st, ref_chunks = _run(m, x, tg, st0.clone(), [37], mode="chunk")
    p, st, chunks = _run(m, x, tg, st0.clone(), parts, mode=mode)
    assert len(ref_chunks) == len(chunks) == 4 and ref_st.pos == 37 and ref_st.pending.x.shape[1] == 5
    assert torch.allclose(p, ref_p, atol=1e-10, rtol=0)
    _assert_states_close(st, ref_st, atol=1e-10)
    for a, b in zip(chunks, ref_chunks):
        assert a.pos == b.pos
        for f in ("inner_loss_before", "inner_loss_after", "dW_norm", "dtheta_norm", "r_end_inf", "r_peak_inf"):
            assert torch.allclose(getattr(a, f), getattr(b, f), atol=1e-10, rtol=0), f
    assert any(float(c.dW_norm.max()) > 0 for c in chunks)


def test_float32_modes_and_partitions_agree():
    m = _model(dtype=torch.float32)
    x, tg = _stream(2, 37, dtype=torch.float32)
    ref_p, ref_st, _ = _run(m, x, tg, m.init_state(2), [37])
    for mode, parts in (("chunk", [13, 24]), ("recurrent", [37])):
        p, st, _ = _run(m, x, tg, m.init_state(2), parts, mode=mode)
        assert torch.allclose(p, ref_p, atol=1e-4, rtol=1e-4)
        _assert_states_close(st, ref_st, atol=1e-4)


def test_batch_elements_adapt_independently():
    m = _model()
    x, tg = _stream(4, 20)
    p, st, rep = m(x, target_delta=tg)
    p1, st1, rep1 = m(x[2:3], target_delta=tg[2:3])
    assert torch.allclose(p[2:3], p1, atol=1e-12, rtol=0)
    for oa, ob in zip(st.omega, st1.omega):
        for k in oa:
            assert torch.allclose(oa[k][2:3], ob[k], atol=1e-12, rtol=0)
    for a, b in zip(rep.chunks, rep1.chunks):
        assert torch.allclose(a.dW_norm[2:3], b.dW_norm, atol=1e-12, rtol=0)
        assert torch.allclose(a.inner_loss_before[2:3], b.inner_loss_before, atol=1e-12, rtol=0)


def test_freeze_keeps_omega_bit_identical_while_the_carry_advances():
    m = _model()
    x, tg = _stream(2, 24)
    st0 = _nonzero_state(m, 2)
    p, st, rep = m(x[:, :20], st0.clone(), target_delta=tg[:, :20], freeze=True)
    for oa, ob in zip(st.omega, st0.omega):
        for k in oa:
            assert torch.equal(oa[k], ob[k]), k
    assert all(not torch.allclose(a, b) for a, b in zip(st.r, st0.r))
    assert st.pos == 20 and st.pending.x.shape[1] == 4 and not st.pending.mask.any()
    assert len(rep.chunks) == 2 and all(c.frozen and not c.stepped for c in rep.chunks)
    assert all(float(c.dW_norm.abs().max()) == 0 and float(c.eta.abs().max()) == 0 for c in rep.chunks)
    assert all(not a.applied for a in rep.accepted) and rep.proposal is None
    # resuming mid-chunk: only the unfrozen rows supervise the next boundary
    # (4 rows, minus the chunk's last row whose next observation has not arrived)
    _, st2, rep2 = m(x[:, 20:], st, target_delta=tg[:, 20:])
    c = rep2.chunks[0]
    assert c.stepped and torch.equal(c.n_observed, torch.tensor([3, 3]))
    # a frozen harness call may cross boundaries without producing a proposal
    _, _, rep3 = m(x[:, :20], st0.clone(), target_delta=tg[:, :20], freeze=True, commit=False)
    assert rep3.proposal is None and len(rep3.chunks) == 2


def test_beta_scale_zero_equals_freeze_for_omega_and_scales_the_step_linearly():
    m = _model()
    x, tg = _stream(2, 24)
    st0 = _nonzero_state(m, 2)
    p0, s0, r0 = m(x, st0.clone(), target_delta=tg, beta_scale=0.0)
    pf, sf, _ = m(x, st0.clone(), target_delta=tg, freeze=True)
    for o0, of, oi in zip(s0.omega, sf.omega, st0.omega):
        for k in oi:
            assert torch.equal(o0[k], oi[k]) and torch.equal(of[k], oi[k])
    assert torch.equal(p0, pf) and all(torch.equal(a, b) for a, b in zip(s0.r, sf.r))
    for c in r0.chunks:
        assert not c.frozen and not c.stepped and bool((c.n_observed > 0).all())
        assert torch.equal(c.inner_loss_before, c.inner_loss_after)
    _, _, half = m(x[:, :8], st0.clone(), target_delta=tg[:, :8], beta_scale=0.5)
    _, _, full = m(x[:, :8], st0.clone(), target_delta=tg[:, :8], beta_scale=1.0)
    assert torch.allclose(half.chunks[0].dW_norm, 0.5 * full.chunks[0].dW_norm, rtol=1e-12, atol=0)
    assert torch.allclose(half.chunks[0].eta, 0.5 * full.chunks[0].eta, rtol=1e-12, atol=0)


def test_observed_target_mask_is_causal():
    m = _model()
    x, tg = _stream(2, 24)
    st0 = _nonzero_state(m, 2)
    base_p, base_st, base_rep = m(x, st0.clone(), target_delta=tg)

    # targets of each chunk's last row have not arrived at its boundary: never used
    tg_late = tg.clone()
    tg_late[:, 7] += 100.0
    tg_late[:, 15] -= 50.0
    p, st, rep = m(x, st0.clone(), target_delta=tg_late)
    assert torch.equal(p, base_p)
    for oa, ob in zip(st.omega, base_st.omega):
        assert all(torch.equal(oa[k], ob[k]) for k in oa)

    # an explicitly unobserved target, even a non-finite one, changes nothing
    mask = m.default_target_mask(0, 2, 24, "cpu").clone()
    mask[:, 3] = False
    tg_zero, tg_nan = tg.clone(), tg.clone()
    tg_zero[:, 3] = 0.0
    tg_nan[:, 3] = float("nan")
    pz, _, _ = m(x, st0.clone(), target_delta=tg_zero, target_mask=mask)
    pn, _, rn = m(x, st0.clone(), target_delta=tg_nan, target_mask=mask)
    assert torch.equal(pz, pn) and bool(torch.isfinite(pn).all())
    assert torch.equal(rn.chunks[0].n_observed, torch.tensor([6, 6]))

    # an observed target of chunk 0 cannot change chunk-0 predictions, only later ones
    tg_obs = tg.clone()
    tg_obs[:, 3] += 5.0
    po, _, _ = m(x, st0.clone(), target_delta=tg_obs)
    assert torch.equal(po[:, :8], base_p[:, :8])
    assert float((po[:, 8:] - base_p[:, 8:]).detach().abs().max()) > 1e-8

    # future inputs cannot change earlier predictions
    x_future = x.clone()
    x_future[:, 10:] += torch.randn(2, 14, 7, dtype=F64)
    pfut, _, _ = m(x_future, st0.clone(), target_delta=tg)
    assert torch.equal(pfut[:, :10], base_p[:, :10])


def test_inner_step_reduces_the_observed_loss_to_first_order():
    m = _model(eta_init=1e-3)
    with torch.no_grad():
        m.head.weight.mul_(10.0)  # larger inner gradients; the check is scale-free
    x, tg = _stream(3, 8)
    _, st, rep = m(x, target_delta=tg, commit=False)
    s = rep.proposal.signals
    assert s.stepped and s.n_observed.tolist() == [7, 7, 7]
    assert bool((s.inner_loss_after < s.inner_loss_before).all())
    # step = -eta * g per layer, so the first-order decrease is sum_l eta_l ||g_l||^2 = ||step_l||^2 / eta_l
    predicted = ((s.dW_norm**2 + s.dtheta_norm**2) / s.eta).sum(dim=1)
    actual = s.inner_loss_before - s.inner_loss_after
    assert torch.allclose(actual, predicted, rtol=1e-2, atol=0)
    assert s.inner_loss_before.shape == (3,) and s.dW_norm.shape == (3, 2) and s.eta.shape == (2,)


def _meta_setup(meta_gradient: str = "full") -> tuple[CoordinateDynamics, torch.Tensor, torch.Tensor]:
    m = _model(chunk=4, eta_init=0.5, coupling_out_std=0.3, meta_gradient=meta_gradient)
    with torch.no_grad():
        m.head.weight.mul_(20.0)
    x, tg = _stream(2, 14, seed=5)
    return m, x, tg


def _meta_objective(m, x, tg):
    # three accepted updates (boundaries 4, 8, 12); the first call ends mid-chunk, so the
    # pending chunk (r_start and embedded rows) must carry the outer graph across calls
    st = m.init_state(2)
    p1, st, _ = m(x[:, :6], st, target_delta=tg[:, :6])
    p2, st, _ = m(x[:, 6:], st, target_delta=tg[:, 6:])
    pred = torch.cat([p1, p2], 1)
    return ((pred[:, 4:] - tg[:, 4:]) ** 2).mean()


def _meta_params(m):
    ps = []
    for blk in m.core.blocks:
        ps += list(blk.W0.values()) + [blk.theta0, blk.eta_logit]
    return ps + [m.embed.weight, m.core.blocks[0].P_v.weight, m.core.blocks[1].P_a.weight]


def _directional(m, x, tg, direction, h=1e-6):
    params = _meta_params(m)
    with torch.no_grad():
        for p, d in zip(params, direction):
            p.add_(h * d)
        fp = _meta_objective(m, x, tg)
        for p, d in zip(params, direction):
            p.sub_(2 * h * d)
        fm = _meta_objective(m, x, tg)
        for p, d in zip(params, direction):
            p.add_(h * d)
    return float((fp - fm) / (2 * h))


def test_meta_gradient_through_inner_updates_matches_finite_differences():
    m, x, tg = _meta_setup("full")
    params = _meta_params(m)
    grads = torch.autograd.grad(_meta_objective(m, x, tg), params)
    g = torch.Generator().manual_seed(11)
    direction = [torch.randn(p.shape, generator=g, dtype=F64) for p in params]
    norm = torch.sqrt(sum((d**2).sum() for d in direction))
    direction = [d / norm for d in direction]
    analytic = float(sum((gr * d).sum() for gr, d in zip(grads, direction)))
    numeric = _directional(m, x, tg, direction)
    assert abs(analytic - numeric) / (abs(analytic) + abs(numeric)) < 1e-6
    # the learned step size receives a correct, nonzero meta-gradient on its own
    eta_idx = [i for i, p in enumerate(params) if p is m.core.blocks[0].eta_logit][0]
    unit = [torch.zeros_like(p) for p in params]
    unit[eta_idx] = torch.ones_like(params[eta_idx])
    eta_num = _directional(m, x, tg, unit)
    assert abs(float(grads[eta_idx])) > 1e-6
    assert abs(float(grads[eta_idx]) - eta_num) / (abs(float(grads[eta_idx])) + abs(eta_num)) < 1e-6
    # the stop-gradient diagnostics are genuinely different estimators
    for variant in ("first_order", "none"):
        mv, _, _ = _meta_setup(variant)
        mv.load_state_dict(m.state_dict())
        gv = torch.autograd.grad(_meta_objective(mv, x, tg), _meta_params(mv), allow_unused=True)
        gv = [torch.zeros_like(p) if t is None else t for t, p in zip(gv, params)]
        approx = float(sum((gr * d).sum() for gr, d in zip(gv, direction)))
        assert abs(approx - numeric) > 1e-3 * abs(numeric), variant  # observed: about 6% apart
        if variant == "none":
            assert float(gv[eta_idx].abs()) == 0.0


def test_amplitude_bound_holds_under_repeated_accepted_changes():
    cfg_kw = dict(rho=0.9, epsilon=0.2, eta_max=100.0, eta_init=50.0, coupling_out_std=1.0)
    m = _model(**cfg_kw)
    cfg = m.cfg
    with torch.no_grad():
        m.head.weight.mul_(50.0)
    x, tg = _stream(3, 12 * 8, seed=9)
    x = x * 20.0
    st = m.init_state(3).detach()
    st.r = [torch.randn(t.shape, dtype=F64, generator=torch.Generator().manual_seed(i)) * 5.0 for i, t in enumerate(st.r)]
    r_init = max(float(t.abs().max()) for t in st.r)
    signals, dW = [], []
    with torch.no_grad():
        for c in range(12):
            sl = slice(8 * c, 8 * c + 8)
            if c % 2 == 0:  # the learner's own large accepted step
                _, st, rep = m(x[:, sl], st, target_delta=tg[:, sl])
            else:  # an arbitrary accepted change of large magnitude
                _, st, rep = m(x[:, sl], st, target_delta=tg[:, sl], commit=False)
                wild = [_random_W(3, cfg, 50.0, seed=100 + 10 * c + i) for i in range(cfg.n_layers)]
                st, acc = m.commit_proposal(st, rep.proposal.with_omega(wild))
                assert acc.applied
            signals.extend(rep.chunks)
            dW.append(float(rep.chunks[0].dW_norm.max()))
    assert max(dW) > 1.0  # the learner's steps are not negligible
    B_C = uniform_bound(r_init, cfg)
    B_1 = uniform_bound(r_init, cfg, chunk=1)
    for s in signals:
        assert bool(torch.isfinite(s.r_peak_inf).all())
        assert bool((s.r_end_inf <= s.bound_end + 1e-9).all())
        assert bool((s.r_peak_inf <= s.bound_peak + 1e-9).all())
        assert float(s.r_end_inf.max()) <= B_C + 1e-9
        assert float(s.r_peak_inf.max()) <= B_1 + 1e-9
    assert float(signals[0].r_start_inf.max()) > 1.0 + cfg.epsilon  # the R > V branch is exercised


def test_falsifier_switches():
    x, tg = _stream(2, 24)
    init = _model().init_state(2)

    def run(**kw):
        m = _model(**kw)
        return m(x, m.init_state(2), target_delta=tg)

    p_full, s_full, _ = run()
    _, s_w, _ = run(adapt_W=False)  # frozen coordinates, adaptive decay
    _, s_t, _ = run(adapt_theta=False)  # adaptive coordinates, frozen decay
    _, s_off, r_off = run(fast_updates=False)
    for layer in range(2):
        assert all(torch.equal(s_w.omega[layer][k], init.omega[layer][k]) for k in W_KEYS)
        assert not torch.equal(s_w.omega[layer][THETA_KEY], init.omega[layer][THETA_KEY])
        assert torch.equal(s_t.omega[layer][THETA_KEY], init.omega[layer][THETA_KEY])
        assert not all(torch.equal(s_t.omega[layer][k], init.omega[layer][k]) for k in W_KEYS)
        assert all(torch.equal(s_off.omega[layer][k], init.omega[layer][k]) for k in init.omega[layer])
    assert all(not c.stepped and float(c.eta.abs().max()) == 0 for c in r_off.chunks)
    assert all(bool((c.inner_loss_before > 0).all()) for c in r_off.chunks)
    # the meta-gradient variants change only the outer gradient, never the forward values
    for mg in ("first_order", "none"):
        p_mg, s_mg, _ = run(meta_gradient=mg)
        assert torch.equal(p_mg, p_full)
        assert all(torch.equal(a[k], b[k]) for a, b in zip(s_mg.omega, s_full.omega) for k in a)


def test_proposals_are_committed_only_explicitly_and_transport_keeps_the_carry():
    m = _model()
    x, tg = _stream(2, 24)
    init = m.init_state(2)
    _, st, rep = m(x[:, :8], init, target_delta=tg[:, :8], commit=False)
    assert rep.proposal is not None and rep.accepted == [] and st.pending is None
    assert all(torch.equal(st.omega[i][k], init.omega[i][k]) for i in range(2) for k in init.omega[i])
    st2, acc = m.commit_proposal(st, rep.proposal)
    assert acc.applied and torch.equal(acc.dW_norm, rep.proposal.signals.dW_norm)
    assert all(torch.equal(a, b) for a, b in zip(st2.r, st.r))  # canonical carry retained exactly
    assert all(st2.omega[i][k] is rep.proposal.omega[i][k] for i in range(2) for k in W_KEYS)
    with pytest.raises(ValueError):  # a harness call may reach one boundary, at its end
        m(x[:, 8:24], st2, target_delta=tg[:, 8:24], commit=False)
    _, st3, _ = m(x[:, 8:11], st2, target_delta=tg[:, 8:11])
    with pytest.raises(ValueError):  # stale proposal
        m.commit_proposal(st3, rep.proposal)
    # accepting every proposal through the harness path equals the unguarded learner
    p_auto, s_auto, _ = m(x, m.init_state(2), target_delta=tg)
    st, outs = m.init_state(2), []
    for c in range(3):
        p, st, r = m(x[:, 8 * c : 8 * c + 8], st, target_delta=tg[:, 8 * c : 8 * c + 8], commit=False)
        st, _ = m.commit_proposal(st, r.proposal)
        outs.append(p)
    assert torch.equal(torch.cat(outs, 1), p_auto)
    _assert_states_close(st, s_auto, atol=0.0)


def test_proposal_is_bound_to_its_model_and_state_version():
    """ASTRA-233 P2: a proposal must not be committable onto another session at the same
    position, twice along one lineage, by another model instance, or with mismatched shapes."""
    m = _model()
    xa, tga = _stream(1, 16, seed=21)
    xb, tgb = _stream(1, 16, seed=22)
    init = m.init_state(1)
    _, sa, ra = m(xa[:, :8], init.clone(), target_delta=tga[:, :8], commit=False)
    _, sb, rb = m(xb[:, :8], init.clone(), target_delta=tgb[:, :8], commit=False)
    pa, pb = ra.proposal, rb.proposal
    assert sa.pos == sb.pos == pa.pos == pb.pos == 8 and sa.pending is None and sb.pending is None
    assert not torch.equal(pa.omega[0]["f_out"], pb.omega[0]["f_out"])
    with pytest.raises(ValueError):  # same position, different session
        m.commit_proposal(sb, pa)
    sa2, acc = m.commit_proposal(sa, pa)
    assert acc.applied and sa2.token != sa.token
    with pytest.raises(ValueError):  # duplicate commit along one lineage
        m.commit_proposal(sa2, pa)
    # a fork (clone/detach) at the boundary is the same version and commits identically;
    # re-committing on the pre-commit snapshot is such a fork, not a second mutation
    for fork in (sa.clone(), sa.detach(), sa):
        sf, accf = m.commit_proposal(fork, pa)
        assert accf.applied and all(torch.equal(a, b) for a, b in zip(sf.r, sa2.r))
        assert all(torch.equal(sf.omega[i][k], sa2.omega[i][k]) for i in range(2) for k in sa2.omega[i])
    # a rejected proposal goes stale as soon as the stream moves on
    _, sa_next, _ = m(xa[:, 8:16], sa, target_delta=tga[:, 8:16], commit=False)
    with pytest.raises(ValueError):
        m.commit_proposal(sa_next, pa)
    # another model instance, even with identical weights, cannot commit it
    other = _model()
    other.load_state_dict(m.state_dict())
    with pytest.raises(ValueError):
        other.commit_proposal(sa, pa)
    # replaced (e.g. projected) weights keep the binding; mismatched shapes are refused
    projected = [{k: v * 0.5 for k, v in om.items()} for om in pa.omega]
    sp, accp = m.commit_proposal(sa, pa.with_omega(projected))
    assert accp.applied and all(sp.omega[i][k] is projected[i][k] for i in range(2) for k in projected[i])
    assert all(torch.equal(a, b) for a, b in zip(sp.r, sa.r))
    wide = [{k: torch.cat([v, v], 0) for k, v in om.items()} for om in pa.omega]
    with pytest.raises(ValueError):
        m.commit_proposal(sa, pa.with_omega(wide))
    # a no-step proposal (beta_scale=0) commits as a no-op but still advances the version
    _, s0, r0 = m(xa[:, :8], init.clone(), target_delta=tga[:, :8], beta_scale=0.0, commit=False)
    s0c, acc0 = m.commit_proposal(s0, r0.proposal)
    assert not acc0.applied and s0c.token != s0.token
    assert all(s0c.omega[i][k] is s0.omega[i][k] for i in range(2) for k in s0.omega[i])


@pytest.mark.parametrize("commit_rule", ["transport", "fixed_z"])
def test_noop_commit_is_decided_by_value_and_keeps_the_carry(commit_rule):
    """ASTRA-234: "applied" must not depend on tensor identity. A no-op reads applied=False on
    every fork of the boundary state and for an unchanged projected candidate, keeps r exactly
    (no fixed_z re-encode), and a computed zero-valued step keeps its differentiable tensors."""
    m = _model(commit_rule=commit_rule)
    x, tg = _stream(2, 8, seed=31)
    init = m.init_state(2)
    _, s0, r0 = m(x, init.clone(), target_delta=tg, beta_scale=0.0, commit=False)
    assert not r0.proposal.signals.stepped
    for fork in (s0, s0.clone(), s0.detach(), s0.to("cpu")):
        sc, acc = m.commit_proposal(fork, r0.proposal)
        assert not acc.applied and float(acc.dW_norm.abs().max()) == 0 and float(acc.dtheta_norm.abs().max()) == 0
        assert all(torch.equal(a, b) for a, b in zip(sc.r, fork.r))
        assert all(sc.omega[i][k] is fork.omega[i][k] for i in range(2) for k in fork.omega[i])
    _, s1, r1 = m(x, init.clone(), target_delta=tg, commit=False)
    unchanged = [{k: v.clone() for k, v in om.items()} for om in s1.omega]
    for fork in (s1, s1.clone(), s1.detach()):
        sc, acc = m.commit_proposal(fork, r1.proposal.with_omega(unchanged))
        assert not acc.applied and all(torch.equal(a, b) for a, b in zip(sc.r, fork.r))
        _, acc_real = m.commit_proposal(fork, r1.proposal)
        assert acc_real.applied and float(acc_real.dW_norm.min()) > 0
    with pytest.raises(ValueError):  # dtype mismatch is refused, not mixed into the state
        m.commit_proposal(s1, r1.proposal.with_omega([{k: v.float() for k, v in om.items()} for om in r1.proposal.omega]))
    # no observed target: the step is computed but zero-valued; not applied, still differentiable
    _, s2, r2 = m(x, m.init_state(2), target_delta=tg, target_mask=torch.zeros(2, 8, dtype=torch.bool), commit=False)
    assert r2.proposal.signals.stepped
    sc, acc = m.commit_proposal(s2, r2.proposal)
    assert not acc.applied and all(torch.equal(a, b) for a, b in zip(sc.r, s2.r))
    assert all(sc.omega[i][k] is r2.proposal.omega[i][k] for i in range(2) for k in s2.omega[i])
    (g_eta,) = torch.autograd.grad(sc.omega[0][THETA_KEY].sum(), m.core.blocks[0].eta_logit)
    assert float(g_eta) == 0.0  # the path to eta exists; its value is zero because g = 0


def test_fixed_z_commit_is_the_discontinuous_diagnostic():
    m = _model()
    mz = _model(commit_rule="fixed_z")
    mz.load_state_dict(m.state_dict())
    x, tg = _stream(2, 24)
    _, st, rep = mz(x[:, :8], mz.init_state(2), target_delta=tg[:, :8], commit=False)
    st_fz, _ = mz.commit_proposal(st, rep.proposal)
    jump = max(float((a - b).detach().abs().max()) for a, b in zip(st_fz.r, st.r))
    assert 1e-8 < jump <= 2 * m.cfg.epsilon + 1e-12
    p_t, _, _ = m(x, m.init_state(2), target_delta=tg)
    p_z, _, _ = mz(x, mz.init_state(2), target_delta=tg)
    assert torch.equal(p_t[:, :8], p_z[:, :8])
    assert float((p_t[:, 8:] - p_z[:, 8:]).detach().abs().max()) > 1e-10


def test_gradients_reach_every_parameter():
    m = _model()
    x, tg = _stream(2, 20)
    m.loss(x, tg).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
        assert float(p.grad.abs().sum()) > 0, name


def test_physics_wrapper_trains_end_to_end_on_cpu():
    torch.manual_seed(0)
    m = CoordinateDynamics(_cfg())
    rng = torch.Generator().manual_seed(0)

    def batch():
        return physics_batch(
            8, seq_len=32, episodes_per_seq=2, mu_range=(0.05, 0.5), nonlinear=False, action_std=0.5, rng=rng
        )

    held = batch()
    pred, st, rep = m(held.inputs, target_delta=held.target_delta)
    assert pred.shape == (8, 32, 4) and st.pos == 32 and len(rep.chunks) == 4 == len(rep.accepted)
    with torch.no_grad():
        before = float(m.loss(held.inputs, held.target_delta))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(25):
        b = batch()
        loss = m.loss(b.inputs, b.target_delta)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        after = float(m.loss(held.inputs, held.target_delta))
    assert after < 0.7 * before, (before, after)
