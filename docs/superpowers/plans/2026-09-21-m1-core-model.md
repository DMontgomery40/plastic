# M1 Core Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `plastic` model family (selective diagonal recurrence + gated delta-rule fast-weight memory + MLP) for text and physics, with a chunk-parallel training path and a recurrent inference path that agree, carried session state, and a smoke-trainable outer loop.

**Architecture:** One `PlasticBlock` composes an RG-LRU style selective scan (activation state `h`) with a fast-weight memory `S` updated by one gradient step per token on `½‖kS − v‖²` with learned write rate β and forget α, then an MLP. `PlasticLM` (tied embedding, BPE tokens) and `PlasticDynamics` (7-d physics input, 4-d delta-observation head) share `PlasticCore`, a stack of blocks. The chunk-parallel path uses the WY form with a nilpotent inverse; the recurrent path is a per-token loop; both accept an incoming `SessionState` and return the outgoing one.

**Tech Stack:** Python 3.12, torch 2.12 (CPU, MPS, CUDA), uv, pytest, hypothesis. Pure PyTorch, no Triton, no `torch.compile` dependence.

**Spec:** `docs/superpowers/specs/2026-09-21-plastic-design.md` (sections 4, 5, 9, 11)

## Global Constraints

- Package name `plastic`; Python `>=3.12`; `torch>=2.12`. Run everything through `uv run`.
- Never form `a^t` or divide by decays: the scan uses pairwise log-space ratios inside a chunk and a sequential carry across chunks.
- Keys and queries are L2-normalized; β ∈ (0,1) via sigmoid; α ∈ (0,1) via `exp(−softplus(·))`.
- No words "legacy", "v1", "backward compatibility", no schema-version fields.
- Every task ends with `uv run pytest` green and a commit on branch `fuse`. Never push.
- Tests run on CPU; MPS tests are parametrized and skipped when MPS is unavailable.

---

### Task 0: Package scaffold

**Files:**
- Create: `pyproject.toml` (replaces the `ttt_eval` one), `.python-version`, `plastic/__init__.py`, `tests/test_package.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `tests/conftest.py::devices` fixture parametrizing over `"cpu"` and `"mps"` (skipped when unavailable); `torch.manual_seed(0)` autouse fixture.

- [ ] **Step 1: Write the failing test** (`tests/test_package.py`)

```python
from plastic import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Write `pyproject.toml`** (name `plastic`, hatchling, deps torch/numpy/fastapi/uvicorn/pydantic/huggingface_hub, dev pytest/hypothesis/httpx, script `plastic = "plastic.cli:main"`, pytest `testpaths = ["tests"]`), `.python-version` = `3.12`, `plastic/__init__.py` with `__version__ = "0.1.0"`.

- [ ] **Step 3: `tests/conftest.py`**

```python
import pytest
import torch


def _available_devices() -> list[str]:
    out = ["cpu"]
    if torch.backends.mps.is_available():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


@pytest.fixture(params=_available_devices())
def device(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
```

- [ ] **Step 4: Run** `uv sync --extra dev && uv run pytest -q` → 1 passed.
- [ ] **Step 5: Commit** `git add pyproject.toml .python-version plastic tests && git commit -m "build: scaffold the plastic package with uv"`

---

### Task 1: Selective diagonal scan (`plastic/model/scan.py`)

**Files:**
- Create: `plastic/model/__init__.py`, `plastic/model/scan.py`
- Test: `tests/test_scan.py`

**Interfaces:**
- Produces:
  - `scan_sequential(a: Tensor, u: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]`: `a, u` of shape `(B, T, D)`, `h0` `(B, D)`; returns `(h_all (B,T,D), h_last (B,D))` for `h_t = a_t ⊙ h_{t−1} + u_t`.
  - `scan_chunked(a: Tensor, u: Tensor, h0: Tensor | None = None, chunk: int = 64) -> tuple[Tensor, Tensor]`: same contract, any `T` (internally pads to a multiple of `chunk` with `a = 1, u = 0` and slices the result).

- [ ] **Step 1: Failing tests** (`tests/test_scan.py`)

```python
import torch
import pytest
from plastic.model.scan import scan_sequential, scan_chunked


@pytest.mark.parametrize("T", [1, 5, 64, 130, 512])
def test_chunked_matches_sequential(device, T):
    B, D = 2, 32
    u = torch.randn(B, T, D, device=device)
    a = torch.sigmoid(torch.randn(B, T, D, device=device) * 2 - 1)
    h0 = torch.randn(B, D, device=device)
    ref, ref_last = scan_sequential(a, u, h0)
    got, got_last = scan_chunked(a, u, h0, chunk=64)
    assert torch.allclose(ref, got, atol=1e-4, rtol=1e-4)
    assert torch.allclose(ref_last, got_last, atol=1e-4, rtol=1e-4)


def test_tiny_decays_do_not_blow_up(device):
    B, T, D = 2, 256, 16
    u = torch.randn(B, T, D, device=device)
    a = torch.rand(B, T, D, device=device) * 0.2 + 1e-3
    ref, _ = scan_sequential(a, u)
    got, _ = scan_chunked(a, u, chunk=64)
    assert torch.isfinite(got).all()
    assert (ref - got).abs().max() < 1e-4


def test_gradients_flow(device):
    B, T, D = 1, 128, 8
    a = torch.sigmoid(torch.randn(B, T, D, device=device)).requires_grad_(True)
    u = torch.randn(B, T, D, device=device, requires_grad=True)
    h, _ = scan_chunked(a, u, chunk=32)
    h.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    assert u.grad is not None and torch.isfinite(u.grad).all()
```

- [ ] **Step 2: Run** `uv run pytest tests/test_scan.py -q` → ImportError.
- [ ] **Step 3: Implement** (reference, verified 2e-5 vs sequential on CPU and MPS)

```python
"""Selective diagonal recurrence h_t = a_t * h_{t-1} + u_t, numerically stable."""
from __future__ import annotations

import torch
from torch import Tensor


def scan_sequential(a: Tensor, u: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
    B, T, D = u.shape
    h = torch.zeros(B, D, dtype=u.dtype, device=u.device) if h0 is None else h0
    out = []
    for t in range(T):
        h = a[:, t] * h + u[:, t]
        out.append(h)
    return torch.stack(out, dim=1), h


def scan_chunked(a: Tensor, u: Tensor, h0: Tensor | None = None, chunk: int = 64) -> tuple[Tensor, Tensor]:
    B, T, D = u.shape
    L = int(chunk)
    pad = (-T) % L
    if pad:
        a = torch.cat([a, torch.ones(B, pad, D, dtype=a.dtype, device=a.device)], dim=1)
        u = torch.cat([u, torch.zeros(B, pad, D, dtype=u.dtype, device=u.device)], dim=1)
    n = (T + pad) // L
    a = a.view(B, n, L, D)
    u = u.view(B, n, L, D)
    log_a = torch.log(a.clamp_min(1e-30))
    C = torch.cumsum(log_a, dim=2)                                  # (B,n,L,D), <= 0
    diff = C.unsqueeze(3) - C.unsqueeze(2)                          # (B,n,L,L,D) [t,s]
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=u.device))
    P = torch.exp(diff.masked_fill(~mask.view(1, 1, L, L, 1), float("-inf")))
    intra = torch.einsum("bntsd,bnsd->bntd", P, u)
    decay_from_start = torch.exp(C)
    h = torch.zeros(B, D, dtype=u.dtype, device=u.device) if h0 is None else h0
    outs = []
    for c in range(n):
        oc = intra[:, c] + decay_from_start[:, c] * h.unsqueeze(1)
        outs.append(oc)
        h = oc[:, -1]
    h_all = torch.stack(outs, dim=1).view(B, T + pad, D)[:, :T]
    return h_all, h_all[:, -1]
```

- [ ] **Step 4: Run** `uv run pytest tests/test_scan.py -q` → all pass.
- [ ] **Step 5: Commit** `git commit -m "feat(model): chunked log-space selective scan with sequential reference"`

---

### Task 2: Gated delta rule (`plastic/model/delta.py`)

**Files:**
- Create: `plastic/model/delta.py`
- Test: `tests/test_delta.py`

**Interfaces:**
- Produces:
  - `delta_recurrent(q, k, v, beta, alpha, S0=None) -> tuple[Tensor, Tensor, Tensor]` with `q,k,v: (B,H,T,d)`, `beta, alpha: (B,H,T)`, `S0: (B,H,d,d) | None`; returns `(o (B,H,T,d), S (B,H,d,d), err (B,H,T))` where `err_t = ‖v_t − k_t (α_t S_{t−1})‖`.
  - `delta_chunk(q, k, v, beta, alpha, S0=None, chunk=64) -> tuple[Tensor, Tensor, Tensor]`: same outputs, any `T` (pads with `beta = 0, alpha = 1` so padded tokens neither write nor decay).
  - Row-vector convention: read is `k @ S`; write is `S += β kᵀ e`.

- [ ] **Step 1: Failing tests** (`tests/test_delta.py`)

```python
import torch
import torch.nn.functional as F
import pytest
from hypothesis import given, settings, strategies as st
from plastic.model.delta import delta_recurrent, delta_chunk


def _inputs(B, H, T, d, device, S0=True):
    q = F.normalize(torch.randn(B, H, T, d, device=device), dim=-1)
    k = F.normalize(torch.randn(B, H, T, d, device=device), dim=-1)
    v = torch.randn(B, H, T, d, device=device)
    beta = torch.sigmoid(torch.randn(B, H, T, device=device))
    alpha = torch.sigmoid(torch.randn(B, H, T, device=device) + 3)
    S = torch.randn(B, H, d, d, device=device) * 0.1 if S0 else None
    return q, k, v, beta, alpha, S


@pytest.mark.parametrize("T", [1, 7, 64, 200])
def test_chunk_matches_recurrent(device, T):
    q, k, v, beta, alpha, S0 = _inputs(2, 3, T, 16, device)
    o1, S1, e1 = delta_recurrent(q, k, v, beta, alpha, S0)
    o2, S2, e2 = delta_chunk(q, k, v, beta, alpha, S0, chunk=64)
    assert torch.allclose(o1, o2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S1, S2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(e1, e2, atol=1e-4, rtol=1e-4)


def test_one_shot_recall(device):
    d = 16
    k1 = F.normalize(torch.randn(1, 1, 1, d, device=device), dim=-1)
    v1 = torch.randn(1, 1, 1, d, device=device)
    beta = torch.ones(1, 1, 1, device=device)
    alpha = torch.ones(1, 1, 1, device=device)
    o, S, _ = delta_recurrent(k1, k1, v1, beta, alpha)
    assert torch.allclose(o[0, 0, 0], v1[0, 0, 0], atol=1e-5)


def test_write_pressure_identity(device):
    q, k, v, beta, alpha, S0 = _inputs(1, 2, 20, 8, device)
    _, _, err = delta_recurrent(q, k, v, beta, alpha, S0)
    # replay to get per-step state deltas
    S = S0.clone()
    for t in range(20):
        Sa = alpha[:, :, t, None, None] * S
        e = v[:, :, t] - torch.einsum("bhk,bhkv->bhv", k[:, :, t], Sa)
        S_new = Sa + beta[:, :, t, None, None] * torch.einsum("bhk,bhv->bhkv", k[:, :, t], e)
        dS = (S_new - Sa).flatten(-2).norm(dim=-1)
        assert torch.allclose(dS, beta[:, :, t] * err[:, :, t], atol=1e-5)
        S = S_new


@settings(max_examples=25, deadline=None)
@given(T=st.integers(1, 96), scale=st.floats(0.1, 50.0))
def test_state_bounded_for_any_input(T, scale):
    B, H, d = 1, 1, 8
    q = F.normalize(torch.randn(B, H, T, d), dim=-1)
    k = F.normalize(torch.randn(B, H, T, d), dim=-1)
    v = torch.randn(B, H, T, d) * scale
    beta = torch.rand(B, H, T)
    alpha = torch.rand(B, H, T)
    _, S, _ = delta_recurrent(q, k, v, beta, alpha)
    # each row-space contraction keeps ||S|| <= max ||v|| * sum of geometric weights < inf; check no blow-up
    assert torch.isfinite(S).all()
    assert S.norm() <= scale * T + 1e-6


def test_gradients_reach_gates(device):
    q, k, v, beta, alpha, S0 = _inputs(1, 2, 64, 8, device)
    beta = beta.clone().requires_grad_(True)
    alpha = alpha.clone().requires_grad_(True)
    q = q.clone().requires_grad_(True)
    o, _, _ = delta_chunk(q, k, v, beta, alpha, S0, chunk=32)
    o.pow(2).mean().backward()
    for g in (beta.grad, alpha.grad, q.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
```

- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement**

```python
"""Gated delta rule: one gradient step per token on ½||k S − v||² with write rate β and forget α."""
from __future__ import annotations

import math

import torch
from torch import Tensor


def delta_recurrent(q, k, v, beta, alpha, S0=None):
    B, H, T, d = q.shape
    S = torch.zeros(B, H, d, d, dtype=q.dtype, device=q.device) if S0 is None else S0
    outs, errs = [], []
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        bt = beta[:, :, t, None, None]
        at = alpha[:, :, t, None, None]
        S = at * S
        e = vt - torch.einsum("bhk,bhkv->bhv", kt, S)
        errs.append(e.norm(dim=-1))
        S = S + bt * torch.einsum("bhk,bhv->bhkv", kt, e)
        outs.append(torch.einsum("bhk,bhkv->bhv", qt, S))
    return torch.stack(outs, dim=2), S, torch.stack(errs, dim=2)


def delta_chunk(q, k, v, beta, alpha, S0=None, chunk=64):
    B, H, T, d = q.shape
    L = int(chunk)
    pad = (-T) % L
    if pad:
        z = lambda x: torch.cat([x, torch.zeros(B, H, pad, d, dtype=x.dtype, device=x.device)], dim=2)
        q, k, v = z(q), z(k), z(v)
        beta = torch.cat([beta, torch.zeros(B, H, pad, dtype=beta.dtype, device=beta.device)], dim=2)
        alpha = torch.cat([alpha, torch.ones(B, H, pad, dtype=alpha.dtype, device=alpha.device)], dim=2)
    n = (T + pad) // L
    q, k, v = (x.view(B, H, n, L, d) for x in (q, k, v))
    beta, alpha = beta.view(B, H, n, L), alpha.view(B, H, n, L)
    g = torch.cumsum(torch.log(alpha.clamp_min(1e-30)), dim=-1)          # γ_t (log)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2)                                # [t,s]
    tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
    stril = torch.tril(tril, -1)
    Dm = torch.exp(diff.masked_fill(~tril, float("-inf")))                 # inclusive
    Ds = torch.exp(diff.masked_fill(~stril, float("-inf")))                # strict
    kb = k * beta.unsqueeze(-1)
    A = torch.einsum("bhntd,bhnsd->bhnts", kb, k) * Ds                      # strictly lower
    I = torch.eye(L, dtype=q.dtype, device=q.device)
    X = -A
    Tinv = I + X
    for _ in range(max(0, int(math.log2(L)) - 1)):
        X = X @ X
        Tinv = Tinv @ (I + X)
    gexp = torch.exp(g)
    w = Tinv @ (kb * gexp.unsqueeze(-1))
    u = Tinv @ (v * beta.unsqueeze(-1))
    S = torch.zeros(B, H, d, d, dtype=q.dtype, device=q.device) if S0 is None else S0
    outs, errs = [], []
    for c in range(n):
        qc, kc = q[:, :, c], k[:, :, c]
        u_eff = u[:, :, c] - w[:, :, c] @ S
        Aqk = torch.einsum("bhtd,bhsd->bhts", qc, kc) * Dm[:, :, c]
        o = torch.einsum("bhts,bhsd->bhtd", Aqk, u_eff) + torch.einsum("bhtd,bhdv->bhtv", qc * gexp[:, :, c].unsqueeze(-1), S)
        outs.append(o)
        errs.append(u_eff.norm(dim=-1) / beta[:, :, c].clamp_min(1e-12))
        dec_to_end = torch.exp(g[:, :, c, -1:] - g[:, :, c])
        S = gexp[:, :, c, -1, None, None] * S + torch.einsum("bhsd,bhsv->bhdv", kc * dec_to_end.unsqueeze(-1), u_eff)
    o = torch.stack(outs, dim=2).view(B, H, T + pad, d)[:, :, :T]
    err = torch.stack(errs, dim=2).view(B, H, T + pad)[:, :, :T]
    return o, S, err
```

Note on padded tokens: β = 0 makes `u_eff = 0`, so `err` is `0/1e-12 = 0` there and they are sliced off anyway.

- [ ] **Step 4: Run** `uv run pytest tests/test_delta.py -q` → pass.
- [ ] **Step 5: Commit** `git commit -m "feat(model): gated delta rule with recurrent and chunk-parallel paths"`

---

### Task 3: Config and session state (`plastic/config.py`, `plastic/model/state.py`)

**Files:**
- Create: `plastic/config.py`, `plastic/model/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Produces:
  - `ModelConfig` dataclass (frozen): `domain: Literal["text","physics"] = "text"`, `d_model=256`, `n_heads=4`, `n_layers=4`, `chunk=64`, `vocab_size=4096`, `tie_embeddings=True`, `rule: Literal["delta","chunk"]="delta"`, `memory: Literal["linear","mlp"]="linear"`, `memory_input: Literal["ssm_out","block_in"]="ssm_out"`, `mlp_mult=4`, `obs_dim=4`, `act_dim=2`, `ssm_c=8.0`, `beta_bias_init=0.0`, `alpha_bias_init=-4.0`, `lam_init=2.197`, `chunk_momentum=0.9`, `chunk_orthogonalize=True`, `chunk_lr=0.1`, `mem_hidden_mult=2`. Methods: `head_dim`, `input_dim` (7 for physics), `to_dict()`, `from_dict()`, `to_json()`, `from_json()`, `signature_material()` (canonical JSON string).
  - `LayerState` dataclass: `h: Tensor (B,D)`, `S: Tensor (B,H,dh,dh)`, `M: Tensor | None` (same shape as `S`, only when `rule="chunk"`).
  - `SessionState`: `layers: list[LayerState]`, `pos: int`; `SessionState.zeros(cfg, batch=1, device=None, dtype=torch.float32)`, `.clone()`, `.detach()`, `.to(device)`, `.state_dict() -> dict[str, Tensor|int]`, `SessionState.from_state_dict(d)`, `.s_delta(other) -> list[Tensor]` (per-layer `S − other.S`), `.norms() -> dict` with `s_norm`, `h_norm` per layer and totals.

- [ ] **Step 1: Failing tests** (`tests/test_state.py`)

```python
import json
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


def test_state_zeros_clone_roundtrip():
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2)
    st = SessionState.zeros(cfg, batch=3)
    assert len(st.layers) == 2 and st.layers[0].S.shape == (3, 2, 16, 16) and st.layers[0].h.shape == (3, 32)
    st.layers[0].S += 1.0
    c = st.clone()
    c.layers[0].S += 1.0
    assert float(st.layers[0].S[0, 0, 0, 0]) == 1.0
    d = st.state_dict()
    back = SessionState.from_state_dict(d)
    assert back.pos == st.pos and torch.equal(back.layers[0].S, st.layers[0].S)
    deltas = c.s_delta(st)
    assert torch.allclose(deltas[0], torch.ones_like(deltas[0]))
    assert set(st.norms().keys()) >= {"s_norm", "h_norm", "s_norm_total"}
```

- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement** both modules (dataclasses with `asdict`, `json.dumps(sort_keys=True)`; `SessionState.zeros` allocates per layer; `state_dict` keys `layer{i}.h`, `layer{i}.S`, `layer{i}.M`, `pos`).
- [ ] **Step 4: Run** → pass.
- [ ] **Step 5: Commit** `git commit -m "feat: model config and session state containers"`

---

### Task 4: Fast-weight memory module (`plastic/model/memory.py`) and chunk rule (`plastic/model/chunk_rule.py`)

**Files:**
- Create: `plastic/model/chunk_rule.py`, `plastic/model/memory.py`
- Test: `tests/test_chunk_rule.py`, `tests/test_memory.py`

**Interfaces:**
- Produces:
  - `newton_schulz(G: Tensor, steps: int = 5) -> Tensor`: orthogonalize the last two dims of `G` (batched), output has singular values ≈ 1.
  - `chunk_rule_step(S, M, k, v, beta, alpha_chunk, *, lr, momentum, orthogonalize) -> tuple[S_new, M_new, err]` for a linear memory: `err = k S − v` at chunk-start weights (rows), mean-scaled gradient `G = (1/L) kᵀ (β ⊙ err)`, `M_new = momentum·M + G` (optionally `newton_schulz(M_new)` scaled to the Frobenius norm of `G`), `S_new = alpha_chunk · S − lr · M_new`.
  - `MemorySignals` dataclass: `err (B,H,T)`, `beta (B,H,T)`, `alpha (B,H,T)`, `write_norm (B,H,T)` (= β·err for rule `delta`).
  - `FastWeightMemory(cfg: ModelConfig)` nn.Module with `forward(u: Tensor (B,T,D), S0: Tensor (B,H,dh,dh), M0: Tensor | None, *, mode: Literal["chunk","recurrent"], beta_scale: float = 1.0) -> tuple[m (B,T,D) (per-head reads, RMSNorm per head, concatenated), S_new, M_new, MemorySignals]`.
  - Learned parameters: `W_q, W_k, W_v` (no bias), `W_beta: Linear(D, H)` (bias init `cfg.beta_bias_init`), `W_alpha: Linear(D, H)` (bias init `cfg.alpha_bias_init`), `norm_mem: RMSNorm(dh)`.

- [ ] **Step 1: Failing tests**

`tests/test_chunk_rule.py`
```python
import torch
import torch.nn.functional as F
from plastic.model.chunk_rule import newton_schulz, chunk_rule_step


def test_newton_schulz_orthogonalizes():
    G = torch.randn(2, 3, 16, 16)
    O = newton_schulz(G, steps=5)
    sv = torch.linalg.svdvals(O)
    assert (sv > 0.6).all() and (sv < 1.4).all()


def test_chunk_step_fixed_norm_when_orthogonalized():
    B, H, L, d = 1, 1, 32, 16
    k = F.normalize(torch.randn(B, H, L, d), dim=-1)
    v = torch.randn(B, H, L, d)
    beta = torch.rand(B, H, L)
    S = torch.zeros(B, H, d, d); M = torch.zeros(B, H, d, d)
    S1, M1, _ = chunk_rule_step(S, M, k, v, beta, alpha_chunk=torch.ones(B, H), lr=0.1, momentum=0.0, orthogonalize=True)
    S2, M2, _ = chunk_rule_step(S, M, k, v * 10, beta, alpha_chunk=torch.ones(B, H), lr=0.1, momentum=0.0, orthogonalize=True)
    n1, n2 = (S1 - S).norm(), (S2 - S).norm()
    assert abs(float(n1 - n2)) / float(n1) < 0.05


def test_chunk_step_is_stable_mean_scaled():
    B, H, L, d = 1, 1, 64, 32
    k = F.normalize(torch.randn(1, d) + 0.1 * torch.randn(L, d), dim=-1).view(B, H, L, d)
    v = torch.randn(B, H, L, d)
    beta = torch.full((B, H, L), 0.9)
    S = torch.randn(B, H, d, d); M = torch.zeros_like(S)
    for _ in range(50):
        S, M, _ = chunk_rule_step(S, M, k, v, beta, alpha_chunk=torch.full((B, H), 0.99), lr=0.5, momentum=0.0, orthogonalize=False)
    assert torch.isfinite(S).all() and S.norm() < 1e3
```

`tests/test_memory.py`
```python
import torch
from plastic.config import ModelConfig
from plastic.model.memory import FastWeightMemory


def test_chunk_and_recurrent_modes_agree(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(2, 40, 32, device=device)
    S0 = torch.randn(2, 2, 16, 16, device=device) * 0.1
    m1, S1, _, sig1 = mem(u, S0, None, mode="chunk")
    m2, S2, _, sig2 = mem(u, S0, None, mode="recurrent")
    assert torch.allclose(m1, m2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S1, S2, atol=1e-4, rtol=1e-4)
    assert torch.allclose(sig1.write_norm, sig2.write_norm, atol=1e-4)


def test_beta_scale_zero_freezes_memory(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(1, 20, 32, device=device)
    S0 = torch.randn(1, 2, 16, 16, device=device) * 0.1
    _, S1, _, sig = mem(u, S0, None, mode="chunk", beta_scale=0.0)
    # with beta=0 the state only decays: S1 = prod(alpha) * S0 elementwise per head
    assert sig.write_norm.abs().max() == 0
    assert (S1.abs() <= S0.abs() + 1e-6).all()


def test_signals_shapes(device):
    cfg = ModelConfig(d_model=32, n_heads=2, chunk=16)
    mem = FastWeightMemory(cfg).to(device)
    u = torch.randn(3, 17, 32, device=device)
    S0 = torch.zeros(3, 2, 16, 16, device=device)
    m, S, M, sig = mem(u, S0, None, mode="chunk")
    assert m.shape == (3, 17, 32) and S.shape == (3, 2, 16, 16) and M is None
    assert sig.err.shape == sig.beta.shape == sig.alpha.shape == (3, 2, 17)
```

- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement.** `newton_schulz` uses coefficients `(3.4445, −4.7750, 2.0315)`, normalizes by Frobenius norm plus 1e-7, transposes when rows > cols. `FastWeightMemory.forward`: `RMSNorm` is a small module in `plastic/model/norm.py` (`RMSNorm(d, eps=1e-6)`, weight init ones). Projections reshape `(B,T,D) -> (B,H,T,dh)`; `q,k = F.normalize(...)`; `beta = sigmoid(W_beta(u)).transpose(1,2) * beta_scale`; `alpha = exp(−softplus(W_alpha(u))).transpose(1,2)`. Rule `delta`: call `delta_chunk` or `delta_recurrent`; `write_norm = beta * err`. Rule `chunk`: split into chunks, call `chunk_rule_step` per chunk with `alpha_chunk = alpha.mean over chunk`, reads within a chunk use chunk-start weights `m_t = q_t S_c` (apply-then-update), `write_norm` broadcast per chunk `‖S_new − S_c‖_F / L`. Output `m`: `norm_mem` on each head read, then `(B,H,T,dh) -> (B,T,D)`.
- [ ] **Step 4: Run** → pass.
- [ ] **Step 5: Commit** `git commit -m "feat(model): fast-weight memory module with delta and chunk rules"`

---

### Task 5: Block (`plastic/model/block.py`)

**Files:**
- Create: `plastic/model/block.py`
- Test: `tests/test_block.py`

**Interfaces:**
- Produces: `PlasticBlock(cfg)` with `forward(x: Tensor (B,T,D), state: LayerState, *, mode: Literal["chunk","recurrent"], beta_scale: float = 1.0) -> tuple[y (B,T,D), LayerState, MemorySignals]`. Parameters: `norm1, norm2, norm3: RMSNorm`; SSM `W_in, W_r (bias), W_i (bias), lam: Parameter(D) init cfg.lam_init, W_g, W_o`; memory `FastWeightMemory`; `W_g2, W_o2`; `mlp: Sequential(Linear(D, mlp_mult·D), GELU, Linear(mlp_mult·D, D))`. SSM equations exactly as spec 4.1; `log_a = −cfg.ssm_c · r · softplus(−lam)`; input scaled by `sqrt(1 − a² + 1e-6)`. `memory_input="ssm_out"` feeds `norm2(x after SSM residual)`; `"block_in"` feeds `norm1(x)`.

- [ ] **Step 1: Failing tests** (`tests/test_block.py`)

```python
import torch
from plastic.config import ModelConfig
from plastic.model.block import PlasticBlock
from plastic.model.state import SessionState


def _cfg(**kw):
    base = dict(d_model=32, n_heads=2, n_layers=1, chunk=16)
    base.update(kw)
    return ModelConfig(**base)


def test_modes_agree_and_state_carries(device):
    cfg = _cfg()
    blk = PlasticBlock(cfg).to(device)
    x = torch.randn(2, 48, 32, device=device)
    st = SessionState.zeros(cfg, batch=2, device=device).layers[0]
    y_full, s_full, _ = blk(x, st, mode="chunk")
    y_a, s_a, _ = blk(x[:, :20], st, mode="chunk")
    y_b, s_b, _ = blk(x[:, 20:], s_a, mode="chunk")
    y_r, s_r, _ = blk(x, st, mode="recurrent")
    assert torch.allclose(y_full, torch.cat([y_a, y_b], 1), atol=1e-4, rtol=1e-4)
    assert torch.allclose(y_full, y_r, atol=1e-4, rtol=1e-4)
    assert torch.allclose(s_full.S, s_b.S, atol=1e-4) and torch.allclose(s_full.h, s_b.h, atol=1e-4)
    assert torch.allclose(s_full.S, s_r.S, atol=1e-4)


def test_beta_scale_changes_output(device):
    cfg = _cfg()
    blk = PlasticBlock(cfg).to(device)
    x = torch.randn(1, 32, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    y1, _, _ = blk(x, st, mode="chunk")
    y0, s0, _ = blk(x, st, mode="chunk", beta_scale=0.0)
    assert not torch.allclose(y1, y0)
    assert torch.allclose(s0.S, torch.zeros_like(s0.S))


def test_block_in_variant_runs(device):
    cfg = _cfg(memory_input="block_in")
    blk = PlasticBlock(cfg).to(device)
    x = torch.randn(1, 16, 32, device=device)
    st = SessionState.zeros(cfg, batch=1, device=device).layers[0]
    y, _, _ = blk(x, st, mode="chunk")
    assert y.shape == x.shape
```

- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement** per the interface. In `recurrent` mode use `scan_sequential` and memory `mode="recurrent"`; in `chunk` mode use `scan_chunked(chunk=cfg.chunk)` and memory `mode="chunk"`.
- [ ] **Step 4: Run** → pass.
- [ ] **Step 5: Commit** `git commit -m "feat(model): PlasticBlock composing selective scan, fast-weight memory, and MLP"`

---

### Task 6: Models (`plastic/model/lm.py`)

**Files:**
- Create: `plastic/model/lm.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `PlasticCore(cfg)`: `blocks: ModuleList`, `norm_f: RMSNorm`; `forward(x (B,T,D), state: SessionState, *, mode, beta_scale=1.0) -> tuple[y (B,T,D), SessionState (pos advanced by T), list[MemorySignals]]`.
  - `PlasticLM(cfg)`: `embed: Embedding(V, D)`, `core`, `head` (tied to `embed.weight` when `cfg.tie_embeddings`); `forward(tokens (B,T), state: SessionState | None = None, *, mode="chunk", beta_scale=1.0) -> tuple[logits (B,T,V), SessionState, list[MemorySignals]]`; `loss(tokens) -> Tensor` (shifted CE); `num_params() -> int`.
  - `PlasticDynamics(cfg)`: `embed: Linear(cfg.input_dim, D)`, `core`, `head: Linear(D, cfg.obs_dim)`; `forward(inputs (B,T,7), state=None, *, mode, beta_scale) -> (pred_delta (B,T,4), SessionState, signals)`; `loss(inputs, target_delta) -> Tensor` (MSE).
  - `build_model(cfg) -> PlasticLM | PlasticDynamics` by `cfg.domain`.

- [ ] **Step 1: Failing tests** (`tests/test_models.py`)

```python
import torch
from plastic.config import ModelConfig
from plastic.model.lm import PlasticLM, PlasticDynamics, build_model


def test_lm_shapes_and_param_count():
    cfg = ModelConfig()
    lm = PlasticLM(cfg)
    n = lm.num_params()
    assert 5.0e6 < n < 6.5e6, n
    toks = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, st, sig = lm(toks)
    assert logits.shape == (2, 64, cfg.vocab_size) and st.pos == 64 and len(sig) == cfg.n_layers


def test_lm_state_carry_matches_full_pass(device):
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=16, vocab_size=50)
    lm = PlasticLM(cfg).to(device)
    toks = torch.randint(0, 50, (1, 40), device=device)
    full, st_full, _ = lm(toks)
    a, st_a, _ = lm(toks[:, :24])
    b, st_b, _ = lm(toks[:, 24:], st_a)
    assert torch.allclose(full, torch.cat([a, b], 1), atol=1e-4, rtol=1e-4)
    rec, st_r, _ = lm(toks, mode="recurrent")
    assert torch.allclose(full, rec, atol=1e-4, rtol=1e-4)
    assert st_full.pos == st_b.pos == st_r.pos == 40


def test_beta_off_ablation_changes_loss():
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=2, chunk=16, vocab_size=50)
    lm = PlasticLM(cfg)
    toks = torch.randint(0, 50, (2, 64))
    l1 = lm.loss(toks)
    logits0, _, _ = lm(toks, beta_scale=0.0)
    l0 = torch.nn.functional.cross_entropy(logits0[:, :-1].reshape(-1, 50), toks[:, 1:].reshape(-1))
    assert not torch.allclose(l1, l0)


def test_dynamics_model():
    cfg = ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=2, chunk=16)
    m = build_model(cfg)
    assert isinstance(m, PlasticDynamics)
    x = torch.randn(2, 33, 7)
    pred, st, _ = m(x)
    assert pred.shape == (2, 33, 4) and st.pos == 33
    assert m.loss(x, torch.randn(2, 33, 4)).ndim == 0
```

- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement.** Init: `nn.init.normal_(embed.weight, std=0.02)`; head tied when configured, else `Linear(D, V, bias=False)`; `PlasticCore.forward` threads `LayerState` through blocks and returns `SessionState(layers=new, pos=state.pos + T)`.
- [ ] **Step 4: Run** → pass.
- [ ] **Step 5: Commit** `git commit -m "feat(model): PlasticLM and PlasticDynamics on a shared PlasticCore"`

---

### Task 7: Smoke training and throughput check

**Files:**
- Create: `plastic/train/__init__.py`, `plastic/train/optim.py`, `scripts/bench_block.py`
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Produces: `build_optimizer(model: nn.Module, *, lr_matrix=2e-2, lr_other=1e-3, weight_decay=0.1, use_muon=True) -> torch.optim.Optimizer` returning a wrapper `SplitOptimizer` exposing `.step()`, `.zero_grad(set_to_none=True)`, `.state_dict()`, `.load_state_dict()`, `.param_groups` over an inner `Muon` (2-D parameters of `core.blocks` only, `weight_decay` set explicitly) and `AdamW` (everything else). `use_muon=False` puts everything in AdamW.

- [ ] **Step 1: Failing test** (`tests/test_train_smoke.py`)

```python
import torch
from plastic.config import ModelConfig
from plastic.model.lm import PlasticLM, PlasticDynamics
from plastic.train.optim import build_optimizer


def _kv_recall_batch(B, T, V, n_pairs, device):
    """Random key->value pairs written once and queried later in the same sequence."""
    keys = torch.randint(0, V // 2, (B, n_pairs), device=device)
    vals = torch.randint(V // 2, V, (B, n_pairs), device=device)
    seq = torch.stack([keys, vals], -1).reshape(B, -1)
    idx = torch.stack([torch.randperm(n_pairs, device=device) for _ in range(B)])
    q = torch.stack([torch.gather(keys, 1, idx), torch.gather(vals, 1, idx)], -1).reshape(B, -1)
    toks = torch.cat([seq, q], 1)
    return toks[:, :T]


def test_lm_learns_recall_smoke():
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_heads=2, n_layers=2, chunk=16, vocab_size=64)
    lm = PlasticLM(cfg)
    opt = build_optimizer(lm, lr_matrix=2e-2, lr_other=3e-3, use_muon=True)
    losses = []
    for step in range(60):
        toks = _kv_recall_batch(16, 64, 64, 16, "cpu")
        loss = lm.loss(toks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    assert min(losses[-10:]) < 0.8 * max(losses[:5]), losses


def test_dynamics_smoke():
    torch.manual_seed(0)
    cfg = ModelConfig(domain="physics", d_model=32, n_heads=2, n_layers=2, chunk=16)
    m = PlasticDynamics(cfg)
    opt = build_optimizer(m, use_muon=False)
    losses = []
    for _ in range(40):
        x = torch.randn(8, 48, 7)
        target = x[..., :4] * 0.5 + 0.1
        loss = m.loss(x, target)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]
```

- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement** `plastic/train/optim.py` and `scripts/bench_block.py` (builds `ModelConfig()`, runs 5 fwd+bwd+step at B=8, T=1024 on the best device, prints ms/step and tok/s).
- [ ] **Step 4: Run** `uv run pytest -q` (all) and `uv run python scripts/bench_block.py` (expect ≈ 8K tok/s on MPS).
- [ ] **Step 5: Commit** `git commit -m "feat(train): Muon/AdamW split optimizer and smoke training tests"`

---

## Self-review

- Spec coverage for M1: 4.1 block equations (Tasks 1, 2, 4, 5), rule switch (Task 4), memory_input flag (Task 5), 4.2 domains (Task 6), state containers (Task 3), first-order meta-training smoke (Task 7), three checkpoint numbers deferred to M2 (`eval.json`), MLP memory (`memory="mlp"`) is accepted by config but only implemented for the linear case in M1; the MLP inner step is an M4 experiment item and `FastWeightMemory` raises `NotImplementedError` for it until then.
- Type consistency: `LayerState` fields `h, S, M`; `SessionState.layers`, `.pos`; `MemorySignals.err/beta/alpha/write_norm`; `mode` literal `"chunk"|"recurrent"`; `beta_scale` float everywhere.
