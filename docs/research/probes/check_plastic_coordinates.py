"""Throwaway mathematical probe; not a trained model or repository implementation."""
import json
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(23)
torch.set_num_threads(1)
D, K, H = 8, 4, 5
RHO, EPS = 0.97, 0.2


def init():
    return {"fa": torch.randn(K, H) * .25,
            "fb": torch.randn(H, K) * .25,
            "ga": torch.randn(K, H) * .25,
            "gb": torch.randn(H, K) * .25,
            "theta": torch.randn(D) * .1}


def shear(x, a, b):
    return EPS * torch.tanh(torch.tanh(x @ a) @ b)


def encode(w, r):
    p, q = r.chunk(2, -1)
    z1 = p + shear(q, w["fa"], w["fb"])
    z2 = q + shear(z1, w["ga"], w["gb"])
    return torch.cat((z1, z2), -1)


def decode(w, z):
    z1, z2 = z.chunk(2, -1)
    q = z2 - shear(z1, w["ga"], w["gb"])
    p = z1 - shear(q, w["fa"], w["fb"])
    return torch.cat((p, q), -1)


def transport(a, b, z):
    return encode(b, decode(a, z))


def scan(a, b, z0):
    # Associative affine prefix, no divisions or in-place autograd writes.
    aa, bb = a, b
    n = a.shape[0]
    offset = 1
    while offset < n:
        aa, bb = (torch.cat((aa[:offset], aa[offset:] * aa[:-offset])),
                  torch.cat((bb[:offset], bb[offset:] + aa[offset:] * bb[:-offset])))
        offset *= 2
    return aa * z0 + bb


def rollout(w, r0, inputs, slow, parallel=True):
    a = RHO * torch.sigmoid(inputs @ slow["a"] + w["theta"])
    v = torch.tanh(inputs @ slow["v"])
    b = (1 - a) * v
    z0 = encode(w, r0)
    if parallel:
        zs = scan(a, b, z0)
    else:
        z = z0
        out = []
        for at, bt in zip(a, b):
            z = at * z + bt
            out.append(z)
        zs = torch.stack(out)
    return decode(w, zs)


slow = {k: torch.randn(D, D) / D**.5 for k in ("a", "v")}
w, w2, w3 = init(), init(), init()
r0 = torch.randn(D)
x = torch.randn(13, D)
target = torch.randn(13, D)
metrics = {}
metrics["inverse_max_error"] = (decode(w, encode(w, x)) - x).abs().max().item()
z = encode(w, r0)
metrics["transport_preservation_error"] = (decode(w2, transport(w, w2, z))-r0).abs().max().item()
metrics["transport_composition_error"] = (transport(w2, w3, transport(w, w2, z))-transport(w, w3, z)).abs().max().item()
metrics["transport_roundtrip_error"] = (transport(w2, w, transport(w, w2, z))-z).abs().max().item()
metrics["scan_max_error"] = (rollout(w, r0, x, slow)-rollout(w, r0, x, slow, False)).abs().max().item()


def loss(w, r, inp, targets, slow):
    states = rollout(w, r, inp, slow)
    return ((states - targets)**2).mean(), states[-1]


gp = torch.func.grad(lambda ww: loss(ww, r0, x, target, slow)[0])(w)
gs = torch.func.grad(lambda ww: ((rollout(ww, r0, x, slow, False)-target)**2).mean())(w)
metrics["scan_gradient_error"] = max((gp[k]-gs[k]).abs().max().item() for k in w)

# Tensor chart weights affect the nonlinear transition (not merely its name).
changed = {k: v.clone() for k, v in w.items()}
changed["fa"] = changed["fa"] + .4
metrics["chart_changes_future_max"] = (rollout(changed, r0, x, slow)-rollout(w, r0, x, slow)).abs().max().item()
instant_grad = torch.func.grad(lambda ww: decode(ww, encode(ww, r0)).square().sum())(w)
metrics["boundary_only_canary_gradient"] = max(v.abs().max().item() for v in instant_grad.values())

# Causality under frozen chunk parameters, perturb all future inputs.
x2 = x.clone()
x2[6:] += torch.randn_like(x2[6:])
metrics["future_input_prefix_error"] = (rollout(w, r0, x, slow)[:6]-rollout(w, r0, x2, slow)[:6]).abs().max().item()

# Three support/update chunks, then a separate query chunk. Every inner call
# differentiates only w, while r remains an independent argument with outer graph.
episodes = [(torch.randn(7, D), torch.randn(7, D)) for _ in range(4)]


def meta_objective(w0, slow0, log_eta):
    ww, rr = w0, r0
    for inp, tgt in episodes[:-1]:
        g, (ll, rr_new) = torch.func.grad_and_value(loss, argnums=0, has_aux=True)(ww, rr, inp, tgt, slow0)
        ww = {k: ww[k] - log_eta.exp()*g[k] for k in ww}
        rr = rr_new  # canonical end state is retained, NOT replayed with new weights
    return loss(ww, rr, *episodes[-1], slow0)[0]


eta = torch.tensor(-1.8)
gw, gslow, geta = torch.func.grad(meta_objective, argnums=(0,1,2))(w, slow, eta)
direction = {k: torch.randn_like(v) for k, v in w.items()}
norm = sum(v.square().sum() for v in direction.values()).sqrt()
direction = {k: v/norm for k,v in direction.items()}
analytic = sum((gw[k]*direction[k]).sum() for k in w)
h = 1e-5
plus = {k:w[k]+h*direction[k] for k in w}
minus = {k:w[k]-h*direction[k] for k in w}
numeric = (meta_objective(plus, slow, eta)-meta_objective(minus, slow, eta))/(2*h)
metrics["meta_directional_relative_error"] = ((analytic-numeric).abs()/(analytic.abs()+numeric.abs()+1e-12)).item()
numeric_eta = (meta_objective(w, slow, eta+h)-meta_objective(w, slow, eta-h))/(2*h)
metrics["meta_eta_relative_error"] = ((geta-numeric_eta).abs()/(geta.abs()+numeric_eta.abs()+1e-12)).item()
metrics["meta_eta_derivative"] = geta.item()
metrics["meta_slow_gradient_norm"] = sum(v.square().sum() for v in gslow.values()).sqrt().item()
metrics["meta_fast_gradient_norm"] = sum(v.square().sum() for v in gw.values()).sqrt().item()

# Scalar chain-rule falsifier: E_w(r)=r+w and z'=a*z.
a, wr, rr = .6, torch.tensor(.4), torch.tensor(.9)
total = torch.func.grad(lambda p: a*(rr+p)-p)(wr)
old_z = (rr+wr).detach()
wrong = torch.func.grad(lambda p: a*old_z-p)(wr)
metrics["scalar_transported_derivative"] = total.item()
metrics["scalar_wrong_fixed_latent_derivative"] = wrong.item()

# Bounded additive shears: canonical state stays bounded even with changes in
# arbitrarily large chart weights. This is a sup-norm boundedness check, not a
# contraction proof or robustness test.
rr = torch.randn(D)
peak = rr.abs().max().item()
for _ in range(100):
    ww = {k: 50*torch.randn_like(v) for k,v in w.items()}
    ss = rollout(ww, rr, 5*torch.randn(8,D), slow)
    peak = max(peak, ss.abs().max().item())
    rr = ss[-1]
bound = max(1. + EPS*(1+RHO)/(1-RHO), 3.)
metrics["bounded_switch_observed_peak"] = peak
metrics["bounded_switch_theoretical_bound"] = bound

# Stable but unrestricted linear conjugacies can grow under switching.
f1 = torch.tensor([[.25, 1.], [0., .75]])
f2 = torch.tensor([[.25, 0.], [1., .75]])
metrics["unbounded_chart_switch_spectral_radius"] = torch.linalg.eigvals(f2@f1).abs().max().item()

assert max(metrics[k] for k in ["inverse_max_error", "transport_preservation_error", "transport_composition_error", "transport_roundtrip_error", "scan_max_error", "scan_gradient_error", "boundary_only_canary_gradient", "future_input_prefix_error"]) < 1e-10
assert metrics["chart_changes_future_max"] > 1e-6
assert metrics["meta_directional_relative_error"] < 1e-5
assert metrics["meta_eta_relative_error"] < 1e-5
assert metrics["meta_fast_gradient_norm"] > 1e-7
assert metrics["meta_slow_gradient_norm"] > 1e-7
assert peak < bound
metrics["torch_version"] = torch.__version__
metrics["device"] = "cpu"
metrics["mps_available"] = torch.backends.mps.is_available()
print(json.dumps(metrics, indent=2))
