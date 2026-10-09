"""Exp 1 sanity checks. Writes results/exp1/sanity.json."""
import json, os, sys, subprocess
import torch
sys.path.insert(0, os.path.dirname(__file__))
from models import SeqModel, scan_signed_chunked
from tasks import batch, group_table
from plastic.model.delta import delta_chunk, delta_recurrent
from plastic.model.scan import scan_chunked, scan_sequential
import torch.nn.functional as F

out = {}
torch.manual_seed(0)
# 1. delta_chunk == delta_recurrent for beta in (0,2) (the WY/solve form is algebraic in beta)
B, H, T, d = 3, 2, 150, 16
q = F.normalize(torch.randn(B, H, T, d), dim=-1); k = F.normalize(torch.randn(B, H, T, d), dim=-1)
v = torch.randn(B, H, T, d)
for name, beta in [("beta01", torch.rand(B, H, T)), ("beta02", 2 * torch.rand(B, H, T)), ("beta_near2", 1.95 + 0.05 * torch.rand(B, H, T))]:
    alpha = torch.exp(-torch.rand(B, H, T) * 0.1)
    o1, S1, e1 = delta_recurrent(q, k, v, beta, alpha)
    o2, S2, e2 = delta_chunk(q, k, v, beta, alpha, chunk=64)
    out[f"delta_chunk_vs_recurrent_maxabs_{name}"] = float(max((o1 - o2).abs().max(), (S1 - S2).abs().max()))
    out[f"state_norm_{name}"] = float(S1.flatten(-2).norm(dim=-1).max())
# 2. eigenvalues of the per-token transition alpha (I - beta k k^T)
kk = F.normalize(torch.randn(1000, d), dim=-1)
for bm in (1.0, 2.0):
    be = bm * torch.rand(1000); al = torch.rand(1000)
    M = al[:, None, None] * (torch.eye(d) - be[:, None, None] * kk[:, :, None] * kk[:, None, :])
    ev = torch.linalg.eigvalsh(M)
    out[f"transition_eig_range_betamax{bm}"] = [float(ev.min()), float(ev.max())]
# 3. signed scan vs sequential reference (and vs repo chunked scan for positive decays)
a = 2 * torch.rand(4, 300, 8) - 1; u = torch.randn(4, 300, 8); h0 = torch.randn(4, 8)
h1, l1 = scan_sequential(a, u, h0); h2, l2 = scan_signed_chunked(a, u, h0, chunk=16)
out["signed_scan_vs_sequential_maxabs"] = float((h1 - h2).abs().max())
ap = torch.rand(4, 300, 8)
out["signed_scan_vs_repo_scan_positive_maxabs"] = float((scan_chunked(ap, u, h0, 16)[0] - scan_signed_chunked(ap, u, h0, 16)[0]).abs().max())
out["repo_scan_with_negative_decay_maxabs_error"] = float((scan_chunked(a, u, h0, 16)[0] - h1).abs().max())
# 4. B with beta_max=1 reproduces A bit for bit (same seed: same init, same forward, same 30 training steps)
def run(variant, beta_max):
    torch.manual_seed(7)
    m = SeqModel(variant, 3, 2, beta_max=beta_max)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(30):
        tok, lab = batch("parity", 32, 24, g)
        lg = m(tok); loss = F.cross_entropy(lg.reshape(-1, 2), lab.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    return losses, m
lA, mA = run("A", None); lB1, mB1 = run("B", 1.0); lB2, _ = run("B", 2.0)
out["B_betamax1_equals_A_losses_bitwise"] = lA == lB1
out["B_betamax1_vs_A_param_maxabs"] = float(max((p - q).abs().max() for p, q in zip(mA.parameters(), mB1.parameters())))
out["B_betamax2_differs_from_A"] = lA != lB2
# 5. group tables
for gname in ("s3", "s5"):
    tab, e = group_table(gname); G = tab.shape[0]
    i, j, l = torch.randint(0, G, (3, 2000))
    out[f"{gname}_associative"] = bool((tab[tab[i, j], l] == tab[i, tab[j, l]]).all())
    out[f"{gname}_identity"] = bool((tab[e] == torch.arange(G)).all() and (tab[:, e] == torch.arange(G)).all())
    out[f"{gname}_nonabelian"] = bool((tab != tab.T).any())
# 6. labels
tok, lab = batch("parity", 4, 50, torch.Generator().manual_seed(0))
out["parity_label_check"] = bool(((tok[:, 1:] - 1).cumsum(1) % 2 == lab[:, 1:]).all())
os.makedirs("results/exp1", exist_ok=True)
json.dump(out, open("results/exp1/sanity.json", "w"), indent=1)
print(json.dumps(out, indent=1))
