"""Aggregate exp2 probe JSONs -> results/exp2/{sessions.csv, summary.csv, summary.md, *.png}."""
import glob, json, os, sys, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/exp2"
# outputs go to a separate dir (default: ../report/exp2) so raw results/ is never touched
O = sys.argv[1] if len(sys.argv) > 1 else "../report/exp2"
os.makedirs(O, exist_ok=True)
rows = [r for f in sorted(glob.glob(f"{R}/probe_seed*.json")) for r in json.load(open(f))]
conds = ["rollback", "commit", "rb_rewind", "rb_zero", "natural"]
keys = ["seed", "session", "type", "n_chunks", "rollback_S_bitexact", "rollback_h_change", "commit_dS"]
with open(f"{O}/sessions.csv", "w") as f:
    cols = keys + [f"{m}_{q}_{c}" for m in ("direct", "frozen") for q in ("kl", "nll") for c in ["never"] + conds] + \
        ["direct_kl_commit_vs_rollback", "frozen_kl_commit_vs_rollback", "natural_decisions"]
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(str(r.get(c, "")).replace(",", ";") for c in cols) + "\n")


def ms(xs):
    xs = [x for x in xs if x == x]
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0, len(xs))


md = ["# Exp 2 summary", ""]
n = len(rows)
md.append(f"Sessions x (type,size) combos: {n}; seeds: {sorted({r['seed'] for r in rows})}")
md.append(f"Committed S after forced rollback bit-identical to pre-chunk S (torch.equal, all layers): "
          f"{sum(r['rollback_S_bitexact'] for r in rows)}/{n}")
nat = [r for r in rows if r["natural_rollback_S_bitexact"] is not None]
md.append(f"Same check on chunks the calibrated harness rolled back on its own (all X chunks rolled back): "
          f"{sum(r['natural_rollback_S_bitexact'] for r in nat)}/{len(nat)}")
md.append(f"Activation state changed by the rolled-back chunk (sum over layers ||h_rb - h_pre||): mean "
          f"{st.mean(r['rollback_h_change'] for r in rows):.3f}, min {min(r['rollback_h_change'] for r in rows):.3f}")
md.append(f"Rewind (restore pre-chunk activations after rollback) KL to never-seen, max over all: "
          f"direct {max(r['direct_kl_rb_rewind'] for r in rows):.3g}, frozen {max(r['frozen_kl_rb_rewind'] for r in rows):.3g}")
md.append("")
summ = []
for tag in ("direct", "frozen"):
    md += [f"## C measured `{tag}` ({'learning on during C' if tag == 'direct' else 'memory frozen during C'})", "",
           "Leakage = KL(never||rollback) / KL(never||commit), per session then averaged; sd across the 3 per-seed means. "
           "ΔNLL = NLL(cond) − NLL(never) on C (nats/token).", "",
           "| type | X chunks | natural decisions (rollback frac) | KL rollback | KL commit | **leakage KL** | KL rb_zero | ΔNLL rollback | ΔNLL commit | leakage ΔNLL | KL rb (tok 0-16) | KL rb (tok 64-256) | KL commit (tok 64-256) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in sorted({r["type"] for r in rows}):
        for nx in sorted({r["n_chunks"] for r in rows}):
            rr = [r for r in rows if r["type"] == t and r["n_chunks"] == nx]
            if not rr:
                continue
            seeds = sorted({r["seed"] for r in rr})
            lk_seed = [st.mean(r[f"{tag}_kl_rollback"] / r[f"{tag}_kl_commit"] for r in rr if r["seed"] == s) for s in seeds]
            dn_rb = [r[f"{tag}_nll_rollback"] - r[f"{tag}_nll_never"] for r in rr]
            dn_cm = [r[f"{tag}_nll_commit"] - r[f"{tag}_nll_never"] for r in rr]
            nat_rb = np.mean([np.mean([d == "rollback" for d in r["natural_decisions"]]) for r in rr])
            klr = ms([r[f"{tag}_kl_rollback"] for r in rr]); klc = ms([r[f"{tag}_kl_commit"] for r in rr])
            klz = ms([r[f"{tag}_kl_rb_zero"] for r in rr])
            lk = ms(lk_seed)
            ldn = st.mean(dn_rb) / st.mean(dn_cm) if abs(st.mean(dn_cm)) > 1e-9 else float("nan")
            e1 = ms([r[f"{tag}_kl_rollback_0_16"] for r in rr]); e3 = ms([r[f"{tag}_kl_rollback_64_256"] for r in rr])
            c3 = ms([r[f"{tag}_kl_commit_64_256"] for r in rr])
            md.append(f"| {t} | {nx} | {nat_rb:.2f} | {klr[0]:.4f}±{klr[1]:.4f} | {klc[0]:.4f}±{klc[1]:.4f} | **{lk[0]:.2f}±{lk[1]:.2f}** | "
                      f"{klz[0]:.4f} | {st.mean(dn_rb):+.4f} | {st.mean(dn_cm):+.4f} | {ldn:.2f} | {e1[0]:.4f} | {e3[0]:.4f} | {c3[0]:.4f} |")
            summ.append(dict(tag=tag, type=t, n_chunks=nx, n=len(rr), natural_rollback_frac=nat_rb, kl_rb=klr[0], kl_rb_sd=klr[1],
                             kl_commit=klc[0], kl_commit_sd=klc[1], leak_kl=lk[0], leak_kl_sd_seeds=lk[1], kl_rb_zero=klz[0],
                             dnll_rb=st.mean(dn_rb), dnll_commit=st.mean(dn_cm), leak_dnll=ldn,
                             kl_rb_0_16=e1[0], kl_rb_16_64=st.mean(r[f"{tag}_kl_rollback_16_64"] for r in rr), kl_rb_64_256=e3[0],
                             kl_cm_0_16=st.mean(r[f"{tag}_kl_commit_0_16"] for r in rr), kl_cm_16_64=st.mean(r[f"{tag}_kl_commit_16_64"] for r in rr), kl_cm_64_256=c3[0]))
    md.append("")
    for t, key in (("repeat_token", "logp_poisontok"), ("repeat_phrase", "logp_phrase")):
        rr = [r for r in rows if r["type"] == t and f"{tag}_{key}_never" in r]
        if rr:
            up = {c: st.mean(r[f"{tag}_{key}_{c}"] - r[f"{tag}_{key}_never"] for r in rr) for c in conds}
            md.append(f"- {t}: mean uplift of log p(poison tokens) on C vs never: " + ", ".join(f"{c} {v:+.4f}" for c, v in up.items()))
    md.append("")
# C decisions divergence
md += ["## Harness decisions on the clean continuation C", ""]
for c in ["never"] + conds:
    ks = [k for r in rows for k in r[f"C_decisions_{c}"]]
    md.append(f"- {c}: " + ", ".join(f"{k}={ks.count(k) / len(ks):.2f}" for k in sorted(set(ks))))
md.append("")
reasons = {}
for r in rows:
    for rs in r["natural_reasons"]:
        for x in rs:
            k = x.split("(")[0]
            reasons[k] = reasons.get(k, 0) + 1
md.append("Natural-decision reasons on X chunks (counts of reason keys): " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))
with open(f"{O}/summary.csv", "w") as f:
    f.write(",".join(summ[0].keys()) + "\n")
    for s in summ:
        f.write(",".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in s.values()) + "\n")
json.dump(summ, open(f"{O}/summary.json", "w"), indent=1)
open(f"{O}/summary.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))

# plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, tag in zip(axes, ("direct", "frozen")):
    ss = [s for s in summ if s["tag"] == tag]
    types = sorted({s["type"] for s in ss}); sizes = sorted({s["n_chunks"] for s in ss})
    w = 0.8 / len(sizes)
    for j, nx in enumerate(sizes):
        vals = [next((s for s in ss if s["type"] == t and s["n_chunks"] == nx), None) for t in types]
        ax.bar(np.arange(len(types)) + j * w, [v["leak_kl"] for v in vals], w, yerr=[v["leak_kl_sd_seeds"] for v in vals],
               capsize=2, label=f"{nx} chunk(s) = {64 * nx} tok")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(np.arange(len(types)) + 0.4 - w / 2); ax.set_xticklabels(types, rotation=20)
    ax.set_ylabel("KL(never||rollback) / KL(never||commit)"); ax.set_title(f"leakage fraction, C {tag}")
    ax.legend(fontsize=7)
fig.suptitle("Exp 2: influence of a rolled-back chunk as a fraction of the committed chunk's influence\n(rewind of activation state after rollback: 0 exactly)")
fig.tight_layout(); fig.savefig(f"{O}/leakage_fraction.png", dpi=130)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, tag in zip(axes, ("direct", "frozen")):
    ss = [s for s in summ if s["tag"] == tag and s["n_chunks"] == max(x["n_chunks"] for x in summ)]
    xs = [8, 40, 160]
    for s in ss:
        l, = ax.plot(xs, [s["kl_rb_0_16"], s["kl_rb_16_64"], s["kl_rb_64_256"]], marker="o", label=f"{s['type']} rollback")
        ax.plot(xs, [s["kl_cm_0_16"], s["kl_cm_16_64"], s["kl_cm_64_256"]], marker="x", ls="--", color=l.get_color())
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("position in C (bucket centre: 0-16, 16-64, 64-256)")
    ax.set_ylabel("KL to never-seen"); ax.set_title(f"decay over C ({tag}); solid=rollback, dashed=commit"); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{O}/kl_by_position.png", dpi=130)
