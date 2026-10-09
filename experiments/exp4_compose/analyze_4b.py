"""Exp 4b extra analysis: training curves, depth-1 two-candidate diagnostic, old-vs-aligned comparison.
Usage: python analyze_4b.py <results_dir>   (expects <dir>/aligned_s{0,1,2}/ and <dir>/old_s0/ each with train_log.json + items_seed*.csv.gz)"""
import csv, glob, gzip, json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aggregate_lib import wilson  # noqa

R = Path(sys.argv[1])
runs = sorted(p.name for p in R.iterdir() if p.is_dir() and (p / "train_log.json").exists())


def items(run):
    out = []
    for f in glob.glob(str(R / run / "items_seed*.csv.gz")):
        with gzip.open(f, "rt") as fh:
            out += [{k: (v if k == "group" else int(v)) for k, v in r.items()} for r in csv.DictReader(fh)]
    return out


res = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for run in runs:
    log = [r for r in json.load(open(R / run / "train_log.json")) if "acc_by_depth" in r]
    steps = [r["step"] for r in log]
    d1 = [r["acc_by_depth"].get("1") for r in log]
    ls = "--" if run.startswith("old") else "-"
    axes[0].plot(steps, d1, ls, label=run, lw=1)
    hard = []
    for r in log:
        h = max(int(k) for k in r["acc_by_depth"])
        hard.append(r["acc_by_depth"][str(h)] if True else None)
    axes[1].plot(steps, [max(r["acc_by_depth"].items(), key=lambda kv: int(kv[0]))[1] for r in log], ls, label=run, lw=1)
    unl = log[-1]["unlock_steps"] if log else {}
    res[run] = {"steps_run": steps[-1] if steps else 0, "final_unlocked": log[-1]["unlocked_depths"] if log else [], "unlock_steps": unl,
                "train_acc_d1_last1000steps_mean": (lambda v: sum(v) / len(v) if v else None)([x for x in d1[-10:] if x is not None]),
                "elapsed_s": log[-1]["elapsed_s"] if log else None,
                "max_train_acc_d1_100step_window": max(x for x in d1 if x is not None)}
    for d, s in unl.items():
        if int(d) > 1:
            axes[0].axvline(s, color="0.8", lw=0.5)
axes[0].axhline(0.5, color="r", ls=":", lw=0.8); axes[0].axhline(0.95, color="g", ls=":", lw=0.8)
axes[0].set_title("training accuracy on depth-1 queries (100-step windows)"); axes[0].set_xlabel("step"); axes[0].legend(fontsize=7)
axes[1].set_title("training accuracy at the hardest unlocked depth"); axes[1].set_xlabel("step")
fig.tight_layout(); fig.savefig(R / "training_curves.png", dpi=130)

for run in runs:
    it = [r for r in items(run)]
    if not it:
        continue
    d1 = [r for r in it if r["depth"] == 1]
    n = len(d1)
    cand = sum(r["pred_fw_reset"] in (r["pred_one_hop"], r["pred_other_gen"]) for r in d1)
    ok = sum(r["pred_fw_reset"] == r["answer"] for r in d1)
    wrong = sum(r["pred_fw_reset"] == r["pred_other_gen"] and r["pred_other_gen"] != r["answer"] for r in d1)
    res[run]["depth1_eval"] = {"n": n, "exact": ok / n, "wilson95": wilson(ok, n), "pred_in_two_successors": cand / n, "pred_is_other_generator_successor": wrong / n,
                               "ambiguous_two_succ_equal": sum(r["pred_one_hop"] == r["pred_other_gen"] for r in d1) / n}
json.dump(res, open(R / "analysis_4b.json", "w"), indent=1)
print(json.dumps(res, indent=1))
