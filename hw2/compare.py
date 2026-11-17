# summary figure for the report: baseline training curves + FID/IS over training for all runs
# (reads history.json from train.py and metrics.json from evaluate.py)
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {"baseline": "results", "minimax G loss": "results/minimax", "label smoothing 0.9": "results/smooth_0.9"}


def smooth(y, k=200):
    return np.convolve(y, np.ones(k) / k, mode="valid")


fig, axes = plt.subplots(1, 4, figsize=(11, 2.5))

# baseline losses and D outputs, moving average only (the raw curves are in results/loss_curves.png)
h = json.load(open("results/history.json"))
ep = np.arange(len(h["loss_d"])) / (len(h["loss_d"]) / 30)
for ax, keys in [(axes[0], [("loss_d", "D loss"), ("loss_g", "G loss")]),
                 (axes[1], [("d_real", "D(x)"), ("d_fake", "D(G(z))")])]:
    # purple/brown so they don't look like the run colors in the other two panels
    for (key, label), c in zip(keys, ["C4", "C5"]):
        y = smooth(h[key])
        ax.plot(ep[len(ep) - len(y):], y, color=c, label=label)
    ax.legend(fontsize=8)
axes[0].set_title("Baseline losses")
axes[0].set_ylabel("BCE loss (nats)")
axes[1].set_title("Baseline D outputs")
axes[1].set_ylabel("probability of real")
axes[1].set_ylim(0, 1)

print(f"{'':22s} {'FID':>8s} {'FID pool':>9s} {'IS':>6s}")
for name, d in runs.items():
    m = json.load(open(f"{d}/metrics.json"))
    epochs = sorted(int(k.split("_")[1]) for k in m if k.startswith("epoch_"))
    axes[2].plot(epochs, [m[f"epoch_{e}"]["fid_pool"] for e in epochs], "o-", ms=3, label=name)
    axes[3].plot(epochs, [m[f"epoch_{e}"]["is"] for e in epochs], "o-", ms=3, label=name)
    last = m[f"epoch_{epochs[-1]}"]
    print(f"{name + ' ep' + str(epochs[-1]):22s} {last['fid']:8.4f} {last['fid_pool']:9.2f} {last['is']:6.3f}")

# the two reference sets are the same in every run, take them from the baseline
ref = json.load(open("results/metrics.json"))
for name in ["real", "noise"]:
    r = ref[name]
    print(f"{name + ' vs real':22s} {r['fid']:8.4f} {r['fid_pool']:9.2f} {r['is']:6.3f}")
# final scores over 3 training seeds (seed 0 is the run in results/ and results/smooth_0.9/)
print("\nmean +- std over seeds 0, 1, 2 (epoch 30)")
for name, dirs in [("baseline", ["results", "results/seed1", "results/seed2"]),
                   ("label smoothing 0.9", ["results/smooth_0.9", "results/smooth_0.9_seed1",
                                            "results/smooth_0.9_seed2"])]:
    last = [json.load(open(f"{d}/metrics.json"))["epoch_30"] for d in dirs]
    print(f"{name:22s}", "  ".join(f"{k} {np.mean([m[k] for m in last]):.4g} +- {np.std([m[k] for m in last], ddof=1):.2g}"
                                   for k in ["fid", "fid_pool", "is"]))

axes[2].axhline(ref["real"]["fid_pool"], color="gray", ls="--", lw=1, label="real paintings")
axes[3].axhline(ref["real"]["is"], color="gray", ls="--", lw=1, label="real paintings")
axes[2].set_title("FID (pool features)")
axes[2].set_ylabel("FID (lower is better)")
axes[3].set_title("Inception score")
axes[3].set_ylabel("IS (higher is better)")

for ax in axes:
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
# one legend under the two metric panels, they share the same runs
fig.legend(*axes[2].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.74, 0),
           ncol=4, fontsize=8, frameon=False)
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.savefig("results/summary.pdf")
plt.savefig("results/summary.png", dpi=150)
