"""Figure: determinism against barrier / noise for the noisy Duffing oscillator and for GPT-2 (collapse.py)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
d = json.load(open("results/collapse.json")); g = json.load(open("results/exp3_gpt2.json"))
gap = {l["prompt"]: (float(np.median(l["cyc_gap"][-3:])), l["k"]) for l in g["loops"]}  # measured barrier: last three cycles
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.3), gridspec_kw=dict(wspace=0.35))
S = sorted({r["sigma"] for r in d["duffing"]})
x = [d["duffing"][[r["sigma"] for r in d["duffing"]].index(s)]["ratio"] for s in S]
med = [np.median([r["DET"] for r in d["duffing"] if r["sigma"] == s]) for s in S]
lo = [np.min([r["DET"] for r in d["duffing"] if r["sigma"] == s]) for s in S]; hi = [np.max([r["DET"] for r in d["duffing"] if r["sigma"] == s]) for s in S]
ax[0].semilogx(x, med, "o-", ms=3, color="#2457c5"); ax[0].fill_between(x, lo, hi, color="#2457c5", alpha=0.15, lw=0)
nper = d["args"]["gen"] * 1.2 / (2 * np.pi)
ax[0].axvline(np.log(nper / np.log(2)), color="0.5", ls=":", lw=0.9)
ax[0].set_xlabel("barrier / noise, $\\Delta U/D$"); ax[0].set_ylabel("determinism DET")
ax[0].set_title("(a) Duffing oscillator, 5 seeds per point", fontsize=7.5, loc="left")
pts = [(gap[r["prompt"]][0] / r["T"], r["DET"], r["prompt"]) for r in d["lm"]]
P = sorted({p for _, _, p in pts})
for p in P:
    q = sorted([(a, b) for a, b, pp in pts if pp == p])
    ax[1].semilogx(*zip(*q), "-", lw=0.6, color="#c25e14", alpha=0.35)
xs = np.array([a for a, _, _ in pts]); ys = np.array([b for _, b, _ in pts])
bins = np.geomspace(xs.min(), xs.max() * 1.001, 7); mid, bm = [], []
for b0, b1 in zip(bins[:-1], bins[1:]):
    m = (xs >= b0) & (xs < b1)
    if m.sum() >= 3:
        mid.append(np.sqrt(b0 * b1)); bm.append(np.median(ys[m]))
ax[1].semilogx(mid, bm, "o-", ms=3.5, color="#c25e14", lw=1.4)
win = d["args"]["gen"] - d["args"]["skip"]; kmed = np.median([gap[p][1] for p in P])
ax[1].axvline(np.log(3 * win / kmed / np.log(2)), color="0.5", ls=":", lw=0.9)
ax[1].set_xlabel("barrier / noise, $\\Delta/T$"); ax[1].set_ylabel("determinism DET")
ax[1].set_title(f"(b) GPT-2, {len(P)} prompts (lines) and binned median", fontsize=7.5, loc="left")
from matplotlib.ticker import NullFormatter
ax[1].xaxis.set_minor_formatter(NullFormatter()); ax[1].set_xticks([2, 3, 5, 10]); ax[1].set_xticklabels(["2", "3", "5", "10"])
for a_ in ax:
    a_.text(0.03, 0.97, "dotted: one escape\nexpected in the window", transform=a_.transAxes, ha="left", va="top", fontsize=6, color="0.4")
fig.savefig("paper/figs/collapse.pdf", dpi=300, bbox_inches="tight"); fig.savefig("paper/figs/collapse.png", dpi=200, bbox_inches="tight")
xd = np.array(x); print("duffing", list(zip(np.round(xd, 2), np.round(med, 3))))
print("lm bins", list(zip(np.round(mid, 2), np.round(bm, 3))), "threshold lm", np.log(3 * win / kmed / np.log(2)), "duffing", np.log(nper / np.log(2)))
corr = np.corrcoef(np.log(xs), ys)[0, 1]; print("lm corr(log ratio, DET)", corr, "n", len(xs))
