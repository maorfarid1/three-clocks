"""Figure for Exp. 2: bifurcation diagram + sharpness vs 2/eta."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

srcs = (sys.argv[1] if len(sys.argv) > 1 else "results/exp2.json").split(",")
src = srcs[0]
out = sys.argv[2] if len(sys.argv) > 2 else "paper/figs/exp2_bifurcation.pdf"
d = json.load(open(src))
runs = d["runs"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))

def detrend(t, w=9):
    """Relative fluctuation of the loss around its running mean (odd window)."""
    t = np.asarray(t)
    k = np.ones(w) / w
    m = np.convolve(t, k, mode="same")
    h = w // 2
    return ((t - m) / m)[h:-h]


def lag1(x):
    x = x - x.mean()
    return float(np.dot(x[1:], x[:-1]) / np.dot(x, x))


# (a) bifurcation diagram: relative loss fluctuations around the running mean
for r in runs:
    if r["diverged"]:
        continue
    f = detrend(r["tail"])
    ax[0].plot(np.full_like(f, r["eta"]), f, ".", ms=0.6, color="k", alpha=0.5, rasterized=True)
div = [r["eta"] for r in runs if r["diverged"]]
if div:
    ax[0].axvspan(min(div), max(div) * 1.05, color="#d62728", alpha=0.12, lw=0)
    ax[0].text(min(div) * 1.05, 0.98, "diverges", transform=ax[0].get_xaxis_transform(),
               color="#d62728", va="top", fontsize=7)
ax[0].set_xscale("log")
ax[0].set_yscale("symlog", linthresh=1e-4)
ax[0].set_xlabel(r"step size $\eta$")
ax[0].set_ylabel(r"$(\mathcal{L}_t - \bar{\mathcal{L}}_t)/\bar{\mathcal{L}}_t$, last %d steps" % d["args"]["tail"])
ax[0].set_title("(a) bifurcation diagram", fontsize=8, loc="left")

# (b) sharpness vs 2/eta, all seeds
seedcols = ["#1f77b4", "#ff7f0e", "#9467bd"]
per_seed = []
for k, sp in enumerate(srcs):
    dd = json.load(open(sp))
    okk = [r for r in dd["runs"] if not r["diverged"]]
    e_k = np.array([r["eta"] for r in okk]); s_k = np.array([r["sharp"] for r in okk])
    ax[1].loglog(e_k, s_k, "o", ms=2.2, color=seedcols[k % 3], alpha=0.85,
                 label=(r"$\lambda_{\max}$, seed %d" % dd["args"]["seed"]))
    per_seed.append((dd, okk, e_k, s_k))
ok = per_seed[0][1]
eta = per_seed[0][2]
sh = per_seed[0][3]
allE = np.concatenate([p[2] for p in per_seed])
ax[1].loglog(np.sort(allE), 2 / np.sort(allE), "-", color="k", lw=0.8, label=r"$2/\eta$")
ax[1].set_xlabel(r"step size $\eta$")
ax[1].set_ylabel("sharpness")
ax[1].legend(frameon=False, fontsize=6)
ax[1].set_title("(b) edge of stability", fontsize=8, loc="left")

# (c) lag-1 autocorrelation of the fluctuations: -1 = clean period-2 orbit, all seeds
summ = []
for k, (dd, okk, e_k, s_k) in enumerate(per_seed):
    ac_k = np.array([lag1(detrend(r["tail"])) for r in okk])
    ax[2].semilogx(e_k, ac_k, "o", ms=2.2, color=seedcols[k % 3], alpha=0.85)
    divs = [r["eta"] for r in dd["runs"] if r["diverged"]]
    summ.append(dict(seed=dd["args"]["seed"], eta=e_k.tolist(), lag1=ac_k.tolist(), sharp=s_k.tolist(),
                     ratio=(s_k * e_k / 2).tolist(), max_trainable=float(e_k.max()),
                     first_diverging=float(min(divs)) if divs else None))
ac = np.array(summ[0]["lag1"])
ax[2].axhline(-1, color="k", lw=0.6, ls=":")
ax[2].axhline(0, color="k", lw=0.6, ls=":")
ax[2].set_ylim(-1.08, 1.0)
ax[2].set_xlabel(r"step size $\eta$")
ax[2].set_ylabel("lag-1 autocorrelation")
ax[2].set_title("(c) period-2 signature", fontsize=8, loc="left")
json.dump(summ if len(summ) > 1 else dict(eta=eta.tolist(), lag1=ac.tolist(), sharp=sh.tolist()),
          open(os.path.join(os.path.dirname(src), "exp2_summary_seeds.json" if len(summ) > 1 else "exp2_summary.json"), "w"), indent=1)
fig.tight_layout()
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300)
fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print("saved", out)
