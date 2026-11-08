"""Figure: the reduced-order model of loop escape (a particle in a well that deepens and saturates) against the full computation."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("results/reduced.json"))
g = json.load(open("results/exp3_gpt2.json"))
C = {"gpt2": "#c25e14", "pythia160m": "#2457c5"}; LAB = {"gpt2": "GPT-2", "pythia160m": "Pythia-160M"}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.42))
rows = {r["prompt"]: r for r in R["gpt2"]["rows"]}
ex = sorted([r for r in rows.values() if r["ncyc"] >= 15], key=lambda r: r["fit_rmse"])
pick = ex[:3]  # the three loops of at least 15 cycles with the smallest fit error
for r, col in zip(pick, ["#c25e14", "#8a5a2b", "#d9a066"]):
    l = next(l for l in g["loops"] if l["prompt"] == r["prompt"])
    y = np.array(l["cyc_gap"])[:30]; c = np.arange(1, len(y) + 1); cc = np.linspace(1, len(y), 200)
    ax[0].plot(c, y, "o", ms=2.2, color=col)
    ax[0].plot(cc, r["Dinf"] - (r["Dinf"] - r["D1"]) * np.exp(-(cc - 1) / r["tau"]), "-", lw=1, color=col)
ax[0].set_xlabel("cycle $c$"); ax[0].set_ylabel("barrier $\\Delta_c$ (logit gap)")
ax[0].set_title("(a) the well deepens and saturates\n(GPT-2, three loops, fits)", fontsize=7.5, loc="left")
for m in ("gpt2", "pythia160m"):
    x = [r["exact"]["16"] for r in R[m]["rows"] if r["exact"]["16"] and r["reduced"]["16"]]
    y = [r["reduced"]["16"] for r in R[m]["rows"] if r["exact"]["16"] and r["reduced"]["16"]]
    ax[1].plot(x, y, "o", ms=2.5, color=C[m], alpha=0.8, label=f"{LAB[m]} ($\\nu={R[m]['nu']}$)")
lim = [0.2, 1.0]; ax[1].plot(lim, lim, ":", color="0.5", lw=0.8); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel("full model: $T_{1/2}(16)$"); ax[1].set_ylabel("reduced model: $T_{1/2}(16)$")
ax[1].legend(fontsize=6, frameon=False, loc="upper left"); ax[1].set_title("(b) one loop per point", fontsize=7.5, loc="left")
Ns = [1, 4, 16, 64, 256]
for m in ("gpt2", "pythia160m"):
    s = R[m]["summary"]
    ax[2].semilogx(Ns, [s[str(N)]["median_exact"] for N in Ns], "o-", ms=3, lw=1.1, color=C[m], label=f"{LAB[m]}, full")
    ax[2].semilogx(Ns, [s[str(N)]["median_reduced"] for N in Ns], "s--", ms=3, lw=1.0, color=C[m], mfc="white", label=f"{LAB[m]}, reduced")
ax[2].set_xlabel("horizon $N$ (cycles)"); ax[2].set_ylabel("median $T_{1/2}(N)$"); ax[2].legend(fontsize=5.8, frameon=False)
ax[2].set_title("(c) locking temperature vs horizon", fontsize=7.5, loc="left")
fig.savefig("paper/figs/reduced.pdf", dpi=300, bbox_inches="tight"); fig.savefig("paper/figs/reduced.png", dpi=200, bbox_inches="tight")
