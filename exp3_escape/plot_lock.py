"""Figure for Exp. 3d: growth law of the barrier and the locking temperature."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
D = json.load(open("results/exp3d_lock.json"))
MODELS = [("gpt2", "GPT-2 (124M)", "#d62728"), ("EleutherAI/pythia-160m", "Pythia-160M", "#1f77b4")]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.25))

# (a) phase diagram: T against the logarithmic growth rate c
c = np.linspace(0.05, 3.2, 300)
ax[0].fill_between(c, 0, c, color="#fde0dd", lw=0)
ax[0].fill_between(c, c, 3.2, color="#deebf7", lw=0)
ax[0].plot(c, c, "k-", lw=1.0)
ax[0].text(2.75, 1.95, "locked\n($T<\\gamma$)", fontsize=6.5, ha="center"); ax[0].text(1.55, 2.95, "escapes a.s. ($T\\geq \\gamma$)", fontsize=6.5, ha="center")
a0 = np.median([D[m]["summary"]["a_median"] for m, _, _ in MODELS])
Tg = np.linspace(0.05, 3.2, 600)
for N, ls in [(10, ":"), (1000, "--")]:
    n = np.arange(3, 3 + N)
    bd = []
    for cc in c:  # smallest T at which the m=1 Arrhenius survival over N cycles drops below 1/2
        hz = np.exp(-(a0 + cc * np.log(n))[:, None] / Tg[None, :]).sum(0)
        i = np.argmax(hz > np.log(2)); bd.append(Tg[i] if hz[i] > np.log(2) else np.nan)
    ax[0].plot(c, bd, ls, color="0.35", lw=0.9, label=f"survival $1/2$, $N={N}$")
for m, name, col in MODELS:
    L = [l for l in D[m]["loops"] if l["T_half_exact"]]
    ax[0].plot([l["c"] for l in L], [l["T_half_exact"] for l in L], "o", ms=2.3, color=col, label=name)
ax[0].set_xlim(0, 3.2); ax[0].set_ylim(0, 3.2)
ax[0].set_xlabel("growth rate $\\gamma$ in $\\Delta_c=a+\\gamma\\ln c$"); ax[0].set_ylabel("temperature $T$")
ax[0].set_title("(a) locking phase diagram", loc="left", fontsize=8)
ax[0].legend(fontsize=5.8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))

# (b) best growth law by AICc
laws = ["log", "linear", "saturating"]
w = 0.38
for i, (m, name, col) in enumerate(MODELS):
    bc = D[m]["summary"]["best_counts"]
    ax[1].bar(np.arange(3) + (i - 0.5) * w, [bc[k] for k in laws], w, color=col, label=name)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["$a+\\gamma\\ln c$", "$a+bc$", "saturating"])
ax[1].set_ylabel("loops where law fits best")
ax[1].set_title("(b) barrier growth law (AICc)", loc="left", fontsize=8)
ax[1].legend(fontsize=6, frameon=False)

# (c) horizon dependence of the locking temperature
for m, name, col in MODELS:
    h = D[m]["summary"]["horizon"]
    ax[2].semilogx(h["N"], h["T_half_median"], "o", ms=3, color=col, label=name)
    ax[2].semilogx(h["N"], h["fit_pred"], "-", lw=0.9, color=col)
ax[2].set_xlabel("horizon $N$ (cycles after burn-in)"); ax[2].set_ylabel("$T$ at which survival $=1/2$")
ax[2].set_title("(c) locking temperature vs horizon", loc="left", fontsize=8)
ax[2].legend(fontsize=6, frameon=False)
fig.tight_layout(w_pad=0.7)
fig.savefig("paper/figs/exp3d_lock.pdf", dpi=300); fig.savefig("paper/figs/exp3d_lock.png", dpi=200)
