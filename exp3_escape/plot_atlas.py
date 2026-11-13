"""Figure: regime atlas, a noisy Duffing oscillator and GPT-2 in the same four representations."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
d = json.load(open("results/atlas.json"))
cols = [("duffing", r, f"Duffing, $D={2*r['sigma']**2:.3g}$\n" + lab) for r, lab in zip(d["duffing"], ["periodic", "noisy, hopping", "noise-dominated"])] + \
       [("lm", r, f"GPT-2, $T={r['T']:g}$\n" + lab) for r, lab in zip(d["lm"], ["loop", "coherent text", "incoherent text"])]
C = ["#2457c5", "#2457c5", "#2457c5", "#c25e14", "#c25e14", "#c25e14"]
fig, ax = plt.subplots(4, 6, figsize=(7.3, 5.4), gridspec_kw=dict(hspace=0.5, wspace=0.18))
for j, (kind, r, title) in enumerate(cols):
    y = np.array(r["x"] if kind == "duffing" else np.array(r["pc"])[:, 0])[:160]
    ax[0, j].plot(y, lw=0.6, color=C[j]); ax[0, j].set_title(title, fontsize=6.5)
    ax[0, j].set_xlabel("time" if kind == "duffing" else "token", labelpad=1); ax[0, j].set_yticks([])
    f, P = np.array(r["f"])[1:], np.array(r["P"])[1:]
    ax[1, j].semilogy(f, P, lw=0.7, color=C[j]); ax[1, j].set_xlabel("frequency", labelpad=1); ax[1, j].set_yticks([]); ax[1, j].minorticks_off()
    ax[1, j].text(0.97, 0.95, f"$\\mathcal{{H}}_s$={r['Hspec']:.2f}", transform=ax[1, j].transAxes, ha="right", va="top", fontsize=6)
    if kind == "duffing":
        X, V = np.array(r["x"]), np.array(r["v"]); ax[2, j].plot(X, V, lw=0.3, color=C[j]); ax[2, j].set_xlabel("$q$", labelpad=1)
    else:
        Z = np.array(r["pc"]); ax[2, j].plot(Z[:, 0], Z[:, 1], lw=0.3, color=C[j]); ax[2, j].plot(Z[:, 0], Z[:, 1], ".", ms=1.2, color=C[j])
        ax[2, j].set_xlabel("PC 1 (PC 2 vertical)", labelpad=1)
    ax[2, j].set_xticks([]); ax[2, j].set_yticks([])
    n = r["n"]; R = np.unpackbits(np.array(r["R"], dtype=np.uint8))[: n * n].reshape(n, n)[:300, :300]
    ax[3, j].imshow(R, cmap="Greys", origin="lower", interpolation="nearest"); ax[3, j].set_xticks([]); ax[3, j].set_yticks([])
    ax[3, j].set_xlabel(f"DET={r['DET_fixedRR']:.2f}", labelpad=1)
for i, lab in enumerate(["(a) time history", "(b) power spectrum\n(log scale)", "(c) phase portrait", "(d) recurrence plot"]):
    ax[i, 0].set_ylabel(lab, fontsize=7.5, fontweight="bold")
ax[2, 0].set_xlabel("$q$ ($\\dot q$ vertical)", labelpad=1)
fig.savefig("paper/figs/atlas.pdf", dpi=300, bbox_inches="tight"); fig.savefig("paper/figs/atlas.png", dpi=200, bbox_inches="tight")
