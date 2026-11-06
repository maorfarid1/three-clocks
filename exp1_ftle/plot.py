"""Figure + summary numbers for Exp. 1 (FTLE of the residual stream across training)."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/exp1.json"
out = sys.argv[2] if len(sys.argv) > 2 else "paper/figs/exp1_ftle.pdf"
runs = json.load(open(src))["runs"]
models = []
for r in runs:
    if r["model"] not in models:
        models.append(r["model"])
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axx = plt.subplots(1, len(models) + 4, figsize=(7.2, 2.35),
                        gridspec_kw=dict(width_ratios=[1] * len(models) + [0.06, 0.32, 1.05, 1.05]))
cax = axx[len(models)]
axx[len(models) + 1].axis("off")
ax = [a for i, a in enumerate(axx) if i not in (len(models), len(models) + 1)]
allv = np.concatenate([np.array(r["lam_rel"]) for r in runs])
vmax = float(np.max(allv))
summary = {}
cols = ["#d62728", "#1f77b4", "#2ca02c"]
for mi, m in enumerate(models):
    rs = sorted([r for r in runs if r["model"] == m], key=lambda r: r["step"])
    steps = [r["step"] for r in rs]
    M = np.array([r["lam_rel"] for r in rs]).T  # layers x checkpoints
    im = ax[mi].imshow(M, aspect="auto", origin="lower", cmap="magma_r", vmin=0, vmax=vmax)
    ax[mi].set_xticks(range(len(steps)))
    ax[mi].set_xticklabels([("0" if s == 0 else (f"{s//1000}k" if s >= 1000 else str(s))) for s in steps],
                           rotation=90, fontsize=6)
    ax[mi].set_yticks(range(M.shape[0]))
    ax[mi].set_yticklabels([str(i + 1) for i in range(M.shape[0])], fontsize=6)
    ax[mi].set_xlabel("training step")
    ax[mi].set_ylabel("layer")
    ax[mi].set_title(f"({'ab'[mi]}) {m.split('/')[-1]}", fontsize=8, loc="left")
    ax[mi].spines[:].set_visible(False)
    st = np.array([r["stack"] for r in rs])
    sd = np.array([r["stack_sd"] for r in rs])
    x = np.array([max(s, 1) for s in steps])
    ax[-2].errorbar(x, st, yerr=sd, fmt="-o", ms=2.5, lw=1, capsize=1.5, color=cols[mi], label=m.split("/")[-1])
    raw = np.array([r["lam"] for r in rs])  # checkpoints x layers
    L = raw.shape[1]
    kappa = st * L / raw.sum(1)
    ax[-1].semilogx(x, kappa, "-o", ms=2.5, lw=1, color=cols[mi], label=m.split("/")[-1])
    summary[m] = dict(steps=steps, stack=st.round(3).tolist(), stack_sd=sd.round(3).tolist(),
                      lam_rel=M.round(3).tolist(), lam=np.array([r["lam"] for r in rs]).T.round(3).tolist(),
                      chain_err=max(r["chain_err"] for r in rs), kappa=kappa.round(3).tolist(),
                      mean_lam_rel=M.mean(0).round(3).tolist())
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$\tilde\lambda_\ell$", fontsize=8)
cb.ax.tick_params(labelsize=6)
ax[-2].set_xscale("log")
ax[-2].set_xlabel("training step (0 at 1)")
ax[-2].set_ylabel(r"$\Lambda$")
ax[-2].set_title(f"({'abcd'[len(models)]}) whole stack", fontsize=8, loc="left")
ax[-2].legend(frameon=False, fontsize=6)
ax[-1].set_xlabel("training step (0 at 1)")
ax[-1].set_ylabel(r"$\kappa$")
ax[-1].set_ylim(0, 1)
ax[-1].set_title(f"({'abcde'[len(models)+1]}) coherence $\\kappa$", fontsize=8, loc="left")
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.27, top=0.88, wspace=0.45)
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300)
fig.savefig(out.replace(".pdf", ".png"), dpi=200)
json.dump(summary, open("results/exp1_summary.json", "w"), indent=1)
for m, s in summary.items():
    print(m, "steps", s["steps"], "stack", s["stack"], "chain_err", s["chain_err"])
