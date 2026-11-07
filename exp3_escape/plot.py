"""Figure + summary numbers for Exp. 3 (escape from repetition loops)."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

srcs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["results/exp3_gpt2.json"]
out = sys.argv[2] if len(sys.argv) > 2 else "paper/figs/exp3_escape.pdf"
names = {"gpt2": "GPT-2 (124M)", "EleutherAI/pythia-160m": "Pythia-160M"}
colors = ["#d62728", "#1f77b4"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.15))
summary = {}

for si, src in enumerate(srcs):
    d = json.load(open(src))
    temps = np.array(d["temps"])
    loops = d["loops"]
    name = names.get(d["args"]["model"], d["args"]["model"])
    col = colors[si % 2]
    S = dict(n_prompts=d["args"]["n_prompts"], n_loops=len(loops),
             periods=[L["k"] for L in loops])

    # (a) barrier per cycle, cycles counted from loop onset
    C = max(len(L["cyc_gap"]) for L in loops)
    G = np.full((len(loops), C), np.nan)
    for i, L in enumerate(loops):
        G[i, : len(L["cyc_gap"])] = L["cyc_gap"]
    cnt = np.sum(~np.isnan(G), 0)
    keep = cnt >= 10  # avoid survivorship effects from the few longest loops
    c = np.arange(1, C + 1)[keep]
    med = np.nanmedian(G, 0)[keep]
    q1, q3 = np.nanpercentile(G, 25, 0)[keep], np.nanpercentile(G, 75, 0)[keep]
    ax[0].plot(c, med, "-o", ms=2, color=col, label=name)
    ax[0].fill_between(c, q1, q3, color=col, alpha=0.15, lw=0)
    S["barrier_median_by_cycle"] = dict(zip(c.tolist(), np.round(med, 3).tolist()))
    S["n_loops_by_cycle"] = dict(zip(c.tolist(), cnt[keep].tolist()))
    # fraction of loops whose barrier at cycle 6 exceeds cycle 2 (when both exist)
    pairs = [(L["cyc_gap"][1], L["cyc_gap"][5]) for L in loops if len(L["cyc_gap"]) >= 6]
    S["frac_barrier_grows_c2_to_c6"] = float(np.mean([b > a for a, b in pairs])) if pairs else None
    S["n_pairs_c2_c6"] = len(pairs)

    # (b)+(c) Arrhenius test on the first timed cycle (after burn-in)
    b = d["args"]["burn_cycles"]
    fit_b, meas_b, shown = [], [], []
    for i, L in enumerate(loops):
        if len(L["cyc_gap"]) <= b:
            continue
        r = np.array(L["cyc_escape"])[b]
        dmin = L["cyc_gap"][b]
        m = (r > 1e-9) & (r < 1e-2)
        if m.sum() >= 3:
            sl = np.polyfit(1 / temps[m], np.log(r[m]), 1)[0]
            fit_b.append(-sl)
            meas_b.append(dmin)
        if si == 0 and 1.0 < dmin < 3.0 and len(shown) < 4:
            cc = ["#440154", "#31688e", "#35b779", "#e1a100"][len(shown)]
            ax[1].semilogy(1 / temps, np.clip(r, 1e-14, 1), "-", lw=1.1, color=cc)
            ax[1].semilogy(1 / temps, np.clip(np.exp(-dmin / temps), 1e-14, 1), ":", lw=1.0, color=cc)
            shown.append(dmin)
    fit_b, meas_b = np.array(fit_b), np.array(meas_b)
    ax[2].plot(meas_b, fit_b, "o", ms=2.5, color=col, label=name)
    S["arrhenius_n"] = int(len(fit_b))
    S["arrhenius_rel_err_median"] = float(np.median(np.abs(fit_b - meas_b) / meas_b)) if len(fit_b) else None
    S["arrhenius_corr"] = float(np.corrcoef(meas_b, fit_b)[0, 1]) if len(fit_b) > 2 else None

    # (d) survival of all stored cycles after burn-in vs T, and free-sampling loop fraction
    surv = np.array([L["surv_end"] for L in loops])
    ax[3].semilogx(temps, np.median(surv, 0), "-", color=col, label=f"{name}: stays locked")
    ax[3].fill_between(temps, np.percentile(surv, 25, 0), np.percentile(surv, 75, 0), color=col, alpha=0.15, lw=0)
    fr = d["free"]
    ft = np.array(sorted(float(k) for k in fr))
    ff = np.array([fr[k]["frac_loop"] for k in sorted(fr, key=float)])
    ax[3].semilogx(ft, ff, "s--", ms=2.5, lw=0.8, color=col, alpha=0.8, label=f"{name}: free runs loop")
    S["surv_median"] = dict(zip(np.round(temps, 3).tolist(), np.round(np.median(surv, 0), 3).tolist()))
    S["free_frac_loop"] = dict(zip(ft.tolist(), ff.tolist()))
    S["mc"] = d["mc"]
    summary[name] = S

ax[0].set_xlabel("cycle since loop onset")
ax[0].set_ylabel(r"barrier $\Delta_c$ (min logit gap)")
ax[0].set_title("(a) the well deepens", fontsize=8, loc="left")
ax[0].legend(frameon=False, fontsize=6)
ax[1].set_xlabel(r"$1/T$")
ax[1].set_ylabel(r"escape prob. per cycle $r$")
ax[1].set_ylim(1e-8, 2)
ax[1].text(0.03, 0.03, "solid: exact $r(T)$\ndotted: $e^{-\\Delta_{\\min}/T}$", transform=ax[1].transAxes,
           ha="left", va="bottom", fontsize=6)
ax[1].set_title("(b) Arrhenius plot", fontsize=8, loc="left")
lim = [0, max(ax[2].get_xlim()[1], ax[2].get_ylim()[1])]
ax[2].plot(lim, lim, "k-", lw=0.6)
ax[2].set_xlabel(r"measured barrier $\Delta_{\min}$")
ax[2].set_ylabel(r"fitted slope of $-\log r$ vs $1/T$")
ax[2].set_title("(c) barrier from the fit", fontsize=8, loc="left")
ax[3].set_xlabel("temperature $T$")
ax[3].set_ylabel("probability")
ax[3].set_title("(d) locking vs temperature", fontsize=8, loc="left")
ax[3].legend(frameon=False, fontsize=5.5, loc="lower left")
ax[3].set_xticks([0.2, 0.5, 1, 2])
ax[3].set_xticklabels(["0.2", "0.5", "1", "2"])
ax[3].minorticks_off()
fig.tight_layout()
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300)
fig.savefig(out.replace(".pdf", ".png"), dpi=200)
json.dump(summary, open("results/exp3_summary.json", "w"), indent=1)
print(json.dumps({k: {kk: v[kk] for kk in v if kk not in ("surv_median", "mc", "periods")} for k, v in summary.items()}, indent=1))
