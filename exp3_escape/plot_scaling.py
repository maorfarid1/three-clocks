"""Exp. 3e: how loop escape and the coherent window change with model size (Pythia 70M to 1B)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.optimize import curve_fit

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
SIZES = [("70m", 70e6), ("160m", 160e6), ("410m", 410e6), ("1b", 1.0e9)]
_vir = plt.get_cmap("viridis")
cmap = lambda f: _vir(0.85 * f)


def t_half(temps, surv):
    for i in range(len(temps) - 1):
        if surv[i] >= 0.5 > surv[i + 1]:
            return float(np.exp(np.log(temps[i]) + (surv[i] - 0.5) / (surv[i] - surv[i + 1]) * (np.log(temps[i + 1]) - np.log(temps[i]))))
    return None


def sat(n, D, a, tau):
    return D - (D - a) * np.exp(-(n - 2) / tau)


def aicc(y, yhat, p):
    n = len(y); rss = float(np.sum((y - yhat) ** 2)) + 1e-12
    return n * np.log(rss / n) + 2 * p + 2 * p * (p + 1) / max(1, n - p - 1)


summary = {}
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.3))
for si, (tag, npar) in enumerate(SIZES):
    f = f"results/exp3_pythia{tag}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); temps = np.array(d["temps"]); L = d["loops"]; b = d["args"]["burn_cycles"]
    col = cmap(si / (len(SIZES) - 1))
    C = max(len(l["cyc_gap"]) for l in L)
    G = np.full((len(L), C), np.nan)
    for i, l in enumerate(L):
        G[i, :len(l["cyc_gap"])] = l["cyc_gap"]
    cnt = np.sum(~np.isnan(G), 0); ok = cnt >= 10
    med = np.nanmedian(G, 0)
    c = np.arange(1, C + 1)
    ax[0].plot(c[ok], med[ok], "-", color=col, lw=1.3, label=f"Pythia-{tag.upper()}")
    # growth laws
    best = {"log": 0, "linear": 0, "saturating": 0}; Ds = []
    for l in L:
        g = np.array(l["cyc_gap"]); n = np.arange(2, len(g) + 1); y = g[1:]
        if len(n) < 5:
            continue
        A = {"log": aicc(y, np.polyval(np.polyfit(np.log(n), y, 1), np.log(n)), 2),
             "linear": aicc(y, np.polyval(np.polyfit(n, y, 1), n), 2)}
        try:
            ps, _ = curve_fit(sat, n, y, p0=[y.max(), y[0], 3.0], bounds=([-5, -5, 0.1], [50, 50, 1e3]), maxfev=20000)
            A["saturating"] = aicc(y, sat(n, *ps), 3); Ds.append(ps[0])
        except Exception:
            pass
        best[min(A, key=A.get)] += 1
    # horizon-dependent locking temperature
    Ns = [1, 4, 16, 64, 256, 1024]; hor = []
    for N in Ns:
        v = []
        for l in L:
            R = np.array(l["cyc_escape"])[b:]
            if len(R) < 3:
                continue
            RR = np.vstack([R[:N]] + [R[-1:]] * max(0, N - len(R)))
            th = t_half(temps, np.prod(1 - RR, 0))
            if th:
                v.append(th)
        hor.append(float(np.median(v)) if v else None)
    ax[2].semilogx(Ns, hor, "o-", ms=2.5, color=col, lw=1)
    def med_at(cc):
        v = G[:, cc - 1]; v = v[~np.isnan(v)]; return float(np.median(v)) if len(v) >= 10 else None
    surv = np.array([l["surv_end"] for l in L])
    summary[tag] = dict(params=npar, n_loops=len(L), n_prompts=d["args"]["n_prompts"],
                        barrier_c2=med_at(2), barrier_c6=med_at(6), barrier_c10=med_at(10),
                        grow_c2_c6=int(sum(1 for l in L if len(l["cyc_gap"]) >= 6 and l["cyc_gap"][5] > l["cyc_gap"][1])),
                        n_c6=int(sum(1 for l in L if len(l["cyc_gap"]) >= 6)),
                        best_law=best, sat_D_median=float(np.median(Ds)) if Ds else None,
                        T_half_horizon=dict(zip(map(str, Ns), hor)),
                        T_half_stored=t_half(temps, np.median(surv, 0)),
                        periods_median=float(np.median([l["k"] for l in L])))
x = [summary[t]["params"] for t, _ in SIZES if t in summary]
for N, mk in (("16", "o-"), ("256", "s--")):
    ax[1].semilogx(x, [summary[t]["T_half_horizon"][N] for t, _ in SIZES if t in summary], mk, ms=3, color="k" if N == "16" else "0.45", lw=1, label=f"$N={N}$ cycles")
ax1b = ax[1].twinx(); ax1b.spines["top"].set_visible(False)
ax1b.semilogx(x, [summary[t]["sat_D_median"] for t, _ in SIZES if t in summary], "^:", ms=3, color="#c25e14", lw=1)
ax1b.set_ylabel(r"saturated barrier", color="#c25e14", fontsize=7, labelpad=1); ax1b.tick_params(axis="y", colors="#c25e14", labelsize=7)
ax[1].set_ylabel("locking temperature $T_{1/2}(N)$")
ax[0].set_xlabel("cycle since loop onset"); ax[0].set_ylabel(r"barrier $\Delta_c$ (median)"); ax[0].set_title("(a) barrier growth", loc="left", fontsize=8)
ax[0].legend(fontsize=6, frameon=False, loc="lower right")
ax[1].set_xlabel("parameters"); ax[1].set_title("(b) deeper wells with size", loc="left", fontsize=8); ax[1].legend(fontsize=6, frameon=False, loc="lower right")
ax[2].set_xlabel("horizon $N$ (cycles)"); ax[2].set_ylabel("$T_{1/2}(N)$"); ax[2].set_title("(c) horizon law per size", loc="left", fontsize=8)
# (d) coherent fraction vs T for each size with a regime map
H = json.load(open("results/exp3b_human.json"))
TAU = float(np.percentile(H["ref_nll"], 90)); REP = float(np.percentile(H["rep4"], 95))
for si, (tag, npar) in enumerate(SIZES):
    f = f"results/exp3b_pythia{tag}.json"
    if not os.path.exists(f):
        continue
    rows = json.load(open(f))["rows"]
    if any(r.get("ref_nll") is None for r in rows):
        continue
    T = sorted({r["T"] for r in rows}); coh = []
    for t in T:
        R = [r for r in rows if r["T"] == t]
        coh.append(np.mean([r["loop"] == 0 and r["rep4"] <= REP and r["ref_nll"] <= TAU for r in R]))
    ax[3].semilogx(T, coh, "o-", ms=2.5, lw=1, color=cmap(si / (len(SIZES) - 1)), label=f"{tag.upper()}")
    i = int(np.argmax(coh))
    summary.setdefault(tag, {})["window"] = dict(T=T, coherent=[float(v) for v in coh], T_best=T[i], coh_best=float(coh[i]),
                                                 width_above_half=[t for t, v in zip(T, coh) if v >= 0.5])
ax[3].set_xlabel("temperature $T$"); ax[3].set_ylabel("coherent fraction"); ax[3].set_title("(d) coherent text", loc="left", fontsize=8)
ax[3].set_xticks([0.3, 0.5, 1]); ax[3].set_xticklabels(["0.3", "0.5", "1"]); ax[3].legend(fontsize=6, frameon=False, loc="upper left")
for a_ in ax:
    a_.xaxis.set_minor_formatter(NullFormatter())
fig.tight_layout(w_pad=1.2)
fig.savefig("paper/figs/exp3e_scaling.pdf", dpi=300); fig.savefig("paper/figs/exp3e_scaling.png", dpi=200)
json.dump(summary, open("results/exp3e_scaling_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1)[:4000])
