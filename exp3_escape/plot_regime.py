"""Figures + summary numbers for Exp. 3b (regime map) and Exp. 3c (loop orbits)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
MODELS = [("gpt2", "GPT-2 (124M)", "#d62728"), ("pythia160m", "Pythia-160M", "#1f77b4")]
C_LOOP, C_REP, C_INC, C_COH = "#e34a33", "#fdae6b", "#7f0000", "#3182bd"
summary = {}

HAVE = os.path.exists("results/exp3b_human.json") and os.path.exists("results/exp3b_gpt2.json")
hum = json.load(open("results/exp3b_human.json")) if HAVE else {"ref_nll": [3.0], "n": 0, "rep4": [0], "loop": [0]}
H = np.array(hum["ref_nll"])
thr = {q: float(np.percentile(H, q)) for q in (50, 75, 90, 95)}
TAU = thr[90]
REP_T = float(np.percentile(hum["rep4"], 95)) if HAVE else 0.1
summary["human"] = dict(n=hum["n"], ref_nll_median=float(np.median(H)), ref_nll_iqr=[float(np.percentile(H, 25)), float(np.percentile(H, 75))],
                        thresholds=thr, rep4_median=float(np.median(hum["rep4"])), loops=int(sum(hum["loop"])))


def classify(r, tau):
    if r["loop"] > 0:
        return "loop"
    if r["rep4"] > REP_T:
        return "rep"
    return "inc" if r["ref_nll"] > tau else "coh"


def cross(x, y, level, rising):
    """First x where y crosses `level` (linear interpolation in log T)."""
    for i in range(len(x) - 1):
        a, b = y[i], y[i + 1]
        if (rising and a < level <= b) or (not rising and a > level >= b):
            lx = np.log(x[i]) + (level - a) / (b - a) * (np.log(x[i + 1]) - np.log(x[i]))
            return float(np.exp(lx))
    return None


fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2))
for mi, (m, name, col) in enumerate(MODELS):
    f = f"results/exp3b_{m}.json"
    if not HAVE or not os.path.exists(f):
        continue
    rows = json.load(open(f))["rows"]
    T = np.array(sorted({r["T"] for r in rows}))
    frac = {k: [] for k in ("loop", "rep", "inc", "coh")}
    nll_med, nll_q1, nll_q3, ent, rep = [], [], [], [], []
    sens = {q: [] for q in thr}
    for t in T:
        R = [r for r in rows if r["T"] == t]
        cl = [classify(r, TAU) for r in R]
        for k in frac:
            frac[k].append(np.mean([c == k for c in cl]))
        for q in thr:
            sens[q].append(np.mean([classify(r, thr[q]) == "coh" for r in R]))
        x = [r["ref_nll"] for r in R]
        nll_med.append(np.median(x)); nll_q1.append(np.percentile(x, 25)); nll_q3.append(np.percentile(x, 75))
        ent.append(np.median([r["entropy"] for r in R])); rep.append(np.median([r["rep4"] for r in R]))
    frac = {k: np.array(v) for k, v in frac.items()}
    # (a)/(b): stacked regime fractions
    a = ax[mi]
    a.stackplot(T, frac["loop"], frac["rep"], frac["coh"], frac["inc"], colors=[C_LOOP, C_REP, C_COH, C_INC], alpha=0.9,
                labels=["loop", "repetitive", "coherent", "incoherent"])
    a.set_xscale("log"); a.set_xlim(T[0], T[-1]); a.set_ylim(0, 1)
    a.set_xticks([0.2, 0.5, 1, 2]); a.set_xticklabels(["0.2", "0.5", "1", "2"])
    a.set_xlabel("temperature $T$"); a.set_title(f"({'ab'[mi]}) {name}", loc="left", fontsize=8)
    if mi == 0:
        a.set_ylabel("fraction of generations")
        a.legend(loc="center right", fontsize=6.5, frameon=False)
    # (c): reference NLL
    ax[2].semilogx(T, nll_med, "-o", ms=2, color=col, label=name)
    ax[2].fill_between(T, nll_q1, nll_q3, color=col, alpha=0.15, lw=0)
    lo = cross(T, frac["loop"], 0.5, rising=False)
    lo_rep = cross(T, frac["loop"] + frac["rep"], 0.5, rising=False)
    hi = cross(T, frac["inc"], 0.5, rising=True)
    best = int(np.argmax(frac["coh"]))
    summary[m] = dict(T=T.tolist(), frac_loop=frac["loop"].tolist(), frac_rep=frac["rep"].tolist(), T_rep_half=lo_rep, rep_threshold=REP_T, frac_inc=frac["inc"].tolist(), frac_coh=frac["coh"].tolist(),
                      ref_nll_median=nll_med, entropy_median=ent, rep4_median=rep,
                      T_loop_half=lo, T_inc_half=hi, T_best=float(T[best]), coh_best=float(frac["coh"][best]),
                      coh_by_threshold={str(q): [float(v) for v in sens[q]] for q in thr},
                      n_per_T=len([r for r in rows if r["T"] == T[0]]))
    # per-prompt windows
    P = sorted({r["prompt"] for r in rows})
    win = []
    for p in P:
        fl = [np.mean([r["loop"] > 0 for r in rows if r["prompt"] == p and r["T"] == t]) for t in T]
        fn = [np.median([r["ref_nll"] for r in rows if r["prompt"] == p and r["T"] == t]) for t in T]
        fr = [np.median([r["rep4"] for r in rows if r["prompt"] == p and r["T"] == t]) for t in T]
        coh = [(fl[i] < 0.5) and (fr[i] <= REP_T) and (fn[i] <= TAU) for i in range(len(T))]
        idx = [i for i, c in enumerate(coh) if c]
        win.append((p, T[idx[0]] if idx else None, T[idx[-1]] if idx else None, len(idx)))
    summary[m]["prompt_windows"] = [dict(prompt=p, T_lo=float(a_) if a_ else None, T_hi=float(b_) if b_ else None, n_T=n) for p, a_, b_, n in win]
    nT = np.array([w[3] for w in win])
    summary[m]["window_nT"] = dict(median=float(np.median(nT)), min=int(nT.min()), max=int(nT.max()), empty=int((nT == 0).sum()))
    if m == "gpt2":
        W = sorted(win, key=lambda w: (w[1] or 9, w[2] or 9))
        for j, (p, a_, b_, n) in enumerate(W):
            if a_ is not None:
                ax[3].plot([a_, b_], [j, j], "-", color=C_COH, lw=1.6, solid_capstyle="round")
                ax[3].plot([a_, b_], [j, j], "|", color=C_COH, ms=3)
        ax[3].set_xscale("log"); ax[3].set_xlim(T[0], T[-1]); ax[3].set_xticks([0.2, 0.5, 1, 2]); ax[3].set_xticklabels(["0.2", "0.5", "1", "2"])
        ax[3].set_yticks([]); ax[3].set_ylabel("prompt (sorted)")
        ax[3].set_xlabel("temperature $T$"); ax[3].set_title("(d) GPT-2, per prompt", loc="left", fontsize=8)

if HAVE: ax[2].axhspan(*summary["human"]["ref_nll_iqr"], color="0.6", alpha=0.25, lw=0)
ax[2].axhline(TAU, color="0.3", ls="--", lw=0.8)
ax[2].text(0.21, TAU + 0.15, "human 90th pct.", fontsize=6, color="0.3")
ax[2].set_xlabel("temperature $T$"); ax[2].set_ylabel("reference NLL (nats/token)")
ax[2].set_xticks([0.2, 0.5, 1, 2]); ax[2].set_xticklabels(["0.2", "0.5", "1", "2"])
ax[2].set_title("(c) reference NLL", loc="left", fontsize=8)
ax[2].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.88))
for a_ in ax:
    a_.xaxis.set_minor_formatter(NullFormatter())
fig.tight_layout(w_pad=0.6)
if HAVE: fig.savefig("paper/figs/exp3b_regimes.pdf", dpi=300); fig.savefig("paper/figs/exp3b_regimes.png", dpi=200)

# ---------- Exp. 3c: orbits ----------
fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.1))
for m, name, col in MODELS:
    f = f"results/exp3c_{m}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); L = d["loops"]
    C = max(len(l["d"]) for l in L)
    cs = [c for c in range(C) if sum(len(l["d"]) > c for l in L) >= 10]
    med = np.array([np.median([l["d"][c] for l in L if len(l["d"]) > c]) for c in cs])
    q1 = [np.percentile([l["d"][c] for l in L if len(l["d"]) > c], 25) for c in cs]
    q3 = [np.percentile([l["d"][c] for l in L if len(l["d"]) > c], 75) for c in cs]
    c = np.array(cs) + 1
    ax[0].semilogy(c, med, "-o", ms=2, color=col, label=name)
    ax[0].fill_between(c, q1, q3, color=col, alpha=0.15, lw=0)
    ok = [l for l in L if len(l["d"]) >= 6]
    pe = np.polyfit(c, np.log(med), 1); pp = np.polyfit(np.log(c), np.log(med), 1)
    summary[m + "_orbit"] = dict(n_loops=len(L), dec_c2_c6=sum(l["d"][5] < l["d"][1] for l in ok), n_c6=len(ok),
                                 median_d=med.tolist(), cycles=c.tolist(),
                                 exp_rate=float(np.exp(pe[0])), exp_r=float(np.corrcoef(c, np.log(med))[0, 1]),
                                 pow_exp=float(pp[0]), pow_r=float(np.corrcoef(np.log(c), np.log(med))[0, 1]),
                                 median_last=float(np.median([l["d"][-1] for l in L])))
    if m == "gpt2" and d.get("example"):
        e = d["example"]; Pc = np.array(e["pc"]); k = e["k"]; off = e["onset"] - e["start"]
        cyc = np.maximum(0, (np.arange(len(Pc)) - off) // k)
        sc = ax[1].scatter(Pc[:, 0], Pc[:, 1], c=cyc, s=3, cmap="viridis", zorder=2)
        last = Pc[-k:]
        ax[1].plot(np.r_[last[:, 0], last[:1, 0]], np.r_[last[:, 1], last[:1, 1]], "-", lw=0.6, color="k", zorder=3)
        cb = fig.colorbar(sc, ax=ax[1], fraction=0.05, pad=0.02); cb.set_label("cycle", fontsize=7); cb.ax.tick_params(labelsize=6)
        ax[1].set_xlabel("PC 1"); ax[1].set_ylabel("PC 2"); ax[1].set_xticks([]); ax[1].set_yticks([])
        ax[1].set_title(f"(b) one GPT-2 loop, period {k}", loc="left", fontsize=8)
        summary["orbit_example"] = dict(prompt=e["prompt"], k=k, var=e["var"], text=e["text"])
ax[0].set_xlabel("cycle $c$ since loop onset"); ax[0].set_ylabel("return distance $\\mathrm{dist}_c$")
ax[0].set_title("(a) distance to previous cycle", loc="left", fontsize=8)
ax[0].legend(fontsize=6.5, frameon=False)
fig.tight_layout(w_pad=0.8)
fig.savefig("paper/figs/exp3c_orbits.pdf", dpi=300); fig.savefig("paper/figs/exp3c_orbits.png", dpi=200)
json.dump(summary, open("results/exp3bc_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "gpt2" and k != "pythia160m"}, indent=1)[:3000])
for m in ("gpt2", "pythia160m"):
    if m in summary:
        s = summary[m]; print(m, {k: s[k] for k in ("T_loop_half", "T_inc_half", "T_best", "coh_best", "window_nT")})
        print("  coh", [round(x, 2) for x in s["frac_coh"]], "loop", [round(x, 2) for x in s["frac_loop"]], "inc", [round(x, 2) for x in s["frac_inc"]])
        print("  nll", [round(x, 2) for x in s["ref_nll_median"]], "H", [round(x, 2) for x in s["entropy_median"]])
