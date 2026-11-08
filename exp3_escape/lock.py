"""Experiment 3d: growth law of the barrier and the locking temperature.

Uses the loops stored by escape.py. For each loop, fits the barrier Delta_n of cycle n
(n = 2..ncyc, cycle 1 = first cycle after onset) with three laws:
  log:         Delta_n = a + c ln n            -> sharp locking temperature T_lock = c (Prop. 3)
  linear:      Delta_n = a + b n               -> locked at every T (Prop. 2)
  saturating:  Delta_n = D - (D - a) e^{-(n-2)/tau} -> escape a.s., time ~ e^{D/T} (Prop. 2)
and compares them by AIC. It then predicts the finite-horizon locking temperature (survival
over the stored cycles after burn-in = 1/2) from the fitted log law with the Arrhenius rate
r_n = m e^{-Delta_n/T}, and compares it with the exact value from the stored logits.

Usage: python exp3_escape/lock.py results/exp3_gpt2.json,results/exp3_pythia160m.json
"""
import json, sys
import numpy as np
from scipy.optimize import curve_fit

srcs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["results/exp3_gpt2.json", "results/exp3_pythia160m.json"]


def aic(y, yhat, p):
    n = len(y); rss = float(np.sum((y - yhat) ** 2)) + 1e-12
    return n * np.log(rss / n) + 2 * p + 2 * p * (p + 1) / max(1, n - p - 1)   # AICc


def sat(n, D, a, tau):
    return D - (D - a) * np.exp(-(n - 2) / tau)


def t_half(temps, surv):
    for i in range(len(temps) - 1):
        if surv[i] >= 0.5 > surv[i + 1]:
            lt = np.log(temps[i]) + (surv[i] - 0.5) / (surv[i] - surv[i + 1]) * (np.log(temps[i + 1]) - np.log(temps[i]))
            return float(np.exp(lt))
    return None


def surv_pred(Delta, temps, m=1.0):
    r = np.clip(m * np.exp(-np.outer(Delta, 1 / temps)), 0, 1)
    return np.prod(1 - r, 0)


out = {}
for src in srcs:
    d = json.load(open(src)); temps = np.array(d["temps"]); b = d["args"]["burn_cycles"]
    name = d["args"]["model"]; rows = []
    for L in d["loops"]:
        g = np.array(L["cyc_gap"]); N = len(g)
        n = np.arange(2, N + 1); y = g[1:]
        if len(n) < 5:
            continue
        A = {}
        pl = np.polyfit(np.log(n), y, 1); A["log"] = aic(y, np.polyval(pl, np.log(n)), 2)
        pn = np.polyfit(n, y, 1); A["linear"] = aic(y, np.polyval(pn, n), 2)
        try:
            ps, _ = curve_fit(sat, n, y, p0=[y.max(), y[0], 3.0], bounds=([-5, -5, 0.1], [50, 50, 1e3]), maxfev=20000)
            A["saturating"] = aic(y, sat(n, *ps), 3)
        except Exception:
            ps = None
        best = min(A, key=A.get)
        # exact finite-horizon locking temperature (survive all stored cycles after burn-in)
        th_exact = t_half(temps, np.array(L["surv_end"]))
        cyc = np.arange(b + 1, N + 1)                               # cycles timed in surv_end
        th_log = t_half(temps, surv_pred(np.polyval(pl, np.log(cyc)), temps))
        th_meas = t_half(temps, surv_pred(g[b:], temps))            # Arrhenius with the measured barriers
        rows.append(dict(prompt=L["prompt"], ncyc=N, best=best, aic=A, c=float(pl[0]), a=float(pl[1]), b_lin=float(pn[0]),
                         sat=None if ps is None else [float(x) for x in ps], T_half_exact=th_exact,
                         T_half_log=th_log, T_half_arrh=th_meas))
    cs = np.array([r["c"] for r in rows])
    pairs = [(r["T_half_exact"], r["T_half_log"]) for r in rows if r["T_half_exact"] and r["T_half_log"]]
    pm = [(r["T_half_exact"], r["T_half_arrh"]) for r in rows if r["T_half_exact"] and r["T_half_arrh"]]
    e, p = np.array(pairs).T; e2, p2 = np.array(pm).T
    S = dict(n_fit=len(rows), best_counts={k: sum(r["best"] == k for r in rows) for k in ("log", "linear", "saturating")},
             c_median=float(np.median(cs)), c_iqr=[float(np.percentile(cs, 25)), float(np.percentile(cs, 75))],
             T_half_exact_median=float(np.median(e)), T_half_log_median=float(np.median(p)),
             log_pred_rel_err_median=float(np.median(np.abs(p - e) / e)), log_pred_corr=float(np.corrcoef(e, p)[0, 1]),
             arrh_pred_rel_err_median=float(np.median(np.abs(p2 - e2) / e2)), arrh_pred_corr=float(np.corrcoef(e2, p2)[0, 1]),
             n_pairs=len(pairs), frac_c_above_Thalf=float(np.mean([r["c"] > r["T_half_exact"] for r in rows if r["T_half_exact"]])),
             sat_D_median=float(np.median([r["sat"][0] for r in rows if r["sat"]])),
             sat_tau_median=float(np.median([r["sat"][2] for r in rows if r["sat"]])))
    # horizon dependence of the exact locking temperature; beyond the stored cycles the last
    # stored per-cycle escape probability is repeated (the saturated barrier)
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]
    hor = {N: [] for N in Ns}
    for L in d["loops"]:
        R = np.array(L["cyc_escape"])[b:]
        if len(R) < 3:
            continue
        for N in Ns:
            RR = np.vstack([R[:N]] + [R[-1:]] * max(0, N - len(R)))
            th = t_half(temps, np.prod(1 - RR, 0))
            if th:
                hor[N].append(th)
    med = [float(np.median(hor[N])) for N in Ns]
    f = lambda N, D, lm: D / (lm + np.log(np.array(N, float) / np.log(2)))
    ph, _ = curve_fit(f, Ns[2:], med[2:], p0=[3, 1])
    S["horizon"] = dict(N=Ns, T_half_median=med, n_loops=[len(hor[N]) for N in Ns], fit_D=float(ph[0]), fit_lnm=float(ph[1]),
                        fit_pred=[float(x) for x in f(Ns, *ph)], max_abs_dev=float(np.max(np.abs(np.array(med[2:]) - f(Ns[2:], *ph)))))
    S["a_median"] = float(np.median([r["a"] for r in rows]))
    out[name] = dict(summary=S, loops=rows)
    print(name, json.dumps(S, indent=1))
json.dump(out, open("results/exp3d_lock.json", "w"), indent=1)
