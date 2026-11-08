"""Reduced-order model of loop escape: a particle in a well that deepens and saturates.

Full system: the exact per-cycle escape probabilities r_c(T) of each greedy loop, computed from the stored logits of
every step of every cycle (Exp. 3; vocabulary of 50k tokens, loop periods up to 64).
Reduced model: one barrier per cycle, crossed with nu independent attempts per cycle,
    Delta_c = Delta_inf - (Delta_inf - Delta_1) exp(-(c - 1) / tau),      r_c(T) = 1 - (1 + e^{-Delta_c/T})^(-nu),
three numbers per loop fitted to the per-cycle minimum logit gaps (no temperature data used), and one attempt number
nu per model (the prefactor of Kramers' rate), chosen on a grid to match the full computation best.
We compare the locking temperature T_1/2(N), at which half of the probability stays in the loop for N cycles after a
burn-in of 2 cycles, between the reduced model and the full computation, and with the closed form of Section 5,
T_1/2(N) ~ Delta_inf / ln(N / ln 2).
Usage: python exp3_escape/reduced.py --out results/reduced.json
"""
import argparse, json
import numpy as np
from scipy.optimize import curve_fit

NS = [1, 4, 16, 64, 256]
TG = np.geomspace(0.1, 3.0, 400)


def sat(c, D, a, tau):
    return D - (D - a) * np.exp(-(c - 1) / tau)


def t_half(temps, surv):
    for i in range(len(temps) - 1):
        if surv[i] >= 0.5 > surv[i + 1]:
            return float(np.exp(np.log(temps[i]) + (surv[i] - 0.5) / (surv[i] - surv[i + 1]) * (np.log(temps[i + 1]) - np.log(temps[i]))))
    return None


def analyze(f, nu, burn=2):
    d = json.load(open(f)); temps = np.array(d["temps"]); rows = []
    for l in d["loops"]:
        y = np.array(l["cyc_gap"]); c = np.arange(1, len(y) + 1)
        R = np.array(l["cyc_escape"])[burn:]
        if len(R) < 3:
            continue
        try:
            p, _ = curve_fit(sat, c, y, p0=[y.max(), y[0], 3.0], bounds=([-5, -5, 0.1], [50, 50, 1e3]), maxfev=20000)
        except Exception:
            continue
        row = dict(prompt=l["prompt"], k=l["k"], ncyc=len(y), Dinf=float(p[0]), D1=float(p[1]), tau=float(p[2]),
                   fit_rmse=float(np.sqrt(np.mean((sat(c, *p) - y) ** 2))), exact={}, reduced={}, closed={})
        for N in NS:
            RR = np.vstack([R[:N]] + [R[-1:]] * max(0, N - len(R)))
            row["exact"][N] = t_half(temps, np.prod(1 - RR, 0))
            cc = np.arange(burn + 1, burn + N + 1)
            e = np.exp(-sat(cc, *p)[:, None] / TG[None, :]); r = 1 - (1 + e) ** (-nu)
            row["reduced"][N] = t_half(TG, np.prod(1 - r, 0))
            row["closed"][N] = float(p[0] / np.log(nu * N / np.log(2)))
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    res = {}
    for m in ("gpt2", "pythia160m"):
        best = None
        for nu in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
            rows = analyze(f"results/exp3_{m}.json", nu)
            err = np.median([abs(np.log(r["reduced"][N] / r["exact"][N])) for r in rows for N in NS if r["exact"][N] and r["reduced"][N]])
            if best is None or err < best[0]:
                best = (err, nu, rows)
        err, nu, rows = best
        summ = {}
        for N in NS:
            pr = [(r["exact"][N], r["reduced"][N]) for r in rows if r["exact"][N] and r["reduced"][N]]
            x, yv = np.array(pr).T
            summ[N] = dict(n=len(pr), median_exact=float(np.median(x)), median_reduced=float(np.median(yv)),
                           median_abs_rel_err=float(np.median(np.abs(yv - x) / x)), corr=float(np.corrcoef(x, yv)[0, 1]))
        res[m] = dict(nu=nu, rows=rows, summary=summ, fit_rmse_median=float(np.median([r["fit_rmse"] for r in rows])))
        print(m, len(rows), "nu", nu, "fit rmse median %.3f" % res[m]["fit_rmse_median"])
        for N, s in summ.items():
            cl = [r["closed"][N] for r in rows if r["exact"][N]]
            print(f"  N={N}: exact {s['median_exact']:.3f} reduced {s['median_reduced']:.3f} closed {np.median(cl):.3f} med|rel err| {s['median_abs_rel_err']:.3f} corr {s['corr']:.3f} (n={s['n']})")
    json.dump(res, open(a.out, "w"), indent=1)
