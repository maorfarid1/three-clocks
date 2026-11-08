"""Edge-of-stability geometry on the Exp. 2 transformer (exploratory, for a follow-up study).

After training with step size eta (same protocol as bifurcation.py) we continue gradient descent
for `window` steps and record, at every step t:
  R_t      = eta * lambda_max(theta_t) / 2              (sharpness ratio at the iterate)
  Rsec_t   = <g_t - g_{t+1}, g_t> / (2 |g_t|^2)         (eta/2 times the curvature of the segment-averaged
                                                          Hessian along the step; exactly 1 on a period-2 orbit)
  cos_t    = |<g_t, u_t>| / |g_t|                       (alignment of the step with the top eigenvector)
and every `fl_every` steps the growth factor per two steps of the linearized two-step map
(I - eta H_{t+1})(I - eta H_t) (a finite-time Floquet multiplier), by power iteration.

Usage: python exp2_bifurcation/eos_geometry.py --seed 0 --out results/eos_geom_seed0.json
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector as p2v, vector_to_parameters as v2p

sys.path.insert(0, os.path.dirname(__file__))
from bifurcation import TinyTransformer, markov_data, lossfn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etas", default="0.05:1.6:9")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--fl_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    x, y, _ = markov_data(12, 12, 64, seed=a.seed)
    torch.manual_seed(a.seed)
    init = TinyTransformer(V=12, T=12).double().state_dict()
    lo, hi, n = a.etas.split(":")
    out = []
    for eta in np.geomspace(float(lo), float(hi), int(n)):
        t0 = time.time()
        m = TinyTransformer(V=12, T=12).double(); m.load_state_dict(init); P = list(m.parameters())
        steps = int(min(30000, max(3000, 600 / eta)))
        ok = True
        for t in range(steps):
            m.zero_grad(); L = lossfn(m, x, y)
            if not torch.isfinite(L) or L.item() > 50:
                ok = False; break
            L.backward()
            with torch.no_grad():
                for q in P:
                    q -= eta * q.grad
        if not ok:
            out.append(dict(eta=float(eta), diverged=True)); print(f"eta={eta:.3f} diverged", flush=True); continue

        def grad(theta):
            v2p(theta, P); return p2v(torch.autograd.grad(lossfn(m, x, y), P)).detach()

        def hvp(theta, d):
            v2p(theta, P); g = torch.autograd.grad(lossfn(m, x, y), P, create_graph=True)
            ds, i = [], 0
            for p in P:
                ds.append(d[i:i + p.numel()].view_as(p)); i += p.numel()
            return p2v(torch.autograd.grad(g, P, grad_outputs=ds)).detach()

        theta = p2v(P).detach().clone(); u = torch.randn_like(theta); u /= u.norm()
        R, Rsec, cos, fl = [], [], [], []
        g = grad(theta)
        for t in range(a.window):
            for _ in range(40 if t == 0 else 10):
                w = hvp(theta, u); lam = float(w @ u); u = w / w.norm()
            th1 = theta - eta * g; g1 = grad(th1)
            R.append(eta * lam / 2)
            Rsec.append(float((g - g1) @ g) / (2 * float(g @ g)))
            cos.append(abs(float(g @ u)) / float(g.norm()))
            if t % a.fl_every == 0:
                th2 = th1 - eta * g1; z = torch.randn_like(theta); z /= z.norm(); gr = []
                for _ in range(30):
                    w = z - eta * hvp(theta, z); w = w - eta * hvp(th1, w); nr = float(w.norm()); gr.append(nr); z = w / nr
                fl.append(float(np.exp(np.mean(np.log(gr[-10:])))))
            theta, g = th1, g1
        rec = dict(eta=float(eta), diverged=False, R=R, Rsec=Rsec, cos=cos, floquet=fl,
                   R_mean=float(np.mean(R)), R_sd=float(np.std(R)), Rsec_mean=float(np.mean(Rsec)), cos_mean=float(np.mean(cos)),
                   floquet_median=float(np.median(fl)), single_sq=float(np.median([(2 * r - 1) ** 2 for r in R])))
        out.append(rec)
        print(f"eta={eta:.3f} R={rec['R_mean']:.3f}+-{rec['R_sd']:.3f} Rsec={rec['Rsec_mean']:.3f} cos={rec['cos_mean']:.2f} "
              f"floquet2={rec['floquet_median']:.3f} vs (eta*lam-1)^2={rec['single_sq']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(dict(args=vars(a), runs=out), open(a.out, "w"))


if __name__ == "__main__":
    main()
