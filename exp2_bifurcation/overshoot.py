"""Experiment 2b: how far above 2/eta does the sharpness sit at the edge of stability?

Same model, data and step-size protocol as bifurcation.py. After training, we keep running
gradient descent for `window` more steps and record at every step
  - the sharpness lambda_t (power iteration on Hessian-vector products, warm-started),
  - the component a_t = <grad L(theta_t), u_t> of the gradient along the top eigenvector u_t.
If the oscillation along the top direction stays bounded and nonzero, its Lyapunov exponent
is zero. With the tangent multiplier 1 - eta*lambda_t this reads <log|eta*lambda_t - 1|> = 0,
and Jensen's inequality then forces <eta*lambda_t> > 2. To second order in the fluctuation,
<R> - 1 ~= Var(R) with R = eta*lambda/2.

Usage: python exp2_bifurcation/overshoot.py --seed 0 --out results/exp2b_seed0.json
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from bifurcation import TinyTransformer, markov_data, lossfn


def top_eig(model, x, y, v=None, iters=12):
    params = list(model.parameters())
    loss = lossfn(model, x, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    if v is None:
        g = torch.Generator().manual_seed(1)
        v = [torch.randn(q.shape, generator=g, dtype=q.dtype) for q in params]
    nrm = math.sqrt(sum((u * u).sum() for u in v)); v = [u / nrm for u in v]
    lam = 0.0
    for _ in range(iters):
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        lam = sum((a * b).sum() for a, b in zip(hv, v)).item()
        nrm = math.sqrt(sum((u * u).sum() for u in hv)); v = [u / nrm for u in hv]
    g_dot_u = sum((gg.detach() * u).sum() for gg, u in zip(grads, v)).item()
    return lam, v, g_dot_u, [gg.detach() for gg in grads], loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=12); ap.add_argument("--T", type=int, default=12); ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--time", type=float, default=600.0)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--etas", type=str, default="0.1:1.6:10")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    lo, hi, n = a.etas.split(":")
    etas = np.geomspace(float(lo), float(hi), int(n)).tolist()
    x, y, H = markov_data(a.V, a.T, a.N, seed=a.seed)
    kw = dict(V=a.V, T=a.T)
    torch.manual_seed(a.seed)
    init = TinyTransformer(**kw).double().state_dict()
    out = []
    for eta in etas:
        t0 = time.time()
        model = TinyTransformer(**kw).double(); model.load_state_dict(init)
        steps = int(min(a.max_steps, max(a.steps, a.time / eta)))
        ok = True
        for t in range(steps):
            model.zero_grad(); loss = lossfn(model, x, y)
            if not torch.isfinite(loss) or loss.item() > 50:
                ok = False; break
            loss.backward()
            with torch.no_grad():
                for q in model.parameters():
                    q -= eta * q.grad
        if not ok:
            out.append(dict(eta=eta, diverged=True)); print(f"eta={eta:.3f} diverged", flush=True); continue
        lam_s, a_s, loss_s, v = [], [], [], None
        for t in range(a.window):
            lam, v, gu, grads, L = top_eig(model, x, y, v, iters=40 if t == 0 else 12)
            lam_s.append(lam); a_s.append(gu); loss_s.append(L)
            with torch.no_grad():
                for q, g in zip(model.parameters(), grads):
                    q -= eta * g
        lam_s = np.array(lam_s); R = eta * lam_s / 2
        a_arr = np.array(a_s)
        rec = dict(eta=eta, diverged=False, steps=steps, R=R.tolist(), a=a_arr.tolist(), loss=loss_s,
                   R_mean=float(R.mean()), R_var=float(R.var()),
                   mean_log_mult=float(np.mean(np.log(np.abs(eta * lam_s - 1)))),
                   mean_log_ratio=float(np.mean(np.log(np.abs(a_arr[1:] / a_arr[:-1])))))
        out.append(rec)
        print(f"eta={eta:.3f} <R>={rec['R_mean']:.4f} Var(R)={rec['R_var']:.4f} <log|eta*lam-1|>={rec['mean_log_mult']:.4f} "
              f"<log|a_t+1/a_t|>={rec['mean_log_ratio']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(dict(args=vars(a), runs=out), open(a.out, "w"))


if __name__ == "__main__":
    main()
