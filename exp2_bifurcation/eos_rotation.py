"""Paper 3, Plan step 1: rotation of the stiff directions and true stability of the edge orbit.

Same training protocol as eos_geometry.py (Paper 1, Exp. 2). After training with step size eta we keep
running gradient descent for `window` steps and measure:

1. Every `every` steps, at a consecutive pair (a = theta_t, b = theta_{t+1}), the top-k Hessian eigenpairs by
   Lanczos (scipy eigsh on Hessian-vector products, residuals ~1e-13). Records eigenvalues, the angle phi between
   the top eigenvectors, principal angles between the top-k eigenspaces, the 2x2 projections onto
   span(u_a, u_b), and the reduced two-step monodromy on that plane and on span(top-k at a, top-k at b).
2. Finite-time Lyapunov spectrum (Benettin, `nlyap` tangent vectors, QR every step) of the GD map over the window,
   reported as multipliers per two steps, plus the curvature eta*v'Hv/2 of the final Lyapunov vectors
   (to tell stiff oscillation modes from flat directions, which give multiplier ~1).
3. At t = `arnoldi_at`: Arnoldi (scipy eigs) on the two-step Jacobian J = (I - eta H_b)(I - eta H_a), top
   `karn` multipliers by modulus, each with the curvature eta*v'H_a v/2 of its eigenvector (real or imaginary part).

Note: power iteration on J (eos_geometry.py) is dominated by the thousands of flat directions with multiplier ~1
and by complex pairs, so it cannot see a multiplier of 1.03 in 30 iterations. Arnoldi can.

Usage: python eos_rotation.py --seed 0 --out results/rot_seed0.json
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh, eigs
from torch.nn.utils import parameters_to_vector as p2v, vector_to_parameters as v2p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bifurcation import TinyTransformer, markov_data, lossfn


class Hess:
    """Hessian-vector products at a fixed parameter vector (graph built once)."""

    def __init__(self, m, P, x, y, theta):
        self.P = P
        v2p(theta, P)
        self.g = torch.autograd.grad(lossfn(m, x, y), P, create_graph=True)
        self.n = theta.numel()
        self.calls = 0

    def __call__(self, d):
        self.calls += 1
        ds, i = [], 0
        for p in self.P:
            ds.append(d[i:i + p.numel()].view_as(p)); i += p.numel()
        return p2v(torch.autograd.grad(self.g, self.P, grad_outputs=ds, retain_graph=True)).detach()

    def np(self, v):
        return self(torch.from_numpy(np.ascontiguousarray(v, dtype=np.float64))).numpy()

    def top(self, k, seed=0):
        op = LinearOperator((self.n, self.n), matvec=self.np, dtype=np.float64)
        v0 = np.random.default_rng(seed).standard_normal(self.n)
        w, U = eigsh(op, k=k, which="LA", tol=1e-10, ncv=max(4 * k, 24), v0=v0)
        o = np.argsort(-w)
        return w[o], U[:, o]


def project(H, Q):
    HQ = np.stack([H.np(Q[:, j]) for j in range(Q.shape[1])], 1)
    S = Q.T @ HQ
    return 0.5 * (S + S.T)


def rho(Ha, Hb, eta):
    I = np.eye(Ha.shape[0])
    return float(np.max(np.abs(np.linalg.eigvals((I - eta * Hb) @ (I - eta * Ha)))))


def train(eta, seed):
    x, y, _ = markov_data(12, 12, 64, seed=seed)
    torch.manual_seed(seed)
    init = TinyTransformer(V=12, T=12).double().state_dict()
    m = TinyTransformer(V=12, T=12).double(); m.load_state_dict(init); P = list(m.parameters())
    steps = int(min(30000, max(3000, 600 / eta)))
    for t in range(steps):
        m.zero_grad(); L = lossfn(m, x, y)
        if not torch.isfinite(L) or L.item() > 50:
            return None
        L.backward()
        with torch.no_grad():
            for q in P:
                q -= eta * q.grad
    return m, P, x, y


def measure(eta, seed, window, every, k, nlyap, arnoldi_at, karn):
    tr = train(eta, seed)
    if tr is None:
        return dict(eta=float(eta), diverged=True)
    m, P, x, y = tr

    def grad(theta):
        v2p(theta, P); L = lossfn(m, x, y); return L.item(), p2v(torch.autograd.grad(L, P)).detach()

    theta = p2v(P).detach().clone(); n = theta.numel()
    Qt = torch.linalg.qr(torch.randn(n, nlyap, generator=torch.Generator().manual_seed(3), dtype=theta.dtype))[0]
    logs = np.zeros(nlyap); cum = []
    pairs, arn, losses = [], None, []
    l, g = grad(theta); Hcur = Hess(m, P, x, y, theta)
    for t in range(window):
        losses.append(l)
        th1 = theta - eta * g
        l1, g1 = grad(th1)
        Hnext = Hess(m, P, x, y, th1)
        # Benettin step with the Jacobian at theta_t
        W = torch.stack([Qt[:, j] - eta * Hcur(Qt[:, j]) for j in range(nlyap)], 1)
        Qt, Rr = torch.linalg.qr(W)
        logs += np.log(np.abs(np.diag(Rr.numpy()))); cum.append((logs / (t + 1)).tolist())
        if t % every == 0:
            la, Ua = Hcur.top(k, seed=t); lb, Ub = Hnext.top(k, seed=t + 1)
            ua, ub = Ua[:, 0], Ub[:, 0]
            phi = math.acos(min(1.0, abs(float(ua @ ub))))
            sv = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
            Qp, _ = np.linalg.qr(np.stack([ua, ub], 1))
            Hap, Hbp = project(Hcur, Qp), project(Hnext, Qp)
            Qs, _ = np.linalg.qr(np.concatenate([Ua, Ub], 1))
            Has, Hbs = project(Hcur, Qs), project(Hnext, Qs)
            step = (th1 - theta).numpy()
            dHu = Hnext.np(ua) - Hcur.np(ua); par = float(ua @ dHu)
            pairs.append(dict(
                t=t, lam_a=la.tolist(), lam_b=lb.tolist(), phi=phi,
                sub_angles=np.degrees(np.arccos(np.clip(sv, -1, 1))).tolist(),
                plane_a=Hap.tolist(), plane_b=Hbp.tolist(), rho_plane=rho(Hap, Hbp, eta), rho_sub=rho(Has, Hbs, eta),
                cos_step=abs(float(step @ ua)) / float(np.linalg.norm(step)), step_norm=float(np.linalg.norm(step)),
                dHu_par=par, dHu_perp=float(np.linalg.norm(dHu - par * ua))))
        if t == arnoldi_at:
            def mv(v):
                z = torch.from_numpy(np.ascontiguousarray(v.real, dtype=np.float64))
                w = z - eta * Hcur(z); w = w - eta * Hnext(w); return w.numpy()
            op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
            try:
                w, V = eigs(op, k=karn, which="LM", tol=1e-7, ncv=max(5 * karn, 40),
                            v0=np.random.default_rng(0).standard_normal(n), maxiter=4000)
                rows = []
                for i in np.argsort(-np.abs(w)):
                    v = V[:, i]; vr = v.real if np.linalg.norm(v.real) >= np.linalg.norm(v.imag) else v.imag
                    vr = vr / np.linalg.norm(vr)
                    rows.append(dict(re=float(w[i].real), im=float(w[i].imag), mod=float(abs(w[i])),
                                     curv=float(eta * (vr @ Hcur.np(vr)) / 2)))
                arn = rows
            except Exception as e:  # ArpackNoConvergence etc.
                arn = dict(error=str(e)[:200])
        theta, g, l, Hcur = th1, g1, l1, Hnext
    # curvature of the final Lyapunov vectors
    lyap_curv = [float(eta * (Qt[:, j] @ Hcur(Qt[:, j])) / 2) for j in range(nlyap)]
    lyap2 = np.exp(2 * logs / window).tolist()  # multiplier per two steps
    R = [eta * (p["lam_a"][0] + p["lam_b"][0]) / 4 for p in pairs]
    nabove = [int(np.sum(np.array(p["lam_a"]) * eta / 2 > 1)) for p in pairs]
    return dict(eta=float(eta), diverged=False, pairs=pairs, arnoldi=arn, lyap2=lyap2, lyap_curv=lyap_curv,
                lyap_running=cum[:: max(1, window // 50)], loss=losses,
                R_top=float(np.mean(R)), n_above_med=float(np.median(nabove)),
                phi_med=float(np.median([p["phi"] for p in pairs])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etas", default="0.05:1.6:9")
    ap.add_argument("--only", default="", help="comma-separated indices into the eta grid")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--nlyap", type=int, default=6)
    ap.add_argument("--arnoldi_at", type=int, default=100)
    ap.add_argument("--karn", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    lo, hi, nn_ = a.etas.split(":")
    grid = np.geomspace(float(lo), float(hi), int(nn_))
    idx = [int(i) for i in a.only.split(",")] if a.only else range(len(grid))
    out = []
    for i in idx:
        t0 = time.time()
        r = measure(grid[i], a.seed, a.window, a.every, a.k, a.nlyap, a.arnoldi_at, a.karn)
        r["seed"] = a.seed; r["secs"] = time.time() - t0
        out.append(r)
        if r["diverged"]:
            print(f"seed={a.seed} eta={grid[i]:.3f} diverged", flush=True)
        else:
            top = r["arnoldi"][0]["mod"] if isinstance(r["arnoldi"], list) else float("nan")
            print(f"seed={a.seed} eta={r['eta']:.3f} R_top={r['R_top']:.3f} n_above={r['n_above_med']:.0f} "
                  f"phi={math.degrees(r['phi_med']):.0f}deg lyap2={r['lyap2'][0]:.4f} (curv {r['lyap_curv'][0]:.3f}) "
                  f"arnoldi_top={top:.4f} ({r['secs']:.0f}s)", flush=True)
        json.dump(dict(args=vars(a), runs=out), open(a.out, "w"))


if __name__ == "__main__":
    main()
