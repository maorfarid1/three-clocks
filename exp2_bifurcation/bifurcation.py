"""Exp. 2: learning-rate bifurcation diagram of a tiny transformer.

Task: next-token prediction on sequences from a fixed random Markov chain
(vocabulary V). The loss has a nonzero floor (the chain's conditional entropy),
so a stationary state exists. Training is full-batch gradient descent, so the
only "noise" is the deterministic dynamics of the map itself.

For each step size eta on a grid we train from the SAME initialization, record
the last `--tail` loss values, estimate their period, and compute the top
Hessian eigenvalue (sharpness) at the end. Plotting the tail losses against eta
gives a bifurcation diagram.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformer(nn.Module):
    """One causal attention block + MLP, pre-LN, explicit attention so that
    Hessian-vector products (double backward) work on CPU."""

    def __init__(self, V, T, d=32, heads=2, mlp=64):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(T, d) * 0.1)
        self.ln1 = nn.LayerNorm(d)
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        self.lnf = nn.LayerNorm(d)
        self.out = nn.Linear(d, V)
        self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))

    def attention(self, a):
        B, T, D = a.shape
        q, k, v = self.qkv(a).split(D, dim=-1)
        sh = lambda z: z.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        s = q @ k.transpose(-1, -2) / math.sqrt(D // self.heads)
        s = s.masked_fill(self.mask[:T, :T], float("-inf"))
        return self.proj((torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, D))

    def forward(self, x):
        h = self.tok(x) + self.pos[: x.shape[1]]
        h = h + self.attention(self.ln1(h))
        h = h + self.mlp(self.ln2(h))
        return self.out(self.lnf(h))


def markov_data(V, T, N, seed=0, conc=0.3):
    g = np.random.default_rng(seed)
    P = g.dirichlet(conc * np.ones(V), size=V)  # transition matrix
    pi = np.ones(V) / V
    X = np.zeros((N, T + 1), dtype=np.int64)
    X[:, 0] = g.choice(V, size=N, p=pi)
    for t in range(T):
        for i in range(N):
            X[i, t + 1] = g.choice(V, p=P[X[i, t]])
    # empirical conditional entropy of the sample = loss floor for a bigram-optimal model
    H = -np.mean([np.log(P[X[i, t], X[i, t + 1]]) for i in range(N) for t in range(T)])
    return torch.tensor(X[:, :-1]), torch.tensor(X[:, 1:]), float(H)


def lossfn(model, x, y):
    lg = model(x)
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))


def period(tail, rtol=1e-7, kmax=32):
    """Smallest k with max|L[t+k]-L[t]| < rtol*mean(L) over the tail; 0 = none found."""
    t = np.asarray(tail)
    s = rtol * max(np.abs(t).mean(), 1e-12)
    for k in range(1, kmax + 1):
        if np.max(np.abs(t[k:] - t[:-k])) < s:
            return k
    return 0


def sharpness(model, x, y, iters=40):
    params = list(model.parameters())
    grads = torch.autograd.grad(lossfn(model, x, y), params, create_graph=True)
    g = torch.Generator().manual_seed(1)
    v = [torch.randn(q.shape, generator=g, dtype=q.dtype) for q in params]
    nrm = math.sqrt(sum((u * u).sum() for u in v))
    v = [u / nrm for u in v]
    lam = 0.0
    for _ in range(iters):
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        lam = sum((a * b).sum() for a, b in zip(hv, v)).item()
        nrm = math.sqrt(sum((u * u).sum() for u in hv))
        v = [u / nrm for u in hv]
    return lam


def run(eta, init_state, model_kw, x, y, steps, tail):
    model = TinyTransformer(**model_kw).double()
    model.load_state_dict(init_state)
    losses = []
    for t in range(steps):
        model.zero_grad()
        loss = lossfn(model, x, y)
        if not torch.isfinite(loss) or loss.item() > 50:
            return dict(eta=eta, diverged=True, diverge_step=t, tail=[], period=None, sharp=None,
                        curve=losses[:: max(1, steps // 300)])
        loss.backward()
        with torch.no_grad():
            for q in model.parameters():
                q -= eta * q.grad
        losses.append(loss.item())
    tl = losses[-tail:]
    return dict(eta=eta, diverged=False, tail=tl, period=period(tl),
                sharp=sharpness(model, x, y), curve=losses[:: max(1, steps // 300)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=12)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000, help="minimum steps")
    ap.add_argument("--time", type=float, default=600.0,
                    help="also train at least time/eta steps, so every eta gets equal gradient-flow time")
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--tail", type=int, default=400)
    ap.add_argument("--etas", type=str, default="0.02:4:56")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/exp2.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    lo, hi, n = a.etas.split(":")
    etas = np.geomspace(float(lo), float(hi), int(n)).tolist()
    x, y, H = markov_data(a.V, a.T, a.N, seed=a.seed)
    model_kw = dict(V=a.V, T=a.T)
    torch.manual_seed(a.seed)
    init = TinyTransformer(**model_kw).double().state_dict()
    print(f"entropy floor ~ {H:.4f}")
    res, t0 = [], time.time()
    for e in etas:
        steps = int(min(a.max_steps, max(a.steps, a.time / e)))
        r = run(e, init, model_kw, x, y, steps, a.tail)
        r["steps"] = steps
        res.append(r)
        last = f"{r['tail'][-1]:.4f}" if r["tail"] else None
        print(f"eta={e:.4f} div={r['diverged']} last={last} per={r['period']} "
              f"sharp={r['sharp']} 2/eta={2/e:.2f} t={time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(a), entropy_floor=H, runs=res), open(a.out, "w"))
