"""Additional paired figures.

basins.pdf       Basins of attraction: forced double-well Duffing oscillator at two forcing amplitudes (smooth and
                 fractal basin boundaries), and the trainability boundary of a small neural network over two
                 learning rates (after Sohl-Dickstein 2024), with a zoom into the boundary.
catapult.pdf     A loss spike in the two-parameter catapult model, drawn on its loss landscape: the iterate bounces
                 across a valley that is too stiff for the step size and is thrown along it to a flatter part.
escape_time.pdf  Time histories of escape: a Brownian particle hopping between two wells, and GPT-2 entering and
                 leaving repetition loops while sampling.
Run from the repository root: python paper/fig_extra.py [basins|catapult|escape ...]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.getcwd())
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
MECH, LEARN = "#2457c5", "#c25e14"
which = sys.argv[1:] or ["basins", "catapult", "escape"]
RES = "results/fig_extra.json"
out = json.load(open(RES)) if os.path.exists(RES) else {}


def duffing_basin(A, n=360, periods=40, w=1.2, d=0.25, lim=2.0):
    """Class of the attractor reached from each initial condition of q'' + d q' - q + q^3 = A cos(w tau):
    +1 right well, -1 left well, 0 motion that visits both wells. RK4, 80 steps per forcing period."""
    q0, p0 = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
    q, p = q0.ravel().copy(), p0.ravel().copy()
    T = 2 * np.pi / w; steps = 80; dt = T / steps; tau = 0.0; lo = np.full(q.size, np.inf); hi = -lo.copy()
    f = lambda q, p, t: (p, -d * p + q - q ** 3 + A * np.cos(w * t))
    for k in range(periods * steps):
        k1q, k1p = f(q, p, tau); k2q, k2p = f(q + dt / 2 * k1q, p + dt / 2 * k1p, tau + dt / 2)
        k3q, k3p = f(q + dt / 2 * k2q, p + dt / 2 * k2p, tau + dt / 2); k4q, k4p = f(q + dt * k3q, p + dt * k3p, tau + dt)
        q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q); p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p); tau += dt
        if k >= (periods - 8) * steps:
            lo = np.minimum(lo, q); hi = np.maximum(hi, q)
    return np.where(lo > 0, 1, np.where(hi < 0, -1, 0)).reshape(n, n)


def trainability(lr0, lr1, h=16, d=8, m=8, steps=500, seed=0):
    """Full-batch gradient descent on a one-hidden-layer tanh network f(x) = W1 tanh(W0 x), squared loss on m random
    points, with separate learning rates for W0 and W1. Returns (converged mask, mean loss, divergence step)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, d)); y = rng.standard_normal(m)
    W0i = rng.standard_normal((h, d)) / np.sqrt(d); W1i = rng.standard_normal(h) / np.sqrt(h)
    G = lr0.size
    W0 = np.broadcast_to(W0i, (G, h, d)).copy(); W1 = np.broadcast_to(W1i, (G, h)).copy()
    a0 = lr0.reshape(G, 1, 1); a1 = lr1.reshape(G, 1)
    alive = np.ones(G, bool); acc = np.zeros(G); div = np.full(G, steps)
    for s in range(steps):
        H = np.tanh(np.einsum('ghd,md->ghm', W0, X)); e = np.einsum('gh,ghm->gm', W1, H) - y
        loss = 0.5 * (e ** 2).mean(1)
        bad = alive & (~np.isfinite(loss) | (loss > 1e6)); div[bad] = s; alive &= ~bad
        acc += np.where(alive, loss, 0)
        gW1 = np.einsum('gm,ghm->gh', e, H) / m
        gW0 = np.einsum('ghm,md->ghd', e[:, None, :] * W1[:, :, None] * (1 - H ** 2) / m, X)
        W1 = np.where(alive[:, None], W1 - a1 * gW1, W1); W0 = np.where(alive[:, None, None], W0 - a0 * gW0, W0)
    return alive, acc / steps, div


def nn_image(e0, e1, n):
    L0, L1 = np.meshgrid(np.logspace(*e0, n), np.logspace(*e1, n))
    al, acc, div = trainability(L0.ravel(), L1.ravel())
    al = al.reshape(n, n); acc = acc.reshape(n, n); div = div.reshape(n, n)
    img = np.zeros((n, n, 3))
    c = np.clip(np.log10(acc + 1e-12), -6, 0); c = (c + 6) / 6
    img[al] = np.stack([0.15 + 0.2 * c[al], 0.3 + 0.35 * c[al], 0.6 + 0.35 * c[al]], -1)
    dv = np.clip(np.log10(div + 1) / np.log10(501), 0, 1)
    img[~al] = np.stack([0.95 - 0.15 * dv[~al], 0.55 + 0.35 * dv[~al], 0.35 + 0.5 * dv[~al]], -1)
    return img, float(al.mean()), al


if "basins" in which:
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.05), gridspec_kw=dict(wspace=0.38))
    cmap = ListedColormap(["#e0a36b", "#d9d9d9", "#7fa2dc"])
    res = {}
    for j, A in enumerate([0.10, 0.25]):
        B = duffing_basin(A)
        ax[j].imshow(B, origin="lower", extent=[-2, 2, -2, 2], cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax[j].set_xlabel("initial $q$"); ax[j].set_ylabel("initial $\\dot q$", labelpad=0)
        ax[j].set_title(f"({'ab'[j]}) Duffing, $A={A:g}$", fontsize=7.5, loc="left")
        res[f"duffing_A{A:g}"] = dict(right=float((B == 1).mean()), left=float((B == -1).mean()), both=float((B == 0).mean()))
    img, frac, mask = nn_image((-1, 1.5), (-1, 1.5), 300)
    ax[2].imshow(img, origin="lower", extent=[-1, 1.5, -1, 1.5], interpolation="nearest", aspect="auto")
    ax[2].set_xlabel("$\\log_{10}$ step size, layer 1"); ax[2].set_ylabel("$\\log_{10}$ step size, layer 2", labelpad=0)
    ax[2].set_title("(c) network: trains or diverges", fontsize=7.5, loc="left")
    # zoom window centred on the boundary at layer-1 step size 10^0.6
    ex = np.linspace(-1, 1.5, 300); col = int(np.argmin(np.abs(ex - 0.6))); yb = ex[np.argmax(~mask[:, col])]
    z0, z1 = (0.55, 0.65), (yb - 0.05, yb + 0.05)
    imgz, fz, _ = nn_image(z0, z1, 300)
    ax[2].add_patch(plt.Rectangle((z0[0], z1[0]), z0[1] - z0[0], z1[1] - z1[0], fill=False, lw=0.8, ec="k"))
    ax[3].imshow(imgz, origin="lower", extent=[*z0, *z1], interpolation="nearest", aspect="auto")
    ax[3].set_xlabel("$\\log_{10}$ step size, layer 1"); ax[3].set_title("(d) zoom into (c)", fontsize=7.5, loc="left")
    res["nn_trained_fraction"] = frac; res["nn_zoom_trained_fraction"] = fz; res["zoom"] = [z0, z1]
    out["basins"] = res
    fig.savefig("paper/figs/basins.pdf", dpi=300, bbox_inches="tight"); fig.savefig("paper/figs/basins.png", dpi=200, bbox_inches="tight")

if "catapult" in which:
    eta = 0.95
    rng = np.random.default_rng(0); u, v = 1.6 + 0.05 * rng.standard_normal(), 0.4
    P, L, S = [], [], []
    for _ in range(60):
        e = u * v - 1; P.append((u, v)); L.append(0.5 * e * e); S.append(u * u + v * v)
        u, v = u - eta * e * v, v - eta * e * u
    P = np.array(P); L = np.array(L); S = np.array(S)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.42, width_ratios=[1.15, 1, 1]))
    U, V = np.meshgrid(np.linspace(0.4, 2.0, 400), np.linspace(-0.3, 1.6, 400)); Z = 0.5 * (U * V - 1) ** 2
    ax[0].contourf(U, V, np.log10(Z + 1e-4), levels=20, cmap="Greys_r", alpha=0.55)
    wv = np.linspace(0.63, 2.0, 300); hv = 1 / wv; sharp = wv ** 2 + hv ** 2 > 2 / eta
    ax[0].plot(wv[sharp], hv[sharp], color="#b22222", lw=1.6, label="stiff: $w_1^2+w_2^2>2/\\eta$")
    ax[0].plot(np.where(~sharp, wv, np.nan), np.where(~sharp, hv, np.nan), color="#2e8b57", lw=1.6, label="soft: $w_1^2+w_2^2<2/\\eta$")
    ax[0].plot(P[:, 0], P[:, 1], "o-", ms=2.2, lw=0.7, color=LEARN)
    ax[0].plot(*P[0], "k^", ms=4); ax[0].set_xlim(0.4, 2.0); ax[0].set_ylim(-0.3, 1.6)
    for k in range(1, 4): ax[0].annotate(str(k), P[k], textcoords="offset points", xytext=(3, 2), fontsize=5.5)
    ax[0].set_xlabel("$w_1$"); ax[0].set_ylabel("$w_2$", labelpad=0); ax[0].legend(fontsize=5.6, frameon=False, loc="lower right")
    ax[0].set_title("(a) path on the loss landscape", fontsize=7.5, loc="left")
    ax[1].semilogy(L + 1e-16, color=LEARN, lw=1.1); ax[1].set_xlabel("step $t$"); ax[1].set_ylabel("loss")
    ax[1].set_title("(b) loss spike", fontsize=7.5, loc="left")
    ax[2].plot(S, color=LEARN, lw=1.1); ax[2].axhline(2 / eta, color="0.5", ls=":", lw=0.8); ax[2].set_xlabel("step $t$")
    ax[2].set_ylabel("curvature $w_1^2+w_2^2$"); ax[2].text(len(S) * 0.98, 2 / eta * 1.03, "$2/\\eta$", ha="right", va="bottom", fontsize=7, color="0.4")
    ax[2].set_title("(c) curvature drops below $2/\\eta$", fontsize=7.5, loc="left")
    out["catapult"] = dict(loss_peak=float(L.max()), loss_start=float(L[0]), curv_start=float(S[0]), curv_end=float(S[-1]),
                           start=[float(P[0, 0]), float(P[0, 1])], end=[float(P[-1, 0]), float(P[-1, 1])])
    fig.savefig("paper/figs/catapult.pdf", dpi=300, bbox_inches="tight"); fig.savefig("paper/figs/catapult.png", dpi=200, bbox_inches="tight")

if "escape" in which:
    # particle: dq = -U'(q) dtau + sqrt(2D) dW, U = (q^2 - 1)^2 / 4
    D, dt, Tm = 0.06, 0.01, 1500.0
    rng = np.random.default_rng(3); q = -1.0; qs = []
    for i in range(int(Tm / dt)):
        q += -(q ** 3 - q) * dt + np.sqrt(2 * D * dt) * rng.standard_normal()
        if i % 50 == 0: qs.append(q)
    qs = np.array(qs); tq = np.arange(len(qs)) * 50 * dt
    hops = int(np.sum(np.abs(np.diff(np.sign(qs[np.abs(qs) > 0.6]))) > 0))
    # GPT-2: sample at temperature T and mark, at each step, the period of the loop the text is in (0 if none)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common.prompts import PROMPTS
    from exp3_escape.escape import find_loop
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained("gpt2"); model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    T, gen = float(os.environ.get("ESC_T", "0.5")), 600

    def loop_segments(seq, kmax=64, reps=3):
        """Label each token with the smallest period k of a k-periodic stretch of at least reps*k tokens that contains it."""
        n = len(seq); per = np.zeros(n, int)
        for k in range(1, kmax + 1):
            run = 0
            for i in range(k, n + 1):
                if i < n and seq[i] == seq[i - k]:
                    run += 1; continue
                if run >= (reps - 1) * k:
                    s = i - run - k
                    seg = per[s:i]; seg[(seg == 0) | (seg > k)] = k
                run = 0
        return per

    best = None
    for pi, prompt in enumerate(PROMPTS[:40]):
        for seed in range(2):
            torch.manual_seed(1000 * seed + pi)
            ids = tok(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                o = model(ids, use_cache=True); past, z = o.past_key_values, o.logits[0, -1]; seq = []
                for _ in range(gen):
                    nxt = int(torch.multinomial(torch.softmax(z.double() / T, -1), 1)); seq.append(nxt)
                    o = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True); past, z = o.past_key_values, o.logits[0, -1]
            per = loop_segments(seq); inl = (per > 0).astype(int)
            # count real escapes: a loop stretch followed by at least 25 tokens outside any loop
            d = np.diff(np.r_[0, inl, 0]); starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
            gaps = [starts[k + 1] - ends[k] for k in range(len(ends) - 1)] + [gen - ends[-1]] if len(ends) else []
            exits = int(sum(g >= 25 for g in gaps if g > 0)); entries = len(starts)
            frac = float(inl.mean())
            score = (min(exits, 2), -abs(frac - 0.55))
            if best is None or score > best[0]:
                best = (score, pi, seed, per, entries, exits, tok.decode(seq[:300]))
        if best[0][0] >= 2 and best[0][1] > -0.2: break
    _, pi, seed, per, entries, exits, text = best
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 2.8), gridspec_kw=dict(hspace=0.95))
    ax[0].plot(tq, qs, color=MECH, lw=0.5); ax[0].axhline(1, color="0.7", ls=":", lw=0.6); ax[0].axhline(-1, color="0.7", ls=":", lw=0.6)
    ax[0].set_xlabel("time $\\tau$"); ax[0].set_ylabel("$q$"); ax[0].set_yticks([-1, 0, 1])
    ax[0].set_title(f"(a) Brownian particle in a double well, $\\Delta U/D={0.25 / D:.1f}$: it rests in a well and hops", fontsize=7.5, loc="left")
    ax[1].fill_between(np.arange(1, gen + 1), 0, per, step="pre", color=LEARN, alpha=0.8, lw=0)
    ax[1].set_xlabel("generated token $n$"); ax[1].set_ylabel("loop period"); ax[1].set_xlim(1, gen)
    ax[1].set_title(f"(b) GPT-2 sampled at $T={T:g}$: it falls into repetition loops and leaves them", fontsize=7.5, loc="left")
    out["escape_time"] = dict(D=D, hops=hops, T=T, prompt=pi, seed=seed, loop_entries=entries, loop_exits=exits,
                              frac_in_loop=float((per > 0).mean()), text=text)
    fig.savefig("paper/figs/escape_time.pdf", dpi=300, bbox_inches="tight"); fig.savefig("paper/figs/escape_time.png", dpi=200, bbox_inches="tight")

json.dump(out, open(RES, "w"), indent=1)
print(json.dumps({k: (v if k != "escape_time" else {kk: vv for kk, vv in v.items() if kk != "text"}) for k, v in out.items()}, indent=1))
