"""Paired figures: a textbook dynamical system and a learning system in the same representation.

pair_depth.pdf     Kuramoto oscillators synchronize over time; tokens of a transformer cluster over depth
                   (idealized attention dynamics, and GPT-2 measured: random initialization vs trained).
pair_training.pdf  Gradient descent on a double well and a transformer at the edge of stability: time history
                   and power spectrum (period two shows up as a peak at frequency 1/2), plus a catapult spike.
pair_escape.pdf    Kramers escape of a Brownian particle and escape from a GPT-2 repetition loop: Arrhenius plots.
Run from the repository root: python paper/fig_pairs.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, os.getcwd())
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
MECH, LEARN = "#2457c5", "#c25e14"
out = {}

# ---------------------------------------------------------------- depth: synchronization
def kuramoto(N=60, K=1.5, T=30.0, dt=0.05, seed=1):
    rng = np.random.default_rng(seed); th = rng.uniform(0, 2 * np.pi, N); w = rng.normal(0, 0.3, N)
    snaps, r = {}, []
    for i in range(int(T / dt) + 1):
        if i in (0, int(3 / dt), int(T / dt)):
            snaps[round(i * dt)] = th.copy()
        r.append(abs(np.exp(1j * th).mean()))
        th = th + dt * (w + K * (np.sin(th[None, :] - th[:, None])).mean(1))
    return snaps, np.array(r), dt


def attention_circle(N=60, beta=1.0, L=60, dt=0.25, seed=1):
    rng = np.random.default_rng(seed); th = rng.uniform(0, 2 * np.pi, N); r = []
    for _ in range(L + 1):
        r.append(abs(np.exp(1j * th).mean()))
        d = th[None, :] - th[:, None]; w = np.exp(beta * (np.cos(d) - 1))
        th = th + dt * (w * np.sin(d)).sum(1) / w.sum(1)
    return np.array(r)


def gpt2_order():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from common.prompts import PROMPTS
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained("gpt2")
    def rs(m):
        R = []
        for i in range(20):
            ids = tok(" ".join(PROMPTS[(i + j) % len(PROMPTS)] for j in range(3)), return_tensors="pt").input_ids
            with torch.no_grad():
                hs = m(ids, output_hidden_states=True).hidden_states[:-1]  # residual stream entering each block
            R.append([float((h[0, 1:] / h[0, 1:].norm(dim=1, keepdim=True)).mean(0).norm()) for h in hs])
        return np.array(R)
    trained = rs(AutoModelForCausalLM.from_pretrained("gpt2").eval())
    torch.manual_seed(0)
    rand = rs(AutoModelForCausalLM.from_config(AutoConfig.from_pretrained("gpt2")).eval())
    return trained, rand


snaps, rk, dt = kuramoto(); ra = attention_circle(); tr, rd = gpt2_order()
out["depth"] = dict(kuramoto_r_end=float(rk[-1]), attention_r_end=float(ra[-1]), gpt2_trained=tr.mean(0).tolist(), gpt2_random=rd.mean(0).tolist())
fig = plt.figure(figsize=(7.2, 2.2)); gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1.25], wspace=0.45)
a0 = fig.add_subplot(gs[0]); a0.set_aspect("equal"); a0.axis("off")
for rad, (t, th) in zip([1.0, 0.72, 0.44], snaps.items()):
    a0.add_patch(plt.Circle((0, 0), rad, fill=False, lw=0.4, color="0.75"))
    a0.scatter(rad * np.cos(th), rad * np.sin(th), s=5, color=plt.cm.Blues(0.4 + 0.5 * (1 - rad)), label=f"$\\tau={t}$")
a0.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, handletextpad=0.1, columnspacing=0.5)
a0.set_title("(a) Kuramoto oscillators\n(phases on the circle)", fontsize=7.5, loc="left")
a1 = fig.add_subplot(gs[1]); a1.plot(np.arange(len(rk)) * dt, rk, color=MECH, lw=1.2)
a1.set_xlabel("time $\\tau$"); a1.set_ylabel("order parameter $\\rho$"); a1.set_ylim(0, 1.02)
a1.set_title("(b) synchronization\nover time", fontsize=7.5, loc="left")
a2 = fig.add_subplot(gs[2]); a2.plot(ra, color=LEARN, lw=1.2); a2.set_xlabel("layer $\\ell$"); a2.set_ylim(0, 1.02)
a2.set_title("(c) idealized attention:\ntokens cluster over depth", fontsize=7.5, loc="left")
a3 = fig.add_subplot(gs[3])
for R, lab, ls in ((rd, "GPT-2, random init.", "--"), (tr, "GPT-2, trained", "-")):
    m, s = R.mean(0), R.std(0); x = np.arange(len(m))
    a3.plot(x, m, ls, color=LEARN, lw=1.2, label=lab); a3.fill_between(x, m - s, m + s, color=LEARN, alpha=0.15, lw=0)
a3.set_xlabel("layer $\\ell$"); a3.set_ylim(0, 1.02); a3.legend(fontsize=6, frameon=False, loc="lower right")
a3.set_title("(d) GPT-2 measured:\ntraining resists collapse", fontsize=7.5, loc="left")
fig.savefig("paper/figs/pair_depth.pdf", dpi=300, bbox_inches="tight"); fig.savefig("paper/figs/pair_depth.png", dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------- training: period two at the edge of stability
def gd_double_well(eta, n=600, u=0.9):
    us = []
    for _ in range(n):
        u = u - eta * (u ** 3 - u); us.append(u)
    return np.array(us)


def catapult(eta=0.95, n=60, seed=0):
    """Two-parameter model f = u v fit to y = 1 with loss (uv - 1)^2 / 2 (Lewkowycz et al. 2020)."""
    rng = np.random.default_rng(seed); u, v = 1.6 + 0.05 * rng.standard_normal(), 0.4
    L, S = [], []
    for _ in range(n):
        e = u * v - 1; L.append(0.5 * e * e); S.append(u * u + v * v)  # sharpness at the minimum manifold
        u, v = u - eta * e * v, v - eta * e * u
    return np.array(L), np.array(S)


fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.6), gridspec_kw=dict(hspace=0.75, wspace=0.4))
ex = {}
for j, eta in enumerate([1.05, 1.27]):  # lambda = U''(1) = 2: eta*lambda = 2.1 (period 2), 2.54 (period 4)
    u = gd_double_well(eta)[-400:]; f, P = welch(u - u.mean(), nperseg=128)
    ax[0, j].plot(u[-40:], "o-", ms=2, lw=0.7, color=MECH); ax[0, j].set_xlabel("step $t$"); ax[0, j].set_ylabel("$u_t$")
    ax[0, j].set_title(f"({'ab'[j]}) GD on a double well,\n$\\eta\\lambda={2 * eta:.2f}$", fontsize=7.5, loc="left")
    ax[1, j].semilogy(f[1:], P[1:] + 1e-30, color=MECH, lw=0.9); ax[1, j].set_xlabel("frequency (1/step)"); ax[1, j].set_ylabel("power (log)")
    ax[1, j].set_title(f"({'de'[j]}) spectrum of ({'ab'[j]})", fontsize=7.5, loc="left")
    ax[1, j].set_yticks([]); ax[1, j].minorticks_off(); ax[1, j].set_ylim(bottom=max(1e-12, P[1:].max() * 1e-12))
d = json.load(open("results/exp2.json"))
run = min((r for r in d["runs"] if not r["diverged"]), key=lambda r: abs(r["eta"] - 0.137))
t = np.array(run["tail"]); n = np.arange(len(t)); dtr = t - np.polyval(np.polyfit(n, t, 5), n)
f, P = welch(dtr, nperseg=128)
ax[0, 2].plot(1e5 * dtr[-40:], "o-", ms=2, lw=0.7, color=LEARN); ax[0, 2].set_xlabel("step $t$"); ax[0, 2].set_ylabel("loss $-$ trend ($10^{-5}$)")
ax[0, 2].set_title(f"(c) transformer at the edge of\nstability, $\\eta={run['eta']:.3f}$", fontsize=7.5, loc="left")
ax[1, 2].semilogy(f[1:], P[1:], color=LEARN, lw=0.9); ax[1, 2].set_xlabel("frequency (1/step)"); ax[1, 2].set_ylabel("power (log)")
ax[1, 2].set_title("(f) spectrum of (c)", fontsize=7.5, loc="left")
ax[1, 2].set_yticks([]); ax[1, 2].minorticks_off()
for a_ in ax[1]:
    a_.axvline(0.5, color="0.6", ls=":", lw=0.8)
out["training"] = dict(eta_transformer=run["eta"], R=run["sharp"] * run["eta"] / 2, peak_freq=float(f[np.argmax(P[1:]) + 1]),
                       share_above_045=float(P[f > 0.45].sum() / P[1:].sum()))
fig.savefig("paper/figs/pair_training.pdf", dpi=300, bbox_inches="tight"); fig.savefig("paper/figs/pair_training.png", dpi=200, bbox_inches="tight")

L, S = catapult()
fig, ax = plt.subplots(1, 2, figsize=(4.6, 1.8), gridspec_kw=dict(wspace=0.45))
ax[0].semilogy(L + 1e-16, color=LEARN, lw=1.1); ax[0].set_xlabel("step $t$"); ax[0].set_ylabel("loss")
ax[0].set_title("(a) loss spike", fontsize=7.5, loc="left")
ax[1].plot(S, color=LEARN, lw=1.1); ax[1].axhline(2 / 0.95, color="0.5", ls=":", lw=0.8); ax[1].set_xlabel("step $t$"); ax[1].set_ylabel("curvature $w_1^2+w_2^2$")
ax[1].text(len(S) * 0.98, 2 / 0.95 * 1.03, "$2/\\eta$", ha="right", va="bottom", fontsize=7, color="0.4")
ax[1].set_title("(b) curvature drops below $2/\\eta$", fontsize=7.5, loc="left")
out["catapult"] = dict(loss_peak=float(L.max()), loss_start=float(L[0]), curv_start=float(S[0]), curv_end=float(S[-1]))
fig.savefig("paper/figs/catapult.pdf", dpi=300, bbox_inches="tight"); fig.savefig("paper/figs/catapult.png", dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------- escape: Arrhenius plots for a particle and for a loop
def kramers_times(D, n=400, dt=0.01, tmax=4e4, seed=0):
    rng = np.random.default_rng(seed); x = -np.ones(n); t = np.zeros(n); done = np.zeros(n, bool); s = np.sqrt(2 * D * dt)
    T = 0.0
    while not done.all() and T < tmax:
        x = x - (x ** 3 - x) * dt + s * rng.standard_normal(n); T += dt
        hit = (~done) & (x > 1.0); t[hit] = T; done |= hit
    return t[done], done.mean()


Ds = [0.05, 0.06, 0.07, 0.085, 0.1, 0.12]; rate = []
for D in Ds:
    tt, frac = kramers_times(D); rate.append(1 / tt.mean())
rate = np.array(rate); slope = np.polyfit(1 / np.array(Ds), np.log(rate), 1)
g = json.load(open("results/exp3_gpt2.json")); temps = np.array(g["temps"])
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.2), gridspec_kw=dict(wspace=0.4))
ax[0].semilogy(1 / np.array(Ds), rate, "o", ms=3.5, color=MECH, label="simulation")
xx = np.linspace(1 / Ds[-1], 1 / Ds[0], 50)
ax[0].semilogy(xx, np.sqrt(2) / (2 * np.pi) * np.exp(-0.25 * xx), "-", color="0.4", lw=0.9, label="Kramers, $\\Delta U=1/4$")
ax[0].set_xlabel("$1/D$ (inverse noise)"); ax[0].set_ylabel("escape rate"); ax[0].legend(fontsize=6, frameon=False)
ax[0].set_title("(a) Brownian particle in a double well", fontsize=7.5, loc="left")
fits = []
for i, l in enumerate(g["loops"][:12]):
    r = np.array(l["cyc_escape"][-1]); ok = (r > 1e-12) & (r < 0.5)
    if ok.sum() < 4:
        continue
    gap = min(l["cyc_gap"][-1:])
    ax[1].semilogy(1 / temps[ok], r[ok], "-", lw=0.8, color=LEARN, alpha=0.6)
    fits.append((gap, -np.polyfit(1 / temps[ok], np.log(r[ok]), 1)[0]))
ax[1].set_xlabel("$1/T$ (inverse temperature)"); ax[1].set_ylabel("escape probability per cycle")
ax[1].set_title("(b) GPT-2 repetition loops", fontsize=7.5, loc="left")
rat = [b / a for a, b in fits]
ax[1].text(0.97, 0.97, f"{len(fits)} loops\nslope / logit gap\n= {min(rat):.2f} to {max(rat):.2f}", transform=ax[1].transAxes, ha="right", va="top", fontsize=6.5)
out["escape"] = dict(kramers_fit_barrier=float(-slope[0]), loops_gap_vs_slope=fits)
fig.savefig("paper/figs/pair_escape.pdf", dpi=300, bbox_inches="tight"); fig.savefig("paper/figs/pair_escape.png", dpi=200, bbox_inches="tight")
json.dump(out, open("results/pairs_summary.json", "w"), indent=1)
print(json.dumps({k: (v if k != "escape" else {"kramers_fit_barrier": v["kramers_fit_barrier"], "n_loops": len(v["loops_gap_vs_slope"]), "gap_slope": [(round(a, 2), round(b, 2)) for a, b in v["loops_gap_vs_slope"]]}) for k, v in out.items()}, indent=0)[:2000])
