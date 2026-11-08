"""Primer figure: four textbook behaviors of nonlinear dynamics and where they appear on the three clocks."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.15))

# (a) gradient descent on a double well: period doubling, chaos, escape between wells, divergence
etas = np.linspace(0.6, 2.2, 1400)
for e in etas:
    x = 0.9; div = False
    for _ in range(600):
        x = x - e * (x ** 3 - x)
        if abs(x) > 1e3:
            div = True; break
    if div:
        ax[0].axvspan(e - 0.0006, e + 0.0006, color="#f2c4c4", lw=0)
        continue
    xs = []
    for _ in range(150):
        x = x - e * (x ** 3 - x); xs.append(x)
    ax[0].plot([e] * len(xs), xs, ",", color="#2457c5", alpha=0.6)
ax[0].axvline(1.0, color="#c25e14", ls="--", lw=0.8)
ax[0].text(1.02, 1.55, r"$\eta\lambda=2$", color="#c25e14", fontsize=7)
ax[0].set_xlabel(r"step size $\eta$"); ax[0].set_ylabel("iterate $u$")
ax[0].set_title("(a) step size as bifurcation\nparameter (clock 2)", loc="left", fontsize=8)
ax[0].set_ylim(-1.8, 1.8)

# (b) attention dynamics on the circle: clustering over depth
rng = np.random.default_rng(3); th0 = rng.uniform(0, 2 * np.pi, 40); beta = 9.0
def step(th, dt=0.3):
    d = th[None, :] - th[:, None]; w = np.exp(beta * (np.cos(d) - 1))
    return th + dt * (w * np.sin(d)).sum(1) / w.sum(1)
snaps, th = {}, th0.copy()
for l in range(201):
    if l in (0, 12, 200):
        snaps[l] = th.copy()
    th = step(th)
for r, (l, s) in zip([1.0, 0.75, 0.5], snaps.items()):
    ax[1].scatter(r * np.cos(s), r * np.sin(s), s=6, color=plt.cm.viridis(1 - r), label=f"layer {l}")
for r in (1.0, 0.75, 0.5):
    ax[1].add_patch(plt.Circle((0, 0), r, fill=False, lw=0.4, color="0.75"))
ax[1].set_aspect("equal"); ax[1].axis("off")
ax[1].legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, handletextpad=0.1, columnspacing=0.6)
ax[1].set_title("(b) synchronization: tokens\ncluster over depth (clock 1)", loc="left", fontsize=8)

# (c) Lyapunov exponent: two nearby orbits of the logistic map separate exponentially
r = 3.9; x, y = 0.3, 0.3 + 1e-9; dist = []
for n in range(60):
    x, y = r * x * (1 - x), r * y * (1 - y); dist.append(abs(x - y) + 1e-16)
ax[2].semilogy(dist, "-", color="#2457c5", lw=1)
n = np.arange(25); lam = np.polyfit(n[2:], np.log(dist[2:25]), 1)[0]
ax[2].semilogy(n, 1e-9 * np.exp(lam * n), ":", color="#c25e14", lw=1)
ax[2].text(26, 1e-6, rf"$e^{{\lambda n}},\ \lambda\approx{lam:.2f}$", color="#c25e14", fontsize=7)
ax[2].set_xlabel("step $n$"); ax[2].set_ylabel("separation")
ax[2].set_title("(c) sensitivity: Lyapunov\nexponent (clocks 1 and 3)", loc="left", fontsize=8)

# (d) noise-driven escape over a barrier; the logit gap plays the barrier on clock 3
u = np.linspace(-1.55, 1.55, 300); U = (u ** 2 - 1) ** 2 / 4
ax[3].plot(u, U, color="k", lw=1.2)
ax[3].annotate("", xy=(0, 0.25), xytext=(0, 0.0), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#c25e14"))
ax[3].plot([-1, 0], [0, 0], ":", color="0.5", lw=0.7)
ax[3].text(0.06, 0.1, r"$\Delta U$", color="#c25e14", fontsize=8)
ax[3].scatter([-1], [0.02], s=25, color="#2457c5", zorder=3)
ax[3].text(-1.5, 0.80, r"rate $\propto e^{-\Delta U/D}$", fontsize=7)
ax[3].text(-1.5, 0.71, "loop: $\\Delta U$ = logit gap,\n         $D$ = temperature", fontsize=6.5, color="#2457c5", va="top")
ax[3].set_ylim(-0.03, 0.9); ax[3].set_xlim(-1.6, 1.6); ax[3].set_yticks([]); ax[3].set_xticks([]); ax[3].set_xlabel("state")
ax[3].set_title("(d) noise-driven escape\n(clocks 2 and 3)", loc="left", fontsize=8)
fig.tight_layout(w_pad=0.8)
fig.savefig("paper/figs/primer_tools.pdf", dpi=300); fig.savefig("paper/figs/primer_tools.png", dpi=200)
print("lambda", lam)
