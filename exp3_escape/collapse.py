"""Regime diagram on one axis: determinism of the trajectory against barrier height over noise intensity.

Duffing: forced double-well oscillator x'' + delta x' - x + x^3 = gamma cos(omega t) + sigma xi(t). The well depth is
dU = 1/4 and the noise acts like a temperature D = sigma^2 / (2 delta) (the stationary density of the unforced
oscillator is proportional to exp(-(v^2/2 + U(x)) / D)). We sweep sigma and record DET at a fixed recurrence rate.
GPT-2: for prompts whose greedy run ends in a loop (Exp. 3), the barrier is the saturated logit gap Delta_inf of that
loop (saturating fit of the per-cycle gaps), and the noise intensity is the temperature T. We sample at a grid of T
and record DET of the hidden-state trajectory, measured exactly as in atlas.py.
If both systems are governed by the same balance between a barrier and noise, DET should fall with the same ratio,
barrier / noise, in both, with the transition near the value at which an escape becomes likely within the window.
Usage: python exp3_escape/collapse.py --out results/collapse.json
"""
import argparse, json, os, sys
import numpy as np
import torch
from scipy.optimize import curve_fit
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS
from exp3_escape.atlas import duffing, lm_run, rqa
from exp3_escape.atlas_stats import lm_state


def sat(c, D, a, tau):
    return D - (D - a) * np.exp(-(c - 1) / tau)


def det_fixed(S, rr):
    Ss = (S - S.mean(0)) / (S[:, :1].std() + 1e-9)
    eps = np.quantile(np.linalg.norm(Ss[:, None] - Ss[None], axis=-1), rr)
    return rqa(Ss, eps)[2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--temps", default="0.3,0.4,0.5,0.6,0.75,0.9,1.1,1.4")
    ap.add_argument("--sigmas", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.6,0.9,1.2")
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--skip", type=int, default=64)
    ap.add_argument("--rr", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = {"duffing": [], "lm": []}
    for s in map(float, a.sigmas.split(",")):
        for seed in range(5):
            X = duffing(s, n=a.gen, seed=seed)
            D = s * s / (2 * 0.25)
            out["duffing"].append(dict(sigma=s, seed=seed, ratio=0.25 / D, DET=det_fixed(X, a.rr)))
    g = json.load(open("results/exp3_gpt2.json"))
    loops = {}
    for l in g["loops"]:
        c = np.arange(1, len(l["cyc_gap"]) + 1); y = np.array(l["cyc_gap"])
        try:
            p, _ = curve_fit(sat, c, y, p0=[y.max(), y[0], 3.0], bounds=([-5, -5, 0.1], [50, 50, 1e3]), maxfev=20000)
            loops[l["prompt"]] = dict(Dinf=float(p[0]), k=l["k"])
        except Exception:
            pass
    tok = AutoTokenizer.from_pretrained("gpt2"); model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    for i in sorted(loops)[: a.n_prompts]:
        for T in map(float, a.temps.split(",")):
            toks, surp, h = lm_run(model, tok, PROMPTS[i], T, a.gen, 0)
            Z = lm_state(h, a.skip)
            out["lm"].append(dict(prompt=i, T=T, Dinf=loops[i]["Dinf"], k=loops[i]["k"], ratio=loops[i]["Dinf"] / T, DET=det_fixed(Z, a.rr)))
            print(i, T, round(loops[i]["Dinf"] / T, 2), round(out["lm"][-1]["DET"], 3), flush=True)
            json.dump(dict(args=vars(a), **out), open(a.out, "w"))
    json.dump(dict(args=vars(a), **out), open(a.out, "w"))
