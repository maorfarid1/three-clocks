"""Regime atlas: the same four pictures for a noisy mechanical oscillator and for a language model.

Mechanical system: forced double-well Duffing oscillator with additive white noise of intensity sigma,
  x'' + delta x' - x + x^3 = gamma cos(omega t) + sigma xi(t),
integrated with Euler-Maruyama. Three noise levels give a periodic orbit in one well, noisy orbits with
hops between the wells, and noise-dominated motion.

Language model: GPT-2 sampling at three temperatures (loop, coherent text, incoherent text). The scalar time
history is the surprisal -log p(x_t) of each generated token under the model at T = 1; the state is the
final-layer hidden state of the last token.

For both: (a) time history, (b) power spectrum (Welch), (c) phase portrait ((x, x') for Duffing, the first two
principal coordinates of the hidden state for the LM; the LM time history and spectrum use the first coordinate), (d) recurrence plot R_ij = 1 if ||s_i - s_j|| < eps, with
eps set so that a fixed fraction (--rr, 5%) of pairs recur in each run (fixed recurrence rate, standardized states); RR and DET
with one eps per system (10% quantile of the middle regime) are stored too.
Quantified by recurrence rate (RR), determinism (DET: fraction of recurrent points on diagonal lines of length
>= 2) and spectral entropy (normalized Shannon entropy of the power spectrum).

Usage: python exp3_escape/atlas.py --out results/atlas.json
"""
import argparse, json, os, sys
import numpy as np
import torch
from scipy.signal import welch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS
from exp3_escape.escape import find_loop


def duffing(sigma, n=512, every=40, dt=0.025, delta=0.25, gamma=0.10, omega=1.2, seed=0, burn=4000):
    rng = np.random.default_rng(seed)
    x, v, t = 1.0, 0.0, 0.0
    out = []
    for i in range(burn * every // 10 + n * every):
        a = -delta * v + x - x ** 3 + gamma * np.cos(omega * t)
        v += a * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        x += v * dt; t += dt
        if i >= burn * every // 10 and (i - burn * every // 10) % every == 0:
            out.append((x, v))
    return np.array(out[:n])


@torch.no_grad()
def lm_run(model, tok, prompt, T, gen, seed):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model(ids, use_cache=True)
    past, z = out.past_key_values, out.logits[0, -1]
    toks, surp, hs = [], [], []
    for _ in range(gen):
        p1 = torch.log_softmax(z.double(), -1)
        nxt = int(torch.multinomial(torch.softmax(z.double() / T, -1), 1)) if T > 0 else int(torch.argmax(z))
        toks.append(nxt); surp.append(float(-p1[nxt]))
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True, output_hidden_states=True)
        past, z = out.past_key_values, out.logits[0, -1]
        hs.append(out.hidden_states[-1][0, -1].float().numpy())
    return toks, np.array(surp), np.array(hs)


def rqa(S, eps, lmin=2):
    D = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=-1)
    R = (D < eps).astype(np.uint8)
    n = len(R); np.fill_diagonal(R, 0)
    rr = R.sum() / (n * n - n)
    on_lines = 0
    for k in range(1, n):
        d = np.diagonal(R, k); run = 0
        for v in list(d) + [0]:
            if v: run += 1
            else:
                if run >= lmin: on_lines += run
                run = 0
    det = 2 * on_lines / max(1, R.sum())
    return R, float(rr), float(det)


def spec(y, fs=1.0):
    f, P = welch(y - y.mean(), fs=fs, nperseg=min(256, len(y)))
    q = P[1:] / P[1:].sum()
    H = float(-(q * np.log(q + 1e-300)).sum() / np.log(len(q)))
    return f, P, H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--temps", default="0.3,0.9,1.6")
    ap.add_argument("--sigmas", default="0.0,0.25,1.2")
    ap.add_argument("--prompt", type=int, default=4)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rr", type=float, default=0.05, help="fixed recurrence rate")
    ap.add_argument("--skip", type=int, default=64, help="transient tokens dropped before the analysis")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    res = {"duffing": [], "lm": []}
    # mechanical
    D = [duffing(float(s), n=a.gen, seed=a.seed) for s in a.sigmas.split(",")]
    sc = D[1].std(0)
    eps = np.quantile(np.linalg.norm((D[1] / sc)[:, None] - (D[1] / sc)[None], axis=-1), a.rr)
    for s, X in zip(a.sigmas.split(","), D):
        R, rr, det = rqa(X / sc, eps); f, P, H = spec(X[:, 0])
        Xs = (X - X.mean(0)) / X.std(0); e2 = np.quantile(np.linalg.norm(Xs[:, None] - Xs[None], axis=-1), a.rr)
        Rf, _, det_fixed = rqa(Xs, e2)
        res["duffing"].append(dict(sigma=float(s), x=X[:, 0].tolist(), v=X[:, 1].tolist(), f=f.tolist(), P=P.tolist(),
                                   RR=rr, DET=det, DET_fixedRR=det_fixed, Hspec=H, R=np.packbits(Rf).tolist(), n=len(Rf)))
    # language model
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model).eval()
    runs = [lm_run(model, tok, PROMPTS[a.prompt], float(T), a.gen, a.seed) for T in a.temps.split(",")]
    for T, (toks, s, h) in zip(a.temps.split(","), runs):
        h = h[a.skip:] / np.linalg.norm(h[a.skip:], axis=1, keepdims=True)  # unit norm: removes GPT-2's outlier scale
        hc = (h - h.mean(0)) / (h.std(0) + 1e-6)  # z-score each feature: keeps a few outlier dimensions from dominating
        hc = np.clip(hc, -4, 4); U, Sv, Vt = np.linalg.svd(hc, full_matrices=False)
        z = hc @ Vt[:20].T  # per-run principal coordinates
        zs = z / z[:, :1].std()
        e2 = np.quantile(np.linalg.norm(zs[:, None] - zs[None], axis=-1), a.rr)
        Rf, rr, det_fixed = rqa(zs, e2)
        f, P, H = spec(z[:, 0]); fS, PS, HS = spec(s[a.skip:])
        k, onset = find_loop(toks, min_reps=3)
        res["lm"].append(dict(T=float(T), surprisal=s.tolist(), pc=(z[:, :2] / z[:, :1].std()).tolist(), f=f.tolist(), P=P.tolist(),
                              Hspec=H, Hspec_surprisal=HS, DET_fixedRR=det_fixed, var_pc12=float((Sv[:2] ** 2).sum() / (Sv ** 2).sum()),
                              mean_surprisal=float(s[a.skip:].mean()), loop=k, onset=onset, R=np.packbits(Rf).tolist(), n=len(Rf),
                              text=tok.decode(toks[:80])))
    json.dump(dict(args=vars(a), **res), open(a.out, "w"))
    for key in ("duffing", "lm"):
        for r in res[key]:
            print(key, r.get("sigma", r.get("T")), {k: round(r[k], 3) for k in ("DET_fixedRR", "Hspec", "Hspec_surprisal", "mean_surprisal", "var_pc12") if k in r}, r.get("loop", ""), r.get("onset", ""))
