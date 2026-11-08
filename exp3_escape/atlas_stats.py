"""Regime atlas statistics: determinism (fixed recurrence rate) and spectral entropy over many prompts and seeds.

Same measurements as atlas.py, for prompts 0..n-1 of the Exp. 3 prompt set (GPT-2) and seeds 0..n-1 (Duffing).
Usage: python exp3_escape/atlas_stats.py --n 20 --out results/atlas_stats.json
"""
import argparse, json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS
from exp3_escape.atlas import duffing, lm_run, rqa, spec
from exp3_escape.escape import find_loop


def lm_state(h, skip):
    h = h[skip:] / np.linalg.norm(h[skip:], axis=1, keepdims=True)
    hc = np.clip((h - h.mean(0)) / (h.std(0) + 1e-6), -4, 4)
    U, Sv, Vt = np.linalg.svd(hc, full_matrices=False)
    return hc @ Vt[:20].T


def measures(S, x, rr):
    Ss = (S - S.mean(0)) / (S[:, :1].std() + 1e-9)
    eps = np.quantile(np.linalg.norm(Ss[:, None] - Ss[None], axis=-1), rr)
    return rqa(Ss, eps)[2], spec(x)[2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temps", default="0.3,0.9,1.6")
    ap.add_argument("--sigmas", default="0.0,0.25,1.2")
    ap.add_argument("--gen", type=int, default=576)
    ap.add_argument("--skip", type=int, default=64)
    ap.add_argument("--rr", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = {"duffing": {}, "lm": {}}
    for s in a.sigmas.split(","):
        rows = []
        for seed in range(a.n):
            X = duffing(float(s), n=a.gen, seed=seed)
            Xs = (X - X.mean(0)) / X.std(0)
            eps = np.quantile(np.linalg.norm(Xs[:, None] - Xs[None], axis=-1), a.rr)
            rows.append((rqa(Xs, eps)[2], spec(X[:, 0])[2]))
        out["duffing"][s] = rows
    tok = AutoTokenizer.from_pretrained("gpt2"); model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    for T in a.temps.split(","):
        rows = []
        for i in range(a.n):
            toks, surp, h = lm_run(model, tok, PROMPTS[i], float(T), a.gen, 0)
            Z = lm_state(h, a.skip)
            det, hs = measures(Z, Z[:, 0], a.rr)
            rows.append((det, hs, find_loop(toks, min_reps=3)[0], float(np.mean(surp[a.skip:]))))
        out["lm"][T] = rows
        print(T, "DET median %.2f, Hspec median %.2f, loops %d" % (np.median([r[0] for r in rows]), np.median([r[1] for r in rows]), sum(r[2] > 0 for r in rows)), flush=True)
        json.dump(dict(args=vars(a), **out), open(a.out, "w"))
    for s, rows in out["duffing"].items():
        print("duffing", s, "DET median %.2f, Hspec median %.2f" % (np.median([r[0] for r in rows]), np.median([r[1] for r in rows])))
    json.dump(dict(args=vars(a), **out), open(a.out, "w"))
