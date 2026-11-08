"""Experiment 3c: repetition loops as orbits in hidden-state space.

Greedy decoding as in Exp. 3, recording the final-layer hidden state of the last token at
every step. For each loop of period k starting at `onset`, the return distance of cycle c is
  d_c = mean_j ||h[onset + c k + j] - h[onset + (c-1) k + j]|| / mean ||h||   (c >= 1),
the distance between the state and its value one period earlier (a Poincare return map).
If the loop converges to a periodic orbit, d_c -> 0; we fit d_c ~ A rho^c and report rho.
Also stores a 2D PCA projection of the states for one example loop.

Usage: python exp3_escape/orbit.py --model gpt2 --out results/exp3c_gpt2.json
"""
import argparse, json, os, sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS
from exp3_escape.escape import find_loop


@torch.no_grad()
def greedy_states(model, ids, gen):
    out = model(ids, use_cache=True, output_hidden_states=True)
    past, logits = out.past_key_values, out.logits[0, -1]
    toks, hs = [], []
    for _ in range(gen):
        nxt = int(torch.argmax(logits))
        toks.append(nxt)
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True, output_hidden_states=True)
        past, logits = out.past_key_values, out.logits[0, -1]
        hs.append(out.hidden_states[-1][0, -1].float().numpy())  # state after reading token nxt
    return toks, np.array(hs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--gen", type=int, default=320)
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--example", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model).eval()
    loops, example = [], None
    for i, p in enumerate(PROMPTS[:a.n_prompts]):
        ids = tok(p, return_tensors="pt").input_ids
        toks, H = greedy_states(model, ids, a.gen)
        k, onset = find_loop(toks, min_reps=3)
        if not k:
            continue
        ncyc = (len(toks) - onset) // k
        scale = float(np.linalg.norm(H[onset:], axis=1).mean())
        d = []
        for c in range(1, ncyc):
            A = H[onset + c * k: onset + (c + 1) * k]
            B = H[onset + (c - 1) * k: onset + c * k]
            d.append(float(np.linalg.norm(A - B, axis=1).mean() / scale))
        rec = dict(prompt=i, k=k, onset=onset, ncyc=ncyc, d=d)
        good = [(c, x) for c, x in enumerate(d, 1) if x > 0]
        if len(good) >= 4:
            c, x = np.array(good).T
            rec["rho"] = float(np.exp(np.polyfit(c, np.log(x), 1)[0]))
        loops.append(rec)
        print(i, "k", k, "onset", onset, "ncyc", ncyc, "d", [round(x, 4) for x in d[:6]], "rho", rec.get("rho"), flush=True)
        if i == a.example:
            seg = H[max(0, onset - 3 * k):]
            Z = (seg - seg.mean(0)) / (seg.std(0) + 1e-6)   # standardize: a few huge dimensions dominate otherwise
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            P = Z @ Vt[:2].T
            example = dict(prompt=i, k=k, onset=onset, start=max(0, onset - 3 * k), pc=P.tolist(),
                           var=(S[:2] ** 2 / (S ** 2).sum()).tolist(),
                           text=tok.decode(toks[onset:onset + k]))
    json.dump(dict(args=vars(a), loops=loops, example=example), open(a.out, "w"))
