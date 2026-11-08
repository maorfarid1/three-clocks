"""Experiment 4c: holding generation in the coherent window by feedback on two order parameters.

The regime map (Exp. 3b) shows that coherent text lives in a narrow band of temperatures whose position differs
from prompt to prompt, so a single fixed temperature cannot keep every prompt inside it. The standard remedy for
a system that must stay near a moving regime boundary is feedback. We measure two quantities over a sliding window
of the last W generated tokens: the recurrence of the text, rho = fraction of 4-grams in the window that already
occurred earlier in the generation (the token-level analogue of the recurrence rate of Section 2), and the mean
surprisal s = -log p(y) of the window under the model at T = 1. Recurrence above rho* means the trajectory is
falling into a repetitive orbit and the temperature is raised; surprisal above s* means it is leaving the coherent
regime and the temperature is lowered:
    T <- clip(T + k_r (rho - rho*) - k_s max(0, s - s*), Tmin, Tmax).
Plain temperature sampling (no top-k or top-p) otherwise. Constant-temperature baselines use the same loop and seeds.

Constant-temperature baselines are taken from Exp. 3b (same prompts, samples, length and scorer); see score_control.py.
Usage: python exp3_escape/regime_control.py --only "feedback T0=0.4,feedback T0=0.6,feedback T0=0.8" --out results/exp4c_gpt2.json
       python exp3_escape/score_control.py results/exp4c_gpt2.json.raw results/exp4c_summary.json
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS
from exp3_escape.escape import find_loop
from exp3_escape.regime import load_ref, ref_nll, rep_n


@torch.no_grad()
def generate(model, prompt_ids, gen, seed, ctrl):
    g = torch.Generator().manual_seed(seed)
    out = model(prompt_ids, use_cache=True); past, z = out.past_key_values, out.logits[0, -1]
    seq, surp, Ts, seen = [], [], [], set()
    T = ctrl["T0"]; rep_flags = []
    for n in range(gen):
        lp = torch.log_softmax(z.double(), -1)
        nxt = int(torch.multinomial(torch.softmax(z.double() / T, -1), 1, generator=g))
        seq.append(nxt); surp.append(float(-lp[nxt])); Ts.append(T)
        if len(seq) >= 4:
            gram = tuple(seq[-4:]); rep_flags.append(gram in seen); seen.add(gram)
        if ctrl.get("feedback") and len(seq) >= ctrl["W"]:
            W = ctrl["W"]
            rho = float(np.mean(rep_flags[-(W - 3):])); s = float(np.mean(surp[-W:]))
            T = float(np.clip(T + ctrl["kr"] * (rho - ctrl["rho"]) - ctrl["ks"] * max(0.0, s - ctrl["s"]), ctrl["Tmin"], ctrl["Tmax"]))
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True); past, z = out.past_key_values, out.logits[0, -1]
    return seq, Ts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--only", default="", help="comma-separated config names to run (default: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    base = dict(W=32, kr=0.05, ks=0.01, rho=0.02, s=3.0, Tmin=0.4, Tmax=1.1)
    configs = [dict(name="feedback T0=0.6", feedback=True, T0=0.6, **base),
               dict(name="feedback T0=0.4", feedback=True, T0=0.4, **base),
               dict(name="feedback T0=0.8", feedback=True, T0=0.8, **base),
               dict(name="constant T=0.7", T0=0.7), dict(name="constant T=0.8", T0=0.8), dict(name="constant T=0.9", T0=0.9)]
    if a.only:
        configs = [c for c in configs if c["name"] in a.only.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model).eval()
    prompts = PROMPTS[:a.n_prompts]
    gens = []
    for cfg in configs:
        t0 = time.time(); rows = []
        for s in range(a.samples):
            for i, p in enumerate(prompts):
                seq, Ts = generate(model, tok(p, return_tensors="pt").input_ids, a.gen, 1000 * s + 7 + i, cfg)
                rows.append(dict(prompt=i, sample=s, seq=seq, Ts=Ts))
        gens.append((cfg, rows))
        print(cfg["name"], f"generated in {time.time() - t0:.0f}s", flush=True)
        json.dump(dict(args=vars(a), raw=[dict(cfg=c, rows=r) for c, r in gens]), open(a.out + ".raw", "w"))
    del model
    ref, rtok = load_ref()
    res = []
    for cfg, rows in gens:
        conts = [tok.decode(r["seq"]) for r in rows]
        nll = ref_nll(ref, rtok, [prompts[r["prompt"]] for r in rows], conts)
        out_rows = [dict(prompt=r["prompt"], sample=r["sample"], loop=find_loop(r["seq"], min_reps=3)[0], rep4=rep_n(r["seq"]),
                         ref_nll=nll[j], T_mean=float(np.mean(r["Ts"])), T_final=r["Ts"][-1], text=conts[j][:300] if r["sample"] == 0 else None)
                    for j, r in enumerate(rows)]
        res.append(dict(cfg=cfg, rows=out_rows))
    json.dump(dict(args=vars(a), configs=res), open(a.out, "w"))
    H = json.load(open("results/exp3b_human.json"))
    NLL_T = float(np.percentile(H["ref_nll"], 90)); REP_T = float(np.percentile(H["rep4"], 95))
    for c in res:
        R = c["rows"]
        loop = np.mean([r["loop"] > 0 for r in R]); rep = np.mean([r["loop"] == 0 and r["rep4"] > REP_T for r in R])
        inc = np.mean([r["loop"] == 0 and r["rep4"] <= REP_T and r["ref_nll"] > NLL_T for r in R])
        print(f"{c['cfg']['name']:18s} good={1 - loop - rep - inc:.3f} loop={loop:.3f} rep={rep:.3f} inc={inc:.3f} "
              f"T_mean={np.median([r['T_mean'] for r in R]):.2f}")
