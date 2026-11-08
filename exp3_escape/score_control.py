"""Score the generations of regime_control.py (raw file) with the four-class criteria of Exp. 3b and compare with
constant temperature from Exp. 3b (same 40 prompts, 3 samples, 200 tokens, same scorer and thresholds).
Usage: python exp3_escape/score_control.py results/exp4c_gpt2.json.raw results/exp4c_gpt2.json
"""
import json, os, sys
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS
from exp3_escape.escape import find_loop
from exp3_escape.regime import load_ref, ref_nll, rep_n

H = json.load(open("results/exp3b_human.json"))
NLL_T = float(np.percentile(H["ref_nll"], 90)); REP_T = float(np.percentile(H["rep4"], 95))


def classes(R):
    loop = np.mean([r["loop"] > 0 for r in R]); rep = np.mean([r["loop"] == 0 and r["rep4"] > REP_T for r in R])
    inc = np.mean([r["loop"] == 0 and r["rep4"] <= REP_T and r["ref_nll"] > NLL_T for r in R])
    good = [r["loop"] == 0 and r["rep4"] <= REP_T and r["ref_nll"] <= NLL_T for r in R]
    per_prompt = [np.mean([g for g, r in zip(good, R) if r["prompt"] == p]) for p in sorted({r["prompt"] for r in R})]
    return dict(good=float(np.mean(good)), loop=float(loop), repetitive=float(rep), incoherent=float(inc), n=len(R),
                prompts_never_good=int(sum(v == 0 for v in per_prompt)), nll_median=float(np.median([r["ref_nll"] for r in R])))


if __name__ == "__main__":
    raw = json.load(open(sys.argv[1])); tok = AutoTokenizer.from_pretrained("gpt2")
    ref, rtok = load_ref(); prompts = PROMPTS[:raw["args"]["n_prompts"]]
    res = {}
    for c in raw["raw"]:
        if not c["cfg"].get("feedback"):
            continue
        rows = c["rows"]; conts = [tok.decode(r["seq"]) for r in rows]
        nll = ref_nll(ref, rtok, [prompts[r["prompt"]] for r in rows], conts)
        R = [dict(prompt=r["prompt"], sample=r["sample"], loop=find_loop(r["seq"], min_reps=3)[0], rep4=rep_n(r["seq"]), ref_nll=nll[j],
                  T_mean=float(np.mean(r["Ts"])), T_min=float(np.min(r["Ts"][32:])), T_max=float(np.max(r["Ts"])), text=conts[j][:400] if r["sample"] == 0 else None)
             for j, r in enumerate(rows)]
        res[c["cfg"]["name"]] = dict(cfg=c["cfg"], summary=classes(R), T_mean_median=float(np.median([r["T_mean"] for r in R])),
                                     T_mean_range=[float(np.percentile([r["T_mean"] for r in R], 10)), float(np.percentile([r["T_mean"] for r in R], 90))], rows=R)
        R2 = [r for r in R if r["prompt"] not in (0, 2, 4)]  # the three prompts used to set the gains
        res[c["cfg"]["name"]]["summary_heldout"] = classes(R2)
    base = json.load(open("results/exp3b_gpt2.json"))["rows"]
    for T in sorted({r["T"] for r in base}):
        R = [r for r in base if r["T"] == T]
        res[f"constant T={T}"] = dict(summary=classes(R), summary_heldout=classes([r for r in R if r["prompt"] not in (0, 2, 4)]))
    json.dump(dict(thresholds=dict(nll=NLL_T, rep4=REP_T), results=res), open(sys.argv[2], "w"))
    for k, v in res.items():
        s = v["summary"]; h = v["summary_heldout"]
        print(f"{k:20s} good={s['good']:.3f} (held-out {h['good']:.3f}) loop={s['loop']:.2f} rep={s['repetitive']:.2f} inc={s['incoherent']:.2f} "
              f"never-good prompts={s['prompts_never_good']}" + (f" T_mean={v['T_mean_median']:.2f} [{v['T_mean_range'][0]:.2f},{v['T_mean_range'][1]:.2f}]" if "T_mean_median" in v else ""))
