"""Summary of Exp. 4: fraction of good generations (no loop, rep4 and reference score within human range)."""
import json, sys, os
import numpy as np
H = json.load(open("results/exp3b_human.json"))
NLL_T = float(np.percentile(H["ref_nll"], 90)); REP_T = float(np.percentile(H["rep4"], 95))

def score(rows):
    loop = np.mean([r["loop"] > 0 for r in rows])
    rep = np.mean([r["loop"] == 0 and r["rep4"] > REP_T for r in rows])
    inc = np.mean([r["loop"] == 0 and r["rep4"] <= REP_T and r["ref_nll"] > NLL_T for r in rows])
    good = 1 - loop - rep - inc
    return dict(loop=float(loop), repetitive=float(rep), incoherent=float(inc), good=float(good),
                nll=float(np.median([r["ref_nll"] for r in rows])), rep4=float(np.median([r["rep4"] for r in rows])), n=len(rows))

out = {"thresholds": dict(nll=NLL_T, rep4=REP_T)}
for m in ("gpt2", "pythia160m"):
    res = {}
    fb = f"results/exp3b_{m}.json"
    if os.path.exists(fb):
        rows = json.load(open(fb))["rows"]
        for t in sorted({r["T"] for r in rows}):
            res[f"temp T={t}"] = score([r for r in rows if r["T"] == t])
    f4 = f"results/exp4_{m}.json"
    if os.path.exists(f4):
        for c in json.load(open(f4))["configs"]:
            cfg = c["cfg"]; lab = {"age": f"AGE T0={cfg['T0']} q={cfg.get('q')}", "penalty": f"penalty {cfg.get('penalty')} T={cfg['T0']}",
                                   "ngram": f"no-repeat-4gram T={cfg['T0']}"}[cfg["method"]]
            res[lab] = score(c["rows"])
    out[m] = res
    print(m)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["good"]):
        print(f"  {k:28s} good={v['good']:.2f} loop={v['loop']:.2f} rep={v['repetitive']:.2f} inc={v['incoherent']:.2f} nll={v['nll']:.2f}")
json.dump(out, open("results/exp4_summary.json", "w"), indent=1)
