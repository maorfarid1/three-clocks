"""Exp. 3f analysis: loop barriers in a reasoning model and the vendor temperature window.

1. For every sampled trace that ends in a loop, re-run the model on prompt + trace once and read the logits
   along the loop: per-cycle barrier (smallest logit gap in the cycle) and exact per-cycle escape probability on
   the temperature grid of Exp. 3. From these we get the locking temperature T_1/2(N): the temperature at which
   half of the loops are still trapped after N further cycles, as in Exp. 3d/3e.
2. Figure: loop fraction, accuracy and locking temperature against T, with the vendor-recommended temperatures.

Usage: python exp3_escape/reasoning_plot.py
"""
import json, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp3_escape.reasoning import build, gsm8k, TEMPS

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
FILES = {"thinking": "results/exp3f_qwen3_think_sample.json", "non-thinking": "results/exp3f_qwen3_nothink_sample.json"}
VENDOR = {"thinking": 0.6, "non-thinking": 0.7}
COL = {"thinking": "#2457c5", "non-thinking": "#c25e14"}
BURN = 2


def t_half(temps, surv):
    for i in range(len(temps) - 1):
        if surv[i] >= 0.5 > surv[i + 1]:
            return float(np.exp(np.log(temps[i]) + (surv[i] - 0.5) / (surv[i] - surv[i + 1]) * (np.log(temps[i + 1]) - np.log(temps[i]))))
    return None


@torch.no_grad()
def loop_barriers(m, tok, prompt, ids, k, onset, temps):
    p = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    full = torch.tensor([p + ids])
    h = m.model(full).last_hidden_state[0]
    W = m.get_output_embeddings().weight
    n0 = len(p)
    ncyc = (len(ids) - onset) // k
    gaps, logps = [], []
    T = torch.tensor(temps, dtype=torch.float64)
    for t in range(onset, onset + ncyc * k):
        z = (h[n0 + t - 1] @ W.T).double()  # logits predicting ids[t]
        a = ids[t]
        za = float(z[a]); z2 = z.clone(); z2[a] = -1e30
        gaps.append(za - float(z2.max()))
        logps.append((za / T - torch.logsumexp(z[None, :] / T[:, None], 1)).tolist())
    gaps = np.array(gaps).reshape(ncyc, k); lp = np.array(logps).reshape(ncyc, k, len(temps))
    return gaps.min(1).tolist(), (1 - np.exp(lp.sum(1))).tolist(), bool(np.all(gaps > 0))


def main():
    summary = {}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    tok = m = probs = None
    for mode, f in FILES.items():
        if not os.path.exists(f):
            continue
        d = json.load(open(f)); rows = d["rows"]; args = d["args"]
        for r in rows:  # a loop must repeat over at least 30 tokens (rules out e.g. the zeros of "120,000" at the cut)
            if r["loop"] and r["n_tokens"] - r["onset"] < max(30, 3 * r["loop"]):
                r["loop"] = 0
        Ts = sorted({r["T"] for r in rows})
        loopf = [np.mean([r["loop"] > 0 for r in rows if r["T"] == t]) for t in Ts]
        acc = [np.mean([r["correct"] for r in rows if r["T"] == t]) for t in Ts]
        fin = [np.mean([r["finished"] for r in rows if r["T"] == t]) for t in Ts]
        n = [sum(1 for r in rows if r["T"] == t) for t in Ts]
        rep = [np.median([r["rep4"] for r in rows if r["T"] == t]) for t in Ts]
        rq = [np.percentile([r["rep4"] for r in rows if r["T"] == t], [25, 75]) for t in Ts]
        ax[0].plot(Ts, fin, "o-", ms=3, lw=1.2, color=COL[mode], label=mode)
        if any(loopf):
            ax[0].plot(Ts, loopf, "x--", ms=3, lw=0.9, color=COL[mode])
        ax[1].plot(Ts, acc, "o-", ms=3, lw=1.2, color=COL[mode], label=mode)
        ax[2].plot(Ts, rep, "o-", ms=3, lw=1.2, color=COL[mode], label=mode)
        ax[2].fill_between(Ts, [q[0] for q in rq], [q[1] for q in rq], color=COL[mode], alpha=0.15, lw=0)
        S = dict(T=Ts, n=n, loop=loopf, acc=acc, finished=fin, rep4_median=rep,
                 tokens_median=[float(np.median([r["n_tokens"] for r in rows if r["T"] == t])) for t in Ts], T_best_acc=Ts[int(np.argmax(acc))], vendor=VENDOR[mode])
        # barriers of looped traces
        looped = [r for r in rows if r["loop"] and r.get("ids")]
        if looped:
            if m is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(args["model"])
                m = AutoModelForCausalLM.from_pretrained(args["model"], dtype=torch.float32).eval()
                probs = gsm8k(max(r["prompt"] for r in rows) + 1)
            L = []
            for r in looped:
                cg, ce, ok = loop_barriers(m, tok, build(tok, probs[r["prompt"]][0], args["think"]), r["ids"], r["loop"], r["onset"], TEMPS)
                L.append(dict(T=r["T"], prompt=r["prompt"], k=r["loop"], ncyc=len(cg), cyc_gap=cg, cyc_escape=ce, greedy_path=ok))
            S["loops"] = L
            Ns = [1, 4, 16, 64, 256]; hor = {}
            for N in Ns:
                v = []
                for l in L:
                    R = np.array(l["cyc_escape"])[BURN:]
                    if len(R) < 2 or not l["greedy_path"]:
                        continue
                    RR = np.vstack([R[:N]] + [R[-1:]] * max(0, N - len(R)))
                    th = t_half(TEMPS, np.prod(1 - RR, 0))
                    if th:
                        v.append(th)
                hor[str(N)] = (float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), len(v)) if v else None
            S["T_half"] = hor
            S["barrier_first"] = float(np.median([l["cyc_gap"][0] for l in L]))
            S["barrier_last"] = float(np.median([l["cyc_gap"][-1] for l in L]))
            S["period_median"] = float(np.median([l["k"] for l in L]))
            S["n_loops_greedy_path"] = int(sum(l["greedy_path"] for l in L))
        summary[mode] = S
    for mode, v in VENDOR.items():
        if mode in summary:
            for a_ in ax:
                a_.axvline(v, color=COL[mode], ls=":", lw=0.9)
    for a_ in ax:
        a_.axvspan(0.5, 0.7, color="0.85", lw=0, zorder=0)
        a_.set_xlabel("sampling temperature $T$")
    ax[0].set_ylabel("finished within budget"); ax[0].set_title("(a) traces that end", loc="left", fontsize=8)
    ax[1].set_ylabel("answer reached (GSM8K)"); ax[1].set_title("(b) answer reached", loc="left", fontsize=8)
    ax[0].legend(fontsize=6, frameon=False, loc="center right")
    ax[2].set_ylabel("repeated 4-grams (median)"); ax[2].set_title("(c) repetition", loc="left", fontsize=8)
    fig.tight_layout(w_pad=1.0)
    fig.savefig("paper/figs/exp3f_reasoning.pdf", dpi=300); fig.savefig("paper/figs/exp3f_reasoning.png", dpi=200)
    json.dump(summary, open("results/exp3f_summary.json", "w"), indent=1)
    for k, v in summary.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "loops"})


if __name__ == "__main__":
    main()
