"""Experiment 4: escape control. Raise the temperature only when a loop starts, by the amount
the escape law says, and compare with standard anti-repetition methods.

Arrhenius-guided escape (AGE). Sample at a base temperature T0. After each step, check whether
the generated tokens end in two full repetitions of a period-k block (k <= 64). If they do, take
the logits stored for the last k steps (the most recent cycle) and choose the smallest temperature
T* on a grid such that the probability of leaving the cycle within one cycle,
    r(T) = 1 - prod_t p_t(T),   p_t(T) = softmax(z_t / T)[a_t],
is at least q, capped at Tmax = 1.5. Sample the next k steps at T*, then return to T0. A loop is
detected when the tokens end in at least max(2, ceil(8/k)) repetitions of a k-block. The stay probabilities use the
top 64 logits of each step, which overestimates p_t and so errs toward a higher T*.

Baselines on the same prompts, lengths and scoring: constant temperature (from Exp. 3b),
the repetition penalty of Keskar et al. (2019), and a ban on repeated 4-grams.

Usage: python exp3_escape/control.py --model gpt2 --out results/exp4_gpt2.json
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS
from exp3_escape.escape import find_loop
from exp3_escape.regime import load_ref, ref_nll, rep_n

TGRID = np.geomspace(0.2, 3.0, 48)


def ends_in_loop(seq, kmax=64, span=8):
    """Smallest k such that the sequence ends in at least max(2, ceil(span/k)) repetitions of a k-block."""
    n = len(seq)
    for k in range(1, kmax + 1):
        reps = max(2, -(-span // k))
        if reps * k > n:
            if 2 * k > n:
                break
            continue
        tail = seq[n - reps * k:]
        if all(tail[i] == tail[i + k] for i in range(len(tail) - k)):
            return k
    return 0


class AGE(LogitsProcessor):
    def __init__(self, T0, q, plen, Tmax=1.5, topk=64):
        self.T0, self.q, self.plen, self.topk, self.Tmax = T0, q, plen, topk, Tmax
        self.hist, self.boost, self.Tb, self.events = None, None, None, None

    def __call__(self, input_ids, scores):
        B = scores.shape[0]
        s = scores.float()
        if self.hist is None:
            self.hist = [[] for _ in range(B)]; self.boost = [0] * B; self.Tb = [self.T0] * B; self.events = [[] for _ in range(B)]
        top = torch.topk(s, self.topk, -1)
        T = torch.full((B, 1), self.T0)
        for b in range(B):
            gen = input_ids[b, self.plen:].tolist()
            # logits stored at step i produced token gen[i]; the logits of this step are appended below
            if self.boost[b] > 0:
                self.boost[b] -= 1; T[b, 0] = self.Tb[b]
            else:
                k = ends_in_loop(gen)
                if k and len(self.hist[b]) >= k:
                    vals, idx = zip(*self.hist[b][-k:])
                    toks = gen[-k:]
                    logp = np.zeros(len(TGRID))
                    for v, ix, a in zip(vals, idx, toks):
                        if a not in ix:
                            logp += -np.inf; continue
                        za = v[ix.index(a)]
                        z = np.array(v)[None, :] / TGRID[:, None]
                        logp += za / TGRID - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) - z.max(1)
                    r = 1 - np.exp(logp)
                    ok = np.where(r >= self.q)[0]
                    Tstar = float(TGRID[ok[0]]) if len(ok) else float(TGRID[-1])
                    Tstar = min(max(Tstar, self.T0), self.Tmax)
                    self.boost[b], self.Tb[b] = k - 1, Tstar
                    T[b, 0] = Tstar
                    self.events[b].append(dict(step=len(gen), k=k, T=Tstar))
            self.hist[b].append((top.values[b].tolist(), top.indices[b].tolist()))
            if len(self.hist[b]) > 70:
                self.hist[b].pop(0)
        return s / T


class Temp(LogitsProcessor):
    def __init__(self, T):
        self.T = T

    def __call__(self, input_ids, scores):
        return scores.float() / self.T


def run_config(model, tok, enc, prompts, cfg, gen, samples, ref, rtok):
    rows = []
    plen = enc.input_ids.shape[1]
    for s in range(samples):
        torch.manual_seed(1000 * s + 7)
        kw = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=gen, min_new_tokens=gen,
                  pad_token_id=tok.pad_token_id)
        if cfg["method"] == "age":
            proc = AGE(cfg["T0"], cfg["q"], plen, cfg.get("Tmax", 1.5))
        else:
            proc = Temp(cfg["T0"])
        if cfg["method"] == "penalty":
            kw["repetition_penalty"] = cfg["penalty"]
        if cfg["method"] == "ngram":
            kw["no_repeat_ngram_size"] = cfg["n"]
        out = model.generate(**enc, logits_processor=LogitsProcessorList([proc]), **kw)
        new = out[:, plen:].tolist()
        conts = [tok.decode(n) for n in new]
        nll = ref_nll(ref, rtok, prompts, conts)
        for i, n in enumerate(new):
            ev = proc.events[i] if cfg["method"] == "age" else []
            rows.append(dict(prompt=i, sample=s, loop=find_loop(n, min_reps=3)[0], rep4=rep_n(n), ref_nll=nll[i],
                             n_boost=len(ev), T_boost=[e["T"] for e in ev], text=conts[i][:300] if s == 0 else None))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    configs = ([dict(method="age", T0=t, q=q) for t in (0.3, 0.5) for q in (0.5, 0.9)] + [dict(method="age", T0=0.2, q=0.9)]
               + [dict(method="penalty", T0=t, penalty=p) for t in (0.3, 0.5) for p in (1.2, 1.4)]
               + [dict(method="ngram", T0=t, n=4) for t in (0.3, 0.5)])
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model).eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = PROMPTS[:a.n_prompts]
    enc = tok(prompts, return_tensors="pt", padding=True)
    ref, rtok = load_ref()
    res = []
    for cfg in configs:
        t0 = time.time()
        rows = run_config(model, tok, enc, prompts, cfg, a.gen, a.samples, ref, rtok)
        res.append(dict(cfg=cfg, rows=rows))
        print(cfg, f"loop={np.mean([r['loop'] > 0 for r in rows]):.3f} nll={np.median([r['ref_nll'] for r in rows]):.2f} "
              f"rep4={np.median([r['rep4'] for r in rows]):.3f} boosts={np.mean([r['n_boost'] for r in rows]):.2f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(dict(args=vars(a), configs=res), open(a.out, "w"))
