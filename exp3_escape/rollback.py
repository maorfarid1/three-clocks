"""Experiment 4b: escape at the entrance of the well (rollback and kick).

Experiment 3 shows that the barrier of a loop is smallest when the loop starts and grows while the model
repeats itself, because the repeated text in the context deepens the well. Arrhenius-guided escape (AGE,
control.py) heats the sampler inside a well that is already deep and leaves the text repetitive. Rollback and
kick acts where the barrier is lowest: when the generated tokens end in a loop, it deletes the periodic tail
back to its first token (the entrance of the loop), forbids that token at that position, and resamples there
at the base temperature T0; then sampling continues at T0. In the language of Section 2 the kick is applied
at the saddle, before the well deepens, and the context no longer carries the repetitions.

Same prompts, length, samples and scoring as control.py (40 prompts, 200 tokens, 3 samples, GPT-2).
Usage: python exp3_escape/rollback.py --out results/exp4b_gpt2.json
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS
from exp3_escape.control import ends_in_loop
from exp3_escape.escape import find_loop
from exp3_escape.regime import load_ref, ref_nll, rep_n


def periodic_start(seq, k):
    """Index of the first token of the k-periodic tail of seq."""
    i = len(seq) - k
    while i - 1 >= 0 and seq[i - 1] == seq[i - 1 + k]:
        i -= 1
    return i


@torch.no_grad()
def generate(model, prompt_ids, T0, gen, max_events, rng_seed):
    g = torch.Generator().manual_seed(rng_seed)
    seq, banned, events = [], {}, []
    ids = prompt_ids
    out = model(ids, use_cache=True); past, z = out.past_key_values, out.logits[0, -1]
    while len(seq) < gen:
        zz = z.double() / T0
        for b in banned.get(len(seq), ()):
            zz[b] = -float("inf")
        nxt = int(torch.multinomial(torch.softmax(zz, -1), 1, generator=g))
        seq.append(nxt)
        k = ends_in_loop(seq) if len(events) < max_events else 0
        if k:
            s0 = periodic_start(seq, k)
            events.append(dict(step=len(seq), k=k, back_to=s0))
            banned.setdefault(s0, set()).add(seq[s0])
            seq = seq[:s0]
            ctx = torch.cat([prompt_ids, torch.tensor([seq], dtype=torch.long)], 1) if seq else prompt_ids
            out = model(ctx, use_cache=True); past, z = out.past_key_values, out.logits[0, -1]
            continue
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True); past, z = out.past_key_values, out.logits[0, -1]
    return seq, events


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--temps", default="0.3,0.5")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--max_events", type=int, default=50)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model).eval()
    prompts = PROMPTS[:a.n_prompts]
    gens = {}
    for T0 in map(float, a.temps.split(",")):
        t0 = time.time(); rows = []
        for s in range(a.samples):
            for i, p in enumerate(prompts):
                seq, ev = generate(model, tok(p, return_tensors="pt").input_ids, T0, a.gen, a.max_events, 1000 * s + 7 + i)
                rows.append(dict(prompt=i, sample=s, seq=seq, events=ev))
        gens[T0] = rows
        print(f"T0={T0} generated in {time.time() - t0:.0f}s", flush=True)
    del model
    ref, rtok = load_ref()
    res = []
    for T0, rows in gens.items():
        conts = [tok.decode(r["seq"]) for r in rows]
        nll = ref_nll(ref, rtok, [prompts[r["prompt"]] for r in rows], conts)
        out_rows = [dict(prompt=r["prompt"], sample=r["sample"], loop=find_loop(r["seq"], min_reps=3)[0], rep4=rep_n(r["seq"]),
                         ref_nll=nll[j], n_events=len(r["events"]), events=r["events"], text=conts[j][:300] if r["sample"] == 0 else None)
                    for j, r in enumerate(rows)]
        res.append(dict(cfg=dict(method="rollback", T0=T0), rows=out_rows))
        print(T0, f"loop={np.mean([r['loop'] > 0 for r in out_rows]):.3f} rep4={np.median([r['rep4'] for r in out_rows]):.3f} "
              f"nll={np.median([r['ref_nll'] for r in out_rows]):.2f} events={np.mean([r['n_events'] for r in out_rows]):.2f}", flush=True)
    json.dump(dict(args=vars(a), configs=res), open(a.out, "w"))
