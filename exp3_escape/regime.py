"""Experiment 3b: regime map of generation against temperature.

For each prompt and temperature we sample several continuations and measure
  - loop:     whether the continuation ends in an exact repetition loop (same test as Exp. 3)
  - rep4:     fraction of repeated 4-grams in the continuation
  - ref_nll:  mean per-token negative log-likelihood of the continuation under a separate
              reference model (Qwen2.5-0.5B), conditioned on the prompt
  - entropy:  mean entropy (nats) of the sampling distribution pi_T along the path
The same reference scoring is applied to human text (OpenWebText, first 16 GPT-2 tokens
as prompt, next `gen` tokens as continuation), which sets the incoherence threshold.

Usage:
  python exp3_escape/regime.py --model gpt2 --out results/exp3b_gpt2.json
  python exp3_escape/regime.py --model EleutherAI/pythia-160m --out results/exp3b_pythia160m.json
  python exp3_escape/regime.py --human --out results/exp3b_human.json
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS
from exp3_escape.escape import find_loop

REF = "Qwen/Qwen2.5-0.5B"


class TempAndEntropy(LogitsProcessor):
    """Divides logits by T and records the entropy of the resulting distribution."""

    def __init__(self, T):
        self.T, self.ent = T, []

    def __call__(self, input_ids, scores):
        s = scores.float() / self.T
        lp = torch.log_softmax(s, -1)
        self.ent.append(-(lp.exp() * lp).nan_to_num().sum(-1))
        return s


def rep_n(seq, n=4):
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


@torch.no_grad()
def ref_nll(ref, rtok, prompts, conts, bs=8):
    """Mean NLL per reference token of each continuation, given its prompt."""
    out = []
    for i in range(0, len(prompts), bs):
        P, C = prompts[i:i + bs], conts[i:i + bs]
        ids, starts = [], []
        for p, c in zip(P, C):
            pi = rtok(p).input_ids
            ci = rtok(p + c).input_ids
            ids.append(ci[:512]); starts.append(len(pi))
        L = max(map(len, ids))
        x = torch.full((len(ids), L), rtok.pad_token_id or 0)
        m = torch.zeros_like(x)
        for j, s in enumerate(ids):
            x[j, :len(s)] = torch.tensor(s); m[j, :len(s)] = 1
        h = ref.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]
        lp = torch.empty(h.shape[:2])
        for c in range(0, h.shape[1], 64):  # chunked head keeps memory small (vocab is 152k)
            z = ref.lm_head(h[:, c:c + 64]).float()
            lp[:, c:c + 64] = z.gather(-1, x[:, 1 + c:1 + c + 64, None])[..., 0] - torch.logsumexp(z, -1)
        for j, s in enumerate(ids):
            seg = lp[j, starts[j] - 1:len(s) - 1]
            out.append(float(-seg.mean()) if len(seg) else float("nan"))
    return out


def load_ref():
    rtok = AutoTokenizer.from_pretrained(REF)
    ref = AutoModelForCausalLM.from_pretrained(REF, dtype=torch.float32).eval()
    return ref, rtok


def human(a):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("stas/openwebtext-10k", "plain_text/train/0000.parquet",
                        repo_type="dataset", revision="refs/convert/parquet")
    docs = pd.read_parquet(f)["text"].tolist()
    gtok = AutoTokenizer.from_pretrained("gpt2")
    P, C, reps, loops = [], [], [], []
    for d in docs:
        t = gtok(d).input_ids
        if len(t) < 16 + a.gen:
            continue
        P.append(gtok.decode(t[:16])); c = t[16:16 + a.gen]; C.append(gtok.decode(c))
        reps.append(rep_n(c)); loops.append(find_loop(c, min_reps=3)[0])
        if len(P) == a.n_human:
            break
    ref, rtok = load_ref()
    nll = ref_nll(ref, rtok, P, C)
    json.dump(dict(source="stas/openwebtext-10k", n=len(P), gen=a.gen, ref_nll=nll, rep4=reps,
                   loop=[int(k > 0) for k in loops]), open(a.out, "w"))
    print("human: n", len(P), "median ref_nll", np.median(nll), "rep4", np.median(reps), "loops", sum(k > 0 for k in loops))


def run(a):
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model).eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = PROMPTS[:a.n_prompts]
    chunks = [list(range(i, min(i + a.batch, len(prompts)))) for i in range(0, len(prompts), a.batch)]
    ref, rtok = (None, None) if a.defer_ref else load_ref()
    rows = []
    for T in a.temps:
        t0 = time.time()
        for s in range(a.samples):
            for ci, idx in enumerate(chunks):
                P = [prompts[i] for i in idx]
                enc = tok(P, return_tensors="pt", padding=True)
                torch.manual_seed(1000 * s + int(round(100 * T)) + 7 * ci)
                proc = TempAndEntropy(T)
                out = model.generate(**enc, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                     max_new_tokens=a.gen, min_new_tokens=a.gen, pad_token_id=tok.pad_token_id,
                                     logits_processor=LogitsProcessorList([proc]))
                new = out[:, enc.input_ids.shape[1]:].tolist()
                ent = torch.stack(proc.ent, 1).mean(1).tolist()
                conts = [tok.decode(n) for n in new]
                nll = [None] * len(conts) if a.defer_ref else ref_nll(ref, rtok, P, conts)
                for j, n in enumerate(new):
                    rows.append(dict(T=T, prompt=idx[j], sample=s, loop=find_loop(n, min_reps=3)[0], rep4=rep_n(n),
                                     ref_nll=nll[j], entropy=ent[j],
                                     text=conts[j][:300] if s == 0 else None, cont=conts[j] if a.defer_ref else None))
        R = [r for r in rows if r["T"] == T]
        print(f"T={T:.2f} loop={np.mean([r['loop'] > 0 for r in R]):.2f} rep4={np.median([r['rep4'] for r in R]):.3f} "
              f"nll={np.median([r['ref_nll'] for r in R]) if not a.defer_ref else float('nan'):.2f} H={np.median([r['entropy'] for r in R]):.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(dict(args=vars(a), rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--n_human", type=int, default=200)
    ap.add_argument("--temps", type=lambda z: [float(x) for x in z.split(",")],
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=40, help="prompts per generate call (40 = all at once, as in the paper runs)")
    ap.add_argument("--defer_ref", action="store_true", help="skip reference scoring here; run --score_file later (saves memory)")
    ap.add_argument("--score_file", default=None, help="fill ref_nll in an existing output produced with --defer_ref")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.score_file:
        d = json.load(open(a.score_file)); rows = d["rows"]; ref, rtok = load_ref()
        todo = [r for r in rows if r.get("ref_nll") is None]
        P = [PROMPTS[r["prompt"]] for r in todo]; C = [r["cont"] for r in todo]
        for r, v in zip(todo, ref_nll(ref, rtok, P, C)):
            r["ref_nll"] = v; r["cont"] = None
        json.dump(d, open(a.out, "w")); print("scored", len(todo))
    else:
        human(a) if a.human else run(a)
