"""Exp. 3f: repetition wells and the temperature window in reasoning models.

Prompts: the first N problems of the GSM8K test set, formatted with each model's chat template
(thinking on or off where the template supports it), asking for a final answer in \\boxed{}.

--mode greedy: greedy decoding for up to --gen tokens. At every step we keep only what the escape
  analysis needs (logit of the chosen token, largest other logit, logsumexp(z/T) on the temperature
  grid of Exp. 3), so long generations with a 152k vocabulary fit in memory. If the output ends in a
  loop, we compute per-cycle barriers and exact escape probabilities exactly as in escape.py.
--mode sample: plain temperature sampling (no top-k or top-p) on a temperature grid, several samples per
  prompt, recording loop, repeated 4-grams, the extracted answer and whether it is correct.

Models in bfloat16 (--bf16) keep a float32 copy of the output head, so logits are computed in float32
from the bfloat16 final hidden state.

Usage:
  python exp3_escape/reasoning.py --model Qwen/Qwen3-0.6B --think 1 --mode greedy --out results/exp3f_qwen3_think_greedy.json
  python exp3_escape/reasoning.py --model Qwen/Qwen3-0.6B --think 1 --mode sample --out results/exp3f_qwen3_think_sample.json
"""
import argparse, json, os, re, sys, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp3_escape.escape import find_loop
from exp3_escape.regime import rep_n

TEMPS = np.round(np.geomspace(0.15, 2.0, 30), 4).tolist()
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def gsm8k(n):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("openai/gsm8k", "main/test/0000.parquet", repo_type="dataset", revision="refs/convert/parquet")
    df = pd.read_parquet(f).head(n)
    return [(q, a.split("####")[-1].strip().replace(",", "")) for q, a in zip(df["question"], df["answer"])]


def build(tok, q, think):
    msgs = [{"role": "user", "content": q + SUFFIX}]
    if tok.chat_template:
        kw = {} if think is None else {"enable_thinking": bool(think)}
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return "Question: " + q + SUFFIX + "\nAnswer:"


def extract(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    s = m[-1] if m else (re.findall(r"-?\d[\d,]*\.?\d*", text) or [""])[-1]
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


class Model:
    def __init__(self, name, bf16):
        self.tok = AutoTokenizer.from_pretrained(name)
        dt = torch.bfloat16 if bf16 else torch.float32
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=dt).eval()
        self.bf16 = bf16
        if bf16:
            self.head = self.m.get_output_embeddings().weight.detach().float()

    @torch.no_grad()
    def step(self, ids, past):
        out = self.m(ids, past_key_values=past, use_cache=True, output_hidden_states=self.bf16)
        if self.bf16:
            h = out.hidden_states[-1][0, -1].float()  # final hidden state (after the last norm in HF Qwen2/3)
            return h @ self.head.T, out.past_key_values
        return out.logits[0, -1].float(), out.past_key_values


@torch.no_grad()
def greedy(M, prompt, gen, eos):
    ids = M.tok(prompt, return_tensors="pt").input_ids
    z, past = M.step(ids, None)
    toks, zl, zo, lse = [], [], [], []
    T = torch.tensor(TEMPS, dtype=torch.float64)
    for _ in range(gen):
        zd = z.double()
        a = int(torch.argmax(zd)); toks.append(a)
        zl.append(float(zd[a])); top2 = torch.topk(zd, 2).values; zo.append(float(top2[1]))
        lse.append(torch.logsumexp(zd[None, :] / T[:, None], 1).tolist())
        if a in eos:
            break
        z, past = M.step(torch.tensor([[a]]), past)
    return toks, np.array(zl), np.array(zo), np.array(lse)


def run_greedy(a, M, probs):
    eos = set([M.tok.eos_token_id] + ([M.m.generation_config.eos_token_id] if isinstance(M.m.generation_config.eos_token_id, int)
                                     else list(M.m.generation_config.eos_token_id or [])))
    rows, t0 = [], time.time()
    for i, (q, ans) in enumerate(probs):
        toks, zl, zo, lse = greedy(M, build(M.tok, q, a.think), a.gen, eos)
        text = M.tok.decode(toks)
        k, onset = find_loop(toks, min_reps=3, kmax=256)
        row = dict(prompt=i, n_tokens=len(toks), ended_eos=toks[-1] in eos, correct=extract(text) == float(ans) if extract(text) is not None else False,
                   k=k, onset=onset, tail=text[-300:])
        if k:
            n = len(toks) - onset; ncyc = n // k
            gap = (zl - zo)[onset:onset + ncyc * k]
            logp = (zl[onset:onset + ncyc * k, None] / np.array(TEMPS)[None, :] - lse[onset:onset + ncyc * k])
            logP = logp.reshape(ncyc, k, len(TEMPS))
            row.update(ncyc=ncyc, cyc_gap=gap.reshape(ncyc, k).min(1).tolist(),
                       cyc_escape=(1 - np.exp(logP.sum(1))).tolist(),
                       surv_end=np.exp(logP[a.burn:].sum((0, 1))).tolist() if ncyc > a.burn else None,
                       loop_text=M.tok.decode(toks[onset:onset + k]))
        rows.append(row)
        print(f"[{i}] n={len(toks)} eos={row['ended_eos']} correct={row['correct']} k={k} onset={onset} "
              + (f"gaps={np.round(row['cyc_gap'][:8], 2).tolist()}" if k else "") + f" t={time.time()-t0:.0f}s", flush=True)
        json.dump(dict(args=vars(a), temps=TEMPS, rows=rows), open(a.out, "w"))


def run_sample(a, M, probs):
    tok = M.tok; tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [build(tok, q, a.think) for q, _ in probs]
    rows, temps = [], [float(x) for x in a.temps.split(",")]
    if os.path.exists(a.out):  # resume: keep finished temperatures
        rows = json.load(open(a.out))["rows"]
    for T in temps:
        if any(r["T"] == T for r in rows):
            continue
        t0 = time.time()
        for s in range(a.samples):
            for c in range(0, len(prompts), a.batch):
                P = prompts[c:c + a.batch]
                enc = tok(P, return_tensors="pt", padding=True)
                torch.manual_seed(1000 * s + int(round(100 * T)) + c)
                out = M.m.generate(**enc, do_sample=True, temperature=T, top_k=0, top_p=1.0, max_new_tokens=a.gen,
                                   pad_token_id=tok.pad_token_id)
                for j, seq in enumerate(out[:, enc.input_ids.shape[1]:].tolist()):
                    eos_pos = [p for p, t in enumerate(seq) if t in (tok.eos_token_id, tok.pad_token_id)]
                    seq = seq[:eos_pos[0]] if eos_pos else seq
                    text = tok.decode(seq); ans = probs[c + j][1]; x = extract(text)
                    k_loop, onset = find_loop(seq, min_reps=3, kmax=256)
                    rows.append(dict(T=T, prompt=c + j, sample=s, n_tokens=len(seq), finished=bool(eos_pos),
                                     loop=k_loop, onset=onset, ids=seq if k_loop else None, rep4=rep_n(seq),
                                     correct=(x is not None and x == float(ans)), tail=text[-200:] if s == 0 else None))
        R = [r for r in rows if r["T"] == T]
        print(f"T={T} loop={np.mean([r['loop'] > 0 for r in R]):.2f} acc={np.mean([r['correct'] for r in R]):.2f} "
              f"finished={np.mean([r['finished'] for r in R]):.2f} rep4={np.median([r['rep4'] for r in R]):.3f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(dict(args=vars(a), rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", type=int, default=None, help="1/0 to set enable_thinking in the chat template")
    ap.add_argument("--mode", choices=["greedy", "sample"], required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--burn", type=int, default=2)
    ap.add_argument("--temps", default="0.2,0.4,0.6,0.8,1.0,1.2")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    M = Model(a.model, a.bf16)
    probs = gsm8k(a.n)
    run_greedy(a, M, probs) if a.mode == "greedy" else run_sample(a, M, probs)
