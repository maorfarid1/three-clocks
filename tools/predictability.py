"""Find the most statistically predictable sentences (what perplexity-based AI detectors flag).
Usage: python predictability.py file.txt [--top 0.3]
Higher = more machine-like. Overall above ~0.25 is in the AI range (14 real pre-2022 introductions: max 0.27, median -0.55)."""
import sys, re, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--top", type=float, default=0.3)
a = ap.parse_args()
name = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(name); mdl = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
for f in a.files:
    text = open(f).read().strip()
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=1500)
    offs = enc.pop("offset_mapping")[0]
    lp = torch.log_softmax(mdl(**enc).logits[0, :-1], -1); p = lp.exp()
    lab = enc["input_ids"][0, 1:]
    ll = lp.gather(-1, lab[:, None]).squeeze(-1)
    mu = (p * lp).sum(-1); var = (p * lp.pow(2)).sum(-1) - mu.pow(2)
    contrib = (ll - mu)
    total = (contrib.sum() / var.sum().sqrt()).item()
    spans = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]+(\s|$)", text)]
    rows = []
    for s, e in spans:
        idx = [i for i in range(len(contrib)) if s <= offs[i + 1][0] < e]
        if len(idx) < 4: continue
        c = contrib[idx]; v = var[idx]
        rows.append(((c.sum() / v.sum().sqrt()).item(), text[s:e].strip()))
    rows.sort(reverse=True)
    k = max(1, int(len(rows) * a.top))
    print(f"== {f}\nOVERALL {total:.3f}  ({'AI range' if total > 0.25 else 'human range'})")
    for sc, sent in rows[:k]: print(f"  {sc:+.2f}  {sent}")
