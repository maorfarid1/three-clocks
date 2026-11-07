"""Exp. 3: escape kinetics from repetition loops versus sampling temperature.

Step 1 (greedy). From each prompt, decode greedily for `--gen` tokens and store,
at every step, the logits. If the tail becomes exactly periodic (a repetition
loop of period k), the run is kept.

Step 2 (exact survival). Once the loop is locked, the context along the loop is
deterministic as long as the sampler keeps choosing the loop token. So the
probability of staying for m more steps at temperature T is a product of
per-step probabilities p_t(T) = softmax(z_t / T)[loop token], computable from
the greedy logits alone. The mean escape (exit) time is

    E[tau](T) = sum_{m >= 0} S(m),   S(m) = prod_{t < m} p_t(T).

Beyond the stored horizon we assume the last stored cycle repeats (a geometric
tail). Because the loop keeps reinforcing itself, this makes E[tau] a lower bound.

Step 3 (barrier). For each step we record the logit gap
Delta_t = z_t[loop] - max_{j != loop} z_t[j]. At low T, escape per step is
dominated by exp(-Delta_t / T), so log E[tau] should be close to linear in 1/T
with slope set by the smallest gap in the cycle (an Arrhenius / Kramers law with
the logit gap as barrier and T as noise intensity).

Step 4 (Monte Carlo check). For a few loops and temperatures we actually sample
and compare measured exit times with E[tau].

Step 5 (free sampling). Fraction of free generations that end in a loop, vs T.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS


def find_loop(seq, min_reps=3, kmax=64):
    """Smallest period k such that the last min_reps*k tokens are k-periodic.
    Returns (k, onset) where onset is the first index from which seq is k-periodic."""
    n = len(seq)
    for k in range(1, kmax + 1):
        if min_reps * k > n:
            break
        tail = seq[n - min_reps * k:]
        if all(tail[i] == tail[i + k] for i in range(len(tail) - k)):
            onset = n - min_reps * k
            while onset > 0 and seq[onset - 1] == seq[onset - 1 + k]:
                onset -= 1
            return k, onset
    return 0, None


@torch.no_grad()
def greedy_with_logits(model, ids, gen):
    out = model(ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[0, -1]
    toks, zs = [], []
    for _ in range(gen):
        nxt = int(torch.argmax(logits))
        toks.append(nxt)
        zs.append(logits.double().clone())
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]
    return toks, zs


def stay_probs(zs, toks, temps):
    """p[t, iT] = softmax(z_t / T)[toks[t]]; gap[t] = z[tok] - max other."""
    Z = torch.stack(zs)  # (n, V)
    idx = torch.tensor(toks)
    zl = Z.gather(1, idx[:, None]).squeeze(1)
    Zo = Z.clone()
    Zo[torch.arange(len(toks)), idx] = -float("inf")
    gap = (zl - Zo.max(1).values).numpy()
    P = np.zeros((len(toks), len(temps)))
    for j, T in enumerate(temps):
        P[:, j] = torch.exp(zl / T - torch.logsumexp(Z / T, 1)).numpy()
    return P, gap


def mean_escape(p, k):
    """E[tau] for per-step stay probs p (1D, starting at loop position 0),
    with the last cycle of length k repeated forever (geometric tail)."""
    logS = np.concatenate([[0.0], np.cumsum(np.log(np.clip(p, 1e-300, 1.0)))])
    S = np.exp(logS)
    head = S[:-1].sum()  # sum_{m=0}^{M-1} S(m)
    last = p[-k:]
    q = float(np.prod(last))
    within = np.concatenate([[1.0], np.cumprod(last)[:-1]]).sum()
    tail = S[-1] * within / max(1.0 - q, 1e-300)
    return head + tail, q


@torch.no_grad()
def monte_carlo(model, ctx_ids, loop_tokens, T, n, horizon, seed):
    """Sample n continuations from the locked-loop context; exit time = first
    step whose token differs from the loop continuation."""
    g = torch.Generator().manual_seed(seed)
    ids = ctx_ids.repeat(n, 1)
    out = model(ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1]
    k = len(loop_tokens)
    exit_t = np.full(n, horizon, dtype=float)
    alive = np.ones(n, dtype=bool)
    for t in range(horizon):
        probs = torch.softmax(logits.double() / T, -1)
        nxt = torch.multinomial(probs, 1, generator=g).squeeze(1)
        want = loop_tokens[t % k]
        left = (nxt != want).numpy() & alive
        exit_t[left] = t
        alive &= ~left
        if not alive.any():
            break
        out = model(nxt[:, None], past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
    return exit_t, alive


@torch.no_grad()
def free_sampling(model, tok, prompts, temps, gen, seed):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True)
    res = {}
    for T in temps:
        torch.manual_seed(seed)
        out = model.generate(**enc, do_sample=True, temperature=T, top_k=0, top_p=1.0,
                             max_new_tokens=gen, min_new_tokens=gen, pad_token_id=tok.pad_token_id)
        new = out[:, enc.input_ids.shape[1]:].tolist()
        loops = [find_loop(s, min_reps=3)[0] for s in new]
        res[T] = dict(frac_loop=float(np.mean([k > 0 for k in loops])), periods=loops)
        print(f"  free T={T}: frac_loop={res[T]['frac_loop']:.2f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--gen", type=int, default=320)
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--burn_cycles", type=int, default=2, help="cycles already in context before timing")
    ap.add_argument("--mc_loops", type=int, default=4)
    ap.add_argument("--mc_n", type=int, default=128)
    ap.add_argument("--mc_temps", type=lambda z: [float(x) for x in z.split(",")], default=[0.6, 1.0])
    ap.add_argument("--free_temps", default="0.3,0.5,0.7,0.85,1.0,1.2,1.5")
    ap.add_argument("--free_gen", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/exp3_gpt2.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(revision=a.revision) if a.revision else {}
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).float().eval()
    temps = np.round(np.geomspace(0.15, 2.0, 30), 4).tolist()
    loops, t0 = [], time.time()
    for i, text in enumerate(PROMPTS[: a.n_prompts]):
        ids = tok(text, return_tensors="pt").input_ids
        toks, zs = greedy_with_logits(model, ids, a.gen)
        k, onset = find_loop(toks, min_reps=3)
        if k == 0:
            print(f"[{i}] no loop", flush=True)
            continue
        start = onset + a.burn_cycles * k  # time escape after burn_cycles full cycles in context
        if start + 2 * k > len(toks):
            print(f"[{i}] loop too late", flush=True)
            continue
        P, gap = stay_probs(zs[onset:], toks[onset:], temps)
        ncyc = (len(toks) - onset) // k
        logP = np.log(np.clip(P[: ncyc * k], 1e-300, 1.0)).reshape(ncyc, k, len(temps))
        cyc_escape = (1.0 - np.exp(logP.sum(1))).tolist()          # r_c(T): escape prob. during cycle c
        cyc_gap = gap[: ncyc * k].reshape(ncyc, k).min(1).tolist()   # barrier of cycle c (min logit gap)
        b = a.burn_cycles
        surv_end = np.exp(logP[b:].sum((0, 1))).tolist()            # survive all stored cycles after burn-in
        loops.append(dict(prompt=i, k=k, onset=onset, start=start, ncyc=ncyc,
                          cyc_escape=cyc_escape, cyc_gap=cyc_gap, surv_end=surv_end,
                          ctx=ids[0].tolist() + toks[:start], loop_tokens=toks[start:start + k],
                          stay_from_start=P[start - onset:, :].tolist(),
                          loop_text=tok.decode(toks[onset:onset + k])))
        print(f"[{i}] k={k} onset={onset} ncyc={ncyc} gap per cycle={np.round(cyc_gap, 2).tolist()} "
              f"t={time.time()-t0:.0f}s", flush=True)
    # Monte Carlo check: sampled survival vs exact survival after 1, 2, 4 cycles
    mc = []
    for L in loops[: a.mc_loops]:
        k = L["k"]
        st = np.array(L["stay_from_start"])
        for T in a.mc_temps:
            j = int(np.argmin(np.abs(np.array(temps) - T)))
            Tq = temps[j]
            ex, alive = monte_carlo(model, torch.tensor([L["ctx"]]), L["loop_tokens"], Tq,
                                    a.mc_n, horizon=min(4 * k, len(st)), seed=a.seed)
            logS = np.concatenate([[0.0], np.cumsum(np.log(np.clip(st[:, j], 1e-300, 1)))])
            row = dict(prompt=L["prompt"], T=Tq, k=k, n=a.mc_n)
            for c in (1, 2, 4):
                m = c * k
                if m <= min(4 * k, len(st)):
                    row[f"exact_S{c}"] = float(np.exp(logS[m]))
                    row[f"mc_S{c}"] = float(np.mean(ex >= m))
            mc.append(row)
            print("  MC", row, flush=True)
    free = free_sampling(model, tok, PROMPTS[: a.n_prompts],
                         [float(x) for x in a.free_temps.split(",")], a.free_gen, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(args={k: v for k, v in vars(a).items()}, temps=temps, loops=loops, mc=mc, free={str(k): v for k, v in free.items()}),
              open(a.out, "w"))
    print("done", time.time() - t0)
