"""Exp. 1: finite-time Lyapunov exponents (FTLEs) of the residual stream,
layer by layer, across public Pythia training checkpoints.

For layer l, the map F_l sends the whole residual stream h_l (T x d) to h_{l+1}.
We estimate the top singular value sigma_l of its Jacobian at the observed
trajectory by power iteration with JVP/VJP, and report

    lambda_l     = log sigma_l                         (one layer = one time step)
    lambda_rel_l = log sigma_l - log(|h_{l+1}| / |h_l|) (stretching beyond norm growth)

plus the depth-averaged FTLE of the full stack, (1/L) log sigma(J_{0->L}),
where J_{0->L} maps the embedding output to the last residual stream (before
the final LayerNorm). Everything runs on CPU.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from torch.func import jvp, vjp
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.prompts import PROMPTS


def top_sv(f, h0, iters, gen):
    """Largest singular value of the Jacobian of f at h0 (power iteration on J^T J)."""
    v = torch.randn(h0.shape, generator=gen, dtype=h0.dtype)
    v = v / v.norm()
    s = 0.0
    for _ in range(iters):
        _, jv = jvp(f, (h0,), (v,))
        _, vjp_fn = vjp(f, h0)
        (jtjv,) = vjp_fn(jv)
        s = jv.norm().item()
        v = jtjv / jtjv.norm()
    return s


def layer_fns(model, T):
    nx = model.gpt_neox
    pos = torch.arange(T).unsqueeze(0)
    neg = torch.finfo(torch.float32).min
    causal = torch.triu(torch.full((T, T), neg), 1)[None, None]

    def make(l):
        layer = nx.layers[l]

        def f(h):
            rot = nx.rotary_emb(h, pos)
            out = layer(h, attention_mask=causal[..., : h.shape[1], : h.shape[1]], position_ids=pos, position_embeddings=rot)
            return out[0] if isinstance(out, tuple) else out
        return f
    return [make(l) for l in range(len(nx.layers))]


@torch.no_grad()
def residuals(model, ids):
    out = model(ids, output_hidden_states=True)
    hs = list(out.hidden_states)
    return hs  # hs[0] = embeddings, hs[L] has final LN applied in HF


def check_chain(model, fns, ids):
    """Verify that chaining our layer calls reproduces HF hidden states."""
    hs = residuals(model, ids)
    h = hs[0]
    errs = []
    with torch.no_grad():
        for l, f in enumerate(fns[:-1]):
            h = f(h)
            errs.append((h - hs[l + 1]).abs().max().item() / hs[l + 1].abs().max().item())
    return max(errs)


def analyze(model_name, revision, prompts, T, iters, seed):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                 attn_implementation="eager").float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fns = layer_fns(model, T)
    L = len(fns)
    gen = torch.Generator().manual_seed(seed)
    per_layer, per_layer_rel, stack = [], [], []
    chain_err = None
    for i, text in enumerate(prompts):
        ids = tok(text, return_tensors="pt").input_ids
        ids = torch.cat([ids, ids, ids], 1)[:, :T] if ids.shape[1] < T else ids[:, :T]
        if chain_err is None:
            chain_err = check_chain(model, fns, ids)
        hs = residuals(model, ids)
        # recompute pre-final-LN last residual by chaining
        h = hs[0].squeeze(0)
        traj = [h]
        with torch.no_grad():
            for f in fns:
                traj.append(f(traj[-1].unsqueeze(0)).squeeze(0))
        lam, lamr = [], []
        for l, f in enumerate(fns):
            g = lambda z, f=f: f(z.unsqueeze(0)).squeeze(0)
            s = top_sv(g, traj[l], iters, gen)
            lam.append(math.log(s))
            lamr.append(math.log(s) - math.log(traj[l + 1].norm().item() / traj[l].norm().item()))
        def full(z):
            for f in fns:
                z = f(z.unsqueeze(0)).squeeze(0)
            return z
        S = top_sv(full, traj[0], iters, gen)
        per_layer.append(lam)
        per_layer_rel.append(lamr)
        stack.append(math.log(S) / L)
    return dict(model=model_name, revision=revision, L=L, chain_err=chain_err,
                lam=np.mean(per_layer, 0).tolist(), lam_sd=np.std(per_layer, 0).tolist(),
                lam_rel=np.mean(per_layer_rel, 0).tolist(), lam_rel_sd=np.std(per_layer_rel, 0).tolist(),
                stack=float(np.mean(stack)), stack_sd=float(np.std(stack)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="EleutherAI/pythia-70m,EleutherAI/pythia-160m")
    ap.add_argument("--steps", default="0,128,512,1000,2000,4000,8000,16000,32000,64000,143000")
    ap.add_argument("--n_prompts", type=int, default=16)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/exp1.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    prompts = PROMPTS[: a.n_prompts]
    res = []
    if os.path.exists(a.out):
        res = json.load(open(a.out))["runs"]
    done = {(r["model"], r["revision"]) for r in res}
    t0 = time.time()
    for m in a.models.split(","):
        for s in a.steps.split(","):
            rev = f"step{s}"
            if (m, rev) in done:
                continue
            r = analyze(m, rev, prompts, a.T, a.iters, a.seed)
            r["step"] = int(s)
            res.append(r)
            print(f"{m} {rev} chain_err={r['chain_err']:.2e} stack={r['stack']:.3f} "
                  f"lam_rel={np.round(r['lam_rel'], 2).tolist()} t={time.time()-t0:.0f}s", flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(dict(args=vars(a), runs=res), open(a.out, "w"))
