# The Three Clocks of a Transformer

Code, data and reading list for

> **Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation**
> Maor Farid

A transformer language model is iterated in three ways: over layers (clock 1), over optimizer steps (clock 2) and over generated tokens (clock 3). The paper reviews the dynamical-systems results on each clock (about 200 papers), collects them in the **Three-Clock Dictionary**, and runs experiments on open models. Everything here runs on a laptop CPU.

## Key results

- **Repetition loops are Kramers escape problems.** Escape from a loop at sampling temperature T follows the Arrhenius law with the logit gap as the barrier (fitted vs measured barrier: correlation 0.997, GPT-2 and Pythia-160M).
- **The well deepens, then saturates.** The barrier grows with every repeat and levels off, so no loop locks forever; the temperature needed to escape within N cycles falls like 1/ln N.
- **Larger models dig deeper wells.** Across Pythia 70M to 1B (same data and order), the saturated barrier rises from 3.0 to 4.8 and the locking temperature by about half.
- **Coherent text lives in a narrow temperature window** between repetition and incoherence (about 0.7 to 0.9), which closes entirely for some prompts.
- **Reasoning models have the window too, without the loops.** Qwen3-0.6B on GSM8K reaches the most answers within a fixed budget at T = 0.6, the value its developers recommend. Below that it rethinks the same step until the budget runs out, a recurrent orbit rather than an exact loop.
- **The edge of stability is multi-mode.** In a small transformer, 2 to 6 Hessian eigenvalues sit 5 to 14% above 2/eta while the curvature along the actual step equals 2/eta; power iteration underestimates the sharpness there.
- **Training reshapes depth dynamics.** Per-layer Lyapunov exponents of Pythia rise over training while the alignment of stretched directions across layers falls.

## Reproduce

```bash
pip install -r requirements.txt   # CPU torch: --index-url https://download.pytorch.org/whl/cpu
make all                          # experiments -> figures -> paper/main.pdf
make arxiv                        # de-anonymized build: paper/main_arxiv.pdf
```

`results/` already contains the raw outputs used in the paper, so `make figures paper` rebuilds the figures and PDF in a minute without rerunning the experiments.

| Experiment | Script | Output |
| --- | --- | --- |
| 1. Finite-time Lyapunov exponents of the residual stream across Pythia checkpoints | `exp1_ftle/ftle.py` | `results/exp1.json` |
| 2. Learning-rate bifurcation diagram of a one-block transformer, 3 seeds | `exp2_bifurcation/bifurcation.py` | `results/exp2*.json` |
| 2b. Edge-of-stability geometry (curvature along the step; Lanczos, Arnoldi, Lyapunov) | `exp2_bifurcation/eos_geometry.py`, `eos_rotation.py` | `results/eos_geom_seed*.json`, `results/eos_rotation/` |
| 3. Escape from repetition loops (GPT-2, Pythia 70M to 1B) | `exp3_escape/escape.py`, `lock.py` | `results/exp3_*.json`, `results/exp3d_lock.json` |
| 3b. Coherent window: loops, repetition, incoherence vs temperature | `exp3_escape/regime.py` | `results/exp3b_*.json` |
| 3c. Loops as orbits in hidden-state space | `exp3_escape/orbit.py` | `results/exp3c_*.json` |
| 3d. Loop-triggered temperature control (negative result) | `exp3_escape/control.py` | `results/exp4_gpt2.json` |
| 3e. Scaling across Pythia sizes | `exp3_escape/plot_scaling.py` | `results/exp3e_scaling_summary.json` |
| 3f. Reasoning model (Qwen3-0.6B, GSM8K) vs the recommended temperature | `exp3_escape/reasoning.py`, `reasoning_plot.py` | `results/exp3f_*.json` |

## Layout

```
common/prompts.py          40 author-written prompts used by Exp. 1 and 3
exp1_ftle/                 Jacobian power iteration per layer (torch.func JVP/VJP)
exp2_bifurcation/          full-batch GD sweep, Hessian power iteration
exp3_escape/               greedy loops, exact stay probabilities, Monte Carlo check
results/                   raw JSON outputs and summaries used in the paper
paper/                     LaTeX source (TMLR style), refs.bib generator, figures
data/arxiv_records.json    arXiv API records behind every arXiv citation
PAPERS.md                  the reading list (189 papers), grouped by clock
DICTIONARY.md              the Three-Clock Dictionary as a markdown table
CITATION.cff               citation metadata (GitHub "Cite this repository")
```

## Reading list

`PAPERS.md` lists every paper in the review with one line on why it matters. Pull requests with new papers are welcome.

## Citation

If you use the code, the data, the dictionary or the reading list, please cite:

```bibtex
@article{farid2026threeclocks,
  title  = {Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation},
  author = {Farid, Maor},
  year   = {2026}
}
```

## License

Code: MIT. Paper text and figures: CC BY 4.0.
