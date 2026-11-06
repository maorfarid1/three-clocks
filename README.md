# The Three Clocks of a Transformer

Code, data and reading list for

> **Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation**
> Maor Farid

A transformer language model is iterated in three ways: over layers (clock 1), over optimizer steps (clock 2) and over generated tokens (clock 3). The paper reviews the dynamical-systems results on each clock, collects them in the Three-Clock Dictionary, and runs three small experiments on open models. Everything here runs on a laptop CPU.

## Reproduce

```bash
pip install -r requirements.txt   # CPU torch: --index-url https://download.pytorch.org/whl/cpu
make all                          # experiments -> figures -> paper/main.pdf
make arxiv                        # de-anonymized build: paper/main_arxiv.pdf
```

`results/` already contains the raw outputs used in the paper, so `make figures paper` rebuilds the figures and PDF in a minute without rerunning the experiments.

| Experiment | Script | Output | Runtime (2 CPU cores) |
| --- | --- | --- | --- |
| 1. Finite-time Lyapunov exponents of the residual stream across Pythia checkpoints | `exp1_ftle/ftle.py` | `results/exp1.json` | about 10 min |
| 2. Learning-rate bifurcation diagram of a one-block transformer, 3 seeds | `exp2_bifurcation/bifurcation.py` | `results/exp2*.json` | about 40 min per seed (3 seeds) |
| 3. Escape from repetition loops in GPT-2 and Pythia-160M | `exp3_escape/escape.py` | `results/exp3_*.json` | about 20 min per model |

## Layout

```
common/prompts.py          40 author-written prompts used by Exp. 1 and 3
exp1_ftle/                 Jacobian power iteration per layer (torch.func JVP/VJP)
exp2_bifurcation/          full-batch GD sweep, Hessian power iteration
exp3_escape/               greedy loops, exact stay probabilities, Monte Carlo check
results/                   raw JSON outputs and summaries used in the paper
paper/                     LaTeX source (TMLR style), refs.bib generator, figures
data/arxiv_records.json    arXiv API records behind every arXiv citation
PAPERS.md                  the reading list (177 papers), grouped by clock
```

## Reading list

`PAPERS.md` lists every paper in the review with one line on why it matters. Pull requests with new papers are welcome.

## Citation

```bibtex
@article{farid2026threeclocks,
  title  = {Large Language Models as Nonlinear Dynamical Systems: The Three Clocks of Depth, Training, and Generation},
  author = {Farid, Maor},
  year   = {2026}
}
```

## License

Code: MIT. Paper text and figures: CC BY 4.0.
