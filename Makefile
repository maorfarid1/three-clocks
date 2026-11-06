# One-command reproduction:  make all
# Every figure and number in the paper is produced by these targets.
PY ?= python3

.PHONY: all experiments figures paper arxiv clean

all: experiments figures paper

experiments: results/exp1.json results/exp2.json results/exp2_seed1.json results/exp2_seed2.json results/exp3_gpt2.json results/exp3_pythia160m.json

results/exp2.json:
	$(PY) exp2_bifurcation/bifurcation.py --out $@

results/exp2_seed%.json:
	$(PY) exp2_bifurcation/bifurcation.py --seed $* --out $@

results/exp1.json:
	$(PY) exp1_ftle/ftle.py --models EleutherAI/pythia-70m --steps 0,128,512,1000,2000,4000,8000,16000,32000,64000,143000 --n_prompts 8 --T 16 --iters 10 --out $@
	$(PY) exp1_ftle/ftle.py --models EleutherAI/pythia-160m --steps 0,512,2000,8000,32000,143000 --n_prompts 8 --T 16 --iters 10 --out $@

results/exp3_gpt2.json:
	$(PY) exp3_escape/escape.py --model gpt2 --out $@

results/exp3_pythia160m.json:
	$(PY) exp3_escape/escape.py --model EleutherAI/pythia-160m --out $@

figures:
	$(PY) exp1_ftle/plot.py results/exp1.json paper/figs/exp1_ftle.pdf
	$(PY) exp2_bifurcation/plot.py results/exp2.json,results/exp2_seed1.json,results/exp2_seed2.json paper/figs/exp2_bifurcation.pdf
	$(PY) exp3_escape/plot.py results/exp3_gpt2.json,results/exp3_pythia160m.json paper/figs/exp3_escape.pdf

paper:
	cd paper && $(PY) make_bib.py && latexmk -pdf -interaction=nonstopmode main.tex

arxiv:
	cd paper && latexmk -pdf -interaction=nonstopmode -jobname=main_arxiv -pdflatex='pdflatex %O "\def\ARXIV{1}\input{%S}"' main.tex

clean:
	cd paper && latexmk -C
