# One-command reproduction:  make all
# Every figure and number in the paper is produced by these targets.
PY ?= python3

.PHONY: all experiments figures paper arxiv clean

all: experiments figures paper

experiments: results/exp1.json results/exp2.json results/exp2_seed1.json results/exp2_seed2.json results/exp3_gpt2.json results/exp3_pythia160m.json results/exp3b_gpt2.json results/exp3b_pythia160m.json results/exp3b_human.json results/exp3c_gpt2.json results/exp3c_pythia160m.json results/exp4_gpt2.json results/eos_geom_seed0.json results/eos_geom_seed1.json results/eos_geom_seed2.json results/eos_rotation/rot_seed0.json results/eos_rotation/rot_seed1.json results/eos_rotation/rot_seed2.json scaling reasoning atlas controllers reduced

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

results/exp3b_gpt2.json:
	$(PY) exp3_escape/regime.py --model gpt2 --temps 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5,2.0 --samples 3 --threads 2 --out $@

results/exp3b_pythia160m.json:
	$(PY) exp3_escape/regime.py --model EleutherAI/pythia-160m --temps 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5,2.0 --samples 3 --threads 2 --out $@

results/exp3b_human.json:
	$(PY) exp3_escape/regime.py --human --threads 2 --out $@

results/exp3c_gpt2.json:
	$(PY) exp3_escape/orbit.py --model gpt2 --out $@

results/exp3c_pythia160m.json:
	$(PY) exp3_escape/orbit.py --model EleutherAI/pythia-160m --out $@

results/exp4_gpt2.json:
	$(PY) exp3_escape/control.py --model gpt2 --out $@

results/eos_geom_seed%.json:
	$(PY) exp2_bifurcation/eos_geometry.py --seed $* --out $@

results/eos_rotation/rot_seed%.json:
	mkdir -p results/eos_rotation && cd exp2_bifurcation && $(PY) eos_rotation.py --seed $* --window 400 --every 50 --arnoldi_at 200 --out ../$@

# Model-size scaling (Exp. 3e). Larger models run without Monte Carlo checks to fit in 6 GB of memory;
# Pythia-1B runs in two halves of 20 prompts that are merged afterwards.
SIZE_ARGS = --mc_loops 0 --free_temps ""
REG_ARGS = --temps 0.3,0.5,0.6,0.7,0.8,0.9,1.0,1.2 --samples 3 --batch 10 --threads 2
.PHONY: scaling reasoning
scaling: results/exp3_pythia70m.json results/exp3_pythia410m.json results/exp3_pythia1b.json results/exp3b_pythia410m.json results/exp3b_pythia1b.json

results/exp3_pythia70m.json:
	$(PY) exp3_escape/escape.py --model EleutherAI/pythia-70m --out $@

results/exp3_pythia410m.json:
	$(PY) exp3_escape/escape.py --model EleutherAI/pythia-410m $(SIZE_ARGS) --out $@

results/exp3_pythia1b.json:
	$(PY) exp3_escape/escape.py --model EleutherAI/pythia-1b $(SIZE_ARGS) --n_prompts 20 --first_prompt 0 --out results/exp3_pythia1b_a.json
	$(PY) exp3_escape/escape.py --model EleutherAI/pythia-1b $(SIZE_ARGS) --n_prompts 40 --first_prompt 20 --out results/exp3_pythia1b_b.json
	$(PY) exp3_escape/merge_parts.py $@ results/exp3_pythia1b_a.json results/exp3_pythia1b_b.json

results/exp3b_pythia%.json:
	$(PY) exp3_escape/regime.py --model EleutherAI/pythia-$* $(REG_ARGS) --defer_ref --out results/exp3b_pythia$*_raw.json
	$(PY) exp3_escape/regime.py --score_file results/exp3b_pythia$*_raw.json --out $@

# Reasoning model (Exp. 3f): Qwen3-0.6B on GSM8K with thinking on and off.
reasoning: results/exp3f_qwen3_think_sample.json results/exp3f_qwen3_nothink_sample.json

results/exp3f_qwen3_think_sample.json:
	$(PY) exp3_escape/reasoning.py --model Qwen/Qwen3-0.6B --think 1 --mode sample --n 18 --batch 6 --gen 1024 --samples 1 --temps 0.2,0.4,0.6,1.0,1.3 --out $@

results/exp3f_qwen3_nothink_sample.json:
	$(PY) exp3_escape/reasoning.py --model Qwen/Qwen3-0.6B --think 0 --mode sample --n 20 --gen 640 --samples 2 --temps 0.2,0.4,0.7,1.0,1.3 --out $@

# Regime atlas (Duffing vs GPT-2) and its statistics; controllers for clock 3 (Exp. 4b, 4c).
.PHONY: atlas controllers
atlas: results/atlas.json results/atlas_stats.json
results/atlas.json:
	$(PY) exp3_escape/atlas.py --gen 576 --out $@
results/atlas_stats.json:
	$(PY) exp3_escape/atlas_stats.py --n 10 --out $@
controllers: results/exp4b_gpt2.json results/exp4c_summary.json
# Reduced-order models (Section 2.3): deepening-well model of loop escape; barrier/noise regime diagram.
.PHONY: reduced
reduced: results/reduced.json results/collapse.json
results/reduced.json: results/exp3_gpt2.json results/exp3_pythia160m.json
	$(PY) exp3_escape/reduced.py --out $@
results/collapse.json: results/exp3_gpt2.json
	$(PY) exp3_escape/collapse.py --n_prompts 5 --out $@
results/exp4b_gpt2.json:
	$(PY) exp3_escape/rollback.py --temps 0.5 --out $@
results/exp4c_summary.json:
	$(PY) exp3_escape/regime_control.py --only "feedback T0=0.4,feedback T0=0.6,feedback T0=0.8" --out results/exp4c_gpt2.json
	$(PY) exp3_escape/score_control.py results/exp4c_gpt2.json.raw $@

figures:
	$(PY) exp1_ftle/plot.py results/exp1.json paper/figs/exp1_ftle.pdf
	$(PY) exp2_bifurcation/plot.py results/exp2.json,results/exp2_seed1.json,results/exp2_seed2.json paper/figs/exp2_bifurcation.pdf
	$(PY) exp3_escape/plot.py results/exp3_gpt2.json,results/exp3_pythia160m.json paper/figs/exp3_escape.pdf
	$(PY) exp3_escape/lock.py results/exp3_gpt2.json,results/exp3_pythia160m.json
	$(PY) exp3_escape/plot_lock.py
	$(PY) exp3_escape/plot_regime.py
	$(PY) exp3_escape/eval_control.py
	$(PY) exp3_escape/plot_scaling.py
	$(PY) exp3_escape/reasoning_plot.py
	$(PY) paper/fig_pairs.py
	$(PY) exp3_escape/plot_atlas.py
	$(PY) exp3_escape/plot_reduced.py
	$(PY) exp3_escape/plot_collapse.py

paper:
	cd paper && $(PY) make_bib.py && latexmk -pdf -interaction=nonstopmode main.tex

arxiv:
	cd paper && latexmk -pdf -interaction=nonstopmode -jobname=main_arxiv -pdflatex='pdflatex %O "\def\ARXIV{1}\input{%S}"' main.tex

clean:
	cd paper && latexmk -C
