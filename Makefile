PY ?= python
DATA ?= data
DEV ?= auto
SEEDS ?= 42 123 456 789 1011 2024
.PHONY: env test smoke run-e1 run-ablation stats figures tune
env:
	$(PY) -m pip install -r requirements.txt
test:
	$(PY) -m pytest -q tests
smoke:
	$(PY) -m gcids.run --dataset toy --config configs/base.yaml configs/smoke.yaml --device $(DEV)
	$(PY) -m gcids.report --results results_smoke
# minutes on a GPU: one seed, NSL-KDD, quick settings
smoke-nsl:
	$(PY) -m gcids.run --dataset nslkdd --config configs/base.yaml configs/nslkdd.yaml configs/smoke.yaml --seeds 42 --data-root $(DATA) --device $(DEV)
tune:
	$(PY) -m gcids.tune --dataset nslkdd --config configs/base.yaml configs/nslkdd.yaml --seed 0 --data-root $(DATA)
# E1 main table + label-role controls + baselines + generator quality (all datasets)
run-e1:
	for d in nslkdd unswnb15 cicids2017; do $(PY) -m gcids.run --dataset $$d --config configs/base.yaml configs/$$d.yaml --seeds $(SEEDS) --data-root $(DATA) --device $(DEV); done
# E4-E9 ablations (NSL-KDD first; repeat for unswnb15 with DS=unswnb15)
DS ?= nslkdd
run-ablation:
	for a in gan pgen dial s1 sensitivity vae; do $(PY) -m gcids.run --dataset $(DS) --config configs/base.yaml configs/$(DS).yaml configs/ablation_$$a.yaml --seeds $(SEEDS) --data-root $(DATA) --device $(DEV); done
stats:
	$(PY) -m gcids.report --results results
figures:
	for d in nslkdd unswnb15 cicids2017; do $(PY) -m gcids.figures --results results --dataset $$d; done

# recompute the numbers of the manuscript from the archived raw results
analysis:
	$(PY) analysis/calc_numbers.py results_archive analysis/outputs/numbers.json
	$(PY) analysis/calc_unsw.py results_archive analysis/outputs/numbers_unsw.json
	$(PY) analysis/calc_ext.py results_archive analysis/outputs/numbers_ext.json
	$(PY) analysis/fig_dial.py analysis/outputs/numbers.json analysis/outputs/fig4_dial.png
.PHONY: analysis
