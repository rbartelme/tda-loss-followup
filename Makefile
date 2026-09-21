# Every target is a thin wrapper over a script + a YAML config. Override
# SEED / N_PAIRS / N_WORKERS on the command line, e.g. `make exp1 N_WORKERS=20`.
UV       ?= uv run
SEED     ?= 0
N_PAIRS  ?= 20000
N_WORKERS ?=

WORKERS_FLAG := $(if $(N_WORKERS),--n-workers $(N_WORKERS),)

.PHONY: pairs exp1-final exp1 exp2 exp3 exp4 figures test lint

pairs:
	$(UV) python scripts/build_pairs.py --config configs/base.yaml --kind random     --n $(N_PAIRS) --seed $(SEED)
	$(UV) python scripts/build_pairs.py --config configs/base.yaml --kind matched    --n $(N_PAIRS) --seed $(SEED)
	$(UV) python scripts/build_pairs.py --config configs/base.yaml --kind mismatched --n $(N_PAIRS) --seed $(SEED)

exp1-final:
	$(UV) python scripts/run_experiment.py --config configs/exp1_tau_sweep.yaml --only-final --resume $(WORKERS_FLAG)

exp1:
	$(UV) python scripts/run_experiment.py --config configs/exp1_tau_sweep.yaml --resume $(WORKERS_FLAG)

exp2:
	$(UV) python scripts/run_experiment.py --config configs/exp2_loss_family.yaml --resume $(WORKERS_FLAG)

exp3:
	$(UV) python scripts/run_experiment.py --config configs/exp3_data_vs_loss.yaml --resume $(WORKERS_FLAG)

exp4:
	$(UV) python scripts/run_experiment.py --config configs/exp4_topo_aux.yaml --resume $(WORKERS_FLAG)

figures:
	$(UV) python scripts/make_figures.py --results results --reference configs/reference.yaml --out figures

test:
	$(UV) pytest -q

lint:
	$(UV) ruff check .
