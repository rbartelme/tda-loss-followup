# Every target is a thin wrapper over a script + a YAML config. Override
# SEED / N_PAIRS / N_WORKERS on the command line, e.g. `make exp1 N_WORKERS=20`.
# EXP_FLAGS passes extra flags to scripts/run_experiment.py, e.g.
# `make exp1 EXP_FLAGS="--tau 0.01 --tau 0.05"` to fill in the intermediate
# checkpoints for the temperatures that moved (protocol step 3).
UV       ?= uv run
SEED     ?= 0
N_PAIRS  ?= 20000
N_WORKERS ?=
EXP_FLAGS ?=

WORKERS_FLAG := $(if $(N_WORKERS),--n-workers $(N_WORKERS),)

.PHONY: pairs dry-run eval-smoke repro-check exp1-final exp1 exp2 exp3 exp4 figures test lint

pairs:
	$(UV) python scripts/build_pairs.py --config configs/base.yaml --kind random     --n $(N_PAIRS) --seed $(SEED)
	$(UV) python scripts/build_pairs.py --config configs/base.yaml --kind matched    --n $(N_PAIRS) --seed $(SEED)
	$(UV) python scripts/build_pairs.py --config configs/base.yaml --kind mismatched --n $(N_PAIRS) --seed $(SEED)

dry-run:
	$(UV) python -m tlf.train --config configs/base.yaml --dry-run --loss $(or $(LOSS),mnrl)

eval-smoke:
	$(UV) python -m tlf.evaluate --config configs/base.yaml --smoke $(WORKERS_FLAG)

# Reproduction check: the untrained starting encoders must reproduce the first
# post's rows (configs/reference.yaml) before any fine-tune result is trusted.
repro-check:
	$(UV) python -m tlf.evaluate --config configs/base.yaml --ckpt microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --reference biomedbert-fulltext $(WORKERS_FLAG)
	$(UV) python -m tlf.evaluate --config configs/base.yaml --ckpt sentence-transformers/all-MiniLM-L6-v2 --reference minilm $(WORKERS_FLAG)

exp1-final:
	$(UV) python scripts/run_experiment.py --config configs/exp1_tau_sweep.yaml --only-final --resume $(WORKERS_FLAG) $(EXP_FLAGS)

exp1:
	$(UV) python scripts/run_experiment.py --config configs/exp1_tau_sweep.yaml --resume $(WORKERS_FLAG) $(EXP_FLAGS)

exp2:
	$(UV) python scripts/run_experiment.py --config configs/exp2_loss_family.yaml --only-final --resume $(WORKERS_FLAG) $(EXP_FLAGS)

exp3:
	$(UV) python scripts/run_experiment.py --config configs/exp3_data_vs_loss.yaml --only-final --resume $(WORKERS_FLAG) $(EXP_FLAGS)

exp4:
	$(UV) python scripts/run_experiment.py --config configs/exp4_topo_aux.yaml --only-final --resume $(WORKERS_FLAG) $(EXP_FLAGS)

figures:
	$(UV) python scripts/make_figures.py --results results --reference configs/reference.yaml --out figures

test:
	$(UV) pytest -q

lint:
	$(UV) ruff check .
