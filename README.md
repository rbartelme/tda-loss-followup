# tda-loss-followup

Follow-up to [Beyond MTEB: A Topology-Aware Embedder Bake-Off Across Two Corpora](https://rbartelme.github.io/blog/tda-embedder-bakeoff/).
The first post argued that cosine bake-offs measure an encoder's training
objective while Mapper topology measures its fit to the corpus. This repo asks
the next question directly: holding architecture fixed, which parts of the
training objective move the topology? It fine-tunes the same encoders under
sweeps of temperature, loss family, pair construction and topological
auxiliaries, and scores every checkpoint with the first post's three-layer
harness (copied verbatim from
[tda-embedder-bakeoff](https://github.com/rbartelme/tda-embedder-bakeoff))
plus a new router layer. One CSV row per (model, loss, τ, pair set, aux, λ,
checkpoint, corpus), each carrying the checkpoint's optimizer step and the
run's total; the post is written from those CSVs.

## The four experiments

| | question | grid |
|---|---|---|
| **Exp 1** temperature | does τ alone move the topology? | `minilm`, `biomedbert-fulltext`, `bge-base` × MNRL × τ ∈ {0.01, 0.05, 0.2, 1.0}, random pairs |
| **Exp 2** loss family | same data and τ, different objective | `biomedbert-fulltext` × {MNRL, triplet, CoSENT, MLM} at τ = 0.05 |
| **Exp 3** data vs loss | does the pair set matter more than the loss? | `biomedbert-fulltext` × MNRL × {matched, mismatched, random} pairs, against the MedCPT reference row |
| **Exp 4** topological auxiliaries | can an explicit topology term buy faithfulness? | `biomedbert-fulltext` × MNRL + λ·{textstat head, distance preservation, persist0 H0, TopoAE H0}, λ ∈ {0.1, 1.0} |

Every run saves checkpoints at fractions {0, 0.1, 0.25, 0.5, 1.0} of training;
each checkpoint is evaluated on SciCUEval (in-domain) and MMLU non-STEM
(contrast) with:

1. **Layer 1** cosine geometry: within/between cosine, gap, LR accuracy.
2. **Layer 2** the bakeoff's 25-seed UMAP → KeplerMapper → HDBSCAN bootstrap: ARI, NMI, coverage, node count.
3. **Layer 3** linguistic anchors: weighted purity, anchor Spearman ρ (Mapper graph distance vs standardized textstat distance), per-feature alignment.
4. **Layer 4** router (new): one logistic regression over the union of both corpora's domain labels (5 scientific + 5 general), trained on a stratified 80% of both eval splits, scored on held-out SciCUEval rows (`router_acc_in`) and MMLU rows (`router_acc_mmlu`), with the cross-corpus misroute fractions (`router_in_to_mmlu_frac`, `router_mmlu_to_in_frac`) beside them.

A Mapper graph with coverage < 0.05 or fewer than 5 nodes is flagged
`disintegrated`; its ARI and ρ are not to be read.

## Quick start

```bash
git clone https://github.com/rbartelme/tda-loss-followup.git
cd tda-loss-followup
uv sync
```

Data. `configs/base.yaml` points at the bakeoff's flattened JSONLs
(`scratch/*_combined.jsonl` in a local clone of tda-embedder-bakeoff) and, as
a fallback, at the raw SciCUEval and MMLU trees produced by the bakeoff's
download scripts. Either source gives the byte-identical evaluation sample; a
parity test proves it against the bakeoff's cached corpus when that is present.
Edit the `paths:` block if yours live elsewhere.

Then, in order:

```bash
make test          # unit tests on synthetic inputs (no model downloads)
make dry-run       # 3 training steps on 64 pairs with MiniLM, all checkpoints saved
make eval-smoke    # the whole evaluator on 50 texts per corpus with a stand-in encoder
make repro-check   # untrained encoders must reproduce the first post (see below)
make pairs         # random / matched / mismatched pair sets from the train split
make exp1-final    # Exp 1, evaluating only ckpt_0.0 and ckpt_1.0
```

## `make` targets

| target | what it does |
|---|---|
| `pairs` | builds `data/pairs/{random,matched,mismatched}_s$(SEED)/pairs.parquet` (default `N_PAIRS=20000`, `SEED=0`) |
| `dry-run` | `python -m tlf.train --dry-run` (`LOSS=mnrl\|triplet\|cosent\|mlm`) |
| `eval-smoke` | `python -m tlf.evaluate --smoke` |
| `repro-check` | evaluates the three untrained starting encoders and prints them beside `configs/reference.yaml` |
| `exp1-final` | Exp 1 with `--only-final` |
| `exp1` | the temperature sweep, all five checkpoints |
| `exp2` `exp3` `exp4` | the loss-family, data and auxiliary grids at init and final only, as the protocol asks; run `scripts/run_experiment.py` without `--only-final` for all five |
| `figures` | the four figures from `results/*.csv` into `figures/` |
| `test` / `lint` | `pytest -q` / `ruff check .` |

All experiment targets skip finished work, so an interrupted sweep is simply
relaunched. Checkpoints and their evaluations are keyed by run, not by
experiment: a run that several grids contain (Exp 1's `biomedbert-fulltext`
τ 0.05 random run is also Exp 2's MNRL run, Exp 3's random run and Exp 4's
no-aux baseline) trains once and runs its Mapper bootstrap once, and each
experiment writes its own CSV row from the cached metrics. Likewise every
run's `ckpt_0.0` is a symlink to `checkpoints/_untrained/<model>/`, the base
model saved once, so the untrained row is evaluated once per base model rather
than once per run. Pass `N_WORKERS=20`
to any of them to size the Mapper bootstrap pool. `EXP_FLAGS` passes anything
else to the driver: after `exp1-final` has shown where anchor ρ moves,
`make exp1 EXP_FLAGS="--tau 0.01 --tau 0.05"` fills in the intermediate
checkpoints for just those temperatures (`--model` narrows likewise; both
repeat and intersect). `scripts/run_experiment.py --dry-run` prints the plan,
including which runs are selected and how many rows will come from cache,
without running anything; `--force` redoes everything.

## Hardware

- **Training runs on the laptop.** The reference machine is the bakeoff's:
  an RTX 4070 Laptop with 8 GB of VRAM. Effective batch 128 fits because MNRL
  uses gradient caching in mini-batches of 32 (`train.micro_batch`), and the
  other losses accumulate over the same micro-batch. Evaluation encodes with
  the bakeoff's `encode_all` at `max_length 256`, batch 32.
- **Mapper sweeps run on Spark.** UMAP single-threads itself when seeded, so
  the 25-seed bootstrap is parallelised across forked worker processes. The
  bakeoff's README reports about 100 minutes for 28 (encoder, corpus)
  evaluations at 8 workers; `docs/protocol.md` budgets more conservatively at
  about 40 minutes per SciCUEval row and 25 per MMLU row, roughly 60 hours for
  the whole sweep on 8 workers or 20–25 on the DGX Spark at 20. Exp 1 with
  `--only-final` is 32 evaluations; the full Exp 1 is 80. On Spark use
  `make exp1 N_WORKERS=20`.
  The distance matrix is 4000 × 4000 float64 per worker under copy-on-write,
  so memory is not the constraint; cores are.
- Training wall time has not been measured yet; the dry-run only proves the
  loop. Expect the first `make exp1-final` to be dominated by the eight
  fine-tunes, each about 470 optimizer steps at batch 128.

## Reproduction check

Before any fine-tune result is trusted, the untrained starting encoders must
reproduce the first post. `ckpt_0.0` of every run is exactly the HF model with
mean pooling, so this is the same check as evaluating the HF ids directly:

```bash
make repro-check
```

prints, for `biomedbert-fulltext` and `minilm` on both corpora, this repo's
Layer 1–3 numbers beside the rows in `configs/reference.yaml` (which were
read from the bakeoff's cached metrics at commit `4625b22`). Every stage is
seeded the way the bakeoff seeded it (sample seed 42, evaluation seed 42,
bootstrap seeds 42–66). The protocol's tolerance is ±0.005 on ARI and ρ. A
larger difference means the evaluation sample or the encoding differs, and
nothing downstream should be run until it is explained. See
`docs/harness-notes.md` §1 for why the sample must be byte-identical and §8
for the MedCPT rows.

## Layout

```
configs/          base.yaml (every knob), experiments.yaml (grid-wide overrides: CV LR), models.yaml, exp1–4, reference.yaml
src/tlf/
  data.py         corpus loaders reproducing the bakeoff eval sample; pair construction
  features.py     the six textstat features, verbatim from the bakeoff
  train.py        fine-tune driver: MNRL (cached), triplet, CoSENT, MLM; checkpoints; aux hook
  losses.py       textstat head, distance preservation, persist0 H0 and TopoAE H0 (pairs from PyPI persist0-tda)
  evaluate.py     layers 1–4 per checkpoint, embedding + metrics caches, --smoke
  experiment.py   config grid → runs → train → evaluate → rows
  results.py      CSV schema, idempotent append, manifests
  plots.py        the four figures
  _bakeoff/       functions copied from tda-embedder-bakeoff @ 4625b22 (see headers)
scripts/          build_pairs.py, run_experiment.py, make_figures.py
docs/             harness-notes.md (what the bakeoff does, exactly), protocol.md
results/          one CSV per experiment (committed)
data/ checkpoints/ embeddings/   gitignored artifacts, each with a manifest.json
```

## Notes on choices

- Mapper parameters, textstat features, the 25-seed bootstrap, the Layer 1
  sampling and the LR classifier are the bakeoff's, unchanged. Where a value
  can differ it is a config field whose default is the bakeoff's: the Layer 1
  LR is the single 80/20 holdout in `base.yaml`, which `make repro-check`
  uses; the four experiment grids inherit `configs/experiments.yaml`, which
  switches it to stratified 5-fold CV (the CSV's `lr_eval` column says which),
  and evaluation encoding is 256 tokens while training uses 128.
- `ari_sd` did not exist in the first post; it is the standard deviation of
  the same 300 pairwise ARIs whose mean the post reported.
- The two corpora share a label schema (`source_subset`, `source_domain`) but
  no label values at either level, so a SciCUEval-trained classifier has no
  MMLU accuracy; the router is a union classifier instead, which matches the
  first post's router (it classifies prompts by domain, and general-domain
  prompts are traffic for a different model, not rejects). The experiment
  grids route over `domain`; `base.yaml` keeps `subset` for `make
  repro-check`. `eval.router.mode: per_corpus` is the alternative; it
  duplicates Layer 1's `lr_acc`.
- Cosine distance is bounded by 2 and standardized textstat distance is not,
  so the three distance-based auxiliaries rescale their fixed reference onto
  the live scale (`ref_scale: match_mean`; `none` is the literal comparison).
  Rescaling never changes the reference's MST, only its units.
- `persist0_h0` and `topoae_h0` are an A/B. Both take H0 (MST) pairs from
  persist0; the first matches the sorted death vector, which is invariant to
  which points a bar joins, the second is TopoAE's exact form (Moor et al.
  2020, dimension 0), distances at each space's own MST edges in both
  directions. The comparison isolates whether pairing information matters.
- The pair builder samples anchors without replacement and only reuses an
  anchor once every anchor has been used; at 20 000 pairs from 7 343 train
  rows each appears about 2.7 times. Batches never contain the same id twice.

## License

MIT. The corpora carry their own licenses (SciCUEval CC-BY 4.0, MMLU MIT).
