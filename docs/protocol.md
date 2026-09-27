# Loss Functions vs. Topology: Experiment Protocol

**Status: walk-through complete, reproduction check run, experiments not yet
run.** The protocol below is as written on 2026-09-21; the discrepancies between
it and the scaffold as built are listed under *To be resolved* at the end. Items
1–6, 9 and 10 were resolved on 2026-09-22 and items 8, 11 and 12 on 2026-09-27,
each struck through with a dated resolution and its own commit; item 7 (HF Hub
push) is deferred until the experiments have run.

Follow-up to *Beyond MTEB: A Topology-Aware Embedder Bake-Off*. The first post argued
that training regime, not architecture, determines whether an encoder's output manifold
tracks linguistic surface features. This experiment tests that causally: same base model,
same pairs, vary only the loss and its temperature, run the same three-layer evaluation.

## Hypotheses

- **H1 (mechanism).** InfoNCE removes the surface-feature gradient. Anchor ρ decays
  monotonically with training steps, and faster at lower temperature τ.
- **H2 (anti-correlation is within-model).** Cosine gap and anchor ρ are anti-correlated
  along the τ axis inside a single model, not just across the roster.
- **H3 (data, not loss, explains MedCPT).** InfoNCE on positives matched on textstat
  features preserves anchor ρ; InfoNCE on mismatched positives destroys it.
- **H4 (topology-aware losses recover both).** An auxiliary structure-preserving term
  yields high cosine gap *and* high anchor ρ.

## Roster

| key | base | role |
| --- | --- | --- |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | Cosine winner, anchor ρ floor. Cheap for dense checkpoint sweeps. |
| `biomedbert-fulltext` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | Pure MLM, never contrastive. Clean starting point for watching ρ decay. |
| `bge-base` *(optional)* | `BAAI/bge-base-en-v1.5` | Same architecture as BiomedBERT; isolates general vs. biomedical pretraining. |
| `medcpt-query` | `ncbi/MedCPT-Query-Encoder` | Reference only (not re-trained). Target to reconstruct in Exp 3. |

Pooling: mean over attention mask for all models, as in the first post. No prefixes.

## Training data

- Positive pairs from SciCUEval: two MCQ stems from the same `source_subset`.
  Hold out the 400/sub-dataset evaluation sample used in the first post so eval prompts
  never appear in training.
- Target ~20k pairs. Same pair set for every run unless the experiment says otherwise.
- Store pair construction seed and the pair file with the results.

## Fixed settings (all runs)

- Effective batch 128 (in-batch negatives). Use `CachedMultipleNegativesRankingLoss`
  if VRAM-limited so effective batch is hardware-independent.
- `max_length` 128. AdamW, lr 2e-5, linear warmup 10%, bf16.
- 3 epochs. Save checkpoints at 0%, 10%, 25%, 50%, 100% of steps.
- Full fine-tuning; no LoRA.
- Log train loss, mean positive cosine, mean in-batch negative cosine per step.

## Experiment 1: Temperature sweep

- Models: `minilm`, `biomedbert-fulltext`.
- Loss: InfoNCE (MNRL). τ ∈ {0.01, 0.05, 0.2, 1.0}.
- Evaluate all five checkpoints per τ → 2 × 4 × 5 = 40 rows.
- Primary plot: anchor ρ vs. training step, one line per τ, one panel per model.
- Secondary: cosine gap vs. anchor ρ scatter over all 40 rows (H2).

## Experiment 2: Loss-family ablation

- Model: `biomedbert-fulltext` (add `minilm` if time allows).
- Losses at τ = 0.05 (or the equivalent margin): MNRL, triplet/margin, CoSENT,
  continued MLM (control, same corpus, no pairs).
- Evaluate init and final only → 4 rows (+1 shared init).
- Question: is the ρ decay specific to InfoNCE-style losses, or does any
  fine-tuning on this corpus produce it?

## Experiment 3: Data vs. loss (the MedCPT question)

- Model: `biomedbert-fulltext`. Loss: MNRL, τ = 0.05.
- Build two pair sets of equal size from the same pool:
  - **matched**: positives within 0.5 SD of each other on standardized textstat distance
  - **mismatched**: positives deliberately > 1.5 SD apart
- Evaluate init and final → 2 rows.
- Compare final anchor ρ and ARI against `medcpt-query` on SciCUEval (0.532 / 0.864).
  Also run both on MMLU to see whether either reproduces the disintegration.

## Experiment 4: Topology-aware auxiliary losses

- Model: `biomedbert-fulltext`. Base loss: MNRL, τ = 0.05.
- Auxiliary terms, each with weight λ ∈ {0.1, 1.0}:
  - textstat regression head (6 features, MSE)
  - distance preservation: MSE between pairwise embedding distance and pairwise
    standardized textstat distance within the batch
  - one persistent-homology regularizer (TopoAE loss or RTD)
- Evaluate final only → 6 rows.
- Report cosine gap, ARI, purity, anchor ρ, and downstream routing accuracy.

## Evaluation (unchanged from post 1)

Run `embedding_diagnostic.py` and `embedding_tda.py` on every checkpoint, both corpora:

- Layer 1: within/between cosine, gap, LR accuracy (switch to 5-fold CV this time)
- Layer 2: 25-seed UMAP → KeplerMapper → HDBSCAN(precomputed cosine); ARI, NMI, coverage, node count
- Layer 3: weighted purity, anchor Spearman ρ, per-feature alignment
- **Layer 4 (new):** router accuracy. Logistic regression on subject label, trained on
  SciCUEval, tested on (a) SciCUEval held-out and (b) MMLU. Reports whether
  topology-faithfulness predicts what the router actually needs.

Keep Mapper parameters identical to post 1. Flag any row with coverage < 0.05 or
mean node count < 5 as "disintegrated" and exclude its ARI from rankings.

## Compute plan

- Fine-tuning: minutes per run; negligible.
- Mapper bootstrap: ~40 min (SciCUEval) + ~25 min (MMLU) per row.
  ~55 rows total → ~60 hours on 8 workers; ~20–25 hours on the DGX Spark at 20 workers
  with the two corpora run concurrently.
- Before the sweep: re-run one first-post encoder on the new hardware and confirm
  ARI / ρ reproduce within ±0.005.

## Order of operations

1. Hardware reproduction check.
2. Exp 1 with init + final checkpoints only (16 rows). Find where ρ moves.
3. Fill in intermediate checkpoints for the τ values that showed movement.
4. Exp 3, then Exp 2, then Exp 4.
5. Layer 4 router eval across every final checkpoint.

## Deliverables

- `results/*.csv` with one row per (model, loss, τ, checkpoint, corpus).
- Figures: ρ-vs-step curves; gap-vs-ρ scatter; Exp 3 bar chart against MedCPT;
  Exp 4 Pareto plot of gap vs. ρ with router accuracy as marker size.
- Trained checkpoints pushed to HF Hub for reproducibility.

## To be resolved

Differences between this protocol, the scaffold brief, and the code as built
(commit `c26cb26`). Each is a decision, not a bug; the code runs either way.

1. ~~**Exp 3 pair sets.** Protocol: matched and mismatched. Built: matched,
   mismatched and random (from the brief). The random run is identical to
   Exp 1's `biomedbert-fulltext` τ 0.05 run, Exp 2's MNRL run and Exp 4's
   no-aux baseline, and the driver keys checkpoints by experiment, so it
   would be trained four times. Proposed: key checkpoint directories by run
   id alone so identical runs train once and each experiment writes its rows
   from cached embeddings; random then stays in Exp 3 for the figure at no cost.~~
   **Resolved 2026-09-22.** Random stays in Exp 3; the shared run trains once
   and is Mapper-evaluated once. Checkpoint directories are keyed by run id
   alone (`checkpoints/<run>/`), so later experiments find the run trained.
   Each evaluation's metrics are cached beside its embeddings
   (`embeddings/<run>/ckpt_<frac>/<corpus>.metrics.json`) under a key
   covering the checkpoint, the eval rows, the encoder, the seed count, the
   router flag and a hash of the result-determining `eval` settings; a later
   experiment reads them back and writes its own CSV row. `--force` ignores
   both caches. The cache only hits if the four experiment configs share one
   `eval` section, which constrains items 2 and 6.
2. ~~**Layer 1 LR.** Protocol: 5-fold CV. Built: the bakeoff's single 80/20
   holdout by default, `eval.lr_eval: cv` opt-in. Proposed: `cv` in the four
   experiment configs, holdout kept in `base.yaml` so `make repro-check`
   compares like with like against the first post. CSV `lr_acc` then differs
   in kind from `reference.yaml`'s; no figure depends on that comparison.~~
   **Resolved 2026-09-22.** As proposed, with the override in one place:
   `configs/experiments.yaml` layers `eval.lr_eval: cv` over `base.yaml` and
   the four experiment configs inherit from it, so the grids share one `eval`
   section by construction (item 1's metrics cache needs that). `base.yaml`
   stays holdout, so `make repro-check` is like for like with the first post.
   The CSV gains an `lr_eval` column (`cv` or `holdout`) so `lr_acc` values
   are never compared across kinds by accident; `load_results` fills it empty
   for CSVs written before the column existed.
3. ~~**Checkpoints evaluated per experiment.** Protocol: init and final for
   Exps 2 and 3, final only for Exp 4. Built: the `exp2`–`exp4` Make targets
   evaluate all five fractions. Proposed: `--only-final` on those targets;
   the untrained row is shared and cached.~~
   **Resolved 2026-09-22.** `--only-final` on the `exp2`–`exp4` targets; Exp 4
   keeps the init row, which is free and is the Pareto plot's baseline. The
   untrained row was *not* shared as built: every run saved its own
   `ckpt_0.0`, and the caches key on checkpoint path and manifest stamp, so
   the 19 distinct runs meant 19 Mapper bootstraps of two distinct models.
   Now the base model is saved once under `checkpoints/_untrained/<model>/`
   (from the roster's `hf_id`, independent of the run's loss path) and every
   run's `ckpt_0.0` is a relative symlink to it; the caches resolve the link,
   so the untrained model is evaluated once per base model while each run
   still writes its own init row.
4. ~~**Filling in intermediate checkpoints for the τ values that moved**
   (order of operations, step 3). Built: the driver fills in, but for every τ.
   Proposed: `--tau` and `--model` filters on `scripts/run_experiment.py`.~~
   **Resolved 2026-09-22.** As proposed: repeatable `--model` and `--tau`
   flags on the driver, intersected and applied after grid expansion; a
   selection that matches nothing is an error naming the grid's values.
   `make exp1 EXP_FLAGS="--tau 0.01 --tau 0.05"` passes them through, and
   `--dry-run` shows the selected runs first.
5. ~~**Persistent-homology regularizer.** Protocol: TopoAE loss or RTD. Built:
   persist0 `TopoH0Loss`, which matches sorted H0 death vectors (MST edge
   lengths) between the cosine-distance matrix of the embedding subsample and
   the Euclidean textstat-distance matrix, with the reference rescaled onto
   the live scale (`ref_scale: match_mean`). This is the TopoAE family;
   TopoAE proper evaluates distances at the pairings selected in each space
   rather than comparing sorted death vectors. persist0 returns the pairing
   indices, so the exact TopoAE form is a small addition if wanted as a
   fourth auxiliary. RTD is not built.~~
   **Resolved 2026-09-22.** A/B. `persist0_h0` stays (it is validated against
   Ripser for H0/MST features) and exact TopoAE joins it as a fourth
   auxiliary, `topoae_h0`: the topological term of Moor et al. (2020) in
   dimension 0, distances at the MST edges each space selects, in both
   directions, all 63 edges of the 64-text subsample, reference rescaled as
   before, pairs from persist0. The two agree on the multiset of MST edge
   lengths and differ only in whether the *same* pairs must die, so the
   comparison isolates whether pairing information matters. Exp 4 is now
   4 auxiliaries × 2 λ = 8 rows. RTD is not built (cross-filtration, H1, a
   new dependency).
6. ~~**Layer 4 router.** Protocol: trained on SciCUEval, tested on SciCUEval
   held-out and on MMLU. MMLU carries none of SciCUEval's subject labels
   (the corpora share the `source_subset` / `source_domain` schema, not
   values), so a SciCUEval-trained classifier has no MMLU accuracy. Built:
   `eval.router.mode: union`, one LR over the union of both corpora's labels
   on a stratified 80% of both eval splits, accuracy reported separately on
   held-out SciCUEval and MMLU rows, plus cross-corpus misroute fractions.
   `per_corpus` and `label: domain` are the alternatives. The intended
   mapping, if different, is still to be stated.~~
   **Resolved 2026-09-22.** No mapping exists: the domain vocabularies are
   disjoint too (biology, biomedicine, chemistry, materials, physics vs.
   business, common_knowledge, humanities, law_policy, social_science). The
   router the first post was choosing an encoder for "classifies prompts by
   domain and complexity", and general-domain prompts are traffic for a
   different model, not rejects, so Layer 4 is the union router at *domain*
   granularity: one LR over the 10 domains, trained on a stratified 80% of
   both eval splits, scored on held-out SciCUEval (`router_acc_in`) and MMLU
   (`router_acc_mmlu`) rows. MedCPT's MMLU disintegration should surface as
   low `router_acc_mmlu`. The misroute fractions (`router_in_to_mmlu_frac`,
   `router_mmlu_to_in_frac`) join the CSV as sanity columns (science vs.
   general is trivially separable), and `router_mode` / `router_label`
   record the recipe. `base.yaml` keeps `label: subset` for
   `make repro-check`; `per_corpus` duplicates Layer 1 and is unused.
   Complexity, the router's other axis, is what anchor ρ measures.
7. **HF Hub push of trained checkpoints.** In the deliverables, not built.
   Needs a namespace and a choice of final-only versus all fractions.
   **Deferred 2026-09-22** until the experiments have run: nothing to push
   before then, and the login lives on whichever machine pushes. Shape
   agreed in principle: `scripts/push_hub.py` and `make push-hub`, one model
   repo per run (`<namespace>/tlf-<run_id>`), the final checkpoint at the
   repo root and any other pushed fraction as a `ckpt_<frac>` branch, with
   the run and checkpoint manifests, the train log and a model card carrying
   the run's CSV rows. Selection rule to confirm then: every checkpoint with
   a CSV row (about 45 checkpoints, ~14 GB) rather than final only (~7.5 GB)
   or all fractions (~30 GB). Untrained checkpoints are the public base
   models and are not pushed. Still open: namespace, public vs. private.
8. ~~**Compute budget.** Protocol: ~40 min (SciCUEval) + ~25 min (MMLU) per
   row. The bakeoff README's own figure (~100 min for 28 encoder-corpus
   evaluations at 8 workers) implies about 3.5 min each. A 10× gap;
   `make repro-check` measures it on the current machine before Spark time
   is booked.~~
   **Deferred 2026-09-22.** A wall-clock question, so it is measured where
   the sweep will run: `make repro-check` on the DGX Spark once it is online
   (this week), at the worker count the sweep will use, with the two corpora
   concurrent. That run is also order-of-operations step 1 (reproduce the
   first post's ARI / ρ within ±0.005; drift would be a library-version
   story, since the eval code is the bakeoff's verbatim). The laptop
   (16 cores, 54 GB, RTX 4070 8 GB) stays for training and smoke runs. The
   compute section is rewritten from the measured number afterwards.
   **Note 2026-09-27.** The Spark's GB10 cannot run the bakeoff's torch 2.6.0,
   which predates Blackwell, so `pyproject.toml` pins torch 2.13.0 for
   aarch64 only; the laptop keeps 2.6.0. That is the one library-version
   difference going into the Spark repro-check, and the ±0.005 check is what
   settles whether it matters.
   **Note 2026-09-27 (fork).** The first two Spark repro-checks lost every
   MMLU Mapper seed. The aarch64 numba has no TBB and falls back to GNU
   OpenMP; once the parent has run parallel numba code (UMAP after the
   SciCUEval bootstrap), numba kills each forked worker that runs parallel
   code. An earlier fix set `OMP_NUM_THREADS=1`, on the mistaken reading that
   the message came from torch; it is replaced by defaulting
   `NUMBA_THREADING_LAYER=forksafe` on import (TBB on the laptop, workqueue
   on the Spark). The laptop control run of the same day evaluated
   `biomedbert-fulltext` and `minilm` under `OMP_NUM_THREADS=1` and
   `bge-base` without it; numba was on TBB throughout.
   **Resolved 2026-09-27** for evaluation; training is timed by the first
   `exp1-final`. Spark times are from the fourth repro-check (20 workers,
   per-seed anchor ρ on, encodings cached), with encoding on the GB10
   estimated from the first Spark run; laptop times are the control run of
   the same day (8 workers, encoding on the RTX 4070 included).

   | model | SciCUEval | MMLU | encoding | Spark per checkpoint | laptop per checkpoint |
   |---|---|---|---|---|---|
   | biomedbert-fulltext | 109 s | 84 s | ~46 s | ~4.0 min | 7.2 min |
   | minilm | 54 s | 18 s | ~12 s | ~1.4 min | 6.0 min |
   | bge-base | 73 s | 40 s | ~33 s | ~2.4 min | 7.5 min |

   The sweep is 25 fine-tunes: Exp 1's 12, then 3, 2 and 8 more from Exps
   2–4, which share Exp 1's `biomedbert-fulltext` τ 0.05 random run. With
   the three untrained base models that is 28 checkpoints at init and final,
   18 of them `biomedbert-fulltext`, or about 1.5 hours of evaluation. Exp 1's
   intermediate fractions add at most 36 checkpoints and 1.6 hours, less in
   practice since step 3 fills only the τ values that moved. So evaluation
   is at most about 3 hours on the Spark, against the protocol's 20–25. The
   protocol's 65 minutes per checkpoint was 9–11 times too high on the laptop
   and 16–46 times on the Spark; the bakeoff README's 3.5 minutes per
   evaluation was close. The corpora run in sequence rather than
   concurrently, which at these times is not worth building. Fine-tuned
   checkpoints may give larger graphs and slower bootstraps; the first
   `exp1-final` will show. Training is 25 runs of 468 optimizer steps (20k
   pairs, batch 128, 3 epochs; Exp 2's continued-MLM control batches texts
   instead), and its wall time has not been measured on either machine.
9. ~~**Exp 1 primary plot x-axis.** Protocol: training step. Built: fraction
   of training, which is how checkpoints are defined; both models take the
   same number of steps on the same pair set, so a step axis is a relabel.~~
   **Resolved 2026-09-22.** Fraction stays the axis, since that is how the
   checkpoints are defined, and the absolute scale travels with it: every
   CSV row now records the checkpoint's optimizer `step` and the run's
   `total_steps` (from the checkpoint and run manifests), and the figure's
   axis label reads "fraction of training · N steps" when every row on the
   panel agrees on N (468 for a 20k-pair set at batch 128 over 3 epochs),
   falling back to the plain label otherwise. The continued-MLM control in
   Exp 2 batches unique texts rather than pairs, so its schedule differs;
   the columns make that visible without a trip to the manifests.
10. ~~**`bge-base`** is in the roster as optional and in `models.yaml`; it is
    not in any experiment grid yet.~~
    **Resolved 2026-09-22.** In Exp 1's grid (now 12 runs). Against
    `biomedbert-fulltext` it holds the architecture fixed and varies only the
    pretraining regime, the first post's central claim tested directly;
    against `minilm` it asks whether the contrastive ρ floor depends on
    model size. Execution is staged with `--model`: the two core models
    first, `bge-base` when time allows. `make repro-check` now covers all
    three starting encoders against `reference.yaml`.
11. ~~**Reproduction tolerance.** Protocol: ARI / ρ reproduce within ±0.005 on
    the new hardware. Measured 2026-09-27 with `make repro-check` on all three
    starting encoders, both corpora, on the laptop (x86_64, 8 workers) and the
    DGX Spark (aarch64, 20 workers). Layer 1 agrees to four decimals on both.
    On the laptop, SciCUEval Layers 2–3 reproduce the first post exactly (ARI,
    anchor ρ, node count); MMLU is close but not exact, within the bound below,
    with evaluation order inside a process the leading suspect. On the Spark,
    ARI differs by up to 0.050 and anchor ρ by up to 0.49. UMAP is seeded but
    not bit-reproducible across CPU architectures, and the lens and the
    clustering both amplify last-bit differences, so the Spark's 25 seeds are
    effectively a fresh sample. A fixed ±0.005 cannot hold across machines.~~

    | row | ARI ref | laptop | Spark | bound | ρ ref | laptop | Spark |
    |---|---|---|---|---|---|---|---|
    | biomedbert-fulltext SciCUEval | 0.818 | 0.818 | 0.799 | 0.031 | 0.249 | 0.249 | 0.199 |
    | biomedbert-fulltext MMLU | 0.695 | 0.701 | 0.692 | 0.058 | 0.476 | 0.476 | 0.322 |
    | minilm SciCUEval | 0.757 | 0.757 | 0.807 | 0.040 | 0.059 | 0.059 | 0.457 |
    | minilm MMLU | 0.822 | 0.829 | 0.798 | 0.041 | 0.006 | 0.036 | 0.015 |
    | bge-base SciCUEval | 0.766 | 0.766 | 0.783 | 0.035 | 0.195 | 0.195 | 0.227 |
    | bge-base MMLU | 0.880 | 0.849 | 0.845 | 0.054 | −0.052 | −0.009 | 0.440 |

    **Resolved 2026-09-27.** Three rules replace the single tolerance.
    Layer 1 (gap, LR accuracy) must match the reference within ±0.005 on any
    machine; a miss means the sample or the encoding differs. Layers 2–3 ARI
    from two machines must agree within two standard errors of the difference
    of their 25-seed means, 2·√(sd₁² + sd₂²)/5, where sd is the per-seed ARI
    spread each run records; the laptop's spread stands in for the first
    post's. Five of six Spark rows meet it; minilm SciCUEval is +0.050 against
    0.040, which one of six comparisons at a two-SE bound produces about a
    quarter of the time. The first post's single-draw anchor ρ must reproduce
    exactly on x86_64, which it does on SciCUEval, and is not compared across
    architectures; the bootstrapped anchor ρ of item 12 takes the ARI rule
    once it exists. Every fine-tuned checkpoint is compared against its own
    `ckpt_0.0` evaluated on the same machine, so each machine's untrained rows
    are the baseline for its own sweep.
    **Note 2026-09-27 (order).** Evaluation order is ruled out. MMLU alone in
    a fresh process reproduces the combined run on all three models to within
    4·10⁻¹⁶ relative, the size of float summation order. That jitter came from
    aggregating seeds in completion order and is gone with item 12, which
    aggregates in seed order. The laptop's MMLU gap to the first post stays
    unexplained and inside the bound.
    **Note 2026-09-27 (jackknife).** The bound above divides the spread of the
    300 pairwise ARIs by √25, as if they were 25 independent values. They
    share seeds, so that is not a standard error; with additive seed effects
    it understates the true one by about √2. The rule now uses the
    leave-one-seed-out jackknife of the mean pairwise ARI, recorded as
    `ari_se`: two machines agree when their means differ by at most
    2·√(se₁² + se₂²). The table above keeps the old reading. From the
    laptop and Spark reruns of the same day, whose aggregates match the
    earlier runs exactly, the jackknife bounds between the two machines are
    0.037 and 0.066 (`biomedbert-fulltext` SciCUEval, MMLU), 0.047 and 0.045
    (`minilm`) and 0.039 and 0.040 (`bge-base`). The jackknife error is 10–20%
    above the old reading on five rows and below it on `bge-base` MMLU. Five
    rows fall inside; `minilm` SciCUEval stays just outside, +0.050 against
    0.047.
12. ~~**Anchor ρ estimator.** Protocol and first post: anchor Spearman ρ between
    Mapper graph distance and textstat distance, on pairs from 500 covered
    documents with a 5000-pair cap. Built, verbatim from the bakeoff: it is
    computed once, on the base-seed graph, so it carries no spread; and the
    cap is filled in `itertools.combinations` order, so nearly all pairs
    involve the first ~11 sampled documents. Anchor ρ is H1's primary metric
    and every checkpoint's UMAP lens is effectively a new draw; the Spark run
    above shows single draws moving by up to 0.49.~~
    **Resolved 2026-09-27.** `tlf.anchor.anchor_rho_uniform` keeps the
    bakeoff's 500-doc subsample and first-node assignment but draws its 5000
    pairs uniformly from all same-component pairs, and the bootstrap worker
    runs it on every seed's graph. Rows carry `anchor_rho_mean` and
    `anchor_rho_sd` beside the verbatim `anchor_rho`; `eval.anchor_per_seed`
    switches it on and is part of the metrics cache key. Seeds are now
    aggregated in seed order, so reruns agree bit for bit. On a synthetic path
    graph with 20 of 500 docs corrupted, the uniform estimator reads +0.92 and
    the bakeoff's −0.82, since every bakeoff pair touches one of the first
    docs. Laptop, 25 seeds:

    | row | single draw | per-seed mean | sd | SE of mean |
    |---|---|---|---|---|
    | biomedbert-fulltext SciCUEval | 0.249 | 0.216 | 0.158 | 0.032 |
    | biomedbert-fulltext MMLU | 0.476 | 0.230 | 0.204 | 0.041 |
    | minilm SciCUEval | 0.059 | 0.262 | 0.126 | 0.025 |
    | minilm MMLU | 0.036 | 0.083 | 0.121 | 0.024 |

    The single draws match the first post exactly except minilm MMLU (the
    laptop's MMLU gap in item 11), and the Spark's single draws (0.199 and
    0.457 on SciCUEval) sit inside these spreads. A per-seed sd of 0.12–0.20
    sets the instrument's resolution: two checkpoints need to differ by about
    0.09 to clear two standard errors. On MMLU the per-seed means still
    separate the two models (0.230 against 0.083, about three standard
    errors of the difference). On SciCUEval they do not (0.216 against 0.262,
    about one); the first post's SciCUEval contrast between them (0.249
    against 0.059) came from single draws.
