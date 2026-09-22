# Loss Functions vs. Topology: Experiment Protocol

**Status: work in progress.** The protocol below is as written; the discrepancies
between it and the scaffold as built are listed under *To be resolved* at the end.

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
8. **Compute budget.** Protocol: ~40 min (SciCUEval) + ~25 min (MMLU) per
   row. The bakeoff README's own figure (~100 min for 28 encoder-corpus
   evaluations at 8 workers) implies about 3.5 min each. A 10× gap;
   `make repro-check` measures it on the current machine before Spark time
   is booked.
9. **Exp 1 primary plot x-axis.** Protocol: training step. Built: fraction
   of training, which is how checkpoints are defined; both models take the
   same number of steps on the same pair set, so a step axis is a relabel.
10. **`bge-base`** is in the roster as optional and in `models.yaml`; it is
    not in any experiment grid yet.
