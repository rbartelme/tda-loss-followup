# Loss Functions vs. Topology: Experiment Protocol

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
