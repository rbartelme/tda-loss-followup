# Bakeoff harness notes

Read-only inspection of `~/00-projects/tda-embedder-bakeoff` at commit `4625b22`
(`updated SciCUEval pull and README`). Everything below is what the first post's
three-layer harness actually does, recorded so this repo can reuse it unchanged.

Layer 1 lives in `scripts/embedding_diagnostic.py`, Layers 2 and 3 in
`scripts/embedding_tda.py`, textstat features in `scripts/complexity.py`. They are
plain scripts under `scripts/` with `sys.path` hacks, not an installable package, so
the plan is to copy the specific functions into `src/tlf/_bakeoff/` with a source
header rather than add a path dependency.

## 1. Corpus loading and sampling

**Source files.** Both corpora are flat JSONL produced by adapters:

- SciCUEval: `scicueval_download.py` pulls the Figshare zip (article 29924687,
  file 57229472, md5-verified) to `~/02-resources/SciCUEval/data/<Subset>/<Competency>.json`;
  `scicueval_to_jsonl.py --root ... --out scratch/scicueval_combined.jsonl` flattens
  it. 10 subsets × 4 competency files. 11,343 rows total.
- MMLU: `mmlu_download.py` pulls 10 non-STEM subjects from `cais/mmlu` via `datasets`,
  concatenating `test + validation + dev` per subject, to `~/02-resources/MMLU/data/<subject>.json`;
  `mmlu_to_jsonl.py` flattens it. 3,134 rows total.

Both adapters emit the same record shape:
`id, question, source_dataset, source_subset, source_domain, source_competency,
source_row_id, license, attribution_url`. The prompt field is `question`, the label
field is `source_subset`.

Row counts per subset (from the local JSONLs):

| SciCUEval subset | rows | | MMLU subject | rows |
|---|---|---|---|---|
| BioText | 968 | | high_school_psychology | 610 |
| GoKG | 1180 | | high_school_us_history | 231 |
| HipKG | 1165 | | international_law | 139 |
| IaeaTab | 1130 | | jurisprudence | 124 |
| MatTab | 936 | | management | 119 |
| MatText | 940 | | marketing | 264 |
| MolTab | 1305 | | miscellaneous | 874 |
| PhaKG | 1217 | | philosophy | 350 |
| PriKG | 1250 | | sociology | 228 |
| ProtTab | 1252 | | world_religions | 195 |

**Sampling** is `load_corpus(path, prompt_field, label_field, sample_per_class=400, seed=42)`:

1. Read every line in file order; drop rows whose prompt is not a non-empty string.
2. Group rows by label, preserving file order within each group.
3. Create one `np.random.default_rng(seed)` with `seed=42`.
4. Iterate groups in `sorted(by_label.items())` order (alphabetical label). For a group
   with more than 400 rows draw `rng.choice(len(group), size=400, replace=False)`;
   for a group with 400 or fewer keep all rows. The chosen indices are used in the
   order `rng.choice` returned them.
5. Concatenate the sampled groups in alphabetical label order.

To reproduce the *identical* eval sample this repo must: read the same JSONL in the
same row order, use seed 42, use a single RNG stream shared across labels in
alphabetical order, and apply the same empty-prompt filter. Any other approach gives a
different sample.

Resulting eval sizes: SciCUEval 4,000 rows (every subset is capped at 400), MMLU 2,450
rows (four subjects are below 400 and are kept whole). The SciCUEval remainder, 7,343
rows, is what this repo will call the `train` split. Both numbers match `n_prompts` in
the cached MedCPT metrics.

**Corpus cache.** With `--cache-corpus`, Layer 1 writes `<out_dir>/corpus.npz` holding
`prompts`, `labels`, `feature_names`, and `textstat` (float32, N×6) in the exact row
order of the embeddings. Layer 2 reads this rather than the JSONL.

## 2. Encoder specification and embedding cache

- Roster is the `ENCODERS: dict[str, str]` map key → HuggingFace id. Keys this repo
  needs: `minilm` → `sentence-transformers/all-MiniLM-L6-v2`; `biomedbert-fulltext` →
  `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`; `bge-base` →
  `BAAI/bge-base-en-v1.5`; `medcpt-query` → `ncbi/MedCPT-Query-Encoder`.
- Loaded with `AutoTokenizer.from_pretrained` and `AutoModel.from_pretrained`;
  `trust_remote_code=True` only for `nomic-embed-v1.5`. Model in eval mode, `torch.no_grad`.
- Pooling is mean over the attention mask for every encoder (`_mean_pool`: masked sum
  of `last_hidden_state` divided by mask count, clamped at 1e-9). No CLS pooling, no
  L2 normalisation at encode time.
- Prefix: identity for every encoder except `embeddinggemma-prescribed`, which applies
  `task: classification | query: {content}`. None of this repo's models use a prefix.
- Tokenisation: `padding=True, truncation=True, max_length=256`. The full run used
  `--batch-size 32`. Output is float32 numpy, `np.vstack` of per-batch pooled outputs.
- Cache: `<out_dir>/<encoder_key>/embeddings.npy` when `--cache-embeddings` is set.
  Layer 2 discovers encoders by scanning `<diagnostic_dir>/*/embeddings.npy`.

Note for this repo: the brief's `base.yaml` sets `max_length 128` for training. The
bakeoff *evaluation* encode used 256. These are different knobs and should be separate
config fields, with the eval one defaulting to 256.

## 3. Layer 1 metrics (cosine geometry)

All computed on the eval sample, per encoder, with `seed=42`:

- Labels are alphabetised and mapped to integer indices.
- A 2-D UMAP scatter (`n_components=2, random_state=seed, metric="cosine"`, other
  defaults) is drawn for plots only. It does not feed any metric.
- `Xn = L2-normalize(X)`. Class centroids are computed on `Xn`, then normalised again;
  `centroid_mean_offdiag` is the mean off-diagonal of the centroid cosine matrix.
- `within_between_cosine(Xn, y, rng, n_pairs=20000)` with a fresh
  `np.random.default_rng(seed)`: 20,000 within-class pairs (pick a random label with
  at least 2 members, then 2 distinct members) and 20,000 between-class pairs (pick
  random `i, j`, reject if same label). `within_mean`, `between_mean`, and
  `cosine_gap = within_mean − between_mean`.
- `lr_accuracy(X, y, seed)`: **one** stratified 80/20 `train_test_split(random_state=seed)`
  on the raw, un-normalised embeddings, then
  `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)`.
  Score is test accuracy.

The brief asks for 5-fold CV accuracy. That differs from the bakeoff's single
holdout. Per the invariants it should be a config field whose default reproduces the
bakeoff (single 80/20 split); 5-fold is the opt-in. Flagged for decision.

**Output**: `<out_dir>/<key>/metrics.json` with
`key, model_id, n_prompts, n_classes, label_names, within_mean, between_mean,
cosine_gap, centroid_mean_offdiag, lr_accuracy`, plus four PNGs. A top-level
`summary.md` ranks by `cosine_gap`.

## 4. Layer 2: the exact Mapper call and bootstrap

Frozen config (`MAPPER_CONFIG`, also dumped to `mapper_config.json`):

```python
MAPPER_CONFIG = {
    "umap": {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
    "cover": {"n_cubes": 15, "perc_overlap": 0.3},
    "clusterer": {"min_cluster_size": 5, "metric": "precomputed"},
}
DEFAULT_N_SEEDS = 25
DEFAULT_BASE_SEED = 42
```

Per encoder:

1. `embeddings = l2_normalize(np.load(embeddings.npy).astype(float32))`, asserting unit
   norms within 1e-4. Zero rows are left as zero.
2. `dist_matrix = sklearn.metrics.pairwise.cosine_distances(embeddings).astype(float64)`,
   computed once and shared across seeds.
3. For each `seed in [42 + i for i in range(25)]`:

   ```python
   lens = umap.UMAP(**MAPPER_CONFIG["umap"], random_state=seed).fit_transform(embeddings)
   graph = km.KeplerMapper(verbose=0).map(
       lens,
       X=dist_matrix,
       cover=km.Cover(n_cubes=15, perc_overlap=0.3),
       clusterer=hdbscan.HDBSCAN(min_cluster_size=5, metric="precomputed"),
       precomputed=True,
   )
   ```

   Warnings are suppressed around the UMAP fit.
4. The graph goes to NetworkX (`nodes` with `size` and `members`, `links` as edges).
   Per-seed statistics: `n_nodes, n_edges, n_components, coverage, avg_node_size,
   median_node_size, largest_component_frac, n_loops_b1, avg_degree, graph_density`.
   `coverage` is the fraction of docs appearing in at least one node; `n_loops_b1` is
   `max(E − V + C, 0)`.
5. Partition per seed: each doc is assigned to the *first* node (in `G.nodes`
   iteration order) that contains it, `-1` if uncovered.
6. Aggregates across the successful seeds: `mean_<stat>` and `cv_<stat>` (std/mean,
   0 if mean is 0) for each statistic; `mean_ari` and `mean_nmi` are the means over
   all C(25,2)=300 seed pairs of `adjusted_rand_score` / `normalized_mutual_info_score`
   on the docs covered by both seeds. `n_successful_seeds` is recorded.

The seed loop is parallelised with a fork-context `ProcessPoolExecutor`; embeddings and
the distance matrix are published to module globals before the fork so only the seed
integer is pickled. `--n-workers` defaults to `min(8, cpu_count // 2)`; `1` runs
serially. The full 14-encoder × 2-corpus run used 8 workers.

The brief's CSV column `ari_sd` does not exist in the bakeoff (it only reports the
mean). It is the std of the same 300 pairwise ARIs and is a pure addition.

## 5. Layer 3: anchors

Computed on one canonical graph built at `base_seed=42` (the same call as above).

**Textstat standardisation.** `standardize(X)` z-scores each of the 6 features using
the mean and std over the eval corpus (`sd == 0` → 1). This is done once on the
`corpus.npz` textstat matrix before any encoder runs. In this repo the same z-scoring
must also exist on the *train* split for pair construction, which is a separate fit.

**Categorical purity** (`categorical_anchor_purity`): for every node, the fraction of
members carrying the node's majority label. `weighted_purity` weights nodes by size,
`mean_purity` does not.

**Anchor Spearman ρ** (`continuous_anchor_correlation(G, textstat_std, n_docs, seed=42, max_pairs=5000)`):

1. Map each doc to its first node. Need at least 50 covered docs, else return
   `rho=0, p=1, n_pairs_used=0`.
2. If more than 500 docs are covered, subsample 500 with `np.random.default_rng(seed)`.
3. `nx.all_pairs_shortest_path_length(G)` gives node-to-node graph distance.
4. Iterate `itertools.combinations(covered_docs, 2)` in order; skip a pair if its nodes
   are in different components; record graph distance and the Euclidean distance
   between the two standardised textstat vectors; stop after 5,000 pairs.
5. Need at least 100 pairs, else return `rho=0, p=1`. Otherwise `scipy.stats.spearmanr`.

So ρ is *not* over all pairs; it is over the first 5,000 same-component pairs of a
500-doc subsample, and a disintegrated graph reports ρ = 0 by construction.

**Per-feature alignment** (`per_feature_alignment`): Spearman ρ of each raw feature
column against UMAP-1, UMAP-2, and the radial norm of the canonical lens. 18 values,
keyed `feat_<name>_corr_{umap1,umap2,radial}`.

**Output**: `<out_dir>/<key>/metrics.json` with `key, n_docs`, the 20 `mean_/cv_` stats,
`mean_ari, mean_nmi, n_successful_seeds, anchor_weighted_purity, anchor_mean_purity,
anchor_spearman_rho, anchor_spearman_p, n_pairs_used`, and the 18 `feat_*` values.
Top-level `comparison.csv` (polars) is `embedder` plus every one of those columns,
one row per encoder, plus `mapper_config.json` and four PNGs.

## 6. The six textstat features

`complexity.extract(prompt) -> dict`, in this key order:

| key | definition |
|---|---|
| `flesch_reading_ease` | `textstat.flesch_reading_ease(prompt)` |
| `syllable_count` | `textstat.syllable_count(prompt)` |
| `lexicon_count` | `textstat.lexicon_count(prompt, removepunct=True)` |
| `sentence_count` | `textstat.sentence_count(prompt)` |
| `avg_sentence_length` | `lexicon_count / sentence_count`, `0.0` when `sentence_count <= 0` |
| `difficult_words` | `textstat.difficult_words(prompt)` |

Pinned `textstat==0.7.4`, which imports `pkg_resources`, so `setuptools<81` is also
pinned. The feature vector this repo needs is that dict's values in that order.

## 7. Disintegration, as observed

The MedCPT row on MMLU shows what a disintegrated Mapper graph looks like in the
bakeoff's numbers: mean nodes 1.84, coverage 0.0056, zero edges, ρ = 0 (fallback), yet
mean ARI 0.96 because a near-empty partition is trivially stable. The brief's flag
`disintegrated = coverage < 0.05 or nodes < 5` catches this row and must be applied
before ARI is read.

## 8. MedCPT reference rows (for `configs/reference.yaml`)

From `scratch/diagnostic_*/medcpt-query/metrics.json` and `scratch/tda_*/comparison.csv`
on this machine. These directories are gitignored in the bakeoff, so this is the only
place they are recorded.

| metric | SciCUEval | MMLU |
|---|---|---|
| within_cos | 0.7856 | 0.7700 |
| between_cos | 0.7345 | 0.7398 |
| gap | 0.0511 | 0.0302 |
| lr_acc (80/20 holdout) | 0.9513 | 0.7694 |
| ari | 0.8639 | 0.9615 |
| nmi | 0.9669 | 0.9615 |
| coverage | 0.3974 | 0.0056 |
| nodes | 173.16 | 1.84 |
| purity (weighted) | 0.9566 | 0.8043 |
| anchor_rho | 0.5316 | 0.0 |
| disintegrated | no | yes |

## 9. Dependency constraints in the bakeoff

`pyproject.toml` pins `transformers==4.51.3`, `torch==2.6.0`, `datasets==3.2.0`,
`textstat==0.7.4`, `setuptools<81`, and floors `scikit-learn>=1.5`, `numpy>=1.26`,
`umap-learn>=0.5.6`, `kmapper>=2.0.1`, `hdbscan>=0.8.40`, `networkx>=3.3`,
`matplotlib>=3.8`, `polars>=1.10`. Dev: `pytest==8.3.4`, `ruff==0.8.4`.

Two things the brief's dependency list omits that the Mapper code needs: `networkx`
(graph statistics) and `setuptools<81` (textstat). `polars` is only used to write the
CSV and can be replaced by pandas. Resolved versions from the bakeoff's `uv.lock` for
the packages that determine Mapper numerics are appended below; matching them is the
safest way to keep the reproduction check honest.

### Resolved versions in the bakeoff lockfile

| package | version |
|---|---|
| `datasets` | 3.2.0 |
| `hdbscan` | 0.8.42 |
| `kmapper` | 2.1.0 |
| `llvmlite` | 0.47.0 |
| `matplotlib` | 3.10.9 |
| `networkx` | 3.6.1 |
| `numba` | 0.65.1 |
| `numpy` | 2.4.4 |
| `pynndescent` | 0.6.0 |
| `scikit-learn` | 1.8.0 |
| `scipy` | 1.17.1 |
| `textstat` | 0.7.4 |
| `torch` | 2.6.0 |
| `transformers` | 4.51.3 |
| `umap-learn` | 0.5.12 |
