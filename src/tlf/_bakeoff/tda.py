"""Copied from rbartelme/tda-embedder-bakeoff @ 4625b22 -- scripts/embedding_tda.py.

Verbatim except where a comment says ``# tlf:``. Excluded from ruff. Do not
edit; if a value must differ, add a parameter whose default is the original.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np

_W_EMB: np.ndarray | None = None
_W_DIST: np.ndarray | None = None
_W_N_DOCS: int = 0
_W_CFG: dict | None = None  # tlf


MAPPER_CONFIG = {
    "umap": {
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "cosine",
    },
    "cover": {"n_cubes": 15, "perc_overlap": 0.3},
    "clusterer": {"min_cluster_size": 5, "metric": "precomputed"},
}


DEFAULT_N_SEEDS = 25


DEFAULT_BASE_SEED = 42


def _default_workers() -> int:
    """Half of available cores by default, capped at 8.

    UMAP forces single-thread when ``random_state`` is set, so we parallelize
    *across* seeds. Half-cores leaves room for BLAS / other work and keeps
    memory pressure reasonable (each worker pays for embeddings + distance
    matrix copies under fork's copy-on-write).
    """
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu // 2))


def _seed_worker(seed: int):
    """One Mapper build inside a worker process."""
    graph, _ = build_mapper_for_seed(_W_EMB, _W_DIST, seed, _W_CFG)  # tlf
    G = mapper_to_networkx(graph)
    return graph_statistics(G, _W_N_DOCS), doc_to_node_partition(G, _W_N_DOCS)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows. Zero-vector rows are mapped to zero vectors safely."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (X / norms).astype(np.float32)


def assert_normalized(X: np.ndarray, tol: float = 1e-4) -> None:
    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=tol):
        raise AssertionError(
            f"embeddings not L2-normalized: norms in "
            f"[{norms.min():.4f}, {norms.max():.4f}]"
        )


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_distances

    assert_normalized(embeddings)
    return cosine_distances(embeddings).astype(np.float64)


def build_mapper_for_seed(
    embeddings: np.ndarray, dist_matrix: np.ndarray, seed: int,
    mapper_config: dict = MAPPER_CONFIG,  # tlf: parameterised, default unchanged
):
    import hdbscan
    import kmapper as km
    import umap

    umap_params = {**mapper_config["umap"], "random_state": seed}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lens = umap.UMAP(**umap_params).fit_transform(embeddings)

    mapper = km.KeplerMapper(verbose=0)
    clusterer = hdbscan.HDBSCAN(**mapper_config["clusterer"])

    graph = mapper.map(
        lens,
        X=dist_matrix,
        cover=km.Cover(**mapper_config["cover"]),
        clusterer=clusterer,
        precomputed=True,
    )
    return graph, lens


def mapper_to_networkx(mapper_graph):
    import networkx as nx

    G = nx.Graph()
    for node_id, members in mapper_graph["nodes"].items():
        G.add_node(node_id, size=len(members), members=list(members))
    for source, targets in mapper_graph["links"].items():
        for t in targets:
            G.add_edge(source, t)
    return G


GRAPH_STAT_KEYS = (
    "n_nodes",
    "n_edges",
    "n_components",
    "coverage",
    "avg_node_size",
    "median_node_size",
    "largest_component_frac",
    "n_loops_b1",
    "avg_degree",
    "graph_density",
)


def graph_statistics(G, n_docs: int) -> dict[str, float]:
    import networkx as nx

    if G.number_of_nodes() == 0:
        return {k: 0.0 for k in GRAPH_STAT_KEYS}

    sizes = [G.nodes[n]["size"] for n in G.nodes]
    covered_docs: set[int] = set()
    for n in G.nodes:
        covered_docs.update(G.nodes[n]["members"])
    components = list(nx.connected_components(G))
    largest = max(components, key=len) if components else set()

    b1 = G.number_of_edges() - G.number_of_nodes() + len(components)

    return {
        "n_nodes": float(G.number_of_nodes()),
        "n_edges": float(G.number_of_edges()),
        "n_components": float(len(components)),
        "coverage": len(covered_docs) / n_docs,
        "avg_node_size": float(np.mean(sizes)),
        "median_node_size": float(np.median(sizes)),
        "largest_component_frac": len(largest) / G.number_of_nodes(),
        "n_loops_b1": float(max(b1, 0)),
        "avg_degree": float(np.mean([d for _, d in G.degree()])),
        "graph_density": float(nx.density(G)),
    }


def doc_to_node_partition(G, n_docs: int) -> np.ndarray:
    """Hard partition: each doc → label of first node it appears in. -1 = uncovered."""
    partition = np.full(n_docs, -1, dtype=np.int64)
    for label, node_id in enumerate(G.nodes):
        for doc_idx in G.nodes[node_id]["members"]:
            if partition[doc_idx] == -1:
                partition[doc_idx] = label
    return partition


def seed_bootstrap_stability(
    embeddings: np.ndarray,
    dist_matrix: np.ndarray,
    n_docs: int,
    n_seeds: int,
    base_seed: int,
    n_workers: int,
    mapper_config: dict = MAPPER_CONFIG,  # tlf: parameterised, default unchanged
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Returns (aggregate metrics dict, per-seed stat rows for plotting).

    Parallelized across seeds via ProcessPoolExecutor. UMAP forces
    single-thread when ``random_state`` is set, so we squeeze multi-core
    throughput by running seeds concurrently in worker processes; each
    worker fork-inherits ``embeddings`` / ``dist_matrix`` from module-level
    globals (COW), so no large pickling per task.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    seeds = [base_seed + i for i in range(n_seeds)]
    stat_rows: list[dict[str, float]] = []
    partitions: list[np.ndarray] = []

    # Publish the heavy arrays to module-level globals before forking.
    global _W_EMB, _W_DIST, _W_N_DOCS, _W_CFG
    _W_EMB = embeddings
    _W_DIST = dist_matrix
    _W_N_DOCS = n_docs
    _W_CFG = mapper_config  # tlf

    if n_workers <= 1:
        # Serial fallback — useful for debugging or low-RAM environments.
        for seed in seeds:
            try:
                stat_row, partition = _seed_worker(seed)
                stat_rows.append(stat_row)
                partitions.append(partition)
            except Exception as e:  # noqa: BLE001
                print(f"    seed {seed} failed: {e!r}", file=sys.stderr)
    else:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_seed_worker, s): s for s in seeds}
            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    stat_row, partition = fut.result()
                    stat_rows.append(stat_row)
                    partitions.append(partition)
                except Exception as e:  # noqa: BLE001
                    print(f"    seed {seed} failed: {e!r}", file=sys.stderr)

    if len(stat_rows) < 2:
        return (
            {"_error": float("nan"), "n_successful_seeds": float(len(stat_rows))},
            stat_rows,
        )

    out: dict[str, float] = {}  # tlf: numpy in place of polars; same mean / sample std
    for k in GRAPH_STAT_KEYS:
        vals = np.array([row[k] for row in stat_rows], dtype=np.float64)
        m = float(vals.mean())
        s = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out[f"mean_{k}"] = m
        out[f"cv_{k}"] = (s / m) if m else 0.0

    aris: list[float] = []
    nmis: list[float] = []
    for p1, p2 in combinations(partitions, 2):
        mask = (p1 >= 0) & (p2 >= 0)
        if int(mask.sum()) > 1:
            aris.append(adjusted_rand_score(p1[mask], p2[mask]))
            nmis.append(normalized_mutual_info_score(p1[mask], p2[mask]))

    out["mean_ari"] = float(np.mean(aris)) if aris else 0.0
    out["mean_nmi"] = float(np.mean(nmis)) if nmis else 0.0
    out["sd_ari"] = float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0  # tlf: addition
    out["sd_nmi"] = float(np.std(nmis, ddof=1)) if len(nmis) > 1 else 0.0  # tlf: addition
    out["n_successful_seeds"] = float(len(stat_rows))
    return out, stat_rows


def categorical_anchor_purity(G, labels: list[str]) -> dict[str, float]:
    from collections import Counter

    purities: list[float] = []
    sizes: list[int] = []
    for n in G.nodes:
        members = G.nodes[n]["members"]
        node_labels = [labels[i] for i in members]
        if not node_labels:
            continue
        most_common = Counter(node_labels).most_common(1)[0][1]
        purities.append(most_common / len(node_labels))
        sizes.append(len(node_labels))
    if not purities:
        return {"weighted_purity": 0.0, "mean_purity": 0.0}
    return {
        "weighted_purity": float(np.average(purities, weights=sizes)),
        "mean_purity": float(np.mean(purities)),
    }


def continuous_anchor_correlation(
    G,
    anchor_features: np.ndarray,
    n_docs: int,
    seed: int,
    max_pairs: int = 5000,
    max_docs: int = 500,  # tlf: parameterised, default unchanged
) -> dict[str, float]:
    import networkx as nx
    from scipy.stats import spearmanr

    doc_to_node: dict[int, str] = {}
    for node_id in G.nodes:
        for doc_idx in G.nodes[node_id]["members"]:
            doc_to_node.setdefault(doc_idx, node_id)

    covered_docs = list(doc_to_node.keys())
    if len(covered_docs) < 50:
        return {"anchor_spearman_rho": 0.0, "anchor_spearman_p": 1.0, "n_pairs_used": 0.0}

    rng = np.random.default_rng(seed)
    if len(covered_docs) > max_docs:
        covered_docs = rng.choice(
            covered_docs, size=max_docs, replace=False
        ).tolist()

    node_dist = dict(nx.all_pairs_shortest_path_length(G))

    graph_dists: list[float] = []
    ling_dists: list[float] = []
    for i, j in combinations(covered_docs, 2):
        ni, nj = doc_to_node[i], doc_to_node[j]
        if nj not in node_dist.get(ni, {}):
            continue
        graph_dists.append(float(node_dist[ni][nj]))
        ling_dists.append(
            float(np.linalg.norm(anchor_features[i] - anchor_features[j]))
        )
        if len(graph_dists) >= max_pairs:
            break

    if len(graph_dists) < 100:
        return {"anchor_spearman_rho": 0.0, "anchor_spearman_p": 1.0, "n_pairs_used": float(len(graph_dists))}

    rho, p = spearmanr(graph_dists, ling_dists)
    return {
        "anchor_spearman_rho": float(rho),
        "anchor_spearman_p": float(p),
        "n_pairs_used": float(len(graph_dists)),
    }


def per_feature_alignment(
    lens: np.ndarray,
    anchor_features: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    from scipy.stats import spearmanr

    radial = np.linalg.norm(lens, axis=1)
    out: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        feat = anchor_features[:, i]
        for axis_label, axis_values in (
            ("umap1", lens[:, 0]),
            ("umap2", lens[:, 1]),
            ("radial", radial),
        ):
            rho, _ = spearmanr(feat, axis_values)
            out[f"feat_{name}_corr_{axis_label}"] = (
                float(rho) if rho is not None and not np.isnan(rho) else 0.0
            )
    return out


def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd
