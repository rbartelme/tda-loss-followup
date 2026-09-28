"""Per-seed anchor rho: Mapper graph distance against anchor-feature distance.

The bakeoff's estimator, ``_bakeoff.tda.continuous_anchor_correlation``, stays
as the single-draw ``anchor_rho`` so rows remain comparable with the first
post. This one differs in two ways (protocol item 12):

* Pairs are drawn uniformly from all connected pairs of the doc subsample. The
  bakeoff walks ``itertools.combinations`` until its 5000-pair cap, which with
  500 docs takes nearly every pair from the first ~11 of them.
* It runs on every bootstrap seed's graph (see ``_bakeoff.tda._seed_worker``),
  so the mean over seeds carries a spread, and a degenerate graph yields NaN,
  which the mean skips, where the bakeoff returns 0.0.

Doc-to-node assignment is the bakeoff's: a doc in several overlapping nodes
belongs to the first one in node order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MIN_COVERED_DOCS = 50  # below this, no rho (the bakeoff's threshold)
MIN_PAIRS = 100  # below this many connected pairs, no rho (likewise)


def _hop_matrix(G: Any) -> tuple[np.ndarray, dict[Any, int]]:
    """All-pairs shortest-path lengths between Mapper nodes.

    Args:
        G: networkx graph.

    Returns:
        ``(hops, index)``: a square float matrix of hop counts, ``inf`` between
        nodes in different components, and the node-to-row mapping.
    """
    import networkx as nx

    index = {node: k for k, node in enumerate(G.nodes)}
    hops = np.full((len(index), len(index)), np.inf)
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        for dst, d in lengths.items():
            hops[index[src], index[dst]] = d
    return hops, index


def anchor_rho_uniform(
    G: Any,
    anchor_features: np.ndarray,
    seed: int,
    *,
    max_docs: int = 500,
    max_pairs: int = 5000,
) -> dict[str, float]:
    """Spearman rho between graph hops and anchor distance on uniform doc pairs.

    Args:
        G: networkx Mapper graph whose nodes carry a ``members`` list of doc
            indices (as built by ``_bakeoff.tda.mapper_to_networkx``).
        anchor_features: ``(n_docs, k)`` standardised anchor features, rows
            indexed like the graph's members.
        seed: Seeds the doc subsample and the pair draw.
        max_docs: Covered docs subsampled before pairing.
        max_pairs: Connected pairs drawn uniformly; all of them when fewer.

    Returns:
        ``anchor_rho`` and ``anchor_p`` (NaN when fewer than ``MIN_COVERED_DOCS``
        docs are covered, fewer than ``MIN_PAIRS`` pairs are connected, or the
        sampled graph or feature distances are all equal) and ``n_pairs_used``.
    """
    from scipy.stats import spearmanr

    doc_to_node: dict[int, Any] = {}
    for node in G.nodes:
        for doc in G.nodes[node]["members"]:
            doc_to_node.setdefault(int(doc), node)
    empty = {
        "anchor_rho": float("nan"),
        "anchor_p": float("nan"),
        "n_pairs_used": 0.0,
    }
    covered = np.array(sorted(doc_to_node), dtype=np.int64)
    if covered.size < MIN_COVERED_DOCS:
        return empty

    rng = np.random.default_rng(seed)
    if covered.size > max_docs:
        covered = rng.choice(covered, size=max_docs, replace=False)

    hops, index = _hop_matrix(G)
    doc_row = np.array([index[doc_to_node[int(d)]] for d in covered])
    iu, ju = np.triu_indices(covered.size, k=1)
    graph_d = hops[doc_row[iu], doc_row[ju]]
    connected = np.flatnonzero(np.isfinite(graph_d))
    if connected.size < MIN_PAIRS:
        return {**empty, "n_pairs_used": float(connected.size)}

    take = rng.choice(connected, size=min(max_pairs, connected.size), replace=False)
    i, j = iu[take], ju[take]
    feats = np.asarray(anchor_features, dtype=np.float64)
    ling_d = np.linalg.norm(feats[covered[i]] - feats[covered[j]], axis=1)
    if np.ptp(graph_d[take]) == 0 or np.ptp(ling_d) == 0:
        # e.g. every connected pair inside a single node, all 0 hops apart:
        # rank correlation is undefined, as scipy would say with a warning.
        return {**empty, "n_pairs_used": float(take.size)}
    rho, p = spearmanr(graph_d[take], ling_d)
    return {
        "anchor_rho": float(rho),
        "anchor_p": float(p),
        "n_pairs_used": float(take.size),
    }
