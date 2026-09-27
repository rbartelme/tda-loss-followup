"""Per-seed uniform anchor rho and its bootstrap summary (protocol item 12)."""

from __future__ import annotations

import time
from math import comb

import networkx as nx
import numpy as np
import pytest

from tlf._bakeoff import tda as bt
from tlf.anchor import anchor_rho_uniform


def _path(n_nodes: int, docs_per_node: int, first_doc: int = 0) -> nx.Graph:
    """A path of Mapper-style nodes, each holding a consecutive block of docs.

    Args:
        n_nodes: Nodes on the path.
        docs_per_node: Member docs per node.
        first_doc: Index of the first node's first doc.

    Returns:
        A networkx path graph with a ``members`` list on every node.
    """
    G = nx.path_graph(n_nodes)
    for k in G.nodes:
        start = first_doc + k * docs_per_node
        G.nodes[k]["members"] = list(range(start, start + docs_per_node))
    return G


def _along(n_docs: int, docs_per_node: int) -> np.ndarray:
    """One anchor feature equal to a doc's node position, plus a little noise.

    Args:
        n_docs: Docs on the path.
        docs_per_node: Member docs per node.

    Returns:
        ``(n_docs, 1)`` float array.
    """
    pos = (np.arange(n_docs) // docs_per_node).astype(np.float64)
    return pos[:, None] + np.random.default_rng(0).normal(0, 0.1, (n_docs, 1))


def test_uniform_pairs_are_not_captured_by_the_first_docs():
    """Corrupting 20 of 500 docs dents uniform rho but flips the bakeoff's.

    The bakeoff fills its 5000-pair cap in ``combinations`` order, so every
    pair it uses touches one of the first ~11 docs.
    """
    G = _path(50, 10)
    feats = _along(500, 10)
    feats[:20] += 1000.0
    ours = anchor_rho_uniform(G, feats, seed=42)
    theirs = bt.continuous_anchor_correlation(
        G, feats, 500, 42, max_pairs=5000, max_docs=500
    )
    assert ours["n_pairs_used"] == 5000
    assert ours["anchor_rho"] > 0.8
    assert theirs["anchor_spearman_rho"] < -0.5


def test_pairs_across_components_never_enter():
    """With the cap lifted, exactly the within-component pairs are used."""
    G = nx.union(_path(25, 10), _path(25, 10, first_doc=250), rename=("a", "b"))
    out = anchor_rho_uniform(G, _along(500, 10), seed=0, max_pairs=10**6)
    assert out["n_pairs_used"] == 2 * comb(250, 2)
    assert np.isfinite(out["anchor_rho"])


def test_too_few_covered_docs_gives_nan():
    """Below 50 covered docs there is no rho, and the seed mean skips it."""
    out = anchor_rho_uniform(_path(3, 10), _along(30, 10), seed=0)
    assert np.isnan(out["anchor_rho"]) and out["n_pairs_used"] == 0


def test_same_seed_same_draw_other_seed_other_draw():
    """The seed fixes the doc subsample and the pairs; another seed moves both."""
    G, feats = _path(50, 10), _along(500, 10)
    feats[::7] += 5.0  # break the perfect monotone so different draws differ
    a = anchor_rho_uniform(G, feats, seed=7, max_docs=100)
    b = anchor_rho_uniform(G, feats, seed=7, max_docs=100)
    c = anchor_rho_uniform(G, feats, seed=8, max_docs=100)
    assert a == b
    assert a["anchor_rho"] != c["anchor_rho"]


_FAKE_RHO = {42: 0.1, 43: float("nan"), 44: 0.3}


def _slow_fake_worker(seed: int):
    """Stand-in for ``tda._seed_worker``: later seeds finish first.

    Args:
        seed: Bootstrap seed.

    Returns:
        A stat row tagged with the seed and a fixed 4-doc partition.
    """
    time.sleep(0.2 * (44 - seed))
    row = {k: float(seed) for k in bt.GRAPH_STAT_KEYS}
    row["anchor_rho"] = _FAKE_RHO[seed]
    return row, np.array([0, 0, 1, 1])


# Python 3.12 warns on any fork from a threaded process; the bootstrap forks by design.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_bootstrap_aggregates_in_seed_order_and_summarises_anchor(monkeypatch):
    """Rows come back in seed order, and NaN seeds drop out of the anchor summary."""
    monkeypatch.setattr(bt, "_seed_worker", _slow_fake_worker)
    out, rows = bt.seed_bootstrap_stability(
        np.zeros((4, 2), np.float32),
        np.zeros((4, 4)),
        4,
        3,
        42,
        3,
        anchor_features=np.zeros((4, 1)),
    )
    assert [r["n_nodes"] for r in rows] == [42.0, 43.0, 44.0]
    assert [r["seed"] for r in rows] == [42, 43, 44]
    assert out["_pairwise"]["seed_a"] == [42, 42, 43]
    assert out["_pairwise"]["seed_b"] == [43, 44, 44]
    assert out["n_anchor_seeds"] == 2
    assert out["mean_anchor_rho"] == pytest.approx(0.2)
    assert out["sd_anchor_rho"] == pytest.approx(np.std([0.1, 0.3], ddof=1))
    assert out["mean_ari"] == 1.0 and out["se_ari"] == 0.0  # identical partitions


def test_no_anchor_features_means_no_anchor_summary(monkeypatch):
    """Passing no features leaves the bootstrap's output as the bakeoff's."""
    monkeypatch.setattr(bt, "_seed_worker", _slow_fake_worker)
    out, _ = bt.seed_bootstrap_stability(
        np.zeros((4, 2), np.float32), np.zeros((4, 4)), 4, 3, 42, 1
    )
    assert "mean_anchor_rho" not in out
