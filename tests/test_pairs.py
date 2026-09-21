"""Pair construction on a 200-row synthetic frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from tlf.data import (
    bakeoff_eval_indices,
    build_pairs,
    build_pairs_with_report,
    eval_split,
    train_split,
)
from tlf.features import FEATURE_NAMES, fit_standardizer, standardize, textstat_vector

N_SUBSETS = 4
PER_SUBSET = 50
N_EVAL_PER_SUBSET = 10


@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, np.ndarray]:
    """200 rows, 4 subsets. Features are 8 tight clusters per subset so that
    matched (<= 0.5 SD) and mismatched (>= 1.5 SD) partners both exist."""
    rng = np.random.default_rng(123)
    rows, feats = [], []
    for s in range(N_SUBSETS):
        centers = rng.normal(0, 1, size=(8, 6)) * 3.0
        for i in range(PER_SUBSET):
            c = centers[i % 8]
            feats.append(c + rng.normal(0, 0.02, size=6))
            rows.append(
                {
                    "id": f"s{s}-r{i}",
                    "text": f"subset {s} row {i}",
                    "subset": f"S{s}",
                    "domain": "d",
                    "split": "eval" if i < N_EVAL_PER_SUBSET else "train",
                    "eval_order": -1,
                }
            )
    df = pd.DataFrame(rows)
    X = np.vstack(feats)
    # Give eval rows absurd features: if standardization leaked them in, the
    # train-only thresholds below would stop holding.
    X[df["split"].to_numpy() == "eval"] += 1e4
    return df, X


def _train_features(synthetic):
    df, X = synthetic
    train = train_split(df)
    Xt = X[np.flatnonzero((df["split"] == "train").to_numpy())]
    return df, train, Xt


def _pair_dists(pairs: pd.DataFrame, train: pd.DataFrame, Xt: np.ndarray) -> np.ndarray:
    mu, sd = fit_standardizer(Xt)
    Z = standardize(Xt, mu, sd)
    pos = {i: k for k, i in enumerate(train["id"])}
    a = np.array([pos[i] for i in pairs["anchor_id"]])
    p = np.array([pos[i] for i in pairs["positive_id"]])
    return np.linalg.norm(Z[a] - Z[p], axis=1)


@pytest.mark.parametrize("kind", ["random", "matched", "mismatched"])
def test_no_eval_leakage(synthetic, kind):
    df, train, Xt = _train_features(synthetic)
    eval_ids = set(eval_split(df)["id"])
    pairs = build_pairs(train, kind, 100, seed=0, feature_matrix=Xt, eval_ids=eval_ids)
    assert not (set(pairs["anchor_id"]) & eval_ids)
    assert not (set(pairs["positive_id"]) & eval_ids)


def test_rejects_eval_rows_in_train_frame(synthetic):
    df, _, _ = _train_features(synthetic)
    with pytest.raises(ValueError):
        build_pairs(df, "random", 10, seed=0, feature_matrix=np.zeros((len(df), 6)))


def test_matched_within_half_sd(synthetic):
    df, train, Xt = _train_features(synthetic)
    pairs = build_pairs(train, "matched", 120, seed=0, feature_matrix=Xt)
    assert len(pairs) == 120
    d = _pair_dists(pairs, train, Xt)
    assert np.all(d <= 0.5)
    assert np.allclose(d, pairs["ts_dist"].to_numpy())


def test_mismatched_beyond_1p5_sd(synthetic):
    df, train, Xt = _train_features(synthetic)
    pairs = build_pairs(train, "mismatched", 120, seed=0, feature_matrix=Xt)
    assert len(pairs) == 120
    d = _pair_dists(pairs, train, Xt)
    assert np.all(d >= 1.5)


def test_random_pairs_are_within_subset(synthetic):
    df, train, Xt = _train_features(synthetic)
    pairs = build_pairs(train, "random", 150, seed=0, feature_matrix=Xt)
    subset_of = dict(zip(train["id"], train["subset"], strict=True))
    assert all(
        subset_of[a] == subset_of[p]
        for a, p in zip(pairs["anchor_id"], pairs["positive_id"], strict=True)
    )
    assert all(
        subset_of[a] == s
        for a, s in zip(pairs["anchor_id"], pairs["subset"], strict=True)
    )
    assert (pairs["anchor_id"] != pairs["positive_id"]).all()


@pytest.mark.parametrize("kind", ["random", "matched", "mismatched"])
def test_fixed_seed_is_reproducible(synthetic, kind):
    df, train, Xt = _train_features(synthetic)
    a = build_pairs(train, kind, 80, seed=7, feature_matrix=Xt)
    b = build_pairs(train, kind, 80, seed=7, feature_matrix=Xt)
    pd.testing.assert_frame_equal(a, b)
    c = build_pairs(train, kind, 80, seed=8, feature_matrix=Xt)
    assert not a.equals(c)


def test_anchors_without_replacement_until_exhausted(synthetic):
    df, train, Xt = _train_features(synthetic)
    n_train = len(train)  # 160
    pairs, rep = build_pairs_with_report(
        train, "random", n_train, seed=0, feature_matrix=Xt
    )
    assert rep["max_anchor_reuse"] == 1
    assert pairs["anchor_id"].is_unique
    # More pairs than anchors: reuse happens, but never a duplicate pair.
    pairs2, rep2 = build_pairs_with_report(
        train, "random", 2 * n_train + 5, seed=0, feature_matrix=Xt
    )
    assert len(pairs2) == 2 * n_train + 5
    assert rep2["max_anchor_reuse"] == 3
    assert not pairs2.duplicated(["anchor_id", "positive_id"]).any()


def test_quota_shortfall_is_filled_from_other_subsets(synthetic):
    df, train, Xt = _train_features(synthetic)
    Xt = Xt.copy()
    # Replace subset S0's clustered features with 40 distinct points at the same
    # scale as the other subsets' cluster centres: no two land within 0.5 SD, so
    # S0 cannot supply a single matched pair and its quota must be filled elsewhere.
    s0 = np.flatnonzero((train["subset"] == "S0").to_numpy())
    Xt[s0] = np.random.default_rng(5).normal(0, 3.0, size=(len(s0), 6))
    mu, sd = fit_standardizer(Xt)
    Z0 = standardize(Xt, mu, sd)[s0]
    assert pdist(Z0).min() > 0.5  # precondition for the assertions below
    pairs, rep = build_pairs_with_report(
        train, "matched", 60, seed=0, feature_matrix=Xt
    )
    assert len(pairs) == 60
    assert rep["quotas"]["S0"] > 0
    assert rep["per_subset_shortfall"]["S0"] == rep["quotas"]["S0"]
    assert rep["filled_outside_quota"] >= rep["quotas"]["S0"]
    assert (pairs["subset"] != "S0").all()
    assert rep["n_dead_anchors"] >= len(s0)


def test_bakeoff_eval_indices_matches_reference_algorithm():
    labels = ["b"] * 7 + ["a"] * 3 + ["c"] * 5
    idx = bakeoff_eval_indices(labels, sample_per_class=4, seed=42)
    # Reference: the bakeoff's load_corpus, inlined.
    by = {}
    for i, lab in enumerate(labels):
        by.setdefault(lab, []).append(i)
    rng = np.random.default_rng(42)
    ref = []
    for _, g in sorted(by.items()):
        if len(g) <= 4:
            ref.extend(g)
        else:
            ref.extend(g[j] for j in rng.choice(len(g), size=4, replace=False))
    assert idx.tolist() == ref
    assert len(set(idx)) == len(idx)


def test_textstat_vector_shape_and_order():
    v = textstat_vector("The cat sat on the mat. It was happy.")
    assert v.shape == (len(FEATURE_NAMES),)
    assert FEATURE_NAMES[2] == "lexicon_count" and v[2] == 9
    assert FEATURE_NAMES[3] == "sentence_count" and v[3] == 2
    assert FEATURE_NAMES[4] == "avg_sentence_length" and v[4] == 4.5
    assert np.isfinite(v).all()
    assert textstat_vector("").shape == (6,)
