"""Evaluator layers on synthetic embeddings, plus an end-to-end smoke when the corpora exist."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlf.config import load_config
from tlf.evaluate import (
    CORE_KEYS,
    RandomProjectionEncoder,
    cached_metrics,
    evaluate_checkpoint,
    layer1,
    layers_2_3,
    metrics_cache_path,
    router_layer,
    stratified_head,
)

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
MAPPER = {
    "umap": {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
    "cover": {"n_cubes": 15, "perc_overlap": 0.3},
    "clusterer": {"min_cluster_size": 5, "metric": "precomputed"},
}


def _blobs(n_per: int, k: int, d: int, seed: int, sep: float = 6.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, size=(k, d))
    X = np.vstack([centers[i] + rng.normal(0, 1, size=(n_per, d)) for i in range(k)])
    labels = [f"c{i}" for i in range(k) for _ in range(n_per)]
    return X.astype(np.float32), labels


def test_random_projection_encoder_is_deterministic_and_shaped():
    """Same texts, same seed, same embeddings; output is (n, dim)."""
    enc = RandomProjectionEncoder(dim=16, seed=1)
    a = enc(["alpha beta", "gamma"])
    b = RandomProjectionEncoder(dim=16, seed=1)(["alpha beta", "gamma"])
    assert a.shape == (2, 16) and np.array_equal(a, b)
    assert not np.array_equal(a[0], a[1])


def test_stratified_head_spreads_rows_across_labels():
    """A grouped eval split is subsampled round-robin, not from one label."""
    ev = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(40)],
            "text": ["t"] * 40,
            "subset": [f"L{i // 10}" for i in range(40)],
            "eval_order": range(40),
        }
    )
    sub = stratified_head(ev, 10)
    assert len(sub) == 10 and sub["subset"].value_counts().to_dict() == {
        "L0": 3,
        "L1": 3,
        "L2": 2,
        "L3": 2,
    }
    assert stratified_head(ev, 100).equals(ev)


@pytest.mark.parametrize("lr_eval", ["holdout", "cv"])
def test_layer1_on_separable_blobs(lr_eval):
    """Well-separated classes give a positive cosine gap and perfect LR accuracy."""
    X, labels = _blobs(30, 3, 16, seed=0)
    out = layer1(X, labels, seed=42, lr_eval=lr_eval, cv_folds=5)
    assert out["gap"] > 0 and out["lr_acc"] == 1.0 and out["n_classes"] == 3
    assert out["lr_eval"] == lr_eval


def test_router_union_and_per_corpus_on_separable_blobs():
    """Both router modes report near-perfect routing on separable synthetic corpora."""
    Xa, la = _blobs(40, 3, 16, seed=1)
    Xb, lb = _blobs(40, 3, 16, seed=2)
    u = router_layer(Xa, la, Xb, lb, mode="union", holdout_frac=0.2, seed=42)
    assert u["router_n_classes"] == 6
    assert u["router_acc_in"] >= 0.9 and u["router_acc_mmlu"] >= 0.9
    assert u["router_in_to_mmlu_frac"] <= 0.1 and u["router_mmlu_to_in_frac"] <= 0.1
    p = router_layer(Xa, la, Xb, lb, mode="per_corpus", seed=42)
    assert p["router_acc_in"] >= 0.9 and p["router_acc_mmlu"] >= 0.9
    with pytest.raises(ValueError):
        router_layer(Xa, la, Xb, lb, mode="nope")


def _hierarchical_cloud(seed: int = 0):
    """6 super-clusters x 5 sub-clusters x 20 points in 8-d: a continuum with density structure.

    Clean, separated blobs give Mapper nothing to cluster inside a cube
    (HDBSCAN never returns a lone cluster); this cloud yields a real graph.
    """
    rng = np.random.default_rng(seed)
    sup = rng.normal(0, 3, (6, 8))
    sub = sup[:, None, :] + rng.normal(0, 1.0, (6, 5, 8))
    X = np.vstack(
        [sub[i // 100, (i % 100) // 20] + rng.normal(0, 0.35, 8) for i in range(600)]
    )
    labels = [f"s{i // 100}" for i in range(600)]
    return X.astype(np.float32), labels


def test_layers_2_3_serial_on_structured_cloud():
    """The Mapper bootstrap runs serially and yields a non-empty, label-pure graph."""
    X, labels = _hierarchical_cloud()
    ts = np.random.default_rng(0).normal(size=(len(labels), 6)).astype(np.float32)
    # 15 cubes per lens axis (the bakeoff default) is sized for 4,000 docs; use 5 here.
    cfg = {**MAPPER, "cover": {"n_cubes": 5, "perc_overlap": 0.3}}
    out = layers_2_3(
        X, labels, ts, n_seeds=2, base_seed=42, n_workers=1, mapper_config=cfg
    )
    for k in (
        "ari",
        "ari_sd",
        "nmi",
        "coverage",
        "nodes",
        "purity",
        "anchor_rho",
        "disintegrated",
    ):
        assert k in out
    assert out["n_seeds_ok"] == 2
    assert out["nodes"] >= 5 and out["coverage"] > 0.5
    assert out["purity"] > 0.5  # nodes follow the super-cluster labels
    assert out["n_pairs_used"] >= 100  # anchor rho actually computed
    assert out["disintegrated"] is False
    assert all(
        f"feat_{f}_corr_umap1" in out
        for f in ("flesch_reading_ease", "difficult_words")
    )


def test_evaluate_checkpoint_smoke_end_to_end(tmp_path):
    """Smoke: both corpora, stand-in encoder, 2 seeds, serial workers, router on."""
    cfg = load_config(BASE)
    if not Path(cfg["paths"]["scicueval_jsonl"]).is_file():
        pytest.skip("corpora not present")
    cfg["paths"]["embeddings"] = str(tmp_path / "emb")
    cfg["paths"]["checkpoints"] = str(tmp_path / "ckpt")
    enc = RandomProjectionEncoder(dim=32, seed=0)
    ckpt = tmp_path / "ckpt" / "_smoke" / "rp"
    res = {}
    for corpus in ("scicueval", "mmlu"):
        res[corpus] = evaluate_checkpoint(
            ckpt, corpus, cfg, seeds=2, n_workers=1, encoder=enc, n_texts=50
        )
        assert set(CORE_KEYS) <= set(res[corpus])
        assert (tmp_path / "emb" / "_smoke" / "rp" / f"{corpus}.npy").is_file()
        assert (tmp_path / "emb" / "_smoke" / "rp" / f"{corpus}.metrics.json").is_file()
    # The router is a checkpoint-level quantity: identical on both corpus rows.
    assert res["scicueval"]["router_acc_in"] == res["mmlu"]["router_acc_in"]
    assert res["scicueval"]["router_acc_mmlu"] == res["mmlu"]["router_acc_mmlu"]

    # A second evaluation is answered from the metrics cache without recomputing.
    path = metrics_cache_path(ckpt, "mmlu", cfg)
    before = path.stat().st_mtime_ns
    kw = dict(seeds=2, n_workers=1, encoder=enc, n_texts=50)
    again = evaluate_checkpoint(ckpt, "mmlu", cfg, **kw)
    assert again["eval_seconds"] == res["mmlu"]["eval_seconds"]
    assert path.stat().st_mtime_ns == before
    # The cache is keyed on what was asked for: seed count and eval settings.
    name = getattr(enc, "name", repr(enc))
    hit = cached_metrics(ckpt, "mmlu", cfg, n_seeds=2, n_texts=50, encoder_name=name)
    assert hit is not None and hit["gap"] == res["mmlu"]["gap"]
    assert (
        cached_metrics(ckpt, "mmlu", cfg, n_seeds=3, n_texts=50, encoder_name=name)
        is None
    )
    mode = cfg["eval"]["router"]["mode"]
    cfg["eval"]["router"]["mode"] = "per_corpus"
    assert (
        cached_metrics(ckpt, "mmlu", cfg, n_seeds=2, n_texts=50, encoder_name=name)
        is None
    )
    cfg["eval"]["router"]["mode"] = mode
    # --force bypasses the cache and rewrites the file.
    evaluate_checkpoint(ckpt, "mmlu", cfg, use_cache=False, **kw)
    assert path.stat().st_mtime_ns != before
