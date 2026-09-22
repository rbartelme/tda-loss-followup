"""Evaluate one checkpoint on one corpus: bakeoff layers 1-3 plus a router layer.

Layer 1 (cosine geometry), Layer 2 (25-seed Mapper bootstrap) and Layer 3
(linguistic anchors) call the functions copied verbatim from the bakeoff in
``tlf._bakeoff``; the encoder is the bakeoff's own ``encode_all`` pointed at
the checkpoint directory, so numbers are comparable with the first post.
Layer 4 is new: a logistic-regression router over the eval-split embeddings
of both corpora (see ``router_layer``).

Embeddings are cached under ``embeddings/<run>/<corpus>.npy`` with a
provenance file, keyed on the checkpoint, the corpus id sequence and the
encoder, so re-evaluating a checkpoint never re-encodes.

``python -m tlf.evaluate --smoke`` runs the whole pipeline on 50 texts per
corpus with 3 seeds and a random-projection stand-in for the encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from tlf._bakeoff import diagnostic as bd
from tlf._bakeoff import tda as bt
from tlf.config import load_config
from tlf.data import CORPORA, corpus_textstat, eval_split, load_corpus
from tlf.features import FEATURE_NAMES
from tlf.results import write_manifest

log = logging.getLogger(__name__)

CORE_KEYS: tuple[str, ...] = (
    "within_cos",
    "between_cos",
    "gap",
    "lr_acc",
    "ari",
    "ari_sd",
    "nmi",
    "coverage",
    "nodes",
    "purity",
    "anchor_rho",
    "disintegrated",
    "router_acc_in",
    "router_acc_mmlu",
)

Encoder = Callable[[list[str]], np.ndarray]


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class CheckpointEncoder:
    """The bakeoff's ``encode_all`` pointed at a checkpoint directory.

    Args:
        ckpt_dir: A sentence-transformers checkpoint; its root holds the HF
            encoder, which ``AutoModel`` loads directly.
        cfg: Resolved config; reads ``eval.max_length``, ``eval.batch_size``,
            ``eval.device``.
    """

    def __init__(self, ckpt_dir: Path | str, cfg: dict[str, Any]) -> None:
        ev = cfg["eval"]
        self.ckpt_dir = str(ckpt_dir)
        self.max_length = int(ev["max_length"])
        self.batch_size = int(ev["batch_size"])
        self.device = bd.resolve_device(str(ev.get("device", "auto")))
        self.name = f"bakeoff.encode_all(max_length={self.max_length})"

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Encode texts to float32 mean-pooled embeddings.

        Args:
            texts: Raw texts.

        Returns:
            ``(len(texts), d)`` float32 array.
        """
        return bd.encode_all(
            list(texts), self.ckpt_dir, self.batch_size, self.max_length, self.device
        )


class RandomProjectionEncoder:
    """Deterministic stand-in encoder for smoke tests (no model, no GPU).

    Hashes character trigrams into ``n_buckets`` counts and projects them with
    a seeded Gaussian matrix, so similar strings land near each other.

    Args:
        dim: Output dimensionality.
        seed: Seed for the projection matrix.
        n_buckets: Hash buckets for the trigram counts.
    """

    def __init__(self, dim: int = 64, seed: int = 0, n_buckets: int = 4096) -> None:
        self.dim = int(dim)
        self.n_buckets = int(n_buckets)
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(
            0.0, 1.0 / np.sqrt(self.n_buckets), size=(self.n_buckets, self.dim)
        )
        self.name = f"random_projection(dim={self.dim}, seed={seed})"

    def _counts(self, text: str) -> np.ndarray:
        v = np.zeros(self.n_buckets, dtype=np.float32)
        t = f"  {text.lower()}  "
        for i in range(len(t) - 2):
            h = int(
                hashlib.blake2b(t[i : i + 3].encode(), digest_size=4).hexdigest(), 16
            )
            v[h % self.n_buckets] += 1.0
        return v

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Encode texts.

        Args:
            texts: Raw texts.

        Returns:
            ``(len(texts), dim)`` float32 array.
        """
        C = (
            np.vstack([self._counts(t) for t in texts])
            if texts
            else np.zeros((0, self.n_buckets))
        )
        return (C @ self.proj).astype(np.float32)


# ---------------------------------------------------------------------------
# Paths and caching
# ---------------------------------------------------------------------------


def run_name_for(ckpt_dir: Path | str, cfg: dict[str, Any]) -> str:
    """Name the run a checkpoint belongs to, for the embeddings cache path.

    Args:
        ckpt_dir: Checkpoint directory.
        cfg: Resolved config (``paths.checkpoints``).

    Returns:
        The checkpoint's path relative to ``paths.checkpoints`` when it lives
        there (``exp1/<run>/ckpt_0.5``), else ``<parent>/<name>``.
    """
    ck = Path(ckpt_dir).resolve()
    root = Path(cfg["paths"]["checkpoints"]).resolve()
    try:
        return str(ck.relative_to(root))
    except ValueError:
        return f"{ck.parent.name}/{ck.name}" if ck.parent.name else ck.name


def embeddings_dir(ckpt_dir: Path | str, cfg: dict[str, Any]) -> Path:
    """Directory holding the cached embeddings for a checkpoint.

    Args:
        ckpt_dir: Checkpoint directory.
        cfg: Resolved config (``paths.embeddings``).

    Returns:
        ``paths.embeddings / run_name_for(ckpt_dir)``.
    """
    return Path(cfg["paths"]["embeddings"]) / run_name_for(ckpt_dir, cfg)


def checkpoint_stamp(ckpt_dir: Path | str) -> str:
    """A cheap identity for a checkpoint's weights, for cache validation.

    Args:
        ckpt_dir: Checkpoint directory.

    Returns:
        ``git_sha:step:timestamp`` from its manifest when present, else the
        mtime of its weights file, else ``"none"`` (no such directory).
    """
    d = Path(ckpt_dir)
    m = d / "manifest.json"
    if m.is_file():
        try:
            j = json.loads(m.read_text())
            return f"{j.get('git_sha')}:{j.get('step')}:{j.get('timestamp')}"
        except (OSError, ValueError):
            pass
    for name in ("model.safetensors", "pytorch_model.bin", "config.json"):
        f = d / name
        if f.is_file():
            return f"mtime:{f.stat().st_mtime_ns}"
    return "none"


def stratified_head(ev: pd.DataFrame, n_texts: int) -> pd.DataFrame:
    """Take about ``n_texts`` rows spread round-robin across labels.

    The eval split is grouped by label, so a plain ``head`` would return a
    single class. Used by smoke runs only.

    Args:
        ev: Eval split in bakeoff order.
        n_texts: Target row count.

    Returns:
        The selected rows in their original relative order.
    """
    if n_texts >= len(ev):
        return ev
    labels = ev["subset"].to_numpy()
    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    picked: list[int] = []
    queues = [list(v) for _, v in sorted(by_label.items())]
    while len(picked) < n_texts and any(queues):
        for q in queues:
            if q and len(picked) < n_texts:
                picked.append(q.pop(0))
    return ev.iloc[sorted(picked)].reset_index(drop=True)


def cached_embeddings(
    ckpt_dir: Path | str,
    corpus: str,
    ev: pd.DataFrame,
    cfg: dict[str, Any],
    encoder: Any,
    seed: int,
) -> np.ndarray:
    """Encode the eval rows with ``encoder``, or load them from the cache.

    Args:
        ckpt_dir: Checkpoint directory (may not exist for stand-in encoders).
        corpus: Corpus name; names the cache files.
        ev: Eval rows to encode, in order.
        cfg: Resolved config.
        encoder: Callable ``texts -> (n, d)`` with a ``name`` attribute.
        seed: Recorded in the provenance file.

    Returns:
        ``(len(ev), d)`` float32 embeddings aligned to ``ev``.
    """
    d = embeddings_dir(ckpt_dir, cfg)
    d.mkdir(parents=True, exist_ok=True)
    npy = d / f"{corpus}.npy"
    meta = d / f"{corpus}.embeddings.json"
    ids = ev["id"].astype(str).tolist()
    key = {
        "ckpt_dir": str(ckpt_dir),
        "ckpt_stamp": checkpoint_stamp(ckpt_dir),
        "corpus": corpus,
        "n": len(ids),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "encoder": getattr(encoder, "name", repr(encoder)),
    }
    if npy.is_file() and meta.is_file():
        try:
            m = json.loads(meta.read_text())
            if all(m.get(k) == v for k, v in key.items()):
                X = np.load(npy)
                if X.shape[0] == len(ids):
                    log.info("embeddings cache hit: %s", npy)
                    return X
        except (OSError, ValueError):
            pass
    log.info("encoding %d %s texts with %s", len(ids), corpus, key["encoder"])
    X = np.asarray(encoder(ev["text"].tolist()), dtype=np.float32)
    np.save(npy, X)
    write_manifest(
        d,
        config_path=cfg["_config_path"],
        seed=seed,
        extra={"artifact": "embeddings", **key, "dim": int(X.shape[1])},
        name=meta.name,
    )
    return X


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def _encode_labels(labels: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    names = sorted(set(labels))
    idx = {n: i for i, n in enumerate(names)}
    return np.array([idx[x] for x in labels], dtype=np.int64), names


def _lr_holdout(X: np.ndarray, y: np.ndarray, seed: int, *, stratify: bool) -> float:
    """The bakeoff's LR holdout with stratification optional (tiny-input fallback)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if stratify else None
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def lr_accuracy_cv(X: np.ndarray, y: np.ndarray, seed: int, folds: int = 5) -> float:
    """Stratified k-fold accuracy with the bakeoff's classifier settings.

    Args:
        X: ``(n, d)`` raw embeddings.
        y: Integer labels.
        seed: Fold shuffling and classifier seed.
        folds: Number of folds. Falls back to the bakeoff's single holdout
            when some class has fewer members than ``folds``.

    Returns:
        Mean test accuracy across folds.
    """
    if np.bincount(y).min() < folds:
        log.warning("class too small for %d-fold CV; using the bakeoff holdout", folds)
        return bd.lr_accuracy(X, y, seed)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        )
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def layer1(
    X: np.ndarray,
    labels: Sequence[str],
    seed: int,
    lr_eval: str = "holdout",
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Cosine geometry: within/between cosine, gap, centroid separation, LR accuracy.

    Same procedure and RNG usage as the bakeoff's ``run_encoder`` minus the plots.

    Args:
        X: ``(n, d)`` raw embeddings.
        labels: Label per row.
        seed: The bakeoff's evaluation seed (42).
        lr_eval: ``"holdout"`` (bakeoff) or ``"cv"``.
        cv_folds: Folds for ``"cv"``.

    Returns:
        ``within_cos, between_cos, gap, centroid_mean_offdiag, lr_acc, lr_eval,
        n_docs, n_classes``.
    """
    y, names = _encode_labels(labels)
    Xn = bd._l2_normalize(X)
    cents = bd.class_centroids(Xn, y, len(names))
    cents_norm = bd._l2_normalize(cents)
    sims = cents_norm @ cents_norm.T
    k = sims.shape[0]
    offdiag = float(sims[~np.eye(k, dtype=bool)].mean()) if k > 1 else float("nan")
    rng = np.random.default_rng(seed)
    within, between = bd.within_between_cosine(Xn, y, rng)
    if lr_eval in ("cv", "cv5"):
        acc = lr_accuracy_cv(X, y, seed, cv_folds)
    else:
        try:
            acc = bd.lr_accuracy(X, y, seed)
        except ValueError:  # stratified 80/20 infeasible on tiny (smoke) inputs
            log.warning(
                "bakeoff holdout infeasible for n=%d; unstratified split", len(y)
            )
            acc = _lr_holdout(X, y, seed, stratify=False)
    return {
        "within_cos": float(within.mean()),
        "between_cos": float(between.mean()),
        "gap": float(within.mean() - between.mean()),
        "centroid_mean_offdiag": offdiag,
        "lr_acc": float(acc),
        "lr_eval": lr_eval,
        "n_docs": int(X.shape[0]),
        "n_classes": len(names),
    }


def layers_2_3(
    X: np.ndarray,
    labels: Sequence[str],
    textstat_raw: np.ndarray,
    *,
    n_seeds: int,
    base_seed: int,
    n_workers: int,
    mapper_config: dict[str, Any],
    anchor_max_docs: int = 500,
    anchor_max_pairs: int = 5000,
    min_coverage: float = 0.05,
    min_nodes: float = 5,
) -> dict[str, Any]:
    """Mapper bootstrap stability and anchor faithfulness, as in the bakeoff.

    Args:
        X: ``(n, d)`` raw embeddings (L2-normalised inside).
        labels: Label per row.
        textstat_raw: ``(n, 6)`` raw textstat features (standardised over the
            eval corpus inside, as the bakeoff does).
        n_seeds: Bootstrap seeds (``base_seed + i``).
        base_seed: First seed; also the canonical graph's seed.
        n_workers: Fork workers for the bootstrap; ``1`` runs serially.
        mapper_config: UMAP / cover / clusterer parameters.
        anchor_max_docs: Docs subsampled for anchor rho.
        anchor_max_pairs: Pair cap for anchor rho.
        min_coverage: Disintegration threshold on mean coverage.
        min_nodes: Disintegration threshold on mean node count.

    Returns:
        Core keys ``ari, ari_sd, nmi, nmi_sd, coverage, nodes, purity,
        mean_purity, anchor_rho, anchor_p, n_pairs_used, disintegrated,
        n_seeds_ok`` plus every ``mean_*``/``cv_*`` statistic and the 18
        per-feature alignment values.
    """
    emb = bt.l2_normalize(np.asarray(X, dtype=np.float32))
    bt.assert_normalized(emb)
    n = emb.shape[0]
    dist = bt.compute_distance_matrix(emb)
    stab, _rows = bt.seed_bootstrap_stability(
        emb, dist, n, n_seeds, base_seed, n_workers, mapper_config
    )
    graph, lens = bt.build_mapper_for_seed(emb, dist, base_seed, mapper_config)
    G = bt.mapper_to_networkx(graph)
    ts = bt.standardize(np.asarray(textstat_raw, dtype=np.float32))
    cat = bt.categorical_anchor_purity(G, list(labels))
    cont = bt.continuous_anchor_correlation(
        G, ts, n, base_seed, max_pairs=anchor_max_pairs, max_docs=anchor_max_docs
    )
    pfa = bt.per_feature_alignment(lens, ts, list(FEATURE_NAMES))
    nan = float("nan")
    coverage = float(stab.get("mean_coverage", nan))
    nodes = float(stab.get("mean_n_nodes", nan))
    disintegrated = bool(
        np.isnan(coverage)
        or np.isnan(nodes)
        or coverage < min_coverage
        or nodes < min_nodes
    )
    out: dict[str, Any] = {
        "ari": float(stab.get("mean_ari", nan)),
        "ari_sd": float(stab.get("sd_ari", nan)),
        "nmi": float(stab.get("mean_nmi", nan)),
        "nmi_sd": float(stab.get("sd_nmi", nan)),
        "coverage": coverage,
        "nodes": nodes,
        "purity": float(cat["weighted_purity"]),
        "mean_purity": float(cat["mean_purity"]),
        "anchor_rho": float(cont["anchor_spearman_rho"]),
        "anchor_p": float(cont["anchor_spearman_p"]),
        "n_pairs_used": float(cont["n_pairs_used"]),
        "disintegrated": disintegrated,
        "n_seeds_ok": float(stab.get("n_successful_seeds", nan)),
    }
    out.update({k: float(v) for k, v in stab.items() if k not in ("_error",)})
    out.update(pfa)
    return out


def router_layer(
    X_a: np.ndarray,
    labels_a: Sequence[str],
    X_b: np.ndarray,
    labels_b: Sequence[str],
    *,
    mode: str = "union",
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Layer 4: can a linear router send a query to the right source?

    ``union`` fits one logistic regression over the union of both corpora's
    labels (prefixed by corpus, since the two label sets are disjoint) on a
    stratified ``1 - holdout_frac`` share of both eval splits, and reports
    accuracy separately on the held-out rows of corpus A (``router_acc_in``)
    and corpus B (``router_acc_mmlu``). It also reports how often held-out rows
    of one corpus were routed to a class of the other. ``per_corpus`` fits two
    independent classifiers with the bakeoff's holdout recipe.

    Args:
        X_a: Corpus A (SciCUEval) embeddings.
        labels_a: Corpus A labels.
        X_b: Corpus B (MMLU) embeddings.
        labels_b: Corpus B labels.
        mode: ``"union"`` or ``"per_corpus"``.
        holdout_frac: Test share.
        seed: Split and classifier seed.

    Returns:
        ``router_acc_in, router_acc_mmlu, router_mode`` and, for ``union``,
        ``router_a_to_b_frac, router_b_to_a_frac, router_n_classes``.

    Raises:
        ValueError: On an unknown mode.
    """
    if mode == "per_corpus":
        ya, _ = _encode_labels(labels_a)
        yb, _ = _encode_labels(labels_b)
        return {
            "router_acc_in": float(bd.lr_accuracy(X_a, ya, seed)),
            "router_acc_mmlu": float(bd.lr_accuracy(X_b, yb, seed)),
            "router_mode": mode,
        }
    if mode != "union":
        raise ValueError(
            f"unknown router mode {mode!r}; expected 'union' or 'per_corpus'"
        )
    if X_a.shape[1] != X_b.shape[1]:
        raise ValueError("both corpora must be embedded by the same encoder")
    X = np.vstack([X_a, X_b])
    lab = np.array([f"a:{x}" for x in labels_a] + [f"b:{x}" for x in labels_b])
    side = np.array([0] * len(labels_a) + [1] * len(labels_b))
    y, names = _encode_labels(lab)
    n_test = int(round(holdout_frac * len(y)))
    strat: np.ndarray | None = y
    if np.bincount(y).min() < 2 or n_test < len(names) or len(y) - n_test < len(names):
        strat = None  # tiny (smoke) inputs: stratification infeasible
    tr, te = train_test_split(
        np.arange(len(y)), test_size=n_test, random_state=seed, stratify=strat
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    is_b_class = np.array([n.startswith("b:") for n in names])
    te_a = side[te] == 0
    te_b = side[te] == 1
    acc_a = float((pred[te_a] == y[te][te_a]).mean()) if te_a.any() else float("nan")
    acc_b = float((pred[te_b] == y[te][te_b]).mean()) if te_b.any() else float("nan")
    return {
        "router_acc_in": acc_a,
        "router_acc_mmlu": acc_b,
        "router_mode": mode,
        "router_n_classes": len(names),
        "router_a_to_b_frac": float(is_b_class[pred[te_a]].mean())
        if te_a.any()
        else float("nan"),
        "router_b_to_a_frac": float((~is_b_class[pred[te_b]]).mean())
        if te_b.any()
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _eval_rows(
    cfg: dict[str, Any], corpus: str, n_texts: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_corpus(cfg, corpus)
    ev = eval_split(df)
    if n_texts is not None:
        ev = stratified_head(ev, int(n_texts))
    return df, ev


def evaluate_router(
    ckpt_dir: Path | str,
    cfg: dict[str, Any],
    *,
    encoder: Any | None = None,
    n_texts: int | None = None,
) -> dict[str, Any]:
    """Layer 4 for a checkpoint, using cached embeddings of both corpora.

    Args:
        ckpt_dir: Checkpoint directory.
        cfg: Resolved config (``eval.router``).
        encoder: Override encoder (smoke runs).
        n_texts: Subsample per corpus (smoke runs).

    Returns:
        The ``router_layer`` dict plus ``router_label``.
    """
    ev_cfg = cfg["eval"]
    r = ev_cfg.get("router", {})
    label_col = "domain" if str(r.get("label", "subset")) == "domain" else "subset"
    encoder = encoder or CheckpointEncoder(ckpt_dir, cfg)
    seed = int(ev_cfg["seed"])
    X: dict[str, np.ndarray] = {}
    lab: dict[str, list[str]] = {}
    for corpus in CORPORA:
        _, ev = _eval_rows(cfg, corpus, n_texts)
        X[corpus] = cached_embeddings(ckpt_dir, corpus, ev, cfg, encoder, seed)
        lab[corpus] = ev[label_col].astype(str).tolist()
    out = router_layer(
        X["scicueval"],
        lab["scicueval"],
        X["mmlu"],
        lab["mmlu"],
        mode=str(r.get("mode", "union")),
        holdout_frac=float(r.get("holdout_frac", 0.2)),
        seed=seed,
    )
    out["router_label"] = label_col
    return out


def evaluate_checkpoint(
    ckpt_dir: Path | str,
    corpus: str,
    cfg: dict[str, Any],
    seeds: int | None = None,
    *,
    n_workers: int | None = None,
    encoder: Any | None = None,
    n_texts: int | None = None,
    with_router: bool = True,
) -> dict[str, Any]:
    """Run layers 1-4 for one checkpoint on one corpus.

    Args:
        ckpt_dir: Checkpoint directory.
        corpus: ``"scicueval"`` or ``"mmlu"``.
        cfg: Resolved config (``eval`` section).
        seeds: Bootstrap seeds; defaults to ``eval.n_seeds`` (25).
        n_workers: Bootstrap workers; defaults to ``eval.n_workers``.
        encoder: Override encoder (smoke runs). Defaults to the bakeoff's
            ``encode_all`` on the checkpoint.
        n_texts: Subsample the eval split (smoke runs).
        with_router: Also compute Layer 4 (encodes the other corpus too).

    Returns:
        One flat dict with the ``CORE_KEYS`` and every supporting statistic,
        also written to ``embeddings/<run>/<corpus>.metrics.json``.
    """
    t0 = time.time()
    ev_cfg = cfg["eval"]
    n_seeds = int(ev_cfg["n_seeds"] if seeds is None else seeds)
    workers = int(ev_cfg["n_workers"] if n_workers is None else n_workers)
    seed = int(ev_cfg["seed"])
    encoder = encoder or CheckpointEncoder(ckpt_dir, cfg)

    df, ev = _eval_rows(cfg, corpus, n_texts)
    X = cached_embeddings(ckpt_dir, corpus, ev, cfg, encoder, seed)
    T_all = corpus_textstat(df, cfg, corpus)
    pos = {i: k for k, i in enumerate(df["id"].astype(str))}
    T = T_all[[pos[i] for i in ev["id"].astype(str)]]
    labels = ev["subset"].astype(str).tolist()

    out: dict[str, Any] = {
        "ckpt_dir": str(ckpt_dir),
        "run": run_name_for(ckpt_dir, cfg),
        "corpus": corpus,
        "encoder": getattr(encoder, "name", repr(encoder)),
        "n_seeds": n_seeds,
    }
    log.info("[%s] layer 1", corpus)
    out.update(
        layer1(
            X,
            labels,
            seed,
            lr_eval=str(ev_cfg.get("lr_eval", "holdout")),
            cv_folds=int(ev_cfg.get("lr_cv_folds", 5)),
        )
    )
    log.info("[%s] layers 2+3: %d seeds, %d workers", corpus, n_seeds, workers)
    dis = ev_cfg.get("disintegrated", {})
    out.update(
        layers_2_3(
            X,
            labels,
            T,
            n_seeds=n_seeds,
            base_seed=int(ev_cfg["base_seed"]),
            n_workers=workers,
            mapper_config=ev_cfg["mapper"],
            anchor_max_docs=int(ev_cfg.get("anchor_max_docs", 500)),
            anchor_max_pairs=int(ev_cfg.get("anchor_max_pairs", 5000)),
            min_coverage=float(dis.get("min_coverage", 0.05)),
            min_nodes=float(dis.get("min_nodes", 5)),
        )
    )
    if with_router:
        log.info("[%s] layer 4 router", corpus)
        out.update(evaluate_router(ckpt_dir, cfg, encoder=encoder, n_texts=n_texts))
    out["eval_seconds"] = round(time.time() - t0, 1)

    d = embeddings_dir(ckpt_dir, cfg)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{corpus}.metrics.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n"
    )
    log.info(
        "[%s] gap=%.4f lr=%.3f ari=%.3f cov=%.3f nodes=%.1f rho=%.3f disint=%s router=(%.3f, %.3f) %.0fs",
        corpus,
        out["gap"],
        out["lr_acc"],
        out["ari"],
        out["coverage"],
        out["nodes"],
        out["anchor_rho"],
        out["disintegrated"],
        out.get("router_acc_in", float("nan")),
        out.get("router_acc_mmlu", float("nan")),
        out["eval_seconds"],
    )
    return out


def main() -> None:
    """CLI: evaluate a checkpoint, or ``--smoke`` the pipeline with a stand-in encoder."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--corpus", choices=[*CORPORA, "both"], default="both")
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument(
        "--n-texts", type=int, default=None, help="subsample the eval split"
    )
    ap.add_argument("--no-router", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="3 seeds, 50 texts, random-projection encoder, both corpora",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    encoder = None
    ckpt = args.ckpt
    n_seeds, n_texts = args.n_seeds, args.n_texts
    if args.smoke:
        encoder = RandomProjectionEncoder(dim=64, seed=int(cfg["eval"]["seed"]))
        ckpt = (
            ckpt or Path(cfg["paths"]["checkpoints"]) / "_smoke" / "random_projection"
        )
        n_seeds = n_seeds or 3
        n_texts = n_texts or 50
    if ckpt is None:
        ap.error("--ckpt is required unless --smoke")
    corpora = list(CORPORA) if args.corpus == "both" else [args.corpus]
    results = {}
    for corpus in corpora:
        results[corpus] = evaluate_checkpoint(
            ckpt,
            corpus,
            cfg,
            seeds=n_seeds,
            n_workers=args.n_workers,
            encoder=encoder,
            n_texts=n_texts,
            with_router=not args.no_router,
        )
    print(f"{'metric':<18}" + "".join(f"{c:>14}" for c in corpora))
    for k in CORE_KEYS:
        vals = []
        for c in corpora:
            v = results[c].get(k, float("nan"))
            vals.append(f"{v:>14}" if isinstance(v, bool) else f"{float(v):>14.4f}")
        print(f"{k:<18}" + "".join(vals))


if __name__ == "__main__":
    main()
