"""Re-run Layers 2-3 on the first post's own cached embeddings, per-seed anchor rho on.

The bakeoff cached every encoder's eval-split embeddings under
``scratch/diagnostic_{subset,mmlu}/<key>/embeddings.npy``; they are
bit-identical to what this repo encodes on the same laptop, so this is the
first post's evaluation rerun end to end, minus the encoding. Each row's
published values from ``scratch/tda_{subset,mmlu}/comparison.csv`` are
stored beside the new metrics, so the single draw can be checked against
them. Rows already written are skipped, so an interrupted run resumes.

Usage::

    uv run python scripts/reanalyse_bakeoff.py --n-workers 8 [--keys KEY ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from tlf.config import load_config
from tlf.data import corpus_textstat
from tlf.evaluate import _eval_rows, layers_2_3

SCRATCH = Path("/home/rbartelme/00-projects/tda-embedder-bakeoff/scratch")
DIRS = {
    "scicueval": ("diagnostic_subset", "tda_subset"),
    "mmlu": ("diagnostic_mmlu", "tda_mmlu"),
}
PUBLISHED_KEYS = ("mean_ari", "anchor_spearman_rho", "mean_coverage", "mean_n_nodes")

log = logging.getLogger("reanalyse_bakeoff")


def discover_keys(scratch: Path) -> list[str]:
    """Encoder keys with cached embeddings for both corpora.

    Args:
        scratch: The bakeoff's ``scratch`` directory.

    Returns:
        Sorted keys present under both diagnostic directories.
    """
    found = [
        {p.parent.name for p in (scratch / diag).glob("*/embeddings.npy")}
        for diag, _ in DIRS.values()
    ]
    return sorted(set.intersection(*found))


def load_published(scratch: Path, corpus: str) -> dict[str, dict[str, float]]:
    """The first post's Layer 2-3 numbers for one corpus, by encoder key.

    Args:
        scratch: The bakeoff's ``scratch`` directory.
        corpus: ``scicueval`` or ``mmlu``.

    Returns:
        ``{key: {column: value}}`` for the columns in ``PUBLISHED_KEYS``.
    """
    path = scratch / DIRS[corpus][1] / "comparison.csv"
    with path.open() as f:
        return {
            row["embedder"]: {k: float(row[k]) for k in PUBLISHED_KEYS}
            for row in csv.DictReader(f)
        }


def reanalyse(
    key: str, corpus: str, cfg: dict[str, Any], scratch: Path, n_workers: int
) -> dict[str, Any]:
    """Layers 2-3 with per-seed anchor rho on one cached embedding matrix.

    Args:
        key: Bakeoff encoder key, e.g. ``medcpt-query``.
        corpus: ``scicueval`` or ``mmlu``.
        cfg: Resolved config (``configs/base.yaml``).
        scratch: The bakeoff's ``scratch`` directory.
        n_workers: Bootstrap workers.

    Returns:
        The ``layers_2_3`` output plus ``key``, ``corpus``, ``source`` and
        ``eval_seconds``.

    Raises:
        ValueError: If the cached matrix has a different row count than the
            eval split.
    """
    ev_cfg = cfg["eval"]
    df, ev = _eval_rows(cfg, corpus, None)
    t_all = corpus_textstat(df, cfg, corpus)
    pos = {i: k for k, i in enumerate(df["id"].astype(str))}
    textstat = t_all[[pos[i] for i in ev["id"].astype(str)]]
    labels = ev["subset"].astype(str).tolist()
    src = scratch / DIRS[corpus][0] / key / "embeddings.npy"
    X = np.load(src)
    if X.shape[0] != len(ev):
        raise ValueError(f"{src}: {X.shape[0]} rows, eval split has {len(ev)}")
    dis = ev_cfg.get("disintegrated", {})
    t0 = time.time()
    out = layers_2_3(
        X,
        labels,
        textstat,
        n_seeds=int(ev_cfg["n_seeds"]),
        base_seed=int(ev_cfg["base_seed"]),
        n_workers=n_workers,
        mapper_config=ev_cfg["mapper"],
        anchor_max_docs=int(ev_cfg["anchor_max_docs"]),
        anchor_max_pairs=int(ev_cfg["anchor_max_pairs"]),
        anchor_per_seed=True,
        min_coverage=float(dis.get("min_coverage", 0.05)),
        min_nodes=float(dis.get("min_nodes", 5)),
    )
    out.update(key=key, corpus=corpus, source=str(src))
    out["eval_seconds"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    """Re-analyse every requested (key, corpus) that has no output yet."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--scratch", type=Path, default=SCRATCH)
    ap.add_argument("--out", type=Path, default=Path("embeddings/_bakeoff_reanalysis"))
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--corpus", choices=[*DIRS, "both"], default="both")
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    keys = args.keys or discover_keys(args.scratch)
    corpora = list(DIRS) if args.corpus == "both" else [args.corpus]
    for corpus in corpora:
        published = load_published(args.scratch, corpus)
        for key in keys:
            path = args.out / key / f"{corpus}.metrics.json"
            if path.is_file():
                log.info("[%s %s] done already", key, corpus)
                continue
            out = reanalyse(key, corpus, cfg, args.scratch, args.n_workers)
            out["published"] = published.get(key, {})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=2, default=float) + "\n")
            pub = out["published"]
            log.info(
                "[%s %s] ari %.4f (published %.4f)  single rho %.4f (published %.4f)"
                "  per-seed rho %.3f ± %.3f  %.0fs",
                key,
                corpus,
                out["ari"],
                pub.get("mean_ari", float("nan")),
                out["anchor_rho"],
                pub.get("anchor_spearman_rho", float("nan")),
                out["anchor_rho_mean"],
                out["anchor_rho_sd"],
                out["eval_seconds"],
            )


if __name__ == "__main__":
    main()
