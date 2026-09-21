"""Build one pair set from the SciCUEval train split.

    uv run python scripts/build_pairs.py --config configs/base.yaml \\
        --kind {random,matched,mismatched} --n 20000 --seed 0

Writes ``data/pairs/<kind>_s<seed>/pairs.parquet`` and a ``manifest.json``
beside it recording the config, git sha, seed, source JSONL hash, package
versions, and the pair builder's report (quotas, shortfalls, reuse).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from tlf.config import load_config
from tlf.data import (
    PAIR_KINDS,
    build_pairs_with_report,
    corpus_source_path,
    corpus_textstat,
    eval_split,
    load_corpus,
    sha256_file,
    train_split,
)
from tlf.results import write_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--kind", choices=PAIR_KINDS, required=True)
    ap.add_argument(
        "--n", type=int, default=None, help="pairs to build (default: pairs.n_pairs)"
    )
    ap.add_argument("--seed", type=int, default=None, help="default: cfg seed")
    ap.add_argument("--corpus", default="scicueval")
    ap.add_argument("--out-root", type=Path, default=None, help="default: paths.pairs")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    seed = int(cfg["seed"] if args.seed is None else args.seed)
    n_pairs = int(cfg["pairs"]["n_pairs"] if args.n is None else args.n)
    out_root = Path(cfg["paths"]["pairs"] if args.out_root is None else args.out_root)

    df = load_corpus(cfg, args.corpus)
    train = train_split(df)
    ev = eval_split(df)
    if train.empty:
        raise SystemExit(f"{args.corpus} has no train split; nothing to pair")
    X_all = corpus_textstat(df, cfg, args.corpus)
    X_train = X_all[np.flatnonzero((df["split"] == "train").to_numpy())]

    pairs, report = build_pairs_with_report(
        train,
        args.kind,
        n_pairs,
        seed,
        feature_matrix=X_train,
        matched_max_sd=float(cfg["pairs"]["matched_max_sd"]),
        mismatched_min_sd=float(cfg["pairs"]["mismatched_min_sd"]),
        eval_ids=ev["id"].tolist(),
    )

    out_dir = out_root / f"{args.kind}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_dir / "pairs.parquet", index=False)
    src = corpus_source_path(cfg, args.corpus)
    write_manifest(
        out_dir,
        config_path=cfg["_config_path"],
        seed=seed,
        extra={
            "artifact": "pairs",
            "corpus": args.corpus,
            "source_jsonl": str(src),
            "source_sha256": sha256_file(src),
            "n_eval": int(len(ev)),
            **report,
        },
    )
    print(
        f"{args.kind}: wrote {len(pairs)} pairs to {out_dir / 'pairs.parquet'} "
        f"(requested {n_pairs}; anchors used {report['n_anchors_used']}/{report['n_train']}, "
        f"max reuse {report['max_anchor_reuse']}, filled outside quota {report['filled_outside_quota']})"
    )


if __name__ == "__main__":
    main()
