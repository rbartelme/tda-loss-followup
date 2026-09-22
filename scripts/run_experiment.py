"""Run one experiment config end to end.

Usage::

    uv run python scripts/run_experiment.py --config configs/exp1_tau_sweep.yaml
        [--only-final] [--resume] [--force] [--dry-run]
        [--n-seeds N] [--n-workers N] [--corpus scicueval|mmlu]

Expands the config grid, trains runs whose checkpoints are missing, evaluates
every (checkpoint, corpus) whose result row is missing, and appends rows to
``results/<exp>.csv``. Finished work is skipped by default, so ``--resume`` is
accepted for readability and changes nothing. Checkpoints and cached metrics
are keyed by run, not by experiment, so a run shared between grids is trained
and evaluated once; ``--force`` retrains and re-evaluates, ignoring both caches.
``--only-final`` evaluates only ``ckpt_0.0`` and ``ckpt_1.0``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tlf.config import load_config
from tlf.experiment import describe_plan, execute, plan


def main() -> None:
    """Parse arguments and run or describe the experiment."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--only-final", action="store_true")
    ap.add_argument(
        "--resume", action="store_true", help="(default behaviour; kept for clarity)"
    )
    ap.add_argument(
        "--force", action="store_true", help="retrain and re-evaluate everything"
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--corpus", choices=["scicueval", "mmlu"], default=None)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    corpora = [args.corpus] if args.corpus else None
    if args.dry_run:
        print(
            describe_plan(
                plan(cfg, only_final=args.only_final, force=args.force, corpora=corpora)
            )
        )
        return
    stats = execute(
        cfg,
        only_final=args.only_final,
        force=args.force,
        n_seeds=args.n_seeds,
        n_workers=args.n_workers,
        corpora=corpora,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
