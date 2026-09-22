"""Run one experiment config end to end.

Usage::

    uv run python scripts/run_experiment.py --config configs/exp1_tau_sweep.yaml
        [--only-final] [--resume] [--force] [--dry-run]
        [--n-seeds N] [--n-workers N] [--corpus scicueval|mmlu]
        [--model KEY]... [--tau T]...

Expands the config grid, trains runs whose checkpoints are missing, evaluates
every (checkpoint, corpus) whose result row is missing, and appends rows to
``results/<exp>.csv``. Finished work is skipped by default, so ``--resume`` is
accepted for readability and changes nothing. Checkpoints and cached metrics
are keyed by run, not by experiment, so a run shared between grids is trained
and evaluated once; ``--force`` retrains and re-evaluates, ignoring both caches.
``--only-final`` evaluates only ``ckpt_0.0`` and ``ckpt_1.0``. ``--model`` and
``--tau`` (each repeatable, intersected) narrow the grid to some runs: after
``--only-final`` has shown where anchor rho moves, rerun without it and with
``--tau`` for those temperatures to fill in their intermediate checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tlf.config import load_config
from tlf.experiment import describe_plan, execute, expand_grid, plan, select_runs


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
    ap.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="KEY",
        help="restrict to this model key (repeatable)",
    )
    ap.add_argument(
        "--tau",
        action="append",
        type=float,
        default=None,
        metavar="T",
        help="restrict to this temperature (repeatable)",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    corpora = [args.corpus] if args.corpus else None
    grid = expand_grid(cfg)
    if (args.model or args.tau) and not select_runs(
        grid, models=args.model, taus=args.tau
    ):
        ap.error(
            f"--model/--tau select no runs; the grid has models "
            f"{sorted({r.model for r in grid})} and taus {sorted({r.tau for r in grid})}"
        )
    if args.dry_run:
        print(
            describe_plan(
                plan(
                    cfg,
                    only_final=args.only_final,
                    force=args.force,
                    corpora=corpora,
                    n_seeds=args.n_seeds,
                    models=args.model,
                    taus=args.tau,
                )
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
        models=args.model,
        taus=args.tau,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
