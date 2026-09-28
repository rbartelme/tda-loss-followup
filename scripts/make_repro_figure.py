r"""Draw the per-seed anchor rho figure for the reproducibility post.

Usage::

    uv run python scripts/make_repro_figure.py \\
        --machine laptop=embeddings --machine "DGX Spark=/path/to/spark/embeddings" \\
        --out figures/anchor-per-seed.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tlf.plots import (
    fig_anchor_per_seed,
    fig_anchor_roster,
    load_machine_metrics,
    load_reanalysis,
    load_reference,
)


def main() -> None:
    """Parse ``label=dir`` machine arguments and write the figure."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--machine",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="machine label and its embeddings/ directory; repeat, in display order",
    )
    ap.add_argument("--reference", type=Path, default=Path("configs/reference.yaml"))
    ap.add_argument("--out", type=Path, default=Path("figures/anchor-per-seed.png"))
    ap.add_argument(
        "--roster",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="scripts/reanalyse_bakeoff.py output per machine; repeat, in display order",
    )
    ap.add_argument(
        "--roster-out", type=Path, default=Path("figures/anchor-roster.png")
    )
    args = ap.parse_args()
    if args.machine:
        ref = load_reference(args.reference)
        machines = {}
        for spec in args.machine:
            label, _, root = spec.partition("=")
            machines[label] = load_machine_metrics(root, ref)
        print(fig_anchor_per_seed(machines, args.out))
    if args.roster:
        rosters = {}
        for spec in args.roster:
            label, _, root = spec.partition("=")
            rosters[label] = load_reanalysis(root)
        print(fig_anchor_roster(rosters, args.roster_out))


if __name__ == "__main__":
    main()
