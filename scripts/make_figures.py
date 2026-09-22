"""Draw the four figures from ``results/*.csv``.

Usage::

    uv run python scripts/make_figures.py --results results
        --reference configs/reference.yaml --out figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tlf.plots import make_all


def main() -> None:
    """Parse arguments and draw every figure whose data exists."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--reference", type=Path, default=Path("configs/reference.yaml"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    written = make_all(args.results, args.reference, args.out)
    for p in written:
        print(p)
    if not written:
        raise SystemExit("no figures written (no result CSVs?)")


if __name__ == "__main__":
    main()
