"""Leave-one-seed-out standard error for a mean over seed pairs.

The bootstrap's stability score is the mean ARI (or NMI) over every pair of
seeds' partitions. Those pairs share seeds, so their spread divided by the
square root of the seed count is not a standard error: with additive seed
effects it understates the true one by a factor of about the square root of
two. Dropping one seed at a time, and every pair it is in, gives a jackknife
standard error that respects the dependence (protocol item 11).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pair_mean_se(
    values: Sequence[float], pairs: Sequence[tuple[int, int]], n_seeds: int
) -> float:
    """Jackknife standard error of the mean of per-pair values, deleting seeds.

    Args:
        values: One value per pair, e.g. the ARI between two seeds' partitions.
        pairs: The ``(i, j)`` seed indices of each value, ``0 <= i, j < n_seeds``.
            Pairs absent from the list (skipped upstream) are simply not used.
        n_seeds: Seeds in the bootstrap.

    Returns:
        ``sqrt((n - 1) / n * sum_k (m_k - mean(m))**2)``, where ``m_k`` is the
        mean over pairs not involving seed ``k``; NaN when fewer than three
        seeds contribute a leave-one-out mean.
    """
    v = np.asarray(values, dtype=np.float64)
    ij = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if v.size != ij.shape[0]:
        raise ValueError(f"{v.size} values for {ij.shape[0]} pairs")
    loo: list[float] = []
    for k in range(n_seeds):
        keep = (ij[:, 0] != k) & (ij[:, 1] != k)
        if keep.any():
            loo.append(float(v[keep].mean()))
    n = len(loo)
    if n < 3:
        return float("nan")
    m = np.asarray(loo)
    return float(np.sqrt((n - 1) / n * np.sum((m - m.mean()) ** 2)))
