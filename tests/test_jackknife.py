"""Jackknife standard error for a mean over seed pairs (protocol item 11)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from tlf.jackknife import pair_mean_se


def _pairs(n: int) -> list[tuple[int, int]]:
    """All seed pairs in ``combinations`` order.

    Args:
        n: Seed count.

    Returns:
        ``(i, j)`` pairs with ``i < j``.
    """
    return list(combinations(range(n), 2))


def test_additive_seed_effects_give_the_exact_standard_error():
    """With value_ij = x_i + x_j the pair mean is 2·mean(x); the jackknife is exact.

    The naive spread-over-root-n reading understates it by about the square
    root of two, which is why the item 11 bound needed this.
    """
    x = np.random.default_rng(0).normal(size=40)
    pairs = _pairs(40)
    vals = [x[i] + x[j] for i, j in pairs]
    se = pair_mean_se(vals, pairs, 40)
    assert se == pytest.approx(2 * np.std(x, ddof=1) / np.sqrt(40))
    naive = np.std(vals, ddof=1) / np.sqrt(40)
    assert se / naive == pytest.approx(np.sqrt(2), rel=0.1)


def test_matches_brute_force_leave_one_out_with_skipped_pairs():
    """Dropping some pairs upstream still gives the direct leave-one-out answer."""
    rng = np.random.default_rng(1)
    pairs = [p for p in _pairs(12) if rng.random() > 0.2]
    vals = rng.random(len(pairs))
    loo = [
        np.mean([v for v, (i, j) in zip(vals, pairs, strict=True) if k not in (i, j)])
        for k in range(12)
    ]
    expect = np.sqrt(11 / 12 * np.sum((np.array(loo) - np.mean(loo)) ** 2))
    assert pair_mean_se(vals, pairs, 12) == pytest.approx(expect)


def test_constant_values_have_zero_error_and_tiny_bootstraps_have_none():
    """Identical partitions give zero; fewer than three seeds give NaN."""
    assert pair_mean_se([1.0] * 10, _pairs(5), 5) == 0.0
    assert np.isnan(pair_mean_se([0.5], _pairs(2), 2))


def test_mismatched_lengths_raise():
    """A value list that does not line up with its pairs is a bug upstream."""
    with pytest.raises(ValueError):
        pair_mean_se([0.1, 0.2], _pairs(3), 3)
