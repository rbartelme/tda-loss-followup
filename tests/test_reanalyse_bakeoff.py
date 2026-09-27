"""Discovery and published-value loading for scripts/reanalyse_bakeoff.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reanalyse_bakeoff as rb  # noqa: E402


def _scratch(tmp_path: Path) -> Path:
    """A minimal bakeoff scratch tree: two keys, one cached on one corpus only.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        The scratch root.
    """
    for diag in ("diagnostic_subset", "diagnostic_mmlu"):
        d = tmp_path / diag / "both"
        d.mkdir(parents=True)
        np.save(d / "embeddings.npy", np.zeros((2, 2), np.float32))
    only = tmp_path / "diagnostic_subset" / "sci-only"
    only.mkdir()
    np.save(only / "embeddings.npy", np.zeros((2, 2), np.float32))
    (tmp_path / "tda_subset").mkdir()
    (tmp_path / "tda_subset" / "comparison.csv").write_text(
        "embedder,mean_ari,anchor_spearman_rho,mean_coverage,mean_n_nodes,extra\n"
        "both,0.8,0.25,0.5,200,x\n"
    )
    return tmp_path


def test_discover_keys_needs_both_corpora(tmp_path):
    """A key cached for one corpus only is left out."""
    assert rb.discover_keys(_scratch(tmp_path)) == ["both"]


def test_load_published_keeps_the_four_columns(tmp_path):
    """Published rows come back as floats under the four compared columns."""
    pub = rb.load_published(_scratch(tmp_path), "scicueval")
    assert pub == {
        "both": {
            "mean_ari": 0.8,
            "anchor_spearman_rho": 0.25,
            "mean_coverage": 0.5,
            "mean_n_nodes": 200.0,
        }
    }
