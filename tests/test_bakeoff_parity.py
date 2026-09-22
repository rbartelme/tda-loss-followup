"""Reproduction check against the first post's cached corpus.

Skipped unless the bakeoff scratch directory and raw corpora exist on this
machine. When they do, this is the proof that `eval_split` is the exact
row sequence the first post encoded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tlf.config import load_config
from tlf.data import (
    eval_split,
    flatten_mmlu,
    flatten_scicueval,
    load_corpus,
    read_jsonl,
)

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
SCRATCH = Path("/home/rbartelme/00-projects/tda-embedder-bakeoff/scratch")
DIAG = {"scicueval": SCRATCH / "diagnostic_subset", "mmlu": SCRATCH / "diagnostic_mmlu"}


@pytest.fixture(scope="module")
def cfg():
    """Load the repo's base config once per module."""
    return load_config(BASE)


@pytest.mark.parametrize("corpus", ["scicueval", "mmlu"])
def test_eval_split_matches_bakeoff_corpus_npz(cfg, corpus):
    """The eval split is the exact prompt and label sequence the first post encoded."""
    npz = DIAG[corpus] / "corpus.npz"
    if not npz.is_file():
        pytest.skip(f"{npz} not present")
    z = np.load(npz, allow_pickle=True)
    ev = eval_split(load_corpus(cfg, corpus))
    assert ev["text"].tolist() == z["prompts"].tolist()
    assert ev["subset"].tolist() == z["labels"].tolist()


@pytest.mark.parametrize("corpus", ["scicueval", "mmlu"])
def test_flatten_from_raw_matches_bakeoff_jsonl(cfg, corpus):
    """Re-flattening the raw tree reproduces the bakeoff's JSONL record for record."""
    root = Path(cfg["paths"][f"{corpus}_root"])
    jsonl = Path(cfg["paths"][f"{corpus}_jsonl"])
    if not (root.is_dir() and jsonl.is_file()):
        pytest.skip("raw tree or bakeoff JSONL not present")
    flat = flatten_scicueval(root) if corpus == "scicueval" else flatten_mmlu(root)
    ref = read_jsonl(jsonl)
    keys = ("id", "question", "source_subset", "source_domain")
    assert [tuple(r[k] for k in keys) for r in flat] == [
        tuple(r[k] for k in keys) for r in ref
    ]


def test_split_sizes(cfg):
    """SciCUEval splits 4000/7343 and MMLU samples 2450 rows, all eval."""
    if not Path(cfg["paths"]["scicueval_jsonl"]).is_file():
        pytest.skip("bakeoff JSONL not present")
    sci = load_corpus(cfg, "scicueval")
    assert (sci["split"] == "eval").sum() == 4000
    assert (sci["split"] == "train").sum() == 7343
    mm = load_corpus(cfg, "mmlu")
    assert len(mm) == 2450 and (mm["split"] == "eval").all()
