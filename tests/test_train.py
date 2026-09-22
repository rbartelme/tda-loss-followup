"""Pure helpers of the training driver (no model downloads)."""

from __future__ import annotations

import numpy as np
import torch

from tlf.train import checkpoint_steps, cosine_stats, frac_tag, no_duplicate_batches


def test_frac_tag_matches_brief_directory_names():
    """Fractions render as ckpt_0.0 ... ckpt_1.0 style tags."""
    assert [frac_tag(f) for f in (0.0, 0.1, 0.25, 0.5, 1.0)] == [
        "0.0",
        "0.1",
        "0.25",
        "0.5",
        "1.0",
    ]


def test_checkpoint_steps_rounds_to_optimizer_steps():
    """Fractions map to rounded step indices with 0 before any update."""
    assert checkpoint_steps([0.0, 0.1, 0.25, 0.5, 1.0], 470) == {
        0.0: 0,
        0.1: 47,
        0.25: 118,
        0.5: 235,
        1.0: 470,
    }


def test_no_duplicate_batches_never_repeats_an_id_in_a_batch():
    """Every anchor and positive id is unique within each batch; trailing partial batch dropped."""
    rng = np.random.default_rng(0)
    anchors = [f"a{i % 20}" for i in range(100)]  # each anchor appears 5 times
    positives = [f"p{i}" for i in range(100)]
    batches = no_duplicate_batches(anchors, positives, 16, rng)
    assert batches and all(len(b) == 16 for b in batches)
    for b in batches:
        ids = [anchors[i] for i in b] + [positives[i] for i in b]
        assert len(ids) == len(set(ids))
    flat = [i for b in batches for i in b]
    assert len(flat) == len(set(flat))


def test_no_duplicate_batches_is_seed_deterministic():
    """Same seed, same batches."""
    a = [f"a{i}" for i in range(50)]
    p = [f"p{i}" for i in range(50)]
    b1 = no_duplicate_batches(a, p, 8, np.random.default_rng(3))
    b2 = no_duplicate_batches(a, p, 8, np.random.default_rng(3))
    assert b1 == b2


def test_cosine_stats_on_identical_and_orthogonal_rows():
    """Identical pairs give pos=1; orthogonal rows give neg=0."""
    e = torch.eye(4)
    pos, neg = cosine_stats(e, e)
    assert pos == 1.0 and abs(neg) < 1e-7
