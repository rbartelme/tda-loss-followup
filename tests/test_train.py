"""Training driver: pure helpers, plus the shared untrained checkpoint on cached MiniLM."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub.constants import HF_HUB_CACHE

from tlf.config import load_config
from tlf.evaluate import checkpoint_stamp, run_name_for
from tlf.experiment import run_is_trained
from tlf.train import (
    aux_schedule,
    checkpoint_steps,
    cosine_stats,
    frac_tag,
    no_duplicate_batches,
    run_finetune,
    untrained_dir,
)

MINILM_CACHED = (
    Path(HF_HUB_CACHE) / "models--sentence-transformers--all-MiniLM-L6-v2"
).is_dir()


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


def test_aux_schedule_survives_the_exp4_grid_merge():
    """In the merged Exp 4 config `aux` is the grid list; the schedule still resolves."""
    cfg = load_config("configs/exp4_topo_aux.yaml")
    assert isinstance(cfg["aux"], list)
    sched = aux_schedule(cfg)
    assert sched["subsample"] == 64 and sched["every"] == 1
    assert aux_schedule({"aux": {"subsample": 8}})["subsample"] == 8  # legacy dict form
    assert aux_schedule({"aux": ["persist0_h0"]}) == {}


@pytest.mark.skipif(not MINILM_CACHED, reason="MiniLM not in the HF cache")
def test_ckpt_0_is_one_shared_symlink_per_base_model(tmp_path):
    """Runs of one base model share one untrained checkpoint via relative symlinks."""
    cfg = load_config("configs/base.yaml")
    if not Path(cfg["paths"]["scicueval_jsonl"]).is_file():
        pytest.skip("corpora not present")
    cfg["paths"]["checkpoints"] = str(tmp_path / "ckpt")
    cfg["train"] = {
        **cfg["train"],
        "batch_size": 16,
        "micro_batch": 8,
        "epochs": 1,
        "checkpoint_fracs": [0.0, 1.0],
        "device": "cpu",
    }
    kw = dict(max_steps=1, n_pairs_limit=32)
    a, b = tmp_path / "ckpt" / "run_a", tmp_path / "ckpt" / "run_b"
    run_finetune("minilm", "mnrl", 0.05, None, cfg, 0, a, **kw)
    shared = untrained_dir("minilm", cfg)
    stamp = checkpoint_stamp(shared)
    run_finetune("minilm", "mnrl", 0.05, None, cfg, 1, b, **kw)
    run_finetune("minilm", "mnrl", 0.05, None, cfg, 0, a, **kw)  # over the old link
    for root in (a, b):
        link = root / "ckpt_0.0"
        assert link.is_symlink() and not os.path.isabs(os.readlink(link))
        assert link.resolve() == shared.resolve()
        assert run_name_for(link, cfg) == "_untrained/minilm"
        assert checkpoint_stamp(link) == stamp  # written once, never rewritten
        assert run_is_trained(root, [0.0, 1.0])
        assert not (root / "ckpt_1.0").is_symlink()
