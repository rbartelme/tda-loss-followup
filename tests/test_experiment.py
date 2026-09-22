"""Grid expansion and plan bookkeeping for the experiment driver."""

from __future__ import annotations

from pathlib import Path

import pytest

from tlf.config import load_config
from tlf.experiment import Run, checkpoint_fracs, expand_grid, plan

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    ("name", "n_runs"),
    [
        ("exp1_tau_sweep", 8),
        ("exp2_loss_family", 4),
        ("exp3_data_vs_loss", 3),
        ("exp4_topo_aux", 6),
    ],
)
def test_grid_sizes_match_the_protocol(name, n_runs):
    """Exp 1: 2 models x 4 tau; Exp 2: 4 losses; Exp 3: 3 pair sets; Exp 4: 3 aux x 2 lambda."""
    cfg = load_config(CONFIGS / f"{name}.yaml")
    runs = expand_grid(cfg)
    assert len(runs) == n_runs
    assert len({r.id for r in runs}) == n_runs


def test_run_id_and_key_shape():
    """Run ids are directory-safe and keys carry NaN lambda when there is no aux."""
    r = Run("exp1_tau_sweep", "minilm", "mnrl", 0.05, "random", None, 0.0, 0)
    assert r.id == "minilm_mnrl_tau0.05_random_s0"
    k = r.key(0.5, "mmlu")
    assert k["aux"] == "" and k["lambda"] != k["lambda"]  # NaN
    r2 = Run(
        "exp4_topo_aux",
        "biomedbert-fulltext",
        "mnrl",
        0.05,
        "random",
        "persist0_h0",
        1.0,
        0,
    )
    assert r2.id == "biomedbert-fulltext_mnrl_tau0.05_random_persist0_h0_lam1_s0"


def test_checkpoint_fracs_only_final():
    """--only-final keeps 0.0 and 1.0."""
    cfg = load_config(CONFIGS / "exp1_tau_sweep.yaml")
    assert checkpoint_fracs(cfg, True) == [0.0, 1.0]
    assert checkpoint_fracs(cfg, False) == [0.0, 0.1, 0.25, 0.5, 1.0]


def test_plan_reports_everything_todo_on_a_fresh_tree(tmp_path):
    """With no checkpoints or results, every (frac, corpus) is to do and nothing is trained."""
    cfg = load_config(CONFIGS / "exp3_data_vs_loss.yaml")
    cfg["paths"]["results"] = str(tmp_path / "results")
    cfg["paths"]["checkpoints"] = str(tmp_path / "ckpt")
    cfg["paths"]["pairs"] = str(tmp_path / "pairs")
    items = plan(cfg, only_final=True)
    assert len(items) == 3
    assert all(
        not it["trained"] and not it["pairs_exist"] and len(it["todo"]) == 4
        for it in items
    )


def test_base_aux_schedule_does_not_leak_into_the_grid():
    """Exps 1-3 inherit base.yaml's schedule section but have no auxiliary in their grid."""
    cfg = load_config(CONFIGS / "exp1_tau_sweep.yaml")
    runs = expand_grid(cfg)
    assert all(r.aux is None and r.lam == 0.0 for r in runs)
    assert runs[0].id == "minilm_mnrl_tau0.01_random_s0"
