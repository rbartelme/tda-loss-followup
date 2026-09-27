"""All four figures render from a synthetic results directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tlf.plots import (
    REPRO_CORPORA,
    REPRO_MODELS,
    _fraction_label,
    fig_anchor_per_seed,
    fig_anchor_roster,
    make_all,
    roster_order,
)
from tlf.results import append_row, make_row

REF = Path(__file__).resolve().parents[1] / "configs" / "reference.yaml"
FRACS = (0.0, 0.1, 0.25, 0.5, 1.0)


def _metrics(rng, frac, disint=False):
    return {
        "within_cos": 0.6,
        "between_cos": 0.4,
        "gap": 0.05 + 0.2 * frac + rng.normal(0, 0.01),
        "lr_acc": 0.9,
        "ari": 0.8,
        "ari_sd": 0.02,
        "nmi": 0.9,
        "coverage": 0.02 if disint else 0.4,
        "nodes": 2.0 if disint else 150.0,
        "purity": 0.9,
        "anchor_rho": 0.3 - 0.2 * frac + rng.normal(0, 0.02),
        "disintegrated": disint,
        "router_acc_in": 0.7 + 0.2 * frac,
        "router_acc_mmlu": 0.6,
    }


def _synthetic_results(root: Path) -> None:
    rng = np.random.default_rng(0)
    for corpus in ("scicueval", "mmlu"):
        for frac in FRACS:
            for model in ("minilm", "biomedbert-fulltext"):
                for tau in (0.01, 0.05, 0.2, 1.0):
                    append_row(
                        make_row(
                            exp="exp1_tau_sweep",
                            model=model,
                            loss="mnrl",
                            tau=tau,
                            pair_set="random",
                            aux=None,
                            lam=0.0,
                            ckpt_frac=frac,
                            seed=0,
                            corpus=corpus,
                            metrics=_metrics(
                                rng, frac, disint=(corpus == "mmlu" and tau == 0.01)
                            ),
                            step=round(frac * 468),
                            total_steps=468,
                        ),
                        root,
                    )
            for loss in ("mnrl", "triplet", "cosent", "mlm"):
                append_row(
                    make_row(
                        exp="exp2_loss_family",
                        model="biomedbert-fulltext",
                        loss=loss,
                        tau=0.05,
                        pair_set="random",
                        aux=None,
                        lam=0.0,
                        ckpt_frac=frac,
                        seed=0,
                        corpus=corpus,
                        metrics=_metrics(rng, frac),
                    ),
                    root,
                )
            for ps in ("matched", "mismatched", "random"):
                append_row(
                    make_row(
                        exp="exp3_data_vs_loss",
                        model="biomedbert-fulltext",
                        loss="mnrl",
                        tau=0.05,
                        pair_set=ps,
                        aux=None,
                        lam=0.0,
                        ckpt_frac=frac,
                        seed=0,
                        corpus=corpus,
                        metrics=_metrics(
                            rng, frac, disint=(ps == "mismatched" and corpus == "mmlu")
                        ),
                    ),
                    root,
                )
            for aux in ("textstat_head", "dist_preserve", "persist0_h0"):
                for lam in (0.1, 1.0):
                    append_row(
                        make_row(
                            exp="exp4_topo_aux",
                            model="biomedbert-fulltext",
                            loss="mnrl",
                            tau=0.05,
                            pair_set="random",
                            aux=aux,
                            lam=lam,
                            ckpt_frac=frac,
                            seed=0,
                            corpus=corpus,
                            metrics=_metrics(rng, frac),
                        ),
                        root,
                    )


def test_make_all_renders_four_figures(tmp_path):
    """Every figure is written and non-empty from a full synthetic sweep."""
    _synthetic_results(tmp_path / "results")
    written = make_all(tmp_path / "results", REF, tmp_path / "figures")
    names = sorted(p.name for p in written)
    assert names == [
        "fig1_rho_vs_step.png",
        "fig2_gap_vs_rho.png",
        "fig3_exp3_pairs_vs_medcpt.png",
        "fig4_exp4_pareto.png",
    ]
    assert all(p.stat().st_size > 10_000 for p in written)


def test_make_all_with_no_results_writes_nothing(tmp_path):
    """An empty results directory produces no figures and no error."""
    (tmp_path / "results").mkdir()
    assert make_all(tmp_path / "results", REF, tmp_path / "figures") == []


def test_fraction_label_carries_the_step_count_only_when_the_panel_agrees():
    """One shared total_steps puts the count in the label; mixed or missing does not."""
    same = pd.DataFrame({"total_steps": [468.0, 468.0, float("nan")]})
    assert _fraction_label(same) == "fraction of training · 468 steps"
    mixed = pd.DataFrame({"total_steps": [468.0, 300.0]})
    assert _fraction_label(mixed) == "fraction of training"
    assert _fraction_label(pd.DataFrame({"ckpt_frac": [0.0]})) == "fraction of training"


def _machine(rng, per_seed=True):
    """Synthetic per-seed metrics for every repro model and corpus.

    Args:
        rng: Random generator.
        per_seed: Include the per-seed anchor lists (METRICS_VERSION 3).

    Returns:
        ``{(model, corpus): metrics}``.
    """
    out = {}
    for model in REPRO_MODELS:
        for corpus in REPRO_CORPORA:
            vals = rng.normal(0.2, 0.12, 25)
            m = {
                "anchor_rho": float(rng.normal(0.2, 0.15)),
                "anchor_rho_mean": float(vals.mean()),
                "anchor_rho_sd": float(vals.std(ddof=1)),
            }
            if per_seed:
                m["per_seed"] = {
                    "seed": list(range(42, 67)),
                    "anchor_rho": vals.tolist(),
                }
            out[(model, corpus)] = m
    return out


def test_fig_anchor_per_seed_renders_two_machines(tmp_path):
    """The reproducibility figure draws from per-seed metrics of two machines."""
    rng = np.random.default_rng(0)
    out = fig_anchor_per_seed(
        {"laptop": _machine(rng), "DGX Spark": _machine(rng)}, tmp_path / "f.png"
    )
    assert out.is_file() and out.stat().st_size > 10_000


def test_fig_anchor_per_seed_refuses_metrics_without_per_seed_values(tmp_path):
    """Metrics from before per-seed persistence cannot be drawn as rugs."""
    import pytest

    with pytest.raises(ValueError, match="no per-seed anchor rho"):
        fig_anchor_per_seed(
            {"laptop": _machine(np.random.default_rng(0), per_seed=False)},
            tmp_path / "f.png",
        )


def _roster(rng):
    """Synthetic re-analysis output: three keys, one disintegrated on MMLU.

    Args:
        rng: Random generator.

    Returns:
        ``{(key, corpus): metrics}`` shaped like ``load_reanalysis`` output.
    """
    out = {}
    for key, centre in (("low", 0.05), ("high", 0.4), ("mid", 0.2)):
        for corpus in REPRO_CORPORA:
            vals = rng.normal(centre, 0.1, 25)
            out[(key, corpus)] = {
                "per_seed": {"seed": list(range(42, 67)), "anchor_rho": vals.tolist()},
                "anchor_rho_mean": float(vals.mean()),
                "anchor_rho_sd": float(vals.std(ddof=1)),
                "n_anchor_seeds": 25.0,
                "disintegrated": False,
                "published": {"anchor_spearman_rho": float(centre)},
            }
    out[("mid", "mmlu")].update(disintegrated=True, n_anchor_seeds=0.0)
    return out


def test_roster_order_sorts_by_per_seed_mean():
    """Highest SciCUEval per-seed mean first; the order is shared by both panels."""
    assert roster_order(_roster(np.random.default_rng(0))) == ["high", "mid", "low"]


def test_roster_order_puts_unscorable_rows_last():
    """A row disintegrated on the ordering corpus sorts to the bottom."""
    assert roster_order(_roster(np.random.default_rng(0)), order_by="mmlu")[-1] == "mid"


def test_fig_anchor_roster_renders_with_a_disintegrated_row(tmp_path):
    """The all-encoder figure draws, writing a word where a graph disintegrated."""
    out = fig_anchor_roster(_roster(np.random.default_rng(0)), tmp_path / "r.png")
    assert out.is_file() and out.stat().st_size > 10_000
