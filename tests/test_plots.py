"""All four figures render from a synthetic results directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tlf.plots import make_all
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
