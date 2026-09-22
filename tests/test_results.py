"""Result CSV: exact columns, idempotent append, typed load."""

from __future__ import annotations

import pandas as pd

from tlf.results import (
    CSV_COLUMNS,
    KEY_COLUMNS,
    append_row,
    load_results,
    make_row,
    results_path,
    row_exists,
    row_key,
)

METRICS = {
    "within_cos": 0.5,
    "between_cos": 0.3,
    "gap": 0.2,
    "lr_acc": 0.9,
    "lr_eval": "holdout",
    "ari": 0.8,
    "ari_sd": 0.01,
    "nmi": 0.95,
    "coverage": 0.4,
    "nodes": 170.0,
    "purity": 0.9,
    "anchor_rho": 0.3,
    "disintegrated": False,
    "router_acc_in": 0.9,
    "router_acc_mmlu": 0.7,
}


def _row(**over):
    kw = dict(
        exp="exp1_tau_sweep",
        model="minilm",
        loss="mnrl",
        tau=0.05,
        pair_set="random",
        aux=None,
        lam=0.0,
        ckpt_frac=1.0,
        seed=0,
        corpus="scicueval",
        metrics=METRICS,
    )
    kw.update(over)
    return make_row(**kw)


def test_csv_columns_are_exactly_the_brief_schema():
    """Column order is the one the blog post is written from."""
    assert CSV_COLUMNS == (
        "exp",
        "model",
        "loss",
        "tau",
        "pair_set",
        "aux",
        "lambda",
        "ckpt_frac",
        "seed",
        "corpus",
        "within_cos",
        "between_cos",
        "gap",
        "lr_acc",
        "lr_eval",
        "ari",
        "ari_sd",
        "nmi",
        "coverage",
        "nodes",
        "purity",
        "anchor_rho",
        "disintegrated",
        "router_acc_in",
        "router_acc_mmlu",
        "git_sha",
        "timestamp",
    )
    assert KEY_COLUMNS == CSV_COLUMNS[:10]
    assert list(_row().keys()) == list(CSV_COLUMNS)


def test_append_twice_with_same_key_yields_one_row(tmp_path):
    """The second append replaces the first in place."""
    append_row(_row(metrics={**METRICS, "gap": 0.1}), tmp_path)
    append_row(_row(metrics={**METRICS, "gap": 0.2}), tmp_path)
    df = pd.read_csv(results_path(tmp_path, "exp1_tau_sweep"))
    assert len(df) == 1 and list(df.columns) == list(CSV_COLUMNS)
    assert float(df["gap"].iloc[0]) == 0.2


def test_different_keys_accumulate_and_order_is_preserved(tmp_path):
    """Rows with different keys are all kept, in first-seen order."""
    append_row(_row(ckpt_frac=0.0), tmp_path)
    append_row(_row(ckpt_frac=1.0), tmp_path)
    append_row(_row(corpus="mmlu"), tmp_path)
    append_row(
        _row(ckpt_frac=0.0, metrics={**METRICS, "gap": 0.42}), tmp_path
    )  # replaces first
    df = load_results(tmp_path)
    assert len(df) == 3
    assert df["ckpt_frac"].tolist() == [0.0, 1.0, 1.0]
    assert df["gap"].iloc[0] == 0.42


def test_key_matching_is_robust_to_float_and_empty_formatting():
    """Keys compare canonically: '0.05' == 0.05, '' == NaN == None, aux None == ''."""
    a = _row()
    b = {**a, "tau": "0.05", "lambda": "", "aux": "", "seed": "0", "ckpt_frac": "1"}
    assert row_key(a) == row_key(b)
    assert row_key(_row(tau=0.2)) != row_key(a)


def test_row_exists_and_load_results_types(tmp_path):
    """row_exists sees appended rows; load_results returns typed columns."""
    assert not row_exists(_row(), tmp_path)
    append_row(
        _row(
            aux="persist0_h0",
            lam=0.1,
            exp="exp4_topo_aux",
            metrics={**METRICS, "disintegrated": True},
        ),
        tmp_path,
    )
    key = {
        c: v
        for c, v in _row(aux="persist0_h0", lam=0.1, exp="exp4_topo_aux").items()
        if c in KEY_COLUMNS
    }
    assert row_exists(key, tmp_path)
    df = load_results(tmp_path)
    assert (
        df["disintegrated"].dtype == bool and bool(df["disintegrated"].iloc[0]) is True
    )
    assert df["lambda"].iloc[0] == 0.1 and df["seed"].dtype.kind == "i"
    assert load_results(tmp_path / "nope").empty


def test_lr_eval_column_is_a_string_and_tolerates_absence(tmp_path):
    """lr_eval records the LR recipe; rows without it (older CSVs) load as empty."""
    append_row(_row(metrics={**METRICS, "lr_eval": "cv"}), tmp_path)
    without = {k: v for k, v in METRICS.items() if k != "lr_eval"}
    append_row(_row(corpus="mmlu", metrics=without), tmp_path)
    df = load_results(tmp_path)
    assert list(df["lr_eval"]) == ["cv", ""]
