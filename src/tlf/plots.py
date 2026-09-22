"""The four figures of the follow-up post, drawn from ``results/*.csv``.

1. ``fig_rho_vs_step``: anchor rho against training fraction, one line per tau,
   small multiples per model x corpus on a shared scale (Exp 1).
2. ``fig_gap_vs_rho``: cosine gap against anchor rho for every row, panels per
   corpus, shape by experiment, rug marginals (all experiments).
3. ``fig_exp3_bars``: final-checkpoint metrics per pair set against the MedCPT
   reference row and the untrained baseline (Exp 3).
4. ``fig_exp4_pareto``: gap against anchor rho for each (aux, lambda) with the
   in-domain router accuracy as marker size (Exp 4).

Design follows Tufte: data ink is the darkest thing on the page, axes are
range frames spanning only the data, there are no gridlines or boxes, series
are labelled directly where they end (labels are spread apart in display
space rather than allowed to collide), small multiples share scales so panels
compare honestly, and marginal distributions are shown as rugs rather than
extra plots. Colour follows the entity, never its rank: tau uses an ordinal
one-hue ramp, corpora and auxiliaries take fixed categorical slots, reference
marks are muted ink. Disintegrated Mapper graphs are drawn hollow, and a
disintegrated reference row is reported as a note rather than a line at zero.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from tlf.results import load_results  # noqa: E402

log = logging.getLogger(__name__)

# Validated palette (dataviz reference instance, light surface).
CATEGORICAL: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
ORDINAL_BLUE: tuple[str, ...] = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "baseline": "#c3c2b7",
    "surface": "#fcfcfb",
}
CORPUS_COLOR = {"scicueval": CATEGORICAL[0], "mmlu": CATEGORICAL[1]}
CORPUS_LABEL = {"scicueval": "SciCUEval", "mmlu": "MMLU non-STEM"}
EXP_MARKER = {
    "exp1_tau_sweep": "o",
    "exp2_loss_family": "s",
    "exp3_data_vs_loss": "^",
    "exp4_topo_aux": "D",
}
EXP_LABEL = {
    "exp1_tau_sweep": "Exp 1 τ sweep",
    "exp2_loss_family": "Exp 2 loss family",
    "exp3_data_vs_loss": "Exp 3 pair sets",
    "exp4_topo_aux": "Exp 4 aux",
}
PAIR_SET_ORDER = ("matched", "mismatched", "random")
EXP3_METRICS = ("gap", "anchor_rho", "ari", "purity", "router_acc_in")
METRIC_LABEL = {
    "gap": "cosine gap",
    "anchor_rho": "anchor ρ",
    "ari": "Mapper ARI",
    "purity": "weighted purity",
    "router_acc_in": "router accuracy, in-domain",
    "lr_acc": "LR accuracy",
}
REFERENCE_KEY = "medcpt-query"
MAPPER_METRICS: tuple[str, ...] = (
    "anchor_rho",
    "ari",
    "ari_sd",
    "nmi",
    "coverage",
    "nodes",
    "purity",
)
FOOT_DISINT = (
    "hollow mark = Mapper graph disintegrated (coverage < 0.05 or fewer than 5 nodes)"
)


def load_reference(path: str | Path) -> dict[str, Any]:
    """Read ``configs/reference.yaml``.

    Args:
        path: The YAML file.

    Returns:
        Model key to ``{"scicueval": {...}, "mmlu": {...}, ...}``.
    """
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Tufte chrome
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes) -> None:
    """Minimal non-data ink: two thin spines, no grid, muted ticks."""
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["baseline"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK["muted"], labelsize=7.5, length=3, width=0.6)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK["secondary"])
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.title.set_color(INK["primary"])
    ax.xaxis.label.set_color(INK["secondary"])
    ax.yaxis.label.set_color(INK["secondary"])


def _finite(values: Iterable[Any]) -> np.ndarray:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return a[np.isfinite(a)]


def _with_endpoints(
    ticks: list[float], lo: float, hi: float, frac: float = 0.12
) -> list[float]:
    """Add the data endpoints as ticks and drop auto ticks that would crowd them."""
    span = hi - lo
    keep = [t for t in ticks if min(abs(t - lo), abs(t - hi)) > frac * span]
    return keep + [lo, hi]


def _range_frame(
    ax: plt.Axes,
    x: Iterable[Any] | None = None,
    y: Iterable[Any] | None = None,
    xticks: Iterable[float] | None = None,
) -> None:
    """Trim each spine to the span of the data it measures (Tufte's range frame).

    Args:
        ax: Axes to trim.
        x: Data along x; ``None`` leaves the bottom spine as is.
        y: Data along y; ``None`` leaves the left spine as is.
        xticks: Explicit x ticks (else the auto ticks inside the range plus
            the endpoints, with crowding ticks removed).
    """
    if x is not None:
        xs = _finite(x)
        if xs.size:
            lo, hi = float(xs.min()), float(xs.max())
            ax.spines["bottom"].set_bounds(lo, hi)
            if xticks is not None:
                ax.set_xticks(sorted(set(xticks)))
            else:
                auto = [t for t in ax.get_xticks() if lo - 1e-9 <= t <= hi + 1e-9]
                ax.set_xticks(sorted(set(_with_endpoints(auto, lo, hi))))
                ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))
    if y is not None:
        ys = _finite(y)
        if ys.size:
            lo, hi = float(ys.min()), float(ys.max())
            if hi > lo:
                ax.spines["left"].set_bounds(lo, hi)
                auto = [t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9]
                ax.set_yticks(sorted(set(_with_endpoints(auto, lo, hi))))
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Nudge 1-D positions apart so neighbours are at least ``min_gap`` apart.

    Order is preserved and the group stays centred on its original mean.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for k in range(1, len(pos)):
        if pos[k] - pos[k - 1] < min_gap:
            pos[k] = pos[k - 1] + min_gap
    shift = float(np.mean([values[i] for i in order]) - np.mean(pos))
    pos = [v + shift for v in pos]
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def _label_points(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    texts: list[str],
    colors: list[str] | None = None,
    dx_pt: float = 7.0,
    gap_pt: float = 9.0,
) -> None:
    """Direct-label points, spreading labels vertically so they never collide.

    Spacing is computed in pixels after ``fig.canvas.draw()`` so transforms are
    final, the group is kept inside the axes, and the positions are then
    inverted back to data coordinates so the labels stay attached to their
    points. Labels that had to move get a hairline connector.
    """
    if not xs:
        return
    fig = ax.figure
    scale = fig.dpi / 72.0
    disp = ax.transData.transform(np.column_stack([xs, ys]))
    px, py = disp[:, 0], disp[:, 1]
    ly = _spread(list(py), gap_pt * scale)
    bb = ax.get_window_extent()
    pad = 4.0 * scale
    if min(ly) < bb.y0 + pad:
        ly = [v + (bb.y0 + pad - min(ly)) for v in ly]
    elif max(ly) > bb.y1 - pad:
        ly = [v - (max(ly) - (bb.y1 - pad)) for v in ly]
    inv = ax.transData.inverted()
    right_third = bb.x0 + 0.68 * (bb.x1 - bb.x0)
    for i, text in enumerate(texts):
        color = colors[i] if colors else INK["secondary"]
        moved = abs(ly[i] - py[i]) > 0.5
        on_left = px[i] > right_third
        tx, ty = inv.transform((px[i] + (-dx_pt if on_left else dx_pt) * scale, ly[i]))
        ax.annotate(
            text,
            xy=(xs[i], ys[i]),
            xycoords="data",
            xytext=(tx, ty),
            textcoords="data",
            fontsize=6.5,
            color=color,
            va="center",
            ha="right" if on_left else "left",
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "color": INK["baseline"],
                    "linewidth": 0.5,
                    "shrinkA": 0,
                    "shrinkB": 3,
                }
                if moved
                else None
            ),
            annotation_clip=False,
        )


def _rug(
    ax: plt.Axes, x: Iterable[Any], y: Iterable[Any], color: str, alpha: float = 0.5
) -> None:
    """Marginal distributions as short ticks on the two axes (Tufte's dot-dash plot)."""
    xs, ys = _finite(x), _finite(y)
    if xs.size:
        ax.plot(
            xs,
            np.zeros_like(xs),
            "|",
            transform=ax.get_xaxis_transform(),
            color=color,
            alpha=alpha,
            markersize=5,
            markeredgewidth=0.8,
            clip_on=False,
            zorder=2,
        )
    if ys.size:
        ax.plot(
            np.zeros_like(ys),
            ys,
            "_",
            transform=ax.get_yaxis_transform(),
            color=color,
            alpha=alpha,
            markersize=5,
            markeredgewidth=0.8,
            clip_on=False,
            zorder=2,
        )


def _fig(nrows: int, ncols: int, w: float = 3.6, h: float = 3.0, **kw):
    return plt.subplots(
        nrows,
        ncols,
        figsize=(w * ncols, h * nrows),
        dpi=150,
        squeeze=False,
        facecolor=INK["surface"],
        **kw,
    )


def _ref_row(
    reference: dict[str, Any], corpus: str, key: str = REFERENCE_KEY
) -> dict[str, Any] | None:
    entry = reference.get(key)
    row = entry.get(corpus) if isinstance(entry, dict) else None
    return row if isinstance(row, dict) else None


def _ref_value(
    reference: dict[str, Any], corpus: str, metric: str, key: str = REFERENCE_KEY
):
    """A reference metric, or ``None`` when absent or meaningless (disintegrated Mapper)."""
    row = _ref_row(reference, corpus, key)
    if row is None or row.get(metric) is None:
        return None
    if metric in MAPPER_METRICS and bool(row.get("disintegrated", False)):
        return None
    try:
        return float(row[metric])
    except (TypeError, ValueError):
        return None


def _ref_disintegrated(reference: dict[str, Any], corpus: str, metric: str) -> bool:
    row = _ref_row(reference, corpus)
    return (
        row is not None
        and metric in MAPPER_METRICS
        and bool(row.get("disintegrated", False))
    )


def _ref_line(
    ax: plt.Axes,
    y: float | None,
    label: str,
    style: str = "--",
    *,
    outside: bool = False,
    below: bool = False,
) -> None:
    """A muted horizontal reference with its value in the label."""
    if y is None or not np.isfinite(y):
        return
    ax.axhline(y, color=INK["muted"], linestyle=style, linewidth=0.9, zorder=1)
    if outside:
        xytext, ha, va = (3, -1 if below else 1), "left", ("top" if below else "bottom")
    else:
        xytext, ha, va = (
            (-2, -2 if below else 2),
            "right",
            ("top" if below else "bottom"),
        )
    ax.annotate(
        f"{label} {y:.3f}",
        xy=(1.0, y),
        xycoords=("axes fraction", "data"),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=6.5,
        color=INK["muted"],
        annotation_clip=False,
    )


def _ref_note(
    ax: plt.Axes,
    reference: dict[str, Any],
    corpus: str,
    metric: str,
    label: str = "MedCPT",
    *,
    outside: bool = False,
) -> None:
    """Say so when the reference graph disintegrated for a Mapper metric.

    Top-right inside the panel by default (where the reference label would
    have been); in the right margin at the baseline for bar panels.
    """
    if _ref_disintegrated(reference, corpus, metric):
        if outside:
            xy, xytext, ha, va = (1.0, 0.0), (3, 0), "left", "bottom"
        else:
            xy, xytext, ha, va = (1.0, 0.98), (-2, 0), "right", "top"
        ax.annotate(
            f"{label}: Mapper disintegrated",
            xy=xy,
            xycoords="axes fraction",
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6.5,
            color=INK["muted"],
            annotation_clip=False,
        )


def _footnote(fig: plt.Figure, text: str, y: float = -0.02) -> None:
    fig.text(0.01, y, text, fontsize=7, color=INK["muted"], ha="left", va="top")


def _title(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, fontsize=10.5, color=INK["primary"], x=0.01, ha="left", y=1.0)


def _save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=fig.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def _final(df: pd.DataFrame) -> pd.DataFrame:
    return df[np.isclose(df["ckpt_frac"], 1.0)]


# ---------------------------------------------------------------------------
# Figure 1: rho vs step per tau per model
# ---------------------------------------------------------------------------


def _fraction_label(sub: pd.DataFrame) -> str:
    """X label for a fraction axis, carrying the step count when the panel agrees on one.

    Args:
        sub: The rows drawn on the panel.

    Returns:
        ``"fraction of training · N steps"`` when every finite ``total_steps``
        is the same ``N``, else ``"fraction of training"``.
    """
    if "total_steps" in sub.columns:
        totals = {int(t) for t in _finite(sub["total_steps"])}
        if len(totals) == 1:
            return f"fraction of training · {totals.pop()} steps"
    return "fraction of training"


def fig_rho_vs_step(
    df: pd.DataFrame, reference: dict[str, Any], out: Path
) -> Path | None:
    """Anchor rho against training fraction, one ordinal-coloured line per tau.

    Small multiples (rows = corpus, columns = model) share the rho scale. Lines
    are labelled at their right end; the MedCPT rho is a muted dashed line.

    Args:
        df: Exp 1 rows (any loss; typically ``mnrl``).
        reference: ``load_reference`` output.
        out: PNG path.

    Returns:
        The path written, or ``None`` if ``df`` is empty.
    """
    if df.empty:
        log.warning("fig_rho_vs_step: no rows")
        return None
    corpora = [c for c in ("scicueval", "mmlu") if c in set(df["corpus"])]
    models = sorted(set(df["model"]))
    taus = sorted(set(df["tau"].dropna()))
    ramp = (
        ORDINAL_BLUE
        if len(taus) <= 4
        else tuple(plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(taus))))
    )
    fracs = sorted(set(df["ckpt_frac"].dropna()))
    fig, axes = _fig(len(corpora), len(models), w=4.0, h=2.9, sharey=True)
    ref_ys = {c: _ref_value(reference, c, "anchor_rho") for c in corpora}
    y_span = np.concatenate([_finite(df["anchor_rho"]), _finite(ref_ys.values())])
    panels: list[tuple[plt.Axes, list[tuple[float, float, str, str]]]] = []
    for i, corpus in enumerate(corpora):
        for j, model in enumerate(models):
            ax = axes[i][j]
            _style(ax)
            sub = df[(df["corpus"] == corpus) & (df["model"] == model)]
            ends: list[tuple[float, float, str, str]] = []
            for k, tau in enumerate(taus):
                s = sub[np.isclose(sub["tau"], tau)]
                if s.empty:
                    continue
                g = s.groupby("ckpt_frac")
                mean = g["anchor_rho"].mean()
                dis = g["disintegrated"].any()
                x = mean.index.to_numpy()
                y = mean.to_numpy()
                ax.plot(x, y, color=ramp[k], linewidth=1.6, zorder=3)
                if s["seed"].nunique() > 1:
                    ax.fill_between(
                        x,
                        g["anchor_rho"].min(),
                        g["anchor_rho"].max(),
                        color=ramp[k],
                        alpha=0.12,
                        linewidth=0,
                    )
                filled = ~dis.to_numpy()
                ax.scatter(
                    x[filled],
                    y[filled],
                    s=22,
                    color=ramp[k],
                    zorder=4,
                    edgecolor=INK["surface"],
                    linewidth=1.0,
                )
                ax.scatter(
                    x[~filled],
                    y[~filled],
                    s=22,
                    facecolor=INK["surface"],
                    edgecolor=ramp[k],
                    linewidth=1.2,
                    zorder=4,
                )
                ends.append((float(x[-1]), float(y[-1]), f"τ = {tau:g}", ramp[k]))
            _ref_line(ax, ref_ys[corpus], "MedCPT")
            _ref_note(ax, reference, corpus, "anchor_rho")
            ax.set_title(
                f"{model} · {CORPUS_LABEL.get(corpus, corpus)}",
                fontsize=8.5,
                loc="left",
            )
            ax.set_xlim(-0.02, 1.22)
            _range_frame(ax, x=fracs, y=y_span, xticks=fracs)
            ax.set_xticklabels([f"{f:g}" for f in fracs])
            if i == len(corpora) - 1:
                ax.set_xlabel(_fraction_label(sub), fontsize=8)
            if j == 0:
                ax.set_ylabel("anchor ρ", fontsize=8)
            panels.append((ax, ends))
    _title(fig, "Exp 1 · does temperature alone move the topology?")
    _footnote(fig, FOOT_DISINT + " · dashed = MedCPT, first post · lines labelled by τ")
    fig.tight_layout()
    fig.canvas.draw()
    for ax, ends in panels:
        _label_points(
            ax,
            [e[0] for e in ends],
            [e[1] for e in ends],
            [e[2] for e in ends],
            colors=[e[3] for e in ends],
            dx_pt=8.0,
            gap_pt=9.0,
        )
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Figure 2: gap vs rho scatter
# ---------------------------------------------------------------------------


def fig_gap_vs_rho(
    df: pd.DataFrame, reference: dict[str, Any], out: Path
) -> Path | None:
    """Cosine gap against anchor rho for every result row, panels per corpus.

    Shape encodes the experiment, hue the corpus, opacity the training fraction
    (final checkpoints opaque), hollow marks disintegrated graphs. Rugs on both
    axes show the marginals. First-post reference encoders are labelled crosses.

    Args:
        df: All result rows.
        reference: ``load_reference`` output.
        out: PNG path.

    Returns:
        The path written, or ``None`` if ``df`` is empty.
    """
    if df.empty:
        log.warning("fig_gap_vs_rho: no rows")
        return None
    corpora = [c for c in ("scicueval", "mmlu") if c in set(df["corpus"])]
    exps = list(dict.fromkeys(df["exp"]))
    markers = {e: EXP_MARKER.get(e, "o") for e in exps}
    fig, axes = _fig(1, len(corpora), w=4.4, h=3.8, sharey=True)
    panels: list[tuple[plt.Axes, list[tuple[float, float, str]]]] = []
    y_all: list[float] = list(_finite(df["anchor_rho"]))
    for entry in reference.values():
        for corpus in corpora:
            r = entry.get(corpus) if isinstance(entry, dict) else None
            if r and r.get("anchor_rho") is not None:
                y_all.append(float(r["anchor_rho"]))
    for j, corpus in enumerate(corpora):
        ax = axes[0][j]
        _style(ax)
        color = CORPUS_COLOR.get(corpus, CATEGORICAL[2])
        sub = df[df["corpus"] == corpus]
        for e in exps:
            s = sub[sub["exp"] == e]
            for final in (False, True):
                ss = s[np.isclose(s["ckpt_frac"], 1.0) == final]
                if ss.empty:
                    continue
                alpha = 0.9 if final else 0.3
                size = 34 if final else 16
                ok = ~ss["disintegrated"].to_numpy()
                ax.scatter(
                    ss["gap"][ok],
                    ss["anchor_rho"][ok],
                    marker=markers[e],
                    s=size,
                    color=color,
                    alpha=alpha,
                    edgecolor=INK["surface"],
                    linewidth=0.6,
                    zorder=3,
                )
                ax.scatter(
                    ss["gap"][~ok],
                    ss["anchor_rho"][~ok],
                    marker=markers[e],
                    s=size,
                    facecolor=INK["surface"],
                    edgecolor=color,
                    alpha=alpha,
                    linewidth=1.0,
                    zorder=3,
                )
        refs: list[tuple[float, float, str]] = []
        for key, entry in reference.items():
            r = entry.get(corpus) if isinstance(entry, dict) else None
            if not r or r.get("gap") is None or r.get("anchor_rho") is None:
                continue
            label = key + (" (disintegrated)" if r.get("disintegrated") else "")
            refs.append((float(r["gap"]), float(r["anchor_rho"]), label))
            ax.scatter(
                [r["gap"]],
                [r["anchor_rho"]],
                marker="x",
                s=42,
                color=INK["secondary"],
                linewidth=1.2,
                zorder=5,
            )
        _rug(ax, sub["gap"], sub["anchor_rho"], color)
        _range_frame(ax, x=list(sub["gap"]) + [r[0] for r in refs], y=y_all)
        ax.set_title(CORPUS_LABEL.get(corpus, corpus), fontsize=9, loc="left")
        ax.set_xlabel("cosine gap (within − between)", fontsize=8)
        if j == 0:
            ax.set_ylabel("anchor ρ", fontsize=8)
        panels.append((ax, refs))
    key_handles = [
        Line2D(
            [],
            [],
            marker=markers[e],
            linestyle="",
            color=INK["secondary"],
            markersize=5,
            label=EXP_LABEL.get(e, e),
        )
        for e in exps
    ]
    fig.legend(
        handles=key_handles,
        loc="lower left",
        ncol=len(key_handles),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.0, -0.06),
        handletextpad=0.3,
        columnspacing=1.2,
    )
    _title(fig, "Cosine separation is not anchor faithfulness")
    _footnote(
        fig,
        "faint = intermediate checkpoints, solid = final · × = first-post reference encoders · "
        + FOOT_DISINT
        + " · rugs show marginals",
        y=-0.10,
    )
    fig.tight_layout()
    fig.canvas.draw()
    for ax, refs in panels:
        _label_points(
            ax,
            [r[0] for r in refs],
            [r[1] for r in refs],
            [r[2] for r in refs],
            dx_pt=6.0,
        )
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Figure 3: Exp 3 bars against the MedCPT reference
# ---------------------------------------------------------------------------


def fig_exp3_bars(
    df: pd.DataFrame, reference: dict[str, Any], out: Path
) -> Path | None:
    """Final-checkpoint metrics per pair set, with MedCPT and the untrained baseline as lines.

    Thin bars, values printed on the bars instead of a y axis, one shared scale
    per metric column so the two corpora compare directly. Reference labels sit
    in the margin to the right of each panel.

    Args:
        df: Exp 3 rows.
        reference: ``load_reference`` output.
        out: PNG path.

    Returns:
        The path written, or ``None`` if there are no final-checkpoint rows.
    """
    fin = _final(df)
    if fin.empty:
        log.warning("fig_exp3_bars: no final-checkpoint rows")
        return None
    corpora = [c for c in ("scicueval", "mmlu") if c in set(fin["corpus"])]
    metrics = [m for m in EXP3_METRICS if m in fin.columns and fin[m].notna().any()]
    pair_sets = [p for p in PAIR_SET_ORDER if p in set(fin["pair_set"])]
    base = df[np.isclose(df["ckpt_frac"], 0.0)]
    fig, axes = _fig(len(corpora), len(metrics), w=2.9, h=2.6, sharey="col")
    x = np.arange(len(pair_sets))
    col_span: dict[int, list[float]] = {}
    for i, corpus in enumerate(corpora):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            _style(ax)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)
            sub = fin[fin["corpus"] == corpus]
            g = sub.groupby("pair_set")
            means = g[metric].mean().reindex(pair_sets)
            sds = g[metric].std().reindex(pair_sets).fillna(0.0)
            dis = g["disintegrated"].any().reindex(pair_sets).fillna(False)
            for k, ps in enumerate(pair_sets):
                v = means[ps]
                if pd.isna(v):
                    continue
                if dis[ps]:
                    ax.bar(
                        x[k],
                        v,
                        width=0.5,
                        facecolor=INK["surface"],
                        edgecolor=CATEGORICAL[0],
                        linewidth=1.2,
                        zorder=3,
                    )
                else:
                    ax.bar(
                        x[k], v, width=0.5, color=CATEGORICAL[0], linewidth=0, zorder=3
                    )
                if sub["seed"].nunique() > 1 and sds[ps] > 0:
                    ax.errorbar(
                        x[k],
                        v,
                        yerr=sds[ps],
                        color=INK["secondary"],
                        linewidth=0.7,
                        capsize=2,
                        zorder=4,
                    )
                ax.annotate(
                    f"{v:.3f}",
                    xy=(x[k], v),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=INK["secondary"],
                    zorder=5,
                    bbox={
                        "boxstyle": "square,pad=0.12",
                        "facecolor": INK["surface"],
                        "edgecolor": "none",
                    },
                )
            ref_v = _ref_value(reference, corpus, metric)
            _ref_line(ax, ref_v, "MedCPT", outside=True)
            _ref_note(ax, reference, corpus, metric)
            b = base[base["corpus"] == corpus][metric]
            base_v = float(b.mean()) if (not b.empty and b.notna().any()) else None
            _ref_line(ax, base_v, "untrained", style=":", outside=True, below=True)
            col_span.setdefault(j, []).extend(
                _finite(list(means.to_numpy(dtype=float)) + [ref_v, base_v]).tolist()
            )
            ax.set_xticks(x)
            ax.set_xticklabels(pair_sets, fontsize=7.5)
            ax.spines["bottom"].set_bounds(x[0] - 0.25, x[-1] + 0.25)
            if i == 0:
                ax.set_title(METRIC_LABEL.get(metric, metric), fontsize=8.5, loc="left")
            if j == 0:
                ax.set_ylabel(CORPUS_LABEL.get(corpus, corpus), fontsize=8.5)
    for j, vals in col_span.items():
        lo = min(0.0, min(vals)) if vals else 0.0
        hi = max(vals) * 1.15 + 1e-9 if vals else 1.0
        for i in range(len(corpora)):
            axes[i][j].set_ylim(lo, hi)
    _title(fig, "Exp 3 · pair construction at fixed loss and τ, final checkpoint")
    _footnote(
        fig,
        "dashed = MedCPT, first post · dotted = untrained ckpt 0.0 · " + FOOT_DISINT,
    )
    # Explicit margins rather than tight_layout: the reference labels live in the
    # panel gaps and must neither squeeze the panels nor be cropped at save time.
    fig.subplots_adjust(
        left=0.05, right=0.93, top=0.86, bottom=0.10, wspace=0.62, hspace=0.35
    )
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Figure 4: Exp 4 Pareto
# ---------------------------------------------------------------------------


def fig_exp4_pareto(
    df: pd.DataFrame,
    reference: dict[str, Any],
    out: Path,
    baseline: pd.DataFrame | None = None,
) -> Path | None:
    """Gap against anchor rho per (aux, lambda) at the final checkpoint; size = router accuracy.

    Args:
        df: Exp 4 rows.
        reference: ``load_reference`` output.
        out: PNG path.
        baseline: Optional rows for the same model/loss/tau/pair set without an
            auxiliary (from Exp 1 or Exp 3), drawn in muted ink.

    Returns:
        The path written, or ``None`` if there are no final-checkpoint rows.
    """
    fin = _final(df)
    if fin.empty:
        log.warning("fig_exp4_pareto: no final-checkpoint rows")
        return None
    corpora = [c for c in ("scicueval", "mmlu") if c in set(fin["corpus"])]
    auxes = list(dict.fromkeys(a for a in fin["aux"] if isinstance(a, str) and a))
    colors = {a: CATEGORICAL[i % len(CATEGORICAL)] for i, a in enumerate(auxes)}
    lams = sorted(set(fin["lambda"].dropna()))

    def size(acc: float) -> float:
        return 24.0 + 260.0 * (0.0 if pd.isna(acc) else float(acc))

    fig, axes = _fig(1, len(corpora), w=4.8, h=4.0, sharey=True)
    panels: list[tuple[plt.Axes, list[tuple[float, float, str]]]] = []
    for j, corpus in enumerate(corpora):
        ax = axes[0][j]
        _style(ax)
        sub = fin[fin["corpus"] == corpus]
        pts: list[tuple[float, float, str]] = []
        for aux in auxes:
            for lam in lams:
                s = sub[(sub["aux"] == aux) & np.isclose(sub["lambda"], lam)]
                if s.empty:
                    continue
                gx, gy = float(s["gap"].mean()), float(s["anchor_rho"].mean())
                acc = float(s["router_acc_in"].mean())
                hollow = bool(s["disintegrated"].any())
                if hollow:
                    fc, ec = INK["surface"], colors[aux]
                elif lam == max(lams):
                    fc, ec = colors[aux], INK["surface"]
                else:
                    fc, ec = "none", colors[aux]
                ax.scatter(
                    [gx],
                    [gy],
                    s=size(acc),
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=3,
                )
                pts.append((gx, gy, f"{aux}, λ {lam:g}"))
        if baseline is not None and not baseline.empty:
            b = _final(baseline[baseline["corpus"] == corpus])
            if not b.empty:
                bx, by = float(b["gap"].mean()), float(b["anchor_rho"].mean())
                ax.scatter(
                    [bx],
                    [by],
                    s=size(float(b["router_acc_in"].mean())),
                    facecolor=INK["muted"],
                    edgecolor=INK["surface"],
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=3,
                )
                pts.append((bx, by, "no auxiliary"))
        r = _ref_row(reference, corpus)
        if r and r.get("gap") is not None and r.get("anchor_rho") is not None:
            ax.scatter(
                [r["gap"]],
                [r["anchor_rho"]],
                marker="x",
                s=50,
                color=INK["secondary"],
                linewidth=1.3,
                zorder=5,
            )
            label = "MedCPT, first post" + (
                " (disintegrated)" if r.get("disintegrated") else ""
            )
            pts.append((float(r["gap"]), float(r["anchor_rho"]), label))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        _rug(ax, xs, ys, INK["secondary"], alpha=0.4)
        ax.set_title(CORPUS_LABEL.get(corpus, corpus), fontsize=9, loc="left")
        ax.set_xlabel("cosine gap (within − between)", fontsize=8)
        if j == 0:
            ax.set_ylabel("anchor ρ", fontsize=8)
        panels.append((ax, pts))
    y_all = [p[1] for _, pts in panels for p in pts]
    for ax, pts in panels:
        _range_frame(ax, x=[p[0] for p in pts], y=y_all)
    key = [
        Line2D([], [], marker="o", linestyle="", color=colors[a], markersize=5, label=a)
        for a in auxes
    ]
    if len(lams) > 1:
        key += [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor=INK["secondary"],
                markersize=5,
                label=f"open = λ {min(lams):g}",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=INK["secondary"],
                markersize=5,
                label=f"filled = λ {max(lams):g}",
            ),
        ]
    fig.legend(
        handles=key,
        loc="lower left",
        ncol=len(key),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.0, -0.08),
        handletextpad=0.3,
        columnspacing=1.2,
    )
    size_key = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=INK["muted"],
            markersize=np.sqrt(size(a)) / 2.0,
            label=f"router acc {a:.2f}",
        )
        for a in (0.5, 0.75, 1.0)
    ]
    fig.legend(
        handles=size_key,
        loc="lower right",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(1.0, -0.08),
        handletextpad=0.6,
        columnspacing=1.6,
    )
    _title(
        fig,
        "Exp 4 · topological auxiliaries: separation against faithfulness, sized by router accuracy",
    )
    _footnote(
        fig,
        FOOT_DISINT
        + " · grey = same run without an auxiliary · × = MedCPT, first post",
        y=-0.14,
    )
    fig.tight_layout()
    fig.canvas.draw()
    for ax, pts in panels:
        _label_points(
            ax,
            [p[0] for p in pts],
            [p[1] for p in pts],
            [p[2] for p in pts],
            dx_pt=8.0,
            gap_pt=9.5,
        )
    return _save(fig, out)


# ---------------------------------------------------------------------------
# All four
# ---------------------------------------------------------------------------


def make_all(
    results_dir: str | Path, reference_path: str | Path, out_dir: str | Path
) -> list[Path]:
    """Draw every figure whose data exists.

    Args:
        results_dir: The ``results/`` directory.
        reference_path: ``configs/reference.yaml``.
        out_dir: Where PNGs go.

    Returns:
        Paths written (figures without data are skipped with a warning).
    """
    out_dir = Path(out_dir)
    ref = load_reference(reference_path)
    df = load_results(results_dir)
    written: list[Path] = []
    if df.empty:
        log.warning("no result CSVs under %s", results_dir)
        return written
    exp1 = df[df["exp"] == "exp1_tau_sweep"]
    exp3 = df[df["exp"] == "exp3_data_vs_loss"]
    exp4 = df[df["exp"] == "exp4_topo_aux"]
    baseline = pd.DataFrame()
    if not exp4.empty:
        m, t, ps = (
            exp4["model"].iloc[0],
            float(exp4["tau"].iloc[0]),
            exp4["pair_set"].iloc[0],
        )
        baseline = df[
            (df["model"] == m)
            & np.isclose(df["tau"], t)
            & (df["pair_set"] == ps)
            & (df["loss"] == "mnrl")
            & (df["aux"] == "")
        ]
    for fn, args, name in (
        (fig_rho_vs_step, (exp1, ref), "fig1_rho_vs_step.png"),
        (fig_gap_vs_rho, (df, ref), "fig2_gap_vs_rho.png"),
        (fig_exp3_bars, (exp3, ref), "fig3_exp3_pairs_vs_medcpt.png"),
        (fig_exp4_pareto, (exp4, ref), "fig4_exp4_pareto.png"),
    ):
        kwargs = {"baseline": baseline} if fn is fig_exp4_pareto else {}
        p = fn(*args, out_dir / name, **kwargs)
        if p is not None:
            written.append(p)
            log.info("wrote %s", p)
    return written
