"""Result rows, idempotent CSV append, and artifact manifests.

One CSV per experiment under ``results/<exp>.csv`` with exactly
``CSV_COLUMNS``. A row is identified by ``KEY_COLUMNS``; appending a row whose
key already exists replaces that row in place, so re-running a sweep never
duplicates results.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "sentence-transformers",
    "datasets",
    "textstat",
    "umap-learn",
    "hdbscan",
    "kmapper",
    "networkx",
    "scikit-learn",
    "scipy",
    "numpy",
    "pandas",
    "persist0-tda",
)


def git_sha(repo: Path = REPO_ROOT) -> str:
    """Return the HEAD commit of a git repository.

    Args:
        repo: Repository root. Defaults to this repo.

    Returns:
        The 40-character sha, suffixed with ``-dirty`` when the working tree
        has uncommitted changes, or ``"unknown"`` if git is unavailable.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def package_versions(names: tuple[str, ...] = MANIFEST_PACKAGES) -> dict[str, str]:
    """Look up installed versions of the packages that affect results.

    Args:
        names: Distribution names to query.

    Returns:
        A mapping from name to version string, or ``"missing"`` when the
        distribution is not installed.
    """
    out: dict[str, str] = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = "missing"
    return out


def write_manifest(
    out_dir: str | Path,
    *,
    config_path: str | Path,
    seed: int,
    extra: dict[str, Any] | None = None,
    name: str = "manifest.json",
) -> Path:
    """Write ``out_dir/<name>`` describing an artifact stored in ``out_dir``.

    Args:
        out_dir: Directory holding the artifact. Created if needed.
        config_path: The YAML config the artifact was produced from.
        seed: The seed used.
        extra: Additional keys merged into the manifest (artifact-specific
            provenance such as source hashes or builder reports).
        name: File name; defaults to ``manifest.json``. Artifacts that share a
            directory (per-corpus embeddings) use a distinct name each.

    Returns:
        The path of the manifest written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "config_path": str(config_path),
        "git_sha": git_sha(),
        "seed": int(seed),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": package_versions(),
    }
    if extra:
        manifest.update(extra)
    path = out_dir / name
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------

CSV_COLUMNS: tuple[str, ...] = (
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
    "router_in_to_mmlu_frac",
    "router_mmlu_to_in_frac",
    "router_mode",
    "router_label",
    "git_sha",
    "timestamp",
)
KEY_COLUMNS: tuple[str, ...] = CSV_COLUMNS[:10]
METRIC_COLUMNS: tuple[str, ...] = CSV_COLUMNS[10:29]
# Metric columns that name a recipe rather than measure something.
STRING_METRICS: tuple[str, ...] = ("lr_eval", "router_mode", "router_label")
NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    c
    for c in CSV_COLUMNS
    if c
    not in (
        "exp",
        "model",
        "loss",
        "pair_set",
        "aux",
        "corpus",
        "lr_eval",
        "router_mode",
        "router_label",
        "disintegrated",
        "git_sha",
        "timestamp",
    )
)


def _norm(v: Any) -> str:
    """Canonical string for key comparison: numbers via ``%.10g``, empties as ``""``."""
    if v is None:
        return ""
    if isinstance(v, float | np.floating) and math.isnan(float(v)):
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int | float | np.integer | np.floating):
        return f"{float(v):.10g}"
    s = str(v).strip()
    if s.lower() in ("", "nan", "none", "null"):
        return ""
    try:
        return f"{float(s):.10g}"
    except ValueError:
        return s


def row_key(row: dict[str, Any]) -> tuple[str, ...]:
    """The identity of a result row.

    Args:
        row: Mapping with at least the ``KEY_COLUMNS``.

    Returns:
        Canonicalised key values in ``KEY_COLUMNS`` order.
    """
    return tuple(_norm(row.get(c)) for c in KEY_COLUMNS)


def results_path(results_dir: str | Path, exp: str) -> Path:
    """Path of an experiment's CSV.

    Args:
        results_dir: The ``results/`` directory.
        exp: Experiment name (``exp1_tau_sweep`` ...).

    Returns:
        ``results_dir/<exp>.csv``.
    """
    return Path(results_dir) / f"{exp}.csv"


def make_row(
    *,
    exp: str,
    model: str,
    loss: str,
    tau: float,
    pair_set: str,
    aux: str | None,
    lam: float,
    ckpt_frac: float,
    seed: int,
    corpus: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a CSV row from a run's identity and ``evaluate_checkpoint`` output.

    Args:
        exp: Experiment name.
        model: Model key.
        loss: Loss name.
        tau: Temperature (recorded as configured even for losses that ignore it).
        pair_set: Pair-set kind.
        aux: Auxiliary loss name or ``None``.
        lam: Auxiliary weight (``0.0`` when ``aux`` is ``None``).
        ckpt_frac: Checkpoint fraction.
        seed: Training seed.
        corpus: Evaluation corpus.
        metrics: Dict from ``evaluate_checkpoint``; missing metrics become NaN
            (the recipe names in ``STRING_METRICS`` become the empty string).

    Returns:
        A dict with exactly ``CSV_COLUMNS``.
    """
    nan = float("nan")
    row: dict[str, Any] = {
        "exp": exp,
        "model": model,
        "loss": loss,
        "tau": float(tau),
        "pair_set": pair_set,
        "aux": aux or "",
        "lambda": float(lam) if aux else nan,
        "ckpt_frac": float(ckpt_frac),
        "seed": int(seed),
        "corpus": corpus,
    }
    for c in METRIC_COLUMNS:
        if c in STRING_METRICS:
            row[c] = str(metrics.get(c, ""))
            continue
        v = metrics.get(c, nan)
        row[c] = bool(v) if c == "disintegrated" else v
    row["git_sha"] = git_sha()
    row["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    return {c: row[c] for c in CSV_COLUMNS}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def append_row(row: dict[str, Any], results_dir: str | Path) -> Path:
    """Append a row to ``results/<exp>.csv``, replacing any row with the same key.

    Args:
        row: A row with the ``CSV_COLUMNS`` (extra keys are ignored, missing
            metrics are written empty).
        results_dir: The ``results/`` directory.

    Returns:
        The CSV path written.

    Raises:
        ValueError: If the row has no ``exp``.
    """
    if not row.get("exp"):
        raise ValueError("row needs an 'exp' to pick its CSV")
    path = results_path(results_dir, str(row["exp"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(
        [{c: row.get(c) for c in CSV_COLUMNS}], columns=list(CSV_COLUMNS)
    )
    new = new.astype(object).where(new.notna(), "")
    new = new.map(lambda v: str(v) if v != "" else "")
    key = row_key(row)
    if path.is_file():
        df = _read_csv(path)
        for c in CSV_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df = df[list(CSV_COLUMNS)]
        hits = [i for i, r in enumerate(df.to_dict("records")) if row_key(r) == key]
        if hits:
            df.iloc[hits[0]] = new.iloc[0]
            if len(hits) > 1:
                df = df.drop(index=df.index[hits[1:]])
        else:
            df = pd.concat([df, new], ignore_index=True)
    else:
        df = new
    df.to_csv(path, index=False)
    return path


def row_exists(key: dict[str, Any], results_dir: str | Path) -> bool:
    """Whether a row with this key is already in its experiment's CSV.

    Args:
        key: Mapping with the ``KEY_COLUMNS`` (at least ``exp``).
        results_dir: The ``results/`` directory.

    Returns:
        ``True`` if present.
    """
    path = results_path(results_dir, str(key.get("exp", "")))
    if not path.is_file():
        return False
    k = row_key(key)
    return any(row_key(r) == k for r in _read_csv(path).to_dict("records"))


def load_results(
    results_dir: str | Path, exps: Sequence[str] | None = None
) -> pd.DataFrame:
    """Load and type one or more experiment CSVs into a single frame.

    Args:
        results_dir: The ``results/`` directory.
        exps: Experiment names to load; all ``*.csv`` when omitted.

    Returns:
        A frame with ``CSV_COLUMNS``: numeric columns as floats (``seed`` as
        int), ``disintegrated`` as bool, ``aux`` empty string for none, the
        ``STRING_METRICS`` empty for CSVs written before those columns
        existed. Empty when nothing is on disk.
    """
    results_dir = Path(results_dir)
    paths = (
        [results_path(results_dir, e) for e in exps]
        if exps is not None
        else sorted(results_dir.glob("*.csv"))
    )
    frames = [_read_csv(p) for p in paths if p.is_file()]
    if not frames:
        return pd.DataFrame(columns=list(CSV_COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[list(CSV_COLUMNS)]
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    df["seed"] = df["seed"].fillna(-1).astype(int)
    df["disintegrated"] = (
        df["disintegrated"].astype(str).str.lower().isin(("true", "1", "yes"))
    )
    return df
