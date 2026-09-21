"""Result rows and artifact manifests.

The CSV schema and idempotent ``append_row`` arrive in Task 6. The manifest
helpers live here from the start because every artifact writer needs them.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    """HEAD sha of this repo, suffixed ``-dirty`` if the tree has changes."""
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
) -> Path:
    """Write ``out_dir/manifest.json`` with provenance for the artifact in ``out_dir``."""
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
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path
