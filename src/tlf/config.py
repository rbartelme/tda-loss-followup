"""YAML config loading with one level of ``base:`` inheritance.

Every script takes ``--config path.yaml``. An experiment config names its
shared settings with ``base: configs/base.yaml``; the experiment keys are
deep-merged over the base. The resolved dict carries ``_config_path`` so
manifests can record where it came from.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a deep copy of ``base``.

    Args:
        base: Mapping providing the defaults.
        override: Mapping whose values win. Nested mappings merge recursively;
            any other value replaces the base value outright.

    Returns:
        A new dict. Neither input is modified.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_base(base: str, relative_to: Path) -> Path:
    """Locate the file named by a ``base:`` entry.

    Args:
        base: Path string from the YAML, absolute or relative.
        relative_to: The config file that named it.

    Returns:
        An existing path. A relative name is tried as given (relative to the
        working directory), then under the config's directory and its parent,
        first by the full relative path and then by bare filename.

    Raises:
        FileNotFoundError: If no candidate exists.
    """
    p = Path(base)
    if p.is_absolute() or p.exists():
        return p
    for parent in (relative_to.parent, relative_to.parent.parent):
        cand = parent / p
        if cand.exists():
            return cand
        cand = parent / p.name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"base config {base!r} not found (from {relative_to})")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, merging it over its ``base:`` config if it names one.

    Args:
        path: Path to the YAML file.

    Returns:
        The merged mapping with ``_config_path`` set to ``str(path)``. The
        ``base`` key itself is removed.

    Raises:
        ValueError: If the top level of the file is not a mapping.
        FileNotFoundError: If a ``base:`` config cannot be found.
    """
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    base = cfg.pop("base", None)
    if base is not None:
        base_cfg = load_config(_resolve_base(base, path))
        base_cfg.pop("_config_path", None)
        cfg = _deep_merge(base_cfg, cfg)
    cfg["_config_path"] = str(path)
    return cfg
