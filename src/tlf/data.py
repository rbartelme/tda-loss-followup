"""Corpus loading, the bakeoff evaluation sample, and pair construction.

The two corpora arrive as the bakeoff's flattened JSONLs (preferred, so the
eval sample is byte-identical to the first post) or are re-flattened from the
raw trees with adapter logic copied from tda-embedder-bakeoff @ 4625b22.

``bakeoff_eval_indices`` reproduces ``load_corpus`` from the bakeoff's
``embedding_diagnostic.py`` exactly: one RNG seeded once, labels walked in
alphabetical order, ``rng.choice(len(group), 400, replace=False)`` per label.
Row order inside the eval split is preserved in ``eval_order`` because the
bakeoff's Layer 1 pair sampling and Layer 2 partitions depend on it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tlf.features import FEATURE_NAMES, fit_standardizer, standardize, textstat_matrix

log = logging.getLogger(__name__)

CORPORA: tuple[str, ...] = ("scicueval", "mmlu")
CORPUS_COLUMNS: tuple[str, ...] = (
    "id",
    "text",
    "subset",
    "domain",
    "split",
    "eval_order",
)

# ---------------------------------------------------------------------------
# Flatteners, copied from the bakeoff adapters (scicueval_to_jsonl.py,
# mmlu_to_jsonl.py @ 4625b22). Same record ids, same row order.
# ---------------------------------------------------------------------------

SCICUEVAL_COMPETENCY_FILES: tuple[str, ...] = (
    "Context-aware_Inference.json",
    "Information-absence_Detection.json",
    "Information_Integration.json",
    "Relevant_Information_Identification.json",
)
SCICUEVAL_SUBSET_DOMAIN: dict[str, str] = {
    "BioText": "biology",
    "GoKG": "biology",
    "HipKG": "biology",
    "ProtTab": "biology",
    "PhaKG": "biomedicine",
    "PriKG": "biomedicine",
    "MolTab": "chemistry",
    "MatTab": "materials",
    "MatText": "materials",
    "IaeaTab": "physics",
}
MMLU_SUBJECT_DOMAIN: dict[str, str] = {
    "philosophy": "humanities",
    "world_religions": "humanities",
    "high_school_psychology": "social_science",
    "sociology": "social_science",
    "international_law": "law_policy",
    "jurisprudence": "law_policy",
    "management": "business",
    "marketing": "business",
    "high_school_us_history": "common_knowledge",
    "miscellaneous": "common_knowledge",
}


def _slugify_competency(name: str) -> str:
    return name.lower().replace("_", "-").replace("--", "-")


def flatten_scicueval(root: Path) -> list[dict[str, Any]]:
    """Walk ``<root>/<Subset>/<Competency>.json`` into flat records."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"SciCUEval root does not exist: {root}")
    out: list[dict[str, Any]] = []
    for subset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subset = subset_dir.name
        for fname in SCICUEVAL_COMPETENCY_FILES:
            path = subset_dir / fname
            if not path.is_file():
                log.warning("missing %s, skipping", path)
                continue
            competency = fname.removesuffix(".json")
            slug = _slugify_competency(competency)
            with path.open() as f:
                entries = json.load(f)
            for i, entry in enumerate(entries):
                q = entry.get("question")
                if not isinstance(q, str) or not q.strip():
                    continue
                out.append(
                    {
                        "id": f"scicueval-{subset.lower()}-{slug}-{i + 1}",
                        "question": q,
                        "source_dataset": "SciCUEval",
                        "source_subset": subset,
                        "source_domain": SCICUEVAL_SUBSET_DOMAIN.get(subset, "unknown"),
                        "source_competency": competency,
                    }
                )
    return out


def flatten_mmlu(root: Path) -> list[dict[str, Any]]:
    """Walk ``<root>/<subject>.json`` into flat records."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MMLU root does not exist: {root}")
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        subject = path.stem
        with path.open() as f:
            entries = json.load(f)
        for i, entry in enumerate(entries):
            q = entry.get("question")
            if not isinstance(q, str) or not q.strip():
                continue
            out.append(
                {
                    "id": f"mmlu-{subject}-{i + 1}",
                    "question": q,
                    "source_dataset": "MMLU",
                    "source_subset": subject,
                    "source_domain": MMLU_SUBJECT_DOMAIN.get(subject, "unknown"),
                    "source_competency": None,
                }
            )
    return out


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_records(
    cfg: dict[str, Any], corpus: str
) -> tuple[list[dict[str, Any]], Path]:
    """Records for ``corpus`` plus the file they came from.

    Order of preference: the bakeoff JSONL named in the config, a previously
    cached copy under ``data/raw``, and finally a fresh flatten of the raw
    tree (which is then cached).
    """
    paths = cfg["paths"]
    jsonl = Path(paths[f"{corpus}_jsonl"])
    cache = Path(paths["data_raw"]) / f"{corpus}_combined.jsonl"
    for cand in (jsonl, cache):
        if cand.is_file():
            return read_jsonl(cand), cand
    root = Path(paths[f"{corpus}_root"])
    flatten = flatten_scicueval if corpus == "scicueval" else flatten_mmlu
    log.info("flattening %s from %s", corpus, root)
    records = flatten(root)
    write_jsonl(cache, records)
    return records, cache


# ---------------------------------------------------------------------------
# The bakeoff eval sample
# ---------------------------------------------------------------------------


def bakeoff_eval_indices(
    labels: Sequence[str], sample_per_class: int, seed: int
) -> np.ndarray:
    """Row indices of the bakeoff's evaluation sample, in the bakeoff's order.

    Mirrors ``load_corpus`` in the bakeoff's ``embedding_diagnostic.py``: rows
    are grouped by label in file order, one ``default_rng(seed)`` is shared
    across labels, labels are visited alphabetically, and a label with more
    than ``sample_per_class`` rows contributes ``rng.choice(n, k, replace=False)``
    in the order returned.
    """
    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(str(lab), []).append(i)
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for _, idxs in sorted(by_label.items()):
        if len(idxs) <= sample_per_class:
            out.extend(idxs)
        else:
            pick = rng.choice(len(idxs), size=sample_per_class, replace=False)
            out.extend(idxs[j] for j in pick)
    return np.asarray(out, dtype=np.int64)


def _records_to_frame(
    records: list[dict[str, Any]], prompt_field: str, label_field: str
) -> pd.DataFrame:
    ids, texts, subsets, domains = [], [], [], []
    for r in records:
        text = r.get(prompt_field)
        if not isinstance(text, str) or not text.strip():
            continue  # same filter as the bakeoff's load_corpus
        ids.append(str(r["id"]))
        texts.append(text)
        subsets.append(str(r[label_field]))
        domains.append(str(r.get("source_domain", "unknown")))
    return pd.DataFrame(
        {"id": ids, "text": texts, "subset": subsets, "domain": domains}
    )


def _assign_splits(df: pd.DataFrame, sample_per_class: int, seed: int) -> pd.DataFrame:
    idx = bakeoff_eval_indices(df["subset"].tolist(), sample_per_class, seed)
    split = np.full(len(df), "train", dtype=object)
    order = np.full(len(df), -1, dtype=np.int64)
    split[idx] = "eval"
    order[idx] = np.arange(len(idx))
    df = df.copy()
    df["split"] = split
    df["eval_order"] = order
    return df


def load_scicueval(cfg: dict[str, Any]) -> pd.DataFrame:
    """SciCUEval with the bakeoff's 400-per-subset ``eval`` sample; the rest is ``train``."""
    records, _ = _source_records(cfg, "scicueval")
    c = cfg["corpus"]
    df = _records_to_frame(records, c["prompt_field"], c["label_field"])
    df = _assign_splits(df, int(c["sample_per_class"]), int(c["sample_seed"]))
    return df[list(CORPUS_COLUMNS)]


def load_mmlu(cfg: dict[str, Any]) -> pd.DataFrame:
    """MMLU with the bakeoff's sampling. Every returned row is ``eval``."""
    records, _ = _source_records(cfg, "mmlu")
    c = cfg["corpus"]
    df = _records_to_frame(records, c["prompt_field"], c["label_field"])
    df = _assign_splits(df, int(c["sample_per_class"]), int(c["sample_seed"]))
    df = df[df["split"] == "eval"].sort_values("eval_order").reset_index(drop=True)
    return df[list(CORPUS_COLUMNS)]


def load_corpus(cfg: dict[str, Any], corpus: str) -> pd.DataFrame:
    if corpus == "scicueval":
        return load_scicueval(cfg)
    if corpus == "mmlu":
        return load_mmlu(cfg)
    raise ValueError(f"unknown corpus {corpus!r}; expected one of {CORPORA}")


def corpus_source_path(cfg: dict[str, Any], corpus: str) -> Path:
    """The JSONL ``load_corpus`` would read for ``corpus`` (for manifests)."""
    return _source_records(cfg, corpus)[1]


def eval_split(df: pd.DataFrame) -> pd.DataFrame:
    """Eval rows in the bakeoff's row order."""
    return df[df["split"] == "eval"].sort_values("eval_order").reset_index(drop=True)


def train_split(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["split"] == "train"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Textstat cache
# ---------------------------------------------------------------------------


def corpus_textstat(df: pd.DataFrame, cfg: dict[str, Any], corpus: str) -> np.ndarray:
    """(N, 6) textstat matrix aligned to ``df`` rows, cached under ``data/raw``.

    The cache is keyed on the id sequence, so a different frame recomputes.
    """
    raw = Path(cfg["paths"]["data_raw"])
    npy = raw / f"{corpus}_textstat.npy"
    ids_path = raw / f"{corpus}_textstat.ids.json"
    ids = df["id"].tolist()
    if npy.is_file() and ids_path.is_file():
        try:
            if json.loads(ids_path.read_text()) == ids:
                X = np.load(npy)
                if X.shape == (len(ids), len(FEATURE_NAMES)):
                    return X
        except (OSError, ValueError):
            pass
    log.info("computing textstat features for %d %s texts", len(ids), corpus)
    X = textstat_matrix(df["text"].tolist())
    raw.mkdir(parents=True, exist_ok=True)
    np.save(npy, X)
    ids_path.write_text(json.dumps(ids))
    return X


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

PAIR_KINDS: tuple[str, ...] = ("random", "matched", "mismatched")
PAIR_COLUMNS: tuple[str, ...] = ("anchor_id", "positive_id", "subset", "ts_dist")


def _proportional_quotas(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder split of ``total`` in proportion to ``sizes``."""
    n = sum(sizes.values())
    if n == 0 or total <= 0:
        return {s: 0 for s in sizes}
    raw = {s: total * c / n for s, c in sizes.items()}
    base = {s: int(v) for s, v in raw.items()}
    rem = total - sum(base.values())
    for s in sorted(sizes, key=lambda s: (-(raw[s] - base[s]), s))[:rem]:
        base[s] += 1
    return base


def build_pairs_with_report(
    df_train: pd.DataFrame,
    kind: str,
    n_pairs: int,
    seed: int,
    *,
    feature_matrix: np.ndarray | None = None,
    matched_max_sd: float = 0.5,
    mismatched_min_sd: float = 1.5,
    eval_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build ``n_pairs`` (anchor, positive) pairs from ``df_train``.

    * ``random``: positive drawn uniformly from the anchor's subset.
    * ``matched``: same subset, standardized textstat distance ``<= matched_max_sd``.
    * ``mismatched``: same subset, standardized textstat distance ``>= mismatched_min_sd``.

    Textstat features are standardized on ``df_train`` only. Anchors are
    sampled without replacement: pass 1 fills a per-subset quota proportional
    to subset size; if a subset cannot supply its quota, the shortfall is
    filled from anchors in other subsets that are still unused. Only once
    every usable anchor has been used once are anchors reused (a second
    round), and no (anchor, positive) pair is ever emitted twice. The report
    records quotas, shortfalls, how many pairs were filled from other subsets,
    and the maximum anchor reuse.
    """
    if kind not in PAIR_KINDS:
        raise ValueError(f"kind must be one of {PAIR_KINDS}, got {kind!r}")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    if "split" in df_train.columns and (df_train["split"] != "train").any():
        raise ValueError("df_train contains rows whose split is not 'train'")
    eval_id_set = {str(x) for x in (eval_ids or ())}
    df = df_train.reset_index(drop=True)
    ids = df["id"].astype(str).to_numpy()
    if eval_id_set & set(ids):
        raise ValueError("df_train contains eval ids")
    subsets = df["subset"].astype(str).to_numpy()
    n = len(df)

    X = (
        textstat_matrix(df["text"].tolist())
        if feature_matrix is None
        else np.asarray(feature_matrix)
    )
    if X.shape != (n, len(FEATURE_NAMES)):
        raise ValueError(
            f"feature_matrix must be ({n}, {len(FEATURE_NAMES)}), got {X.shape}"
        )
    mu, sd = fit_standardizer(X)
    Z = standardize(X, mu, sd)

    rng = np.random.default_rng(seed)
    subset_names = sorted(set(subsets))
    members = {s: np.flatnonzero(subsets == s) for s in subset_names}
    quotas = _proportional_quotas({s: len(m) for s, m in members.items()}, n_pairs)

    used_by_anchor: dict[int, set[int]] = {}
    use_count = np.zeros(n, dtype=np.int64)
    dead = np.zeros(n, dtype=bool)
    anchors: list[int] = []
    positives: list[int] = []
    dists: list[float] = []

    def draw(a: int) -> tuple[int, float] | None:
        m = members[subsets[a]]
        d = np.linalg.norm(Z[m] - Z[a], axis=1)
        if kind == "matched":
            ok = d <= matched_max_sd
        elif kind == "mismatched":
            ok = d >= mismatched_min_sd
        else:
            ok = np.ones(len(m), dtype=bool)
        ok &= m != a
        used = used_by_anchor.get(a)
        if used:
            ok &= ~np.isin(m, list(used))
        cand = m[ok]
        if cand.size == 0:
            dead[a] = True
            return None
        j = int(rng.integers(cand.size))
        return int(cand[j]), float(d[ok][j])

    def take(a: int) -> bool:
        r = draw(a)
        if r is None:
            return False
        p, dist = r
        anchors.append(a)
        positives.append(p)
        dists.append(dist)
        used_by_anchor.setdefault(a, set()).add(p)
        use_count[a] += 1
        return True

    # Pass 1: per-subset quotas, anchors without replacement inside each subset.
    per_subset: Counter[str] = Counter()
    for s in subset_names:
        got = 0
        for a in rng.permutation(members[s]):
            if got >= quotas[s]:
                break
            if take(int(a)):
                got += 1
        per_subset[s] = got
    shortfall = {s: quotas[s] - per_subset[s] for s in subset_names}
    n_after_pass1 = len(anchors)

    # Fill rounds: unused anchors first (round 0), then once-used (round 1), ...
    round_ = 0
    max_rounds = int(np.ceil(n_pairs / max(1, n))) + 1
    while len(anchors) < n_pairs and round_ <= max_rounds:
        eligible = np.flatnonzero((use_count == round_) & ~dead)
        if eligible.size == 0:
            round_ += 1
            continue
        progressed = False
        for a in rng.permutation(eligible):
            if len(anchors) >= n_pairs:
                break
            if take(int(a)):
                per_subset[subsets[a]] += 1
                progressed = True
        if not progressed:
            round_ += 1
    n_built = len(anchors)
    if n_built < n_pairs:
        log.warning(
            "%s: built %d of %d requested pairs (no more admissible pairs)",
            kind,
            n_built,
            n_pairs,
        )
    filled = n_built - n_after_pass1
    if filled:
        log.info(
            "%s: %d pairs filled outside per-subset quotas; shortfalls %s",
            kind,
            filled,
            {s: v for s, v in shortfall.items() if v},
        )

    order = rng.permutation(n_built)
    a_idx = np.asarray(anchors, dtype=np.int64)[order]
    p_idx = np.asarray(positives, dtype=np.int64)[order]
    out = pd.DataFrame(
        {
            "anchor_id": ids[a_idx],
            "positive_id": ids[p_idx],
            "subset": subsets[a_idx],
            "ts_dist": np.asarray(dists, dtype=np.float64)[order],
        }
    )
    report: dict[str, Any] = {
        "kind": kind,
        "seed": int(seed),
        "n_train": int(n),
        "n_pairs_requested": int(n_pairs),
        "n_pairs_built": int(n_built),
        "matched_max_sd": float(matched_max_sd),
        "mismatched_min_sd": float(mismatched_min_sd),
        "quotas": quotas,
        "per_subset_pairs": dict(per_subset),
        "per_subset_shortfall": shortfall,
        "filled_outside_quota": int(filled),
        "max_anchor_reuse": int(use_count.max()) if n else 0,
        "n_anchors_used": int((use_count > 0).sum()),
        "n_dead_anchors": int(dead.sum()),
        "standardizer_mu": mu.tolist(),
        "standardizer_sd": sd.tolist(),
        "feature_names": list(FEATURE_NAMES),
    }
    return out, report


def build_pairs(
    df_train: pd.DataFrame,
    kind: str,
    n_pairs: int,
    seed: int,
    **kwargs: Any,
) -> pd.DataFrame:
    """``build_pairs_with_report`` without the report. See that function."""
    return build_pairs_with_report(df_train, kind, n_pairs, seed, **kwargs)[0]
