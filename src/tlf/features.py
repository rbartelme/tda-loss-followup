"""Textstat surface features used as the linguistic anchor axis.

``extract`` and its helper are copied verbatim from tda-embedder-bakeoff
``scripts/complexity.py`` at commit 4625b22 so that this repo measures the
same six features the first post did. Do not edit them.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np

with warnings.catch_warnings():
    # textstat 0.7.4 imports pkg_resources; the deprecation notice is noise here.
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    import textstat

FEATURE_NAMES: tuple[str, ...] = (
    "flesch_reading_ease",
    "syllable_count",
    "lexicon_count",
    "sentence_count",
    "avg_sentence_length",
    "difficult_words",
)

# ---------------------------------------------------------------------------
# Copied from tda-embedder-bakeoff scripts/complexity.py @ 4625b22 (unchanged)
# ---------------------------------------------------------------------------


def _safe_avg_sentence_length(lexicon_count: int, sentence_count: int) -> float:
    """Compute words-per-sentence, protecting against zero-sentence input."""
    if sentence_count <= 0:
        return 0.0
    return float(lexicon_count) / float(sentence_count)


def extract(prompt: str) -> dict[str, float | int]:
    """Extract CohMetrix proxy features for a prompt."""
    lexicon = textstat.lexicon_count(prompt, removepunct=True)
    sentences = textstat.sentence_count(prompt)
    return {
        "flesch_reading_ease": float(textstat.flesch_reading_ease(prompt)),
        "syllable_count": int(textstat.syllable_count(prompt)),
        "lexicon_count": int(lexicon),
        "sentence_count": int(sentences),
        "avg_sentence_length": _safe_avg_sentence_length(lexicon, sentences),
        "difficult_words": int(textstat.difficult_words(prompt)),
    }


# ---------------------------------------------------------------------------
# Thin wrappers used by this repo
# ---------------------------------------------------------------------------


def textstat_vector(text: str) -> np.ndarray:
    """The six features as a float64 vector in ``FEATURE_NAMES`` order."""
    d = extract(text)
    return np.array([float(d[k]) for k in FEATURE_NAMES], dtype=np.float64)


def textstat_matrix(texts: Iterable[str]) -> np.ndarray:
    """Stack ``textstat_vector`` over ``texts`` into an (N, 6) matrix."""
    rows = [textstat_vector(t) for t in texts]
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack(rows)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and std with the bakeoff's convention ``std == 0 -> 1``."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def standardize(
    X: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None
) -> np.ndarray:
    """Z-score ``X``. Without ``mu``/``sd`` this is the bakeoff's ``standardize``."""
    X = np.asarray(X, dtype=np.float64)
    if mu is None or sd is None:
        mu, sd = fit_standardizer(X)
    return (X - mu) / sd
