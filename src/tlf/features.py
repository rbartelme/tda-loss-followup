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
    """Compute words-per-sentence, protecting against zero-sentence input.

    Args:
        lexicon_count: Number of words in the prompt.
        sentence_count: Number of sentences detected by textstat.

    Returns:
        Average sentence length in words. ``0.0`` when ``sentence_count`` is
        zero (e.g. empty or punctuation-free input) rather than dividing by
        zero.
    """
    if sentence_count <= 0:
        return 0.0
    return float(lexicon_count) / float(sentence_count)


def extract(prompt: str) -> dict[str, float | int]:
    """Extract CohMetrix proxy features for a prompt.

    The six features are surface readability/lexical density signals,
    designed to complement (not replace) a semantic embedding when
    evaluating manifold structure.

    Args:
        prompt: Raw user prompt. May be empty.

    Returns:
        A dictionary with the following keys:

            flesch_reading_ease (float): Overall readability score.
            syllable_count (int): Total syllables — lexical density proxy.
            lexicon_count (int): Word count (punctuation excluded).
            sentence_count (int): Number of sentences.
            avg_sentence_length (float): Mean words per sentence.
            difficult_words (int): Count of Dale-Chall difficult words —
                a domain-terminology density proxy.
    """
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
    """Compute the six textstat features for one text.

    Args:
        text: Raw text. May be empty.

    Returns:
        A float64 array of shape ``(6,)`` in ``FEATURE_NAMES`` order.
    """
    d = extract(text)
    return np.array([float(d[k]) for k in FEATURE_NAMES], dtype=np.float64)


def textstat_matrix(texts: Iterable[str]) -> np.ndarray:
    """Stack ``textstat_vector`` over many texts.

    Args:
        texts: Iterable of raw texts.

    Returns:
        A float64 array of shape ``(N, 6)``; ``(0, 6)`` when ``texts`` is empty.
    """
    rows = [textstat_vector(t) for t in texts]
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack(rows)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-column z-scoring statistics with the bakeoff's convention.

    Args:
        X: Feature matrix of shape ``(N, D)``.

    Returns:
        A ``(mu, sd)`` tuple of shape-``(D,)`` arrays. A column with zero
        standard deviation gets ``sd = 1`` so it standardizes to zero rather
        than NaN, exactly as the bakeoff's ``standardize`` does.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def standardize(
    X: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None
) -> np.ndarray:
    """Z-score a feature matrix.

    Args:
        X: Feature matrix of shape ``(N, D)``.
        mu: Per-column means. Fitted from ``X`` when omitted.
        sd: Per-column standard deviations. Fitted from ``X`` when omitted.

    Returns:
        ``(X - mu) / sd`` as float64. With ``mu`` and ``sd`` omitted this is
        the bakeoff's ``standardize``; with them supplied it applies
        statistics fitted elsewhere (for example, on the train split only).
    """
    X = np.asarray(X, dtype=np.float64)
    if mu is None or sd is None:
        mu, sd = fit_standardizer(X)
    return (X - mu) / sd
