"""Copied from rbartelme/tda-embedder-bakeoff @ 4625b22 -- scripts/embedding_diagnostic.py.

Verbatim except where a comment says ``# tlf:``. Excluded from ruff. Do not
edit; if a value must differ, add a parameter whose default is the original.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def resolve_device(arg: str) -> torch.device:
    """Map a CLI device string to a torch.device, honoring 'auto'."""
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _batched(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_all(
    prompts: list[str],
    model_id: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    trust_remote_code: bool = False,
) -> np.ndarray:
    print(f"  loading {model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    ).to(device)
    model.eval()

    out_chunks: list[np.ndarray] = []
    n = len(prompts)
    for batch_idx, batch in enumerate(_batched(prompts, batch_size)):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        out_chunks.append(pooled.cpu().numpy().astype(np.float32))
        done = min((batch_idx + 1) * batch_size, n)
        if batch_idx % 10 == 0 or done == n:
            print(f"    encoded {done}/{n}")

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(out_chunks)


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return X / norms


def class_centroids(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    cents = np.zeros((n_classes, X.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            cents[c] = X[mask].mean(axis=0)
    return cents


def within_between_cosine(
    Xn: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_pairs: int = 20000
) -> tuple[np.ndarray, np.ndarray]:
    """Sampled within-class and between-class cosine similarities.

    Computing every pair is O(n^2). For the diagnostic we sample ``n_pairs``
    of each kind — enough resolution to compare distributions.
    """
    n = Xn.shape[0]
    within = np.empty(n_pairs, dtype=np.float32)
    between = np.empty(n_pairs, dtype=np.float32)

    # Pre-bucket by label for fast within-class sampling.
    by_label: dict[int, np.ndarray] = {}
    for label in np.unique(y):
        by_label[int(label)] = np.where(y == label)[0]
    label_keys = list(by_label.keys())

    w_filled = 0
    while w_filled < n_pairs:
        # Pick a random label that has ≥2 members.
        c = int(rng.choice(label_keys))
        idxs = by_label[c]
        if idxs.size < 2:
            continue
        i, j = rng.choice(idxs, size=2, replace=False)
        within[w_filled] = float(np.dot(Xn[i], Xn[j]))
        w_filled += 1

    b_filled = 0
    while b_filled < n_pairs:
        i, j = rng.integers(0, n, size=2)
        if y[i] == y[j]:
            continue
        between[b_filled] = float(np.dot(Xn[i], Xn[j]))
        b_filled += 1

    return within, between


def lr_accuracy(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Cheap separability score: stratified 80/20 LR accuracy on raw features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    stratify = y if min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify
    )
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    )
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))
