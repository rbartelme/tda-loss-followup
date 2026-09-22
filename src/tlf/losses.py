"""Auxiliary losses for Exp 4.

Each loss is an ``AuxLoss`` module with ``forward(embeddings, texts)`` where
``embeddings`` is the ``(n, d)`` subsample the driver hands over (with
gradient) and ``texts`` are the matching raw texts. Textstat targets are
computed from the texts on the fly, memoised per text, and standardized with
the train-split statistics the pair builder used.

The distance-based losses (``dist_preserve``, ``persist0_h0``, ``topoae_h0``)
compare a live cosine-distance matrix to a fixed reference built from
standardized textstat vectors. ``persist0_h0`` and ``topoae_h0`` are an A/B:
both take H0 (MST) pairs from ``persist0``; the first matches the sorted
death vector, which is invariant to *which* points a bar joins, the second is
TopoAE's exact form, distances at each space's own MST edges in both
directions, which is not. Those live on
different scales (cosine distance is bounded by 2; textstat Euclidean
distance is not), so both take a ``ref_scale`` policy. The default,
``"match_mean"``, rescales the reference so its mean pairwise distance equals
the live mean (detached), which makes the loss compare relative structure
rather than absolute scale. ``"none"`` compares raw values; a float is a
fixed multiplier.

``persist0`` is consumed from PyPI as ``persist0-tda`` and never vendored.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from persist0 import TopoH0Loss, h0_persistence
from torch import Tensor, nn

from tlf.features import FEATURE_NAMES, textstat_vector

AUX_LOSS_NAMES: tuple[str, ...] = (
    "textstat_head",
    "dist_preserve",
    "persist0_h0",
    "topoae_h0",
)
RefScale = str | float


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


class TextstatTargets:
    """Standardized textstat vectors for texts, memoised per text.

    Args:
        mu: Train-split feature means, shape ``(6,)``.
        sd: Train-split feature standard deviations, shape ``(6,)``.
    """

    def __init__(self, mu: np.ndarray, sd: np.ndarray) -> None:
        self.mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        self.sd = np.asarray(sd, dtype=np.float64).reshape(-1)
        if self.mu.shape != (len(FEATURE_NAMES),) or self.sd.shape != (
            len(FEATURE_NAMES),
        ):
            raise ValueError(f"mu and sd must have shape ({len(FEATURE_NAMES)},)")
        self._cache: dict[str, np.ndarray] = {}

    def raw(self, text: str) -> np.ndarray:
        """Raw six-feature vector for one text, cached.

        Args:
            text: Raw text.

        Returns:
            Float64 array of shape ``(6,)``.
        """
        v = self._cache.get(text)
        if v is None:
            v = textstat_vector(text)
            self._cache[text] = v
        return v

    def __call__(self, texts: Sequence[str], device: torch.device) -> Tensor:
        """Standardized targets for a batch of texts.

        Args:
            texts: Raw texts.
            device: Device for the returned tensor.

        Returns:
            Float32 tensor of shape ``(len(texts), 6)`` without gradient.
        """
        X = np.vstack([self.raw(t) for t in texts]) if texts else np.zeros((0, 6))
        Z = (X - self.mu) / self.sd
        return torch.as_tensor(Z, dtype=torch.float32, device=device)


def cosine_distance_matrix(embeddings: Tensor) -> Tensor:
    """Symmetric, zero-diagonal cosine distance matrix with gradient.

    Args:
        embeddings: ``(n, d)`` tensor.

    Returns:
        ``(n, n)`` float32 tensor ``1 - cos(e_i, e_j)`` clamped at 0.
    """
    z = F.normalize(embeddings.float(), dim=-1)
    D = 1.0 - z @ z.T
    D = 0.5 * (D + D.T)
    eye = torch.eye(D.shape[0], device=D.device, dtype=D.dtype)
    return (D * (1.0 - eye)).clamp_min(0.0)


def euclidean_distance_matrix(X: Tensor) -> Tensor:
    """Symmetric, zero-diagonal Euclidean distance matrix (no gradient intended).

    Args:
        X: ``(n, k)`` tensor.

    Returns:
        ``(n, n)`` float32 tensor.
    """
    D = torch.cdist(X.float(), X.float())
    D = 0.5 * (D + D.T)
    eye = torch.eye(D.shape[0], device=D.device, dtype=D.dtype)
    return D * (1.0 - eye)


def scale_reference(D_live: Tensor, D_ref: Tensor, mode: RefScale) -> Tensor:
    """Bring a fixed reference distance matrix onto the live matrix's scale.

    Args:
        D_live: Live ``(n, n)`` distance matrix (may carry gradient).
        D_ref: Reference ``(n, n)`` distance matrix.
        mode: ``"match_mean"`` scales ``D_ref`` so its mean off-diagonal
            distance equals the detached live mean; ``"none"`` returns
            ``D_ref`` unchanged; a float multiplies ``D_ref`` by that value.

    Returns:
        The scaled reference, detached.

    Raises:
        ValueError: On an unknown mode string.
    """
    D_ref = D_ref.detach()
    if isinstance(mode, int | float) and not isinstance(mode, bool):
        return D_ref * float(mode)
    if mode == "none":
        return D_ref
    if mode == "match_mean":
        n = D_ref.shape[0]
        if n < 2:
            return D_ref
        mask = ~torch.eye(n, dtype=torch.bool, device=D_ref.device)
        live_mean = D_live.detach()[mask].mean()
        ref_mean = D_ref[mask].mean()
        if ref_mean <= 0:
            return D_ref
        return D_ref * (live_mean / ref_mean)
    raise ValueError(
        f"unknown ref_scale {mode!r}; expected 'match_mean', 'none' or a float"
    )


def _no_autocast(device: torch.device):
    """Disable autocast so distance matrices and MSEs are computed in float32."""
    return torch.autocast(device_type=device.type, enabled=False)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class AuxLoss(nn.Module):
    """Base class for auxiliary losses computed on a batch subsample.

    Subclasses implement ``forward(embeddings, texts)`` and may own learnable
    parameters; the driver adds them to the optimizer.
    """

    name: str = "base"

    def forward(self, embeddings: Tensor, texts: Sequence[str]) -> Tensor:
        """Compute the auxiliary loss.

        Args:
            embeddings: Float tensor of shape ``(n, d)`` with gradient.
            texts: The ``n`` raw texts the embeddings came from, same order.

        Returns:
            A scalar tensor.
        """
        raise NotImplementedError


class TextstatHeadLoss(AuxLoss):
    """Linear probe from the pooled embedding to the six standardized textstat features.

    The head is trained jointly with the encoder; the gradient that reaches the
    encoder rewards embeddings from which surface complexity is linearly
    decodable.

    Args:
        embed_dim: Embedding dimensionality.
        mu: Train-split textstat means.
        sd: Train-split textstat standard deviations.
    """

    name = "textstat_head"

    def __init__(self, embed_dim: int, mu: np.ndarray, sd: np.ndarray) -> None:
        super().__init__()
        self.targets = TextstatTargets(mu, sd)
        self.head = nn.Linear(int(embed_dim), len(FEATURE_NAMES))

    def forward(self, embeddings: Tensor, texts: Sequence[str]) -> Tensor:
        """MSE between the head's prediction and the standardized textstat targets.

        Args:
            embeddings: ``(n, d)`` with gradient.
            texts: ``n`` raw texts.

        Returns:
            Scalar MSE.
        """
        with _no_autocast(embeddings.device):
            pred = self.head(embeddings.float())
            target = self.targets(texts, embeddings.device)
            return F.mse_loss(pred, target)


class DistancePreserveLoss(AuxLoss):
    """MSE between pairwise cosine distances and pairwise textstat distances.

    Args:
        mu: Train-split textstat means.
        sd: Train-split textstat standard deviations.
        ref_scale: See ``scale_reference``.
    """

    name = "dist_preserve"

    def __init__(
        self, mu: np.ndarray, sd: np.ndarray, ref_scale: RefScale = "match_mean"
    ) -> None:
        super().__init__()
        self.targets = TextstatTargets(mu, sd)
        self.ref_scale = ref_scale

    def from_distances(self, D_live: Tensor, D_ref: Tensor) -> Tensor:
        """The loss given both distance matrices.

        Args:
            D_live: ``(n, n)`` live distances with gradient.
            D_ref: ``(n, n)`` reference distances (scaled inside).

        Returns:
            Scalar MSE over the ``i < j`` entries; zero when ``n < 2``.
        """
        n = D_live.shape[0]
        if n < 2:
            return D_live.sum() * 0.0
        D_ref = scale_reference(D_live, D_ref, self.ref_scale)
        iu = torch.triu_indices(n, n, offset=1, device=D_live.device)
        return F.mse_loss(D_live[iu[0], iu[1]], D_ref[iu[0], iu[1]])

    def forward(self, embeddings: Tensor, texts: Sequence[str]) -> Tensor:
        """Build both matrices from the subsample and compare them.

        Args:
            embeddings: ``(n, d)`` with gradient.
            texts: ``n`` raw texts.

        Returns:
            Scalar loss.
        """
        with _no_autocast(embeddings.device):
            D_live = cosine_distance_matrix(embeddings)
            with torch.no_grad():
                D_ref = euclidean_distance_matrix(
                    self.targets(texts, embeddings.device)
                )
            return self.from_distances(D_live, D_ref)


class Persist0H0Loss(AuxLoss):
    """Match the H0 persistence (MST death vector) of the embedding subsample to textstat space.

    Wraps ``persist0.TopoH0Loss(top_k)``: ``D_live`` is the cosine distance
    matrix of the subsample's embeddings, ``D_ref`` the Euclidean distance
    matrix of their standardized textstat vectors (fixed, no gradient). The
    gradient reaches the encoder through the gathered death edges.

    Args:
        mu: Train-split textstat means.
        sd: Train-split textstat standard deviations.
        top_k: Compare only the ``k`` most persistent bars (``None`` = all).
        ref_scale: See ``scale_reference``.
    """

    name = "persist0_h0"

    def __init__(
        self,
        mu: np.ndarray,
        sd: np.ndarray,
        top_k: int | None = 32,
        ref_scale: RefScale = "match_mean",
    ) -> None:
        super().__init__()
        self.targets = TextstatTargets(mu, sd)
        self.topo = TopoH0Loss(top_k=None if top_k is None else int(top_k))
        self.ref_scale = ref_scale

    def from_distances(self, D_live: Tensor, D_ref: Tensor) -> Tensor:
        """The loss given both distance matrices.

        Args:
            D_live: ``(n, n)`` symmetric, zero-diagonal, with gradient.
            D_ref: ``(n, n)`` symmetric, zero-diagonal (scaled inside).

        Returns:
            Scalar loss; zero when ``n < 2``.
        """
        n = D_live.shape[0]
        if n < 2:
            return D_live.sum() * 0.0
        D_ref = scale_reference(D_live, D_ref, self.ref_scale)
        return self.topo(D_live.float().unsqueeze(0), D_ref.float().unsqueeze(0))

    def forward(self, embeddings: Tensor, texts: Sequence[str]) -> Tensor:
        """Build both matrices from the subsample and compare their H0 deaths.

        Args:
            embeddings: ``(n, d)`` with gradient.
            texts: ``n`` raw texts.

        Returns:
            Scalar loss.
        """
        with _no_autocast(embeddings.device):
            D_live = cosine_distance_matrix(embeddings)
            with torch.no_grad():
                D_ref = euclidean_distance_matrix(
                    self.targets(texts, embeddings.device)
                )
            return self.from_distances(D_live, D_ref)


class TopoAEH0Loss(AuxLoss):
    """TopoAE's topological term: distances at each space's own MST edges, both ways.

    The exact form of Moor et al. (2020) in dimension 0, which is all the
    original uses. With ``pi_live`` and ``pi_ref`` the death edges (MST pairs)
    that ``persist0.h0_persistence`` selects in each space::

        L = mse(D_live[pi_ref], D_ref[pi_ref]) + mse(D_live[pi_live], D_ref[pi_live])

    Unlike ``Persist0H0Loss`` this is not invariant to which points a bar
    joins: the two agree on the multiset of MST edge lengths, and only this
    one also asks for the same pairs. Exp 4 runs both for that reason. All
    ``n - 1`` edges are used; the gradient reaches the encoder through the
    gathered entries, as in ``Persist0H0Loss``.

    Args:
        mu: Train-split textstat means.
        sd: Train-split textstat standard deviations.
        ref_scale: See ``scale_reference``.
    """

    name = "topoae_h0"

    def __init__(
        self, mu: np.ndarray, sd: np.ndarray, ref_scale: RefScale = "match_mean"
    ) -> None:
        super().__init__()
        self.targets = TextstatTargets(mu, sd)
        self.ref_scale = ref_scale

    def from_distances(self, D_live: Tensor, D_ref: Tensor) -> Tensor:
        """The loss given both distance matrices.

        Args:
            D_live: ``(n, n)`` symmetric, zero-diagonal, with gradient.
            D_ref: ``(n, n)`` symmetric, zero-diagonal (scaled inside).

        Returns:
            Scalar loss; zero when ``n < 2``.
        """
        n = D_live.shape[0]
        if n < 2:
            return D_live.sum() * 0.0
        D_ref = scale_reference(D_live, D_ref, self.ref_scale)
        live = D_live.float().unsqueeze(0)
        ref = D_ref.float().unsqueeze(0)
        idx_live, live_at_live = h0_persistence(live)
        with torch.no_grad():
            idx_ref, ref_at_ref = h0_persistence(ref)
        il, jl = idx_live[0, :, 0], idx_live[0, :, 1]
        ir, jr = idx_ref[0, :, 0], idx_ref[0, :, 1]
        return F.mse_loss(live[0, ir, jr], ref_at_ref[0]) + F.mse_loss(
            live_at_live[0], ref[0, il, jl]
        )

    def forward(self, embeddings: Tensor, texts: Sequence[str]) -> Tensor:
        """Build both matrices from the subsample and compare them at MST edges.

        Args:
            embeddings: ``(n, d)`` with gradient.
            texts: ``n`` raw texts.

        Returns:
            Scalar loss.
        """
        with _no_autocast(embeddings.device):
            D_live = cosine_distance_matrix(embeddings)
            with torch.no_grad():
                D_ref = euclidean_distance_matrix(
                    self.targets(texts, embeddings.device)
                )
            return self.from_distances(D_live, D_ref)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_aux_loss(
    name: str,
    *,
    embed_dim: int,
    ts_mu: np.ndarray,
    ts_sd: np.ndarray,
    params: dict[str, Any] | None = None,
) -> AuxLoss:
    """Construct an auxiliary loss by name.

    Args:
        name: One of ``AUX_LOSS_NAMES``.
        embed_dim: Embedding dimensionality of the encoder being trained.
        ts_mu: Train-split textstat means, shape ``(6,)``.
        ts_sd: Train-split textstat standard deviations, shape ``(6,)``.
        params: Loss-specific parameters from ``aux_params.<name>`` in the
            config: ``ref_scale`` for ``dist_preserve`` and ``topoae_h0``;
            ``top_k`` and ``ref_scale`` for ``persist0_h0``; none for
            ``textstat_head``.

    Returns:
        An ``AuxLoss`` module.

    Raises:
        ValueError: If ``name`` is not a known auxiliary loss.
    """
    params = dict(params or {})
    if name == "textstat_head":
        return TextstatHeadLoss(embed_dim, ts_mu, ts_sd)
    if name == "dist_preserve":
        return DistancePreserveLoss(
            ts_mu, ts_sd, ref_scale=params.get("ref_scale", "match_mean")
        )
    if name == "persist0_h0":
        return Persist0H0Loss(
            ts_mu,
            ts_sd,
            top_k=params.get("top_k", 32),
            ref_scale=params.get("ref_scale", "match_mean"),
        )
    if name == "topoae_h0":
        return TopoAEH0Loss(
            ts_mu, ts_sd, ref_scale=params.get("ref_scale", "match_mean")
        )
    raise ValueError(f"unknown aux loss {name!r}; expected one of {AUX_LOSS_NAMES}")
