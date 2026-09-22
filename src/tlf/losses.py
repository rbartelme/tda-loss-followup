"""Auxiliary losses for Exp 4.

The three concrete losses (``textstat_head``, ``dist_preserve``,
``persist0_h0``) arrive in Task 4. This module already fixes the contract the
training driver relies on::

    aux = build_aux_loss(name, embed_dim=768, ts_mu=mu, ts_sd=sd, params={...})
    value = aux(embeddings, texts)   # embeddings [n, d], texts of length n -> scalar

``ts_mu`` / ``ts_sd`` are the train-split textstat standardizer statistics, so
every auxiliary sees the same feature scale the pair builder used.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from torch import Tensor, nn

AUX_LOSS_NAMES: tuple[str, ...] = ("textstat_head", "dist_preserve", "persist0_h0")


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
        params: Loss-specific parameters from ``aux_params.<name>`` in the config.

    Returns:
        An ``AuxLoss`` module.

    Raises:
        ValueError: If ``name`` is not a known auxiliary loss.
        NotImplementedError: Until Task 4 lands the concrete classes.
    """
    if name not in AUX_LOSS_NAMES:
        raise ValueError(f"unknown aux loss {name!r}; expected one of {AUX_LOSS_NAMES}")
    raise NotImplementedError(f"aux loss {name!r} is implemented in Task 4")
