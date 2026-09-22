"""Auxiliary losses: finite, differentiable, and zero at identity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tlf.features import fit_standardizer, textstat_matrix
from tlf.losses import (
    AUX_LOSS_NAMES,
    DistancePreserveLoss,
    Persist0H0Loss,
    TextstatHeadLoss,
    build_aux_loss,
    cosine_distance_matrix,
    scale_reference,
)

N, D = 64, 768


@pytest.fixture(scope="module")
def texts() -> list[str]:
    """64 texts with varied length and vocabulary so textstat targets are not constant."""
    rng = np.random.default_rng(0)
    words = "protein enzyme catalysis membrane transcription kinase ligand cell receptor pathway".split()
    hard = "phosphorylation electrophoresis chromatography immunoprecipitation spectrophotometry".split()
    out = []
    for i in range(N):
        n_sent = 1 + i % 4
        sents = []
        for _ in range(n_sent):
            k = int(rng.integers(4, 18))
            toks = list(rng.choice(words, size=k)) + list(
                rng.choice(hard, size=int(rng.integers(0, 3)))
            )
            sents.append(" ".join(toks).capitalize() + ".")
        out.append(" ".join(sents))
    return out


@pytest.fixture(scope="module")
def stats(texts):
    """Train-style standardizer statistics fitted on the fixture texts."""
    return fit_standardizer(textstat_matrix(texts))


@pytest.fixture
def embeddings() -> torch.Tensor:
    """Random [64, 768] embeddings requiring gradient."""
    torch.manual_seed(0)
    return torch.randn(N, D, requires_grad=True)


@pytest.mark.parametrize("name", AUX_LOSS_NAMES)
def test_loss_is_finite_with_nonzero_gradient(name, embeddings, texts, stats):
    """Every loss is finite on random [64, 768] input and back-propagates a non-zero gradient."""
    mu, sd = stats
    loss_fn = build_aux_loss(name, embed_dim=D, ts_mu=mu, ts_sd=sd, params={})
    value = loss_fn(embeddings, texts)
    assert value.ndim == 0 and torch.isfinite(value)
    value.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_textstat_head_parameters_receive_gradient(embeddings, texts, stats):
    """The linear head is trainable: its weight gets a gradient."""
    mu, sd = stats
    loss_fn = TextstatHeadLoss(D, mu, sd)
    loss_fn(embeddings, texts).backward()
    assert (
        loss_fn.head.weight.grad is not None
        and loss_fn.head.weight.grad.abs().sum() > 0
    )


def _random_distance_matrix(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 5, generator=g)
    Dm = torch.cdist(X, X)
    return 0.5 * (Dm + Dm.T)


@pytest.mark.parametrize("ref_scale", ["match_mean", "none", 1.0])
def test_persist0_h0_is_zero_when_live_equals_ref(ref_scale, stats):
    """Identical distance matrices give exactly zero H0 loss under every scaling policy."""
    mu, sd = stats
    loss_fn = Persist0H0Loss(mu, sd, top_k=32, ref_scale=ref_scale)
    Dm = _random_distance_matrix(N, 1)
    live = Dm.clone().requires_grad_(True)
    value = loss_fn.from_distances(live, Dm)
    assert float(value) == 0.0


@pytest.mark.parametrize("ref_scale", ["match_mean", "none", 1.0])
def test_dist_preserve_is_zero_when_live_equals_ref(ref_scale, stats):
    """Identical distance matrices give exactly zero distance-preservation loss."""
    mu, sd = stats
    loss_fn = DistancePreserveLoss(mu, sd, ref_scale=ref_scale)
    Dm = _random_distance_matrix(N, 2)
    assert float(loss_fn.from_distances(Dm.clone().requires_grad_(True), Dm)) == 0.0


def test_persist0_h0_positive_and_differentiable_when_matrices_differ(stats):
    """Different matrices give a positive loss with gradient on the live matrix."""
    mu, sd = stats
    loss_fn = Persist0H0Loss(mu, sd, top_k=16, ref_scale="none")
    live = _random_distance_matrix(N, 3).requires_grad_(True)
    ref = _random_distance_matrix(N, 4)
    value = loss_fn.from_distances(live, ref)
    assert float(value) > 0
    value.backward()
    assert live.grad is not None and live.grad.abs().sum() > 0
    # gradient touches only MST death edges: at most 2*(N-1) non-zero entries (symmetric)
    assert int((live.grad != 0).sum()) <= 2 * (N - 1)


def test_persist0_h0_decreases_under_optimisation(texts, stats):
    """Optimising free embeddings against a fixed textstat reference lowers the loss."""
    mu, sd = stats
    loss_fn = Persist0H0Loss(mu, sd, top_k=32)
    torch.manual_seed(0)
    E = torch.nn.Parameter(torch.randn(N, 32))
    opt = torch.optim.Adam([E], lr=0.05)
    first = float(loss_fn(E, texts))
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(E, texts)
        loss.backward()
        opt.step()
    last = float(loss_fn(E, texts))
    assert last < first


def test_scale_reference_match_mean_matches_means():
    """match_mean makes the reference's mean off-diagonal distance equal the live one."""
    live = _random_distance_matrix(20, 5)
    ref = _random_distance_matrix(20, 6) * 7.0
    scaled = scale_reference(live, ref, "match_mean")
    mask = ~torch.eye(20, dtype=torch.bool)
    assert torch.isclose(scaled[mask].mean(), live[mask].mean())
    assert torch.allclose(scale_reference(live, ref, "none"), ref)
    assert torch.allclose(scale_reference(live, ref, 0.5), ref * 0.5)
    with pytest.raises(ValueError):
        scale_reference(live, ref, "bogus")


def test_cosine_distance_matrix_properties(embeddings):
    """Cosine distance matrix is symmetric, zero on the diagonal, and in [0, 2]."""
    Dm = cosine_distance_matrix(embeddings)
    assert torch.allclose(Dm, Dm.T)
    assert torch.all(Dm.diagonal() == 0)
    assert float(Dm.min()) >= 0 and float(Dm.max()) <= 2.0 + 1e-6


def test_build_aux_loss_rejects_unknown_name(stats):
    """An unknown name is a ValueError, not a silent no-op."""
    mu, sd = stats
    with pytest.raises(ValueError):
        build_aux_loss("nope", embed_dim=D, ts_mu=mu, ts_sd=sd)


def test_losses_run_under_bf16_autocast_on_cpu(embeddings, texts, stats):
    """Inside an autocast region the losses still compute in float32 and stay finite."""
    mu, sd = stats
    for name in AUX_LOSS_NAMES:
        loss_fn = build_aux_loss(name, embed_dim=D, ts_mu=mu, ts_sd=sd)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            value = loss_fn(embeddings, texts)
        assert value.dtype == torch.float32 and torch.isfinite(value)
