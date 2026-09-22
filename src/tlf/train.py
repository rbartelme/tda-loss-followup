"""Fine-tune driver: sentence-transformers losses, step-fraction checkpoints, aux hook.

One ``run_finetune`` call trains one encoder with one loss on one pair set and
writes ``out_dir/ckpt_<frac>/`` sentence-transformers checkpoints (each with a
``manifest.json``) plus ``out_dir/train_log.csv``.

Loss paths:

* ``mnrl``: ``CachedMultipleNegativesRankingLoss(scale=1/tau)``, subclassed so
  the auxiliary term is back-propagated into the same cached embedding
  gradients, and so the anchor/positive embeddings are available for the
  cosine log without a second forward pass.
* ``triplet`` / ``cosent``: the sentence-transformers losses, driven through
  their ``compute_loss_from_embeddings`` API in micro-batches with gradient
  accumulation. Negatives are drawn uniformly from the train split each epoch.
* ``mlm``: a plain HF masked-LM loop over the unique texts of the pair set;
  checkpoints save the encoder wrapped with mean pooling.

Run ``python -m tlf.train --dry-run`` for a three-step smoke test on MiniLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer import modules as st_modules
from sentence_transformers.util import batch_to_device
from torch import Tensor
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from tlf.config import load_config
from tlf.data import (
    build_pairs,
    corpus_textstat,
    eval_split,
    load_scicueval,
    sha256_file,
    train_split,
)
from tlf.features import fit_standardizer
from tlf.losses import AuxLoss, build_aux_loss
from tlf.results import write_manifest

log = logging.getLogger(__name__)

LOSSES: tuple[str, ...] = ("mnrl", "triplet", "cosent", "mlm")
LOG_COLUMNS: tuple[str, ...] = (
    "step",
    "epoch",
    "lr",
    "loss",
    "base_loss",
    "aux_loss",
    "pos_cos",
    "neg_cos",
)

# ---------------------------------------------------------------------------
# Small pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def frac_tag(frac: float) -> str:
    """Format a checkpoint fraction for a directory name.

    Args:
        frac: Fraction of total steps in ``[0, 1]``.

    Returns:
        ``"0.0"``, ``"0.1"``, ``"0.25"``, ``"0.5"``, ``"1.0"`` and so on.
    """
    s = f"{frac:.2f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def checkpoint_steps(fracs: Sequence[float], total_steps: int) -> dict[float, int]:
    """Map checkpoint fractions to optimizer-step indices.

    Args:
        fracs: Fractions of training, ``0.0`` meaning the untrained model.
        total_steps: Number of optimizer steps in the run.

    Returns:
        Fraction to step, where step ``0`` is before any update.
    """
    return {float(f): int(round(float(f) * total_steps)) for f in fracs}


def no_duplicate_batches(
    anchor_ids: Sequence[str],
    positive_ids: Sequence[str],
    batch_size: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Shuffle pairs into batches in which no text id appears twice.

    A duplicated anchor or positive inside a batch would be a false in-batch
    negative, so a pair whose ids already occur in the batch under construction
    is deferred to a later batch. Mirrors sentence-transformers'
    ``NoDuplicatesBatchSampler`` but keyed on ids rather than text values.
    Incomplete trailing batches are dropped.

    Args:
        anchor_ids: Anchor id per pair.
        positive_ids: Positive id per pair.
        batch_size: Pairs per batch.
        rng: RNG for the initial permutation.

    Returns:
        A list of batches, each a list of pair indices.
    """
    remaining = rng.permutation(len(anchor_ids)).tolist()
    batches: list[list[int]] = []
    while len(remaining) >= batch_size:
        batch: list[int] = []
        used: set[str] = set()
        leftover: list[int] = []
        for i in remaining:
            a, p = anchor_ids[i], positive_ids[i]
            if len(batch) == batch_size or a in used or p in used:
                leftover.append(i)
                continue
            batch.append(i)
            used.add(a)
            used.add(p)
        if len(batch) < batch_size:
            break
        batches.append(batch)
        remaining = leftover
    return batches


@torch.no_grad()
def cosine_stats(anchors: Tensor, positives: Tensor) -> tuple[float, float]:
    """Mean positive-pair cosine and mean in-batch negative cosine.

    Args:
        anchors: ``(n, d)`` anchor embeddings.
        positives: ``(n, d)`` positive embeddings, row-aligned.

    Returns:
        ``(pos_cos, neg_cos)``; ``neg_cos`` is NaN when ``n < 2``.
    """
    a = F.normalize(anchors.detach().float(), dim=-1)
    p = F.normalize(positives.detach().float(), dim=-1)
    sims = a @ p.T
    n = sims.shape[0]
    pos = float(sims.diagonal().mean()) if n else float("nan")
    neg = (
        float((sims.sum() - sims.diagonal().sum()) / (n * (n - 1)))
        if n > 1
        else float("nan")
    )
    return pos, neg


# ---------------------------------------------------------------------------
# Config-facing helpers
# ---------------------------------------------------------------------------


def resolve_device(spec: str) -> torch.device:
    """Turn ``"auto"``, ``"cuda"``, ``"cuda:0"`` or ``"cpu"`` into a device.

    Args:
        spec: Device string from the config or CLI.

    Returns:
        ``cuda`` when requested or available under ``auto``, else ``cpu``.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def aux_schedule(cfg: dict[str, Any]) -> dict[str, Any]:
    """The auxiliary-loss schedule section of a config.

    Reads ``aux_schedule`` (``subsample``, ``every``, ``lr``). For backward
    compatibility a dict-valued ``aux`` is accepted too; a list-valued ``aux``
    is an experiment grid and is ignored here.

    Args:
        cfg: Resolved config.

    Returns:
        The schedule dict, possibly empty.
    """
    sched = cfg.get("aux_schedule")
    if isinstance(sched, dict):
        return sched
    legacy = cfg.get("aux")
    return legacy if isinstance(legacy, dict) else {}


def load_model_roster(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read ``models.yaml`` named by ``paths.models_yaml``.

    Args:
        cfg: Resolved config.

    Returns:
        Model key to its entry (``hf_id``, ``pooling``, ``dim``, ...).
    """
    with Path(cfg["paths"]["models_yaml"]).open() as f:
        return yaml.safe_load(f)


def build_sentence_transformer(
    hf_id: str, max_length: int, device: torch.device
) -> SentenceTransformer:
    """Wrap an HF encoder with mean pooling, no prefixes, no normalisation.

    Args:
        hf_id: HuggingFace model id or a local directory.
        max_length: Tokenizer truncation length.
        device: Where to place the model.

    Returns:
        A two-module ``SentenceTransformer`` (Transformer, Pooling).
    """
    transformer = st_modules.Transformer(hf_id, max_seq_length=max_length)
    get_dim = getattr(transformer, "get_embedding_dimension", None)
    dim = (
        int(get_dim())
        if callable(get_dim)
        else int(transformer.auto_model.config.hidden_size)
    )
    pooling = st_modules.Pooling(dim, pooling_mode="mean")
    return SentenceTransformer(modules=[transformer, pooling], device=str(device))


def make_optimizer(
    model_params: list[tuple[str, torch.nn.Parameter]],
    lr: float,
    weight_decay: float,
    aux_params: list[torch.nn.Parameter] | None = None,
    aux_lr: float | None = None,
) -> torch.optim.AdamW:
    """AdamW with HF-style no-decay groups for biases and LayerNorm weights.

    Args:
        model_params: ``named_parameters()`` of the encoder.
        lr: Peak learning rate.
        weight_decay: Decay for the decayed group.
        aux_params: Parameters of the auxiliary loss, if it has any.
        aux_lr: Learning rate for ``aux_params``; defaults to ``lr``.

    Returns:
        The optimizer.
    """
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay = [
        p
        for n, p in model_params
        if p.requires_grad and not any(k in n for k in no_decay)
    ]
    nodecay = [
        p for n, p in model_params if p.requires_grad and any(k in n for k in no_decay)
    ]
    groups: list[dict[str, Any]] = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    if aux_params:
        groups.append(
            {"params": list(aux_params), "weight_decay": 0.0, "lr": aux_lr or lr}
        )
    return torch.optim.AdamW(groups, lr=lr)


# ---------------------------------------------------------------------------
# Pair data
# ---------------------------------------------------------------------------


@dataclass
class PairData:
    """Pairs plus the text lookup and train pool the loss paths need."""

    anchor_ids: np.ndarray
    positive_ids: np.ndarray
    text_of: dict[str, str]
    train_ids: np.ndarray
    ts_mu: np.ndarray
    ts_sd: np.ndarray
    source: str
    source_sha256: str | None = None

    @property
    def n(self) -> int:
        """Number of pairs."""
        return int(len(self.anchor_ids))

    def texts(self, ids: Sequence[str]) -> list[str]:
        """Look up texts for ids, preserving order."""
        return [self.text_of[i] for i in ids]


def load_pair_data(
    cfg: dict[str, Any],
    pair_path: Path | None,
    *,
    seed: int,
    n_limit: int | None = None,
) -> PairData:
    """Load a pair set and join it to the SciCUEval train texts.

    Args:
        cfg: Resolved config.
        pair_path: ``pairs.parquet`` written by ``scripts/build_pairs.py``.
            ``None`` builds ``n_limit`` random pairs in memory (dry runs).
        seed: Seed for in-memory pair construction.
        n_limit: Keep only the first ``n_limit`` pairs.

    Returns:
        The pair data, with textstat standardizer statistics taken from the
        pair set's manifest when present and fitted on the train split otherwise.

    Raises:
        ValueError: If any pair touches an eval id.
    """
    corpus = load_scicueval(cfg)
    train = train_split(corpus)
    text_of = dict(zip(corpus["id"], corpus["text"], strict=True))
    eval_ids = set(eval_split(corpus)["id"])
    mu = sd = None
    sha = None
    if pair_path is not None:
        pairs = pd.read_parquet(pair_path)
        sha = sha256_file(pair_path)
        manifest = pair_path.parent / "manifest.json"
        if manifest.is_file():
            m = json.loads(manifest.read_text())
            if "standardizer_mu" in m and "standardizer_sd" in m:
                mu = np.asarray(m["standardizer_mu"], dtype=np.float64)
                sd = np.asarray(m["standardizer_sd"], dtype=np.float64)
        source = str(pair_path)
    else:
        n = n_limit or 64
        X = corpus_textstat(corpus, cfg, "scicueval")
        X_train = X[np.flatnonzero((corpus["split"] == "train").to_numpy())]
        pairs = build_pairs(train, "random", n, seed, feature_matrix=X_train)
        mu, sd = fit_standardizer(X_train)
        source = f"in-memory random pairs (n={n}, seed={seed})"
    if n_limit is not None:
        pairs = pairs.head(n_limit)
    if mu is None or sd is None:
        X = corpus_textstat(corpus, cfg, "scicueval")
        mu, sd = fit_standardizer(
            X[np.flatnonzero((corpus["split"] == "train").to_numpy())]
        )
    a = pairs["anchor_id"].astype(str).to_numpy()
    p = pairs["positive_id"].astype(str).to_numpy()
    if eval_ids & (set(a) | set(p)):
        raise ValueError(f"pair set {source} touches eval ids")
    return PairData(
        anchor_ids=a,
        positive_ids=p,
        text_of=text_of,
        train_ids=train["id"].astype(str).to_numpy(),
        ts_mu=mu,
        ts_sd=sd,
        source=source,
        source_sha256=sha,
    )


# ---------------------------------------------------------------------------
# Aux hook
# ---------------------------------------------------------------------------


@dataclass
class AuxHook:
    """Auxiliary-loss schedule: ``lam * module(subsample)`` every ``every`` steps."""

    module: AuxLoss | None = None
    lam: float = 0.0
    subsample: int = 64
    every: int = 1
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    @property
    def active(self) -> bool:
        """Whether an auxiliary term is configured at all."""
        return self.module is not None and self.lam != 0.0

    def term(
        self, step: int, embeddings: Tensor, texts: Sequence[str]
    ) -> Tensor | None:
        """The weighted auxiliary term for this step, or ``None`` when not due.

        Args:
            step: Zero-based optimizer step.
            embeddings: Candidate embeddings ``(m, d)`` with gradient.
            texts: Their texts.

        Returns:
            ``lam * module(sub, sub_texts)`` on a random subsample of size
            ``min(subsample, m)``, or ``None``.
        """
        if not self.active or step % self.every != 0:
            return None
        m = embeddings.shape[0]
        k = min(self.subsample, m)
        idx = np.sort(self.rng.choice(m, size=k, replace=False))
        sub = embeddings[torch.as_tensor(idx, device=embeddings.device)]
        assert self.module is not None
        return self.lam * self.module(sub, [texts[i] for i in idx])


class HookedCachedMNRL(losses.CachedMultipleNegativesRankingLoss):
    """Gradient-cached MNRL that adds an auxiliary term and exposes embeddings.

    ``calculate_loss`` runs after the mini-batched, no-grad embedding pass, on
    detached leaf embeddings whose ``.grad`` becomes the gradient cache. Adding
    ``lam * aux`` here and back-propagating it accumulates into that same cache,
    so the re-forward pass propagates base and auxiliary gradients together.
    The concatenated anchor and positive embeddings are also read here for the
    cosine log at no extra forward cost.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        *,
        scale: float,
        mini_batch_size: int,
        aux: AuxHook,
    ) -> None:
        """Construct the loss.

        Args:
            model: The sentence-transformer being trained.
            scale: ``1 / tau``.
            mini_batch_size: Rows per embedding mini-batch (memory knob only).
            aux: The auxiliary schedule; may be inactive.
        """
        super().__init__(model, scale=scale, mini_batch_size=mini_batch_size)
        self.aux = aux
        self.step = 0
        self.texts: list[str] = []
        self.last: dict[str, float] = {}

    def calculate_loss(
        self,
        reps: list[list[Tensor]],
        labels: Tensor | None = None,
        *,
        with_backward: bool = False,
    ) -> Tensor:
        """Base InfoNCE plus the auxiliary term, recording stats in ``self.last``.

        Args:
            reps: Per-column lists of mini-batch embeddings (anchors, positives).
            labels: Unused by MNRL.
            with_backward: Back-propagate into ``reps`` and return a detached total.

        Returns:
            The total loss.
        """
        base = super().calculate_loss(reps, labels, with_backward=with_backward)
        anchors = torch.cat(reps[0])
        positives = torch.cat(reps[1])
        pos_cos, neg_cos = cosine_stats(anchors, positives)
        total = base
        aux_val = float("nan")
        term = self.aux.term(self.step, torch.cat([anchors, positives]), self.texts)
        if term is not None:
            if with_backward:
                term.backward()
                term = term.detach()
            aux_val = float(term)
            total = base + term
        self.last = {
            "base_loss": float(base),
            "aux_loss": aux_val,
            "pos_cos": pos_cos,
            "neg_cos": neg_cos,
        }
        return total


# ---------------------------------------------------------------------------
# Shared loop
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    """Everything that identifies one fine-tuning run (goes into manifests)."""

    model_key: str
    hf_id: str
    loss: str
    tau: float
    aux: str | None
    lam: float
    seed: int
    pair_source: str
    pair_sha256: str | None
    n_pairs: int
    device: str
    dry_run: bool = False


def _autocast(device: torch.device, bf16: bool):
    """Autocast context for bf16 mixed precision on the given device."""
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=bool(bf16)
    )


def _train_loop(
    *,
    plan: list[list[Any]],
    step_fn: Callable[[Any, int], dict[str, float]],
    save_fn: Callable[[Path], None],
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
    clip_params: list[torch.nn.Parameter],
    cfg: dict[str, Any],
    spec: RunSpec,
    out_dir: Path,
    max_steps: int | None,
) -> dict[str, Any]:
    """Run optimizer steps over ``plan`` with warmup, checkpoints and logging.

    Args:
        plan: Per-epoch lists of batches; each batch is passed to ``step_fn``.
        step_fn: Performs forward and backward for one batch and returns a
            stats dict with ``loss`` and optionally ``base_loss``, ``aux_loss``,
            ``pos_cos``, ``neg_cos``.
        save_fn: Writes a checkpoint into the given directory.
        optimizer: Optimizer over the model (and aux) parameters.
        max_grad_norm: Gradient clipping threshold; ``<= 0`` disables.
        clip_params: Parameters to clip.
        cfg: Resolved config (``train`` section is read).
        spec: Run identity for manifests.
        out_dir: Run directory.
        max_steps: Truncate the run to this many steps (dry runs).

    Returns:
        Summary with checkpoint directories, steps and wall time.
    """
    tr = cfg["train"]
    total_steps = sum(len(b) for b in plan)
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))
    if total_steps == 0:
        raise ValueError("no training steps: fewer pairs than one batch?")
    warmup = int(round(float(tr["warmup_ratio"]) * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    ckpts = checkpoint_steps(tr["checkpoint_fracs"], total_steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    def save_due(step: int) -> None:
        for frac, s in ckpts.items():
            if s != step:
                continue
            d = out_dir / f"ckpt_{frac_tag(frac)}"
            if d.exists():
                shutil.rmtree(d)
            save_fn(d)
            write_manifest(
                d,
                config_path=cfg["_config_path"],
                seed=spec.seed,
                extra={
                    "artifact": "checkpoint",
                    **spec.__dict__,
                    "ckpt_frac": frac,
                    "step": step,
                    "total_steps": total_steps,
                    "train": tr,
                    "aux_schedule": aux_schedule(cfg),
                },
            )
            saved[frac_tag(frac)] = str(d)
            log.info("saved %s at step %d/%d", d.name, step, total_steps)

    write_manifest(
        out_dir,
        config_path=cfg["_config_path"],
        seed=spec.seed,
        extra={
            "artifact": "run",
            "status": "running",
            **spec.__dict__,
            "total_steps": total_steps,
        },
    )
    save_due(0)
    t0 = time.time()
    step = 0
    print_every = int(tr.get("print_every", 10))
    with (out_dir / "train_log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        done = False
        for epoch, batches in enumerate(plan):
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                stats = step_fn(batch, step)
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                step += 1
                row = {
                    "step": step,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "loss": stats.get("loss", float("nan")),
                    "base_loss": stats.get("base_loss", float("nan")),
                    "aux_loss": stats.get("aux_loss", float("nan")),
                    "pos_cos": stats.get("pos_cos", float("nan")),
                    "neg_cos": stats.get("neg_cos", float("nan")),
                }
                writer.writerow(row)
                f.flush()
                if step % print_every == 0 or step == total_steps:
                    log.info(
                        "step %d/%d loss=%.4f pos=%.3f neg=%.3f aux=%.4f lr=%.2e",
                        step,
                        total_steps,
                        row["loss"],
                        row["pos_cos"],
                        row["neg_cos"],
                        row["aux_loss"],
                        row["lr"],
                    )
                save_due(step)
                if step >= total_steps:
                    done = True
                    break
            if done:
                break
    summary = {
        "status": "complete",
        "steps": step,
        "total_steps": total_steps,
        "wall_seconds": round(time.time() - t0, 1),
        "checkpoints": saved,
    }
    write_manifest(
        out_dir,
        config_path=cfg["_config_path"],
        seed=spec.seed,
        extra={"artifact": "run", **spec.__dict__, **summary},
    )
    return summary


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------


def _embed(
    model: SentenceTransformer, texts: Sequence[str], device: torch.device
) -> Tensor:
    """Preprocess (tokenize) and embed texts with gradient."""
    features = batch_to_device(model.preprocess(list(texts)), device)
    return model(features)["sentence_embedding"]


def _pair_step_cached(
    model: SentenceTransformer,
    loss_obj: HookedCachedMNRL,
    data: PairData,
    device: torch.device,
    bf16: bool,
) -> Callable[[list[int], int], dict[str, float]]:
    """Build the MNRL step function."""

    def step_fn(batch: list[int], step: int) -> dict[str, float]:
        a_texts = data.texts(data.anchor_ids[batch])
        p_texts = data.texts(data.positive_ids[batch])
        loss_obj.step = step
        loss_obj.texts = a_texts + p_texts
        with _autocast(device, bf16):
            fa = batch_to_device(model.preprocess(a_texts), device)
            fp = batch_to_device(model.preprocess(p_texts), device)
            loss = loss_obj([fa, fp], labels=None)
            loss.backward()
        return {"loss": float(loss), **loss_obj.last}

    return step_fn


def _pair_step_uncached(
    model: SentenceTransformer,
    loss_obj: losses.TripletLoss | losses.CoSENTLoss,
    kind: str,
    data: PairData,
    device: torch.device,
    bf16: bool,
    micro: int,
    aux: AuxHook,
    rng: np.random.Generator,
) -> Callable[[list[int], int], dict[str, float]]:
    """Build the triplet / CoSENT step function (micro-batched accumulation).

    Each pair gets a negative drawn uniformly from the train split, excluding
    its own anchor and positive, so the negative distribution matches MNRL's
    in-batch negatives (mostly other subsets). CoSENT ranks the (anchor,
    positive) pair with label 1 above the (anchor, negative) pair with label 0
    within each micro-batch. When an auxiliary is active it is applied per
    micro-batch on the anchor+positive embeddings of that micro-batch.
    """

    def step_fn(batch: list[int], step: int) -> dict[str, float]:
        n = len(batch)
        a_ids = data.anchor_ids[batch]
        p_ids = data.positive_ids[batch]
        neg = rng.choice(data.train_ids, size=n)
        for i in range(n):
            while neg[i] == a_ids[i] or neg[i] == p_ids[i]:
                neg[i] = rng.choice(data.train_ids)
        a_all = data.texts(a_ids)
        p_all = data.texts(p_ids)
        n_all = data.texts(neg)
        total = 0.0
        base_total = 0.0
        aux_total = 0.0
        aux_seen = False
        A_log: list[Tensor] = []
        P_log: list[Tensor] = []
        for s in range(0, n, micro):
            e = min(s + micro, n)
            w = (e - s) / n
            with _autocast(device, bf16):
                A = _embed(model, a_all[s:e], device)
                P = _embed(model, p_all[s:e], device)
                N = _embed(model, n_all[s:e], device)
                if kind == "triplet":
                    base = loss_obj.compute_loss_from_embeddings([A, P, N], labels=None)
                else:
                    e1 = torch.cat([A, A])
                    e2 = torch.cat([P, N])
                    labels = torch.cat(
                        [
                            torch.ones(e - s, device=device),
                            torch.zeros(e - s, device=device),
                        ]
                    )
                    base = loss_obj.compute_loss_from_embeddings([e1, e2], labels)
                loss = base * w
                term = aux.term(step, torch.cat([A, P]), a_all[s:e] + p_all[s:e])
                if term is not None:
                    loss = loss + term * w
                    aux_total += float(term) * w
                    aux_seen = True
                loss.backward()
            total += float(loss)
            base_total += float(base) * w
            A_log.append(A.detach())
            P_log.append(P.detach())
        pos_cos, neg_cos = cosine_stats(torch.cat(A_log), torch.cat(P_log))
        return {
            "loss": total,
            "base_loss": base_total,
            "aux_loss": aux_total if aux_seen else float("nan"),
            "pos_cos": pos_cos,
            "neg_cos": neg_cos,
        }

    return step_fn


def _mlm_step(
    mlm: torch.nn.Module,
    tokenizer: Any,
    collator: DataCollatorForLanguageModeling,
    texts: list[str],
    device: torch.device,
    bf16: bool,
    micro: int,
    max_length: int,
) -> Callable[[list[int], int], dict[str, float]]:
    """Build the masked-LM step function (micro-batched accumulation)."""

    def step_fn(batch: list[int], step: int) -> dict[str, float]:
        n = len(batch)
        total = 0.0
        for s in range(0, n, micro):
            e = min(s + micro, n)
            w = (e - s) / n
            enc = tokenizer(
                [texts[i] for i in batch[s:e]],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs, labels = collator.torch_mask_tokens(enc["input_ids"].clone())
            with _autocast(device, bf16):
                out = mlm(
                    input_ids=inputs.to(device),
                    attention_mask=enc["attention_mask"].to(device),
                    labels=labels.to(device),
                )
                loss = out.loss * w
                loss.backward()
            total += float(loss)
        return {"loss": total, "base_loss": total}

    return step_fn


def _save_mlm_encoder_as_st(
    mlm: torch.nn.Module, tokenizer: Any, ckpt_dir: Path, max_length: int
) -> None:
    """Save the MLM's encoder (without its head) as a mean-pooled ST checkpoint."""
    tmp = ckpt_dir.parent / f"_{ckpt_dir.name}_hf"
    if tmp.exists():
        shutil.rmtree(tmp)
    mlm.base_model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    st = build_sentence_transformer(str(tmp), max_length, torch.device("cpu"))
    st.save(str(ckpt_dir), create_model_card=False)
    shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_finetune(
    model_key: str,
    loss: str,
    tau: float,
    pair_path: Path | str | None,
    cfg: dict[str, Any],
    seed: int,
    out_dir: Path | str,
    *,
    aux: str | None = None,
    lam: float = 0.0,
    max_steps: int | None = None,
    n_pairs_limit: int | None = None,
) -> dict[str, Any]:
    """Fine-tune one encoder with one loss on one pair set.

    Args:
        model_key: Key in ``models.yaml``.
        loss: One of ``LOSSES``.
        tau: Temperature for ``mnrl`` (``scale = 1 / tau``). Ignored otherwise.
        pair_path: ``pairs.parquet`` to train on; ``None`` builds a small
            in-memory random pair set (dry runs).
        cfg: Resolved config (``train``, ``aux``, ``paths``, ``loss_params``).
        seed: Seed for torch, numpy, batch order, negatives and aux subsampling.
        out_dir: Run directory; checkpoints go to ``out_dir/ckpt_<frac>/``.
        aux: Auxiliary loss name for Exp 4, or ``None``.
        lam: Weight of the auxiliary term.
        max_steps: Truncate training (dry runs).
        n_pairs_limit: Use only the first ``n_pairs_limit`` pairs (dry runs).

    Returns:
        The run summary written to ``out_dir/manifest.json``.

    Raises:
        ValueError: On an unknown loss or model key.
    """
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
    roster = load_model_roster(cfg)
    if model_key not in roster:
        raise ValueError(f"unknown model key {model_key!r}; have {sorted(roster)}")
    hf_id = roster[model_key]["hf_id"]
    tr = cfg["train"]
    out_dir = Path(out_dir)
    device = resolve_device(str(tr.get("device", "auto")))
    bf16 = bool(tr.get("bf16", True))
    max_length = int(tr["max_length"])
    batch_size = int(tr["batch_size"])
    micro = int(tr.get("micro_batch", 32))
    epochs = int(tr["epochs"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    data = load_pair_data(
        cfg, Path(pair_path) if pair_path else None, seed=seed, n_limit=n_pairs_limit
    )
    spec = RunSpec(
        model_key=model_key,
        hf_id=hf_id,
        loss=loss,
        tau=float(tau),
        aux=aux,
        lam=float(lam),
        seed=int(seed),
        pair_source=data.source,
        pair_sha256=data.source_sha256,
        n_pairs=data.n,
        device=str(device),
        dry_run=max_steps is not None,
    )
    log.info(
        "run: %s (%s) loss=%s tau=%s aux=%s lam=%s pairs=%d device=%s",
        model_key,
        hf_id,
        loss,
        tau,
        aux,
        lam,
        data.n,
        device,
    )

    if loss == "mlm":
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        mlm = AutoModelForMaskedLM.from_pretrained(hf_id).to(device)
        mlm.train()
        collator = DataCollatorForLanguageModeling(
            tokenizer, mlm_probability=float(tr.get("mlm_probability", 0.15))
        )
        texts = sorted(
            set(data.texts(data.anchor_ids)) | set(data.texts(data.positive_ids))
        )
        plan = []
        for _ in range(epochs):
            perm = rng.permutation(len(texts)).tolist()
            n_full = len(perm) // batch_size
            plan.append(
                [perm[i * batch_size : (i + 1) * batch_size] for i in range(n_full)]
            )
        optimizer = make_optimizer(
            list(mlm.named_parameters()), float(tr["lr"]), float(tr["weight_decay"])
        )
        return _train_loop(
            plan=plan,
            step_fn=_mlm_step(
                mlm, tokenizer, collator, texts, device, bf16, micro, max_length
            ),
            save_fn=lambda d: _save_mlm_encoder_as_st(mlm, tokenizer, d, max_length),
            optimizer=optimizer,
            max_grad_norm=float(tr.get("max_grad_norm", 1.0)),
            clip_params=[p for p in mlm.parameters() if p.requires_grad],
            cfg=cfg,
            spec=spec,
            out_dir=out_dir,
            max_steps=max_steps,
        )

    model = build_sentence_transformer(hf_id, max_length, device)
    model.train()
    embed_dim = int(model.get_embedding_dimension() or 0)

    aux_cfg = aux_schedule(cfg)
    hook = AuxHook(
        subsample=int(aux_cfg.get("subsample", 64)),
        every=int(aux_cfg.get("every", 1)),
        lam=float(lam),
        rng=np.random.default_rng(seed + 1),
    )
    aux_params: list[torch.nn.Parameter] = []
    if aux is not None and lam != 0.0:
        params = (cfg.get("aux_params") or {}).get(aux, {}) or {}
        hook.module = build_aux_loss(
            aux, embed_dim=embed_dim, ts_mu=data.ts_mu, ts_sd=data.ts_sd, params=params
        ).to(device)
        aux_params = [p for p in hook.module.parameters() if p.requires_grad]

    plan = [
        no_duplicate_batches(data.anchor_ids, data.positive_ids, batch_size, rng)
        for _ in range(epochs)
    ]
    optimizer = make_optimizer(
        list(model.named_parameters()),
        float(tr["lr"]),
        float(tr["weight_decay"]),
        aux_params=aux_params,
        aux_lr=aux_cfg.get("lr"),
    )
    clip_params = [p for p in model.parameters() if p.requires_grad] + aux_params

    loss_params = cfg.get("loss_params") or {}
    if loss == "mnrl":
        loss_obj = HookedCachedMNRL(
            model, scale=1.0 / float(tau), mini_batch_size=micro, aux=hook
        )
        step_fn = _pair_step_cached(model, loss_obj, data, device, bf16)
    else:
        if loss == "triplet":
            lp = loss_params.get("triplet", {}) or {}
            metric = getattr(
                losses.TripletDistanceMetric,
                str(lp.get("distance", "euclidean")).upper(),
            )
            loss_obj = losses.TripletLoss(
                model,
                distance_metric=metric,
                triplet_margin=float(lp.get("margin", 5.0)),
            )
        else:
            lp = loss_params.get("cosent", {}) or {}
            loss_obj = losses.CoSENTLoss(model, scale=float(lp.get("scale", 20.0)))
        step_fn = _pair_step_uncached(
            model,
            loss_obj,
            loss,
            data,
            device,
            bf16,
            micro,
            hook,
            np.random.default_rng(seed + 2),
        )

    return _train_loop(
        plan=plan,
        step_fn=step_fn,
        save_fn=lambda d: model.save(str(d), create_model_card=False),
        optimizer=optimizer,
        max_grad_norm=float(tr.get("max_grad_norm", 1.0)),
        clip_params=clip_params,
        cfg=cfg,
        spec=spec,
        out_dir=out_dir,
        max_steps=max_steps,
    )


def main() -> None:
    """CLI: fine-tune one run, or ``--dry-run`` three steps of MiniLM on 64 pairs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--model", default=None, help="key in models.yaml")
    ap.add_argument("--loss", choices=LOSSES, default="mnrl")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--pairs", type=Path, default=None, help="pairs.parquet")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--aux", default=None)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="minilm, 64 in-memory pairs (or --pairs), batch 16, 3 steps, out checkpoints/_dry_run/<loss>",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    seed = int(cfg["seed"] if args.seed is None else args.seed)
    max_steps = None
    n_limit = None
    model_key = args.model
    out = args.out
    if args.dry_run:
        model_key = model_key or "minilm"
        cfg["train"] = {**cfg["train"], "batch_size": 16, "micro_batch": 8, "epochs": 1}
        max_steps, n_limit = 3, 64
        out = out or Path(cfg["paths"]["checkpoints"]) / "_dry_run" / args.loss
    if model_key is None or out is None:
        ap.error("--model and --out are required unless --dry-run")
    if not args.dry_run and args.pairs is None and args.loss != "mlm":
        ap.error("--pairs is required unless --dry-run")
    summary = run_finetune(
        model_key,
        args.loss,
        args.tau,
        args.pairs,
        cfg,
        seed,
        out,
        aux=args.aux,
        lam=args.lam,
        max_steps=max_steps,
        n_pairs_limit=n_limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
