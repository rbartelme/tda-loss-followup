"""Experiment driver: config grid -> runs -> train -> evaluate -> result rows.

An experiment config (``configs/exp*.yaml``) names lists (or scalars) for
``models``, ``loss``, ``tau``, ``pair_set``, ``aux``, ``lambda``, ``seeds`` and
``corpora``. ``expand_grid`` takes their Cartesian product into ``Run`` objects;
``execute`` trains each run whose checkpoints are missing, evaluates every
(checkpoint fraction, corpus) whose result row is missing, and appends rows
idempotently. Finished work is always skipped, so a sweep can be interrupted
and re-launched; ``force=True`` redoes everything.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from tlf.data import CORPORA
from tlf.evaluate import cached_metrics, evaluate_checkpoint
from tlf.results import append_row, make_row, row_exists
from tlf.train import frac_tag, run_finetune

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Run:
    """One fine-tuning run in an experiment grid."""

    exp: str
    model: str
    loss: str
    tau: float
    pair_set: str
    aux: str | None
    lam: float
    seed: int

    @property
    def id(self) -> str:
        """Directory-safe run identifier."""
        s = f"{self.model}_{self.loss}_tau{self.tau:g}_{self.pair_set}"
        if self.aux:
            s += f"_{self.aux}_lam{self.lam:g}"
        return f"{s}_s{self.seed}"

    def key(self, ckpt_frac: float, corpus: str) -> dict[str, Any]:
        """The result-row key for one checkpoint and corpus."""
        return {
            "exp": self.exp,
            "model": self.model,
            "loss": self.loss,
            "tau": self.tau,
            "pair_set": self.pair_set,
            "aux": self.aux or "",
            "lambda": self.lam if self.aux else float("nan"),
            "ckpt_frac": ckpt_frac,
            "seed": self.seed,
            "corpus": corpus,
        }


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list | tuple):
        return list(v)
    return [v]


def _first(cfg: dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    return None


def expand_grid(cfg: dict[str, Any]) -> list[Run]:
    """Expand an experiment config into its runs.

    Args:
        cfg: Resolved experiment config with ``exp`` and grid fields. Accepts
            singular or plural names (``model``/``models``, ``loss``/``losses``,
            ``pair_set``/``pair_sets``, ``seed``/``seeds``). ``aux`` defaults to
            none; ``lambda`` applies only when an aux is set.

    Returns:
        Runs in a stable order (models, losses, taus, pair sets, auxes,
        lambdas, seeds).

    Raises:
        ValueError: If ``exp``, a model, a loss, a tau or a pair set is missing.
    """
    exp = cfg.get("exp")
    models = _as_list(_first(cfg, "models", "model"))
    losses = _as_list(_first(cfg, "loss", "losses"))
    taus = _as_list(_first(cfg, "tau", "taus"))
    pair_sets = _as_list(_first(cfg, "pair_set", "pair_sets"))
    aux_raw = _first(cfg, "aux", "auxes")
    auxes = _as_list(None if isinstance(aux_raw, dict) else aux_raw) or [None]
    lambdas = _as_list(_first(cfg, "lambda", "lambdas")) or [0.0]
    seeds = _as_list(_first(cfg, "seeds", "seed"))
    missing = [
        n
        for n, v in (
            ("exp", exp),
            ("models", models),
            ("loss", losses),
            ("tau", taus),
            ("pair_set", pair_sets),
            ("seed", seeds),
        )
        if not v
    ]
    if missing:
        raise ValueError(f"experiment config is missing {missing}")
    runs: list[Run] = []
    seen: set[Run] = set()
    for model, loss, tau, ps, aux in product(models, losses, taus, pair_sets, auxes):
        for lam in lambdas if aux else [0.0]:
            for seed in seeds:
                r = Run(
                    str(exp),
                    str(model),
                    str(loss),
                    float(tau),
                    str(ps),
                    None if aux in (None, "", "none") else str(aux),
                    float(lam),
                    int(seed),
                )
                if r not in seen:
                    seen.add(r)
                    runs.append(r)
    return runs


def select_runs(
    runs: Sequence[Run],
    *,
    models: Sequence[str] | None = None,
    taus: Sequence[float] | None = None,
) -> list[Run]:
    """Narrow a grid to the runs matching every given filter.

    Args:
        runs: Output of ``expand_grid``.
        models: Keep only these model keys; ``None`` or empty keeps all.
        taus: Keep only these temperatures (``math.isclose``); ``None`` or
            empty keeps all.

    Returns:
        The matching runs in their original order.
    """
    out = list(runs)
    if models:
        keep = set(models)
        out = [r for r in out if r.model in keep]
    if taus:
        want = [float(t) for t in taus]
        out = [r for r in out if any(math.isclose(r.tau, t) for t in want)]
    return out


def checkpoint_fracs(cfg: dict[str, Any], only_final: bool) -> list[float]:
    """Checkpoint fractions to evaluate.

    Args:
        cfg: Resolved config (``train.checkpoint_fracs``).
        only_final: Restrict to ``0.0`` and ``1.0``.

    Returns:
        Sorted fractions.
    """
    fracs = sorted(float(f) for f in cfg["train"]["checkpoint_fracs"])
    return [f for f in fracs if f in (0.0, 1.0)] if only_final else fracs


def pair_path_for(run: Run, cfg: dict[str, Any]) -> Path:
    """Where ``scripts/build_pairs.py`` put this run's pair set.

    Args:
        run: The run.
        cfg: Resolved config; the pair seed is ``pairs.seed`` or the top-level seed.

    Returns:
        ``paths.pairs/<pair_set>_s<pair_seed>/pairs.parquet``.
    """
    pair_seed = int(cfg["pairs"].get("seed", cfg["seed"]))
    return (
        Path(cfg["paths"]["pairs"]) / f"{run.pair_set}_s{pair_seed}" / "pairs.parquet"
    )


def checkpoint_root(run: Run, cfg: dict[str, Any]) -> Path:
    """Run directory ``paths.checkpoints/<run.id>/``.

    Keyed by the run alone, not the experiment, so a run that several
    experiment grids contain trains once, and each grid reads the same
    checkpoints and, through the embeddings and metrics caches, the same
    evaluation.
    """
    return Path(cfg["paths"]["checkpoints"]) / run.id


def checkpoint_steps_for(root: Path, ckpt: Path) -> tuple[float, float]:
    """A checkpoint's optimizer step and its run's total, from the manifests.

    Args:
        root: Run directory; its ``manifest.json`` carries ``total_steps``.
        ckpt: Checkpoint directory; its ``manifest.json`` carries ``step`` and,
            except for the shared untrained checkpoint, ``total_steps``.

    Returns:
        ``(step, total_steps)``, NaN where a manifest or field is missing.
    """

    def field(path: Path, key: str) -> float:
        try:
            v = json.loads(path.read_text()).get(key)
        except (OSError, ValueError):
            return float("nan")
        return float("nan") if v is None else float(v)

    step = field(ckpt / "manifest.json", "step")
    total = field(ckpt / "manifest.json", "total_steps")
    if math.isnan(total):
        total = field(root / "manifest.json", "total_steps")
    return step, total


def run_is_trained(root: Path, fracs: Sequence[float]) -> bool:
    """Whether a run finished training and has every requested checkpoint.

    Args:
        root: Run directory.
        fracs: Fractions that must exist as ``ckpt_<frac>/``.

    Returns:
        ``True`` if the run manifest says complete and all checkpoints exist.
    """
    m = root / "manifest.json"
    if not m.is_file():
        return False
    try:
        if json.loads(m.read_text()).get("status") != "complete":
            return False
    except (OSError, ValueError):
        return False
    return all(
        (root / f"ckpt_{frac_tag(f)}" / "manifest.json").is_file() for f in fracs
    )


def plan(
    cfg: dict[str, Any],
    *,
    only_final: bool = False,
    force: bool = False,
    corpora: Sequence[str] | None = None,
    n_seeds: int | None = None,
    models: Sequence[str] | None = None,
    taus: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Describe what ``execute`` would do, without doing it.

    Args:
        cfg: Resolved experiment config.
        only_final: Evaluate only ``ckpt_0.0`` and ``ckpt_1.0``.
        force: Treat everything as missing, cached metrics included.
        corpora: Corpora to evaluate; defaults to ``cfg["corpora"]`` or both.
        n_seeds: Mapper bootstrap seeds override, as ``execute`` will pass it;
            decides which cached metrics count as hits.
        models: Restrict to these model keys (see ``select_runs``).
        taus: Restrict to these temperatures (see ``select_runs``).

    Returns:
        One dict per run: the run, whether it is trained, its pair path and
        whether that exists, the (frac, corpus) pairs still to write
        (``todo``) and the subset of those whose metrics are already cached
        (``cached``).
    """
    fracs = checkpoint_fracs(cfg, only_final)
    corp = list(corpora or cfg.get("corpora") or CORPORA)
    results_dir = cfg["paths"]["results"]
    out = []
    for run in select_runs(expand_grid(cfg), models=models, taus=taus):
        root = checkpoint_root(run, cfg)
        pp = pair_path_for(run, cfg)
        todo = [
            (f, c)
            for f in fracs
            for c in corp
            if force or not row_exists(run.key(f, c), results_dir)
        ]
        trained = (not force) and run_is_trained(root, fracs)
        cached = [
            (f, c)
            for f, c in todo
            if trained
            and cached_metrics(root / f"ckpt_{frac_tag(f)}", c, cfg, n_seeds=n_seeds)
            is not None
        ]
        out.append(
            {
                "run": run,
                "root": root,
                "trained": trained,
                "pair_path": pp,
                "pairs_exist": pp.is_file(),
                "todo": todo,
                "cached": cached,
            }
        )
    return out


def execute(
    cfg: dict[str, Any],
    *,
    only_final: bool = False,
    force: bool = False,
    n_seeds: int | None = None,
    n_workers: int | None = None,
    corpora: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    taus: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run an experiment: train missing runs, evaluate missing rows, append.

    Args:
        cfg: Resolved experiment config.
        only_final: Evaluate only ``ckpt_0.0`` and ``ckpt_1.0``.
        force: Retrain and re-evaluate everything, ignoring cached metrics.
        n_seeds: Mapper bootstrap seeds override.
        n_workers: Mapper bootstrap workers override.
        corpora: Corpora to evaluate; defaults to the config's list or both.
        models: Restrict to these model keys (see ``select_runs``); used to
            fill in intermediate checkpoints for the runs that moved.
        taus: Restrict to these temperatures (see ``select_runs``).

    Returns:
        Counts of runs trained / skipped and rows written / skipped, how many
        of the written rows came from cached metrics, plus wall time.

    Raises:
        FileNotFoundError: If a run's pair set has not been built.
    """
    t0 = time.time()
    fracs = checkpoint_fracs(cfg, only_final)
    results_dir = cfg["paths"]["results"]
    stats = {
        "runs": 0,
        "trained": 0,
        "runs_skipped": 0,
        "rows_written": 0,
        "rows_cached": 0,
        "rows_skipped": 0,
    }
    for item in plan(
        cfg,
        only_final=only_final,
        force=force,
        corpora=corpora,
        n_seeds=n_seeds,
        models=models,
        taus=taus,
    ):
        run: Run = item["run"]
        root: Path = item["root"]
        stats["runs"] += 1
        n_possible = len(fracs) * len(list(corpora or cfg.get("corpora") or CORPORA))
        if not item["todo"]:
            log.info("[%s] all %d rows present; skipping", run.id, n_possible)
            stats["runs_skipped"] += 1
            stats["rows_skipped"] += n_possible
            continue
        stats["rows_skipped"] += n_possible - len(item["todo"])
        if not item["trained"]:
            if not item["pairs_exist"]:
                raise FileNotFoundError(
                    f"{item['pair_path']} not found; run `make pairs` first"
                )
            if root.exists():
                log.info("[%s] incomplete run on disk; retraining from scratch", run.id)
                shutil.rmtree(root)
            log.info("[%s] training", run.id)
            run_finetune(
                run.model,
                run.loss,
                run.tau,
                item["pair_path"],
                cfg,
                run.seed,
                root,
                aux=run.aux,
                lam=run.lam,
            )
            stats["trained"] += 1
        else:
            log.info(
                "[%s] checkpoints present; evaluating %d rows (%d from cached metrics)",
                run.id,
                len(item["todo"]),
                len(item["cached"]),
            )
        cached = set(item["cached"])
        for frac, corpus in item["todo"]:
            ckpt = root / f"ckpt_{frac_tag(frac)}"
            metrics = evaluate_checkpoint(
                ckpt,
                corpus,
                cfg,
                seeds=n_seeds,
                n_workers=n_workers,
                use_cache=not force,
            )
            step, total_steps = checkpoint_steps_for(root, ckpt)
            append_row(
                make_row(
                    exp=run.exp,
                    model=run.model,
                    loss=run.loss,
                    tau=run.tau,
                    pair_set=run.pair_set,
                    aux=run.aux,
                    lam=run.lam,
                    ckpt_frac=frac,
                    seed=run.seed,
                    corpus=corpus,
                    metrics=metrics,
                    step=step,
                    total_steps=total_steps,
                ),
                results_dir,
            )
            stats["rows_written"] += 1
            stats["rows_cached"] += (frac, corpus) in cached
    stats["wall_seconds"] = round(time.time() - t0, 1)
    return stats


def describe_plan(items: list[dict[str, Any]]) -> str:
    """Render a plan as a text table for ``--dry-run``.

    Args:
        items: Output of ``plan``.

    Returns:
        Multi-line string.
    """
    lines = [f"{'run':<64} {'trained':>7} {'pairs':>5} {'todo':>4} {'cached':>6}"]
    for it in items:
        lines.append(
            f"{it['run'].id:<64} {str(it['trained']):>7} {str(it['pairs_exist']):>5} "
            f"{len(it['todo']):>4} {len(it['cached']):>6}"
        )
    n_todo = sum(len(it["todo"]) for it in items)
    n_cached = sum(len(it["cached"]) for it in items)
    n_train = sum(1 for it in items if it["todo"] and not it["trained"])
    lines.append(
        f"{len(items)} runs; {n_train} to train; {n_todo} result rows to write, "
        f"{n_cached} of them from cached metrics"
    )
    return "\n".join(lines)


def run_to_dict(run: Run) -> dict[str, Any]:
    """Plain-dict view of a run (for manifests and logs)."""
    return asdict(run)
