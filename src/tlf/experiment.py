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
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from tlf.data import CORPORA
from tlf.evaluate import evaluate_checkpoint
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
    """Run directory under ``paths.checkpoints/<exp>/``."""
    return Path(cfg["paths"]["checkpoints"]) / run.exp / run.id


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
) -> list[dict[str, Any]]:
    """Describe what ``execute`` would do, without doing it.

    Args:
        cfg: Resolved experiment config.
        only_final: Evaluate only ``ckpt_0.0`` and ``ckpt_1.0``.
        force: Treat everything as missing.
        corpora: Corpora to evaluate; defaults to ``cfg["corpora"]`` or both.

    Returns:
        One dict per run: the run, whether it is trained, its pair path and
        whether that exists, and the (frac, corpus) pairs still to evaluate.
    """
    fracs = checkpoint_fracs(cfg, only_final)
    corp = list(corpora or cfg.get("corpora") or CORPORA)
    results_dir = cfg["paths"]["results"]
    out = []
    for run in expand_grid(cfg):
        root = checkpoint_root(run, cfg)
        pp = pair_path_for(run, cfg)
        todo = [
            (f, c)
            for f in fracs
            for c in corp
            if force or not row_exists(run.key(f, c), results_dir)
        ]
        out.append(
            {
                "run": run,
                "root": root,
                "trained": (not force) and run_is_trained(root, fracs),
                "pair_path": pp,
                "pairs_exist": pp.is_file(),
                "todo": todo,
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
) -> dict[str, Any]:
    """Run an experiment: train missing runs, evaluate missing rows, append.

    Args:
        cfg: Resolved experiment config.
        only_final: Evaluate only ``ckpt_0.0`` and ``ckpt_1.0``.
        force: Retrain and re-evaluate everything.
        n_seeds: Mapper bootstrap seeds override.
        n_workers: Mapper bootstrap workers override.
        corpora: Corpora to evaluate; defaults to the config's list or both.

    Returns:
        Counts of runs trained / skipped and rows written / skipped, plus wall time.

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
        "rows_skipped": 0,
    }
    for item in plan(cfg, only_final=only_final, force=force, corpora=corpora):
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
                "[%s] checkpoints present; evaluating %d rows",
                run.id,
                len(item["todo"]),
            )
        for frac, corpus in item["todo"]:
            ckpt = root / f"ckpt_{frac_tag(frac)}"
            metrics = evaluate_checkpoint(
                ckpt, corpus, cfg, seeds=n_seeds, n_workers=n_workers
            )
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
                ),
                results_dir,
            )
            stats["rows_written"] += 1
    stats["wall_seconds"] = round(time.time() - t0, 1)
    return stats


def describe_plan(items: list[dict[str, Any]]) -> str:
    """Render a plan as a text table for ``--dry-run``.

    Args:
        items: Output of ``plan``.

    Returns:
        Multi-line string.
    """
    lines = [f"{'run':<64} {'trained':>7} {'pairs':>5} {'todo':>4}"]
    for it in items:
        lines.append(
            f"{it['run'].id:<64} {str(it['trained']):>7} {str(it['pairs_exist']):>5} {len(it['todo']):>4}"
        )
    n_todo = sum(len(it["todo"]) for it in items)
    n_train = sum(1 for it in items if it["todo"] and not it["trained"])
    lines.append(
        f"{len(items)} runs; {n_train} to train; {n_todo} result rows to evaluate"
    )
    return "\n".join(lines)


def run_to_dict(run: Run) -> dict[str, Any]:
    """Plain-dict view of a run (for manifests and logs)."""
    return asdict(run)
