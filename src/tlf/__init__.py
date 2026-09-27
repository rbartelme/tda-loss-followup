"""tda-loss-followup: training-objective sweeps evaluated with the bakeoff harness."""

import os

# Set before numba or tokenizers load; every entry point imports tlf first.
#
# The bakeoff's Mapper bootstrap forks its workers after the parent has run
# parallel numba code (UMAP / pynndescent), and numba's threading layer decides
# whether that fork survives. On the laptop numba finds the system TBB library,
# which is fork-safe. The aarch64 numba wheel on the DGX Spark has no TBB
# support and falls back to GNU OpenMP, which is not, so numba kills every
# forked child with
#   "Terminating: fork() called from a process already using GNU OpenMP, this
#    is unsafe."
# and each seed comes back as BrokenProcessPool. "forksafe" makes numba take
# TBB where it exists and its own workqueue layer otherwise: the laptop is
# unchanged and the Spark never touches OpenMP. The same fork makes tokenizers
# print a warning block per worker unless its parallelism is set explicitly.
#
# setdefault, so a value exported in the shell still wins.
os.environ.setdefault("NUMBA_THREADING_LAYER", "forksafe")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
