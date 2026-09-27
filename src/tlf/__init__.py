"""tda-loss-followup: training-objective sweeps evaluated with the bakeoff harness."""

import os

# Set before anything imports torch; every entry point imports tlf first.
#
# Newer PyTorch (2.13 on the DGX Spark; 2.6.0 on the laptop has no such check)
# kills a forked child if the parent has already used GNU OpenMP:
#   "Terminating: fork() called from a process already using GNU OpenMP, this
#    is unsafe."
# The bakeoff's Mapper bootstrap forks its workers after the model has run CPU
# ops, so on the Spark every worker died and each seed came back as
# BrokenProcessPool. With one OpenMP thread torch never starts a thread team,
# and nothing is lost: encoding runs on the GPU and the bootstrap workers are
# single-threaded by design. The same fork makes tokenizers print a warning
# block per worker unless its parallelism is set explicitly.
#
# setdefault, so a value exported in the shell still wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
