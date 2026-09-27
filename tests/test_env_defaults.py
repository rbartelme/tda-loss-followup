"""Import-time environment defaults that keep the Mapper bootstrap's fork safe.

Each check runs in a fresh interpreter, since the defaults only matter if they
are in place before numba and tokenizers are first imported.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

# The shape of the failure on the DGX Spark: the parent runs parallel numba code
# (UMAP between two bootstraps), forks, and the child runs parallel numba code
# too (each Mapper worker). Under GNU OpenMP numba kills that child.
_PROBE = textwrap.dedent(
    """
    import multiprocessing as mp
    import os

    import numpy as np

    import tlf  # noqa: F401  (must precede numba)
    from numba import njit, prange, threading_layer


    @njit(parallel=True)
    def total(a):
        s = 0.0
        for i in prange(a.size):
            s += a[i]
        return s


    total(np.ones(10_000))
    child = mp.get_context("fork").Process(target=total, args=(np.ones(10_000),))
    child.start()
    child.join()
    print(
        os.environ["NUMBA_THREADING_LAYER"],
        os.environ["TOKENIZERS_PARALLELISM"],
        threading_layer(),
        child.exitcode,
    )
    """
)

_DEFAULTED = {"NUMBA_THREADING_LAYER", "TOKENIZERS_PARALLELISM"}


def _probe(**env: str) -> list[str]:
    """Run the fork probe in a clean subprocess and report what it saw.

    Args:
        **env: Variables to set in the child's environment. The variables tlf
            defaults are removed first, so only these are present.

    Returns:
        The probe's NUMBA_THREADING_LAYER, TOKENIZERS_PARALLELISM, the numba
        threading layer actually selected, and the forked child's exit code,
        as strings.
    """
    child = {k: v for k, v in os.environ.items() if k not in _DEFAULTED}
    child.update(env)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=child,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def test_default_layer_survives_fork_after_parallel_numba():
    """With nothing exported, numba takes a fork-safe layer and the child lives."""
    numba_env, tok_env, layer, exitcode = _probe()
    assert (numba_env, tok_env) == ("forksafe", "false")
    assert layer in {"tbb", "workqueue"}
    assert exitcode == "0"


def test_shell_value_wins():
    """Exported values override the defaults."""
    out = _probe(NUMBA_THREADING_LAYER="workqueue", TOKENIZERS_PARALLELISM="true")
    assert out == ["workqueue", "true", "workqueue", "0"]
