"""Import-time environment defaults that keep the Mapper bootstrap's fork safe.

Each check runs in a fresh interpreter, since the defaults only matter if they
are in place before torch is first imported.
"""

from __future__ import annotations

import os
import subprocess
import sys

_PROBE = (
    "import os, tlf, torch; "
    "print(os.environ['OMP_NUM_THREADS'], os.environ['TOKENIZERS_PARALLELISM'], "
    "torch.get_num_threads())"
)


def _probe(**env: str) -> list[str]:
    """Import tlf then torch in a clean subprocess and report the thread settings.

    Args:
        **env: Variables to set in the child's environment. OMP_NUM_THREADS and
            TOKENIZERS_PARALLELISM are removed first, so only these are present.

    Returns:
        The child's OMP_NUM_THREADS, TOKENIZERS_PARALLELISM and
        ``torch.get_num_threads()``, as strings.
    """
    child = {
        k: v
        for k, v in os.environ.items()
        if k not in {"OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM"}
    }
    child.update(env)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=child,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def test_importing_tlf_makes_torch_single_threaded():
    """With nothing exported, torch starts with one OpenMP thread."""
    assert _probe() == ["1", "false", "1"]


def test_shell_value_wins():
    """An exported value overrides the default."""
    assert _probe(OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="true") == ["4", "true", "4"]
