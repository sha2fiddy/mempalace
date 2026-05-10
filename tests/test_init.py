"""__init__-level guards that must take effect before transitive imports."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "pythonpath",
    [
        "/tmp/some/leaked/path",
        None,
    ],
)
def test_init_strips_leaked_pythonpath(pythonpath):
    """Package init must drop PYTHONPATH before compiled extensions load,
    regardless of whether it was set in the parent environment."""
    env = os.environ.copy()
    if pythonpath is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mempalace, os; print(repr(os.environ.get('PYTHONPATH')))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        result.stdout.strip() == "None"
    ), f"PYTHONPATH not stripped (input was {pythonpath!r}): {result.stdout!r}"
