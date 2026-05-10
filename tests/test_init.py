"""__init__-level guards that must take effect before transitive imports."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "pythonpath",
    [
        "/tmp/some/leaked/path",
        f"/tmp/leak-a{os.pathsep}/tmp/leak-b",
        None,
    ],
)
def test_init_strips_leaked_pythonpath(pythonpath):
    """Package init must clear PYTHONPATH (env) AND remove its entries
    from sys.path so compiled-extension imports cannot resolve from a
    leaked location."""
    env = os.environ.copy()
    if pythonpath is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = pythonpath
    code = (
        "import mempalace, os, sys; "
        f"leaked = {pythonpath!r}; "
        "print('ENV:', repr(os.environ.get('PYTHONPATH'))); "
        "entries = leaked.split(os.pathsep) if leaked else []; "
        "leaked_in_path = any(e in sys.path for e in entries); "
        "print('SYSPATH_LEAK:', leaked_in_path)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    assert "ENV: None" in out, f"PYTHONPATH not cleared (input={pythonpath!r}): {out!r}"
    assert (
        "SYSPATH_LEAK: False" in out
    ), f"sys.path retains leaked entry (input={pythonpath!r}): {out!r}"
