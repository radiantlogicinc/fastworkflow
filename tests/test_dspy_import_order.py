"""Regression test for the dspy-first import order (litellm vs. dspy's lazy openai stand-in).

``import dspy`` installs a lazy stand-in for ``openai`` in ``sys.modules``; if
litellm then imports ``openai._models`` through it, the import fails with a
circular ImportError. ``fastworkflow/__init__.py`` repairs this by touching the
stand-in when dspy came first. These tests guard that repair against dspy
upgrades that change the stand-in. Each check runs in a fresh subprocess, since
import order is process-global.
"""

import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(source: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=300,
    )


def test_dspy_then_fastworkflow_then_litellm_imports_cleanly():
    completed = _run(
        "import dspy; import fastworkflow; import litellm; "
        "import fastworkflow.utils.dspy_utils"
    )
    assert completed.returncode == 0, completed.stderr


def test_dspy_then_dspy_utils_imports_cleanly():
    completed = _run("import dspy; import fastworkflow.utils.dspy_utils")
    assert completed.returncode == 0, completed.stderr


def test_bare_fastworkflow_import_does_not_load_dspy_openai_or_litellm():
    completed = _run(
        "import sys; import fastworkflow; "
        "print(sorted(m for m in ('dspy', 'openai', 'litellm') if m in sys.modules))"
    )
    print(completed.stdout)
    assert completed.returncode == 0, completed.stderr
    loaded = [m for m in ("dspy", "openai", "litellm") if f"'{m}'" in completed.stdout]
    assert loaded == [], f"bare import loaded {loaded}"
