"""Shared helpers so FastAPI tests do not pollute the developer's state tree.

Pending Topology-B blobs under ``<state-root>/workflows/<id>/session_state/`` are
gitignored and survive across pytest runs. A leftover
``nonexistent_user.pending.json`` (awaiting_user=true) is restored on the next
session create and makes unrelated FastAPI tests fail with shifting names
(fix-j83). Every FastAPI ``app_module`` fixture must point the state root at a
per-test temp directory — same pattern as ``tests/test_checkpoint_integration.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import dotenv_values

import fastworkflow


def init_fastapi_hermetic_env(
    env_file: str,
    passwords_file: str,
    speedict_dir: Path | str,
) -> dict[str, Any]:
    """``fastworkflow.init`` with FASTWORKFLOW_STATE_ROOT overridden to ``speedict_dir``.

    Returns the previous ``fastworkflow._env_vars`` so callers can restore in a
    fixture ``finally`` (an interpreter left on a deleted temp dir poisons the
    next test file).
    """
    previous_env = fastworkflow._env_vars
    env_vars = {
        **dotenv_values(env_file),
        **dotenv_values(passwords_file),
        "FASTWORKFLOW_STATE_ROOT": str(speedict_dir),
    }
    fastworkflow.init(env_vars)
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    return previous_env or {}


def restore_fastapi_env(previous_env: dict[str, Any]) -> None:
    """Undo ``init_fastapi_hermetic_env`` for the next test / file."""
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.init(previous_env)
