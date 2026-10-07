"""The persisted command directory is rebuilt when a module it recorded is gone.

The snapshot's fingerprint covers the workflow's own ``_commands`` only, but the
snapshot also records absolute paths to fastworkflow's core commands. Moving or
reinstalling fastworkflow must not leave a snapshot that every turn then fails
to import from.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_directory import CommandDirectory, get_cached_command_directory

HELLO_WORLD = Path(fastworkflow.__file__).parent / "examples" / "hello_world"


@pytest.fixture
def workflow(tmp_path) -> str:
    root = tmp_path / "hello_world"
    shutil.copytree(
        HELLO_WORLD, root, ignore=shutil.ignore_patterns("___command_info", "___convo_info", "__pycache__")
    )
    get_cached_command_directory.cache_clear()
    yield str(root.resolve())
    get_cached_command_directory.cache_clear()


def _cache_file(workflow: str) -> Path:
    return Path(CommandDirectory.get_commandinfo_folderpath(workflow)) / "command_directory.json"


def _core_command(snapshot: dict) -> str:
    internal = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    return next(
        name for name, meta in snapshot["map_command_2_metadata"].items()
        if meta["response_generation_module_path"].startswith(internal)
    )


def test_an_intact_snapshot_is_reused(workflow):
    get_cached_command_directory(workflow)
    cache = _cache_file(workflow)
    marked = json.loads(cache.read_text())
    marked["map_command_2_utterance_metadata"] = {}
    cache.write_text(json.dumps(marked))
    get_cached_command_directory.cache_clear()

    get_cached_command_directory(workflow)

    assert json.loads(cache.read_text())["map_command_2_utterance_metadata"] == {}


def test_a_snapshot_pointing_at_a_moved_fastworkflow_is_rebuilt(workflow, tmp_path):
    get_cached_command_directory(workflow)
    cache = _cache_file(workflow)
    stale = json.loads(cache.read_text())
    core = _core_command(stale)
    real_path = stale["map_command_2_metadata"][core]["response_generation_module_path"]
    gone = str(tmp_path / "old-site-packages" / "fastworkflow" / os.path.basename(real_path))
    stale["map_command_2_metadata"][core]["response_generation_module_path"] = gone
    cache.write_text(json.dumps(stale))
    get_cached_command_directory.cache_clear()

    directory = get_cached_command_directory(workflow)

    assert directory.map_command_2_metadata[core].response_generation_module_path == real_path
    rewritten = json.loads(cache.read_text())
    assert rewritten["map_command_2_metadata"][core]["response_generation_module_path"] == real_path
