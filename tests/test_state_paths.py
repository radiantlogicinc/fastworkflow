"""Unit coverage for the centralized persistent-state path module.

These assert the 3.0.0 storage contract: one absolute root
(FASTWORKFLOW_STATE_ROOT), a per-workflow namespace derived from the folder name
(or FASTWORKFLOW_WORKFLOW_ID), and fingerprinted function caches.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import state_paths
from fastworkflow.storage_keys import encode_path_component


@pytest.fixture(autouse=True)
def _restore_env():
    previous = fastworkflow._env_vars
    yield
    fastworkflow.init(previous or {})


def test_state_root_from_env_file_is_expanded_and_absolute(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "st")})
    assert state_paths.state_root() == str((tmp_path / "st").resolve())


def test_state_root_defaults_under_home(monkeypatch):
    # No dict value and no OS env -> XDG-style default under the user's home.
    monkeypatch.delenv("FASTWORKFLOW_STATE_ROOT", raising=False)
    fastworkflow.init({})
    expected = os.path.abspath(os.path.expanduser("~/.local/state/fastworkflow"))
    assert state_paths.state_root() == expected


def test_os_environ_used_when_absent_from_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "shell"))
    fastworkflow.init({})  # dict does not carry it; OS env must win over default
    assert state_paths.state_root() == str((tmp_path / "shell").resolve())


def test_env_file_wins_over_os_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "shell"))
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "file")})
    assert state_paths.state_root() == str((tmp_path / "file").resolve())


def test_workflow_id_defaults_to_folder_basename(monkeypatch):
    monkeypatch.delenv("FASTWORKFLOW_WORKFLOW_ID", raising=False)
    fastworkflow.init({})
    assert state_paths.workflow_id("/srv/apps/retail_workflow/") == "retail_workflow"


def test_workflow_id_override_takes_precedence():
    fastworkflow.init({"FASTWORKFLOW_WORKFLOW_ID": "stable-id"})
    assert state_paths.workflow_id("/anything/else") == encode_path_component("stable-id")


def test_workflow_id_is_encoded():
    fastworkflow.init({"FASTWORKFLOW_WORKFLOW_ID": "Tenant/One"})
    encoded = state_paths.workflow_id("/x")
    assert encoded == encode_path_component("Tenant/One")
    assert "/" not in encoded  # safe as a single path component


def test_subdirs_are_namespaced_and_created(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path)})
    wp = "/srv/apps/my_workflow"
    base = str((tmp_path / "workflows" / "my_workflow").resolve())
    assert state_paths.workflow_state_dir(wp) == base
    for leaf, fn in (
        ("conversations", state_paths.conversations_dir),
        ("session_state", state_paths.session_state_dir),
        ("checkpoints", state_paths.checkpoints_dir),
    ):
        path = fn(wp)
        assert path == os.path.join(base, leaf)
        assert os.path.isdir(path)


def test_function_cache_is_fingerprint_partitioned(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path)})
    # A non-workflow path cannot be fingerprinted; the module falls back rather
    # than raising, so the cache is still addressable.
    path = state_paths.function_cache_dir("/not/a/workflow", "add")
    parts = Path(path).parts
    assert parts[-1] == "add"
    assert parts[-3] == "function_cache"
    assert os.path.isdir(path)
