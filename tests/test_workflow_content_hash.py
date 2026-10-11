"""The content hash that names a workflow (observability/provenance.py).

The identity is a hash of what determines a workflow's behavior and of nothing
else: not where the tree lives, not timestamps, not what a deployment keeps beside
it. These tests pin the recipe and the selection rule.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time

import pytest

from fastworkflow.observability.provenance import (
    canonical_content_hash,
    imported_package_entries,
    workflow_content_entries,
    workflow_identity,
)


def _write_workflow(root, files: dict[str, str]) -> str:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(root)


# ----------------------------------------------------------------------
# The recipe
# ----------------------------------------------------------------------


def test_length_prefix_separates_trees_that_nul_alone_would_merge():
    one_file = canonical_content_hash([("f", b"b\0c\0d")])
    two_files = canonical_content_hash([("f", b"b"), ("d", b"")])
    assert one_file != two_files


def test_order_and_content_matter_but_collection_order_does_not():
    forward = canonical_content_hash([("a.py", b"1"), ("b.py", b"2")])
    assert forward == canonical_content_hash([("b.py", b"2"), ("a.py", b"1")])
    assert forward != canonical_content_hash([("a.py", b"2"), ("b.py", b"1")])
    assert canonical_content_hash([("a.py", b"x")]) != canonical_content_hash([("b.py", b"x")])


def test_windows_separators_normalize_and_colons_in_names_are_not_drive_letters():
    assert canonical_content_hash([("a\\b.py", b"x")]) == canonical_content_hash([("a/b.py", b"x")])
    assert canonical_content_hash([("odd:name.py", b"x")]).startswith("sha256:")


@pytest.mark.parametrize("path", ["/tmp/wf/a.py", "C:/wf/a.py", "../outside.py", "a/../../b.py"])
def test_paths_outside_the_tree_are_rejected(path):
    with pytest.raises(ValueError, match="tree-relative"):
        canonical_content_hash([(path, b"x")])


def test_duplicate_paths_are_rejected():
    with pytest.raises(ValueError, match="duplicate path"):
        canonical_content_hash([("a.py", b"x"), ("a.py", b"y")])


def test_empty_tree_hashes_to_the_empty_sha256():
    assert canonical_content_hash([]) == f"sha256:{hashlib.sha256().hexdigest()}"


# ----------------------------------------------------------------------
# The selection
# ----------------------------------------------------------------------


def test_identity_survives_copying_the_tree_and_touching_its_files(tmp_path):
    original = _write_workflow(tmp_path / "here", {"_commands/a.py": "x = 1\n"})
    before = workflow_identity(original)

    future = time.time() + 3600
    for dirpath, _, filenames in os.walk(original):
        for filename in filenames:
            os.utime(os.path.join(dirpath, filename), (future, future))
    shutil.copytree(original, str(tmp_path / "there"))

    assert workflow_identity(original) == before
    assert workflow_identity(str(tmp_path / "there")) == before


def test_deployment_and_derived_files_are_not_identity(tmp_path):
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    before = workflow_identity(root)

    _write_workflow(
        tmp_path / "wf",
        {
            "___command_info/model.json": '{"weights": 1}',
            "Insights/note.md": "learned something",
            "_commands/__pycache__/a.json": "{}",
            "fastworkflow.env": "LLM_AGENT=whatever\n",
            ".generated_manifest.json": '{"generated_at": "2026-08-27T12:00:00Z"}',
            "observability.sqlite3-wal": "journal",
            "benchmarks/corpus.json": "[]",
        },
    )
    assert workflow_identity(root) == before


def test_a_benchmarks_package_below_the_root_is_source(tmp_path):
    root = _write_workflow(tmp_path / "wf", {"_commands/a.py": "x = 1\n"})
    before = workflow_identity(root)

    _write_workflow(tmp_path / "wf", {"_commands/benchmarks/b.py": "y = 2\n"})
    assert workflow_identity(root) != before


def test_a_missing_workflow_folder_raises_rather_than_hashing_nothing(tmp_path):
    """os.walk yields nothing for a missing directory, so without a guard a typo
    would be recorded as the well-formed hash of the empty tree."""
    with pytest.raises(ValueError, match="does not exist"):
        workflow_content_entries(str(tmp_path / "no-such-workflow"))


def test_the_imported_package_is_hashed_by_relative_path():
    paths = [path for path, _ in imported_package_entries()]
    assert "fastworkflow/observability/provenance.py" in paths
    assert all(path.startswith("fastworkflow") for path in paths)
    assert canonical_content_hash(imported_package_entries()).startswith("sha256:")
