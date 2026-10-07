"""Zero-argument browsing: `run_chatbot` with no state root and no CLI args.

The picker browses folders from a bare `run_chatbot`; these pin what a listing
holds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastworkflow.run_chatbot.server import browse_directories


@pytest.fixture
def collections_tree(tmp_path) -> Path:
    root = tmp_path / "collections"
    root.mkdir()
    for name in ("exp029-trial-3.3-lifecycle-2026-09-06", "exp028-flat", "no-manifest-here"):
        (root / name).mkdir()
    (root / ".scratch").mkdir()
    (root / "loose.json").write_text("{}", encoding="utf-8")
    return root


def test_picker_skips_dot_directories(collections_tree):
    assert not any(entry["name"] == ".scratch" for entry in
                   browse_directories(str(collections_tree))["entries"])


def test_directory_entries_list_only_folders(collections_tree):
    listing = browse_directories(str(collections_tree))
    assert [entry["name"] for entry in listing["entries"]] == [
        "exp028-flat",
        "exp029-trial-3.3-lifecycle-2026-09-06",
        "no-manifest-here",
    ]
    assert all(entry["is_workflow"] is False for entry in listing["entries"])


def test_directory_listing_keeps_the_300_entry_cap(tmp_path):
    root = tmp_path / "many"
    root.mkdir()
    for index in range(320):
        (root / f"collection-{index:04d}").mkdir()
    assert len(browse_directories(str(root))["entries"]) == 300
