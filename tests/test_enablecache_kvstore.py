"""enablecache decorator against KVStore (JSON-only values)."""

from __future__ import annotations

from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.workflow import enablecache


class _Host:
    def __init__(self, folderpath: str):
        self._folderpath = folderpath

    def get_cachedb_folderpath(self, function_name: str) -> str:
        return str(Path(self._folderpath) / "function_cache" / function_name)

    @enablecache
    def add(self, a: int, b: int) -> int:
        return a + b

    @enablecache
    def bad(self) -> object:
        return object()


def test_enablecache_round_trip(tmp_path: Path):
    fastworkflow.init({})
    host = _Host(str(tmp_path))
    assert host.add(1, 2) == 3
    assert host.add(1, 2) == 3  # cache hit


def test_enablecache_rejects_non_json(tmp_path: Path):
    fastworkflow.init({})
    host = _Host(str(tmp_path))
    with pytest.raises(TypeError, match="JSON-serialisable"):
        host.bad()
