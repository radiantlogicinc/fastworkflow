"""Utterance cache matching uses float32 BLOB rows, not a JSON blob."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fastworkflow.cache_matching import cache_match
from fastworkflow.kvstore import UtteranceCacheStore


class _FakePipeline:
    """Minimal stand-in: get_embedding path is bypassed via direct upsert."""


def test_store_and_match_via_blob_rows(tmp_path: Path, monkeypatch):
    path = str(tmp_path / "cache.db")
    vec_a = np.ones(8, dtype=np.float32)
    vec_b = np.zeros(8, dtype=np.float32)
    vec_b[0] = 1.0

    with UtteranceCacheStore(path) as store:
        store.upsert(
            "1",
            utterance="alpha",
            command_mapping={"cmd_a": {"frequency": 2, "feedback_date": "t1"}},
            embedding=vec_a,
        )
        store.upsert(
            "2",
            utterance="beta",
            command_mapping={"cmd_b": {"frequency": 1, "feedback_date": "t2"}},
            embedding=vec_b,
        )

    def fake_get_embedding(text, model_pipeline):
        # Near-identical to vec_a
        return np.ones(8, dtype=np.float32).reshape(1, -1)

    monkeypatch.setattr(
        "fastworkflow.cache_matching.get_embedding", fake_get_embedding
    )
    label = cache_match(path, "query", _FakePipeline(), threshold=0.5)
    assert label == "cmd_a"

    # Mechanical JSON whole-cache key must not exist
    from fastworkflow.kvstore import KVStore

    with KVStore(path) as kv:
        assert kv.get("cache") is None
