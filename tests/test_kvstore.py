"""Integration tests for the SQLite KVStore replacing speedict."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from fastworkflow.kvstore import KVStore, UtteranceCacheStore


def test_kvstore_round_trip_and_keyerror(tmp_path: Path):
    path = str(tmp_path / "store.sqlite3")
    with KVStore(path) as db:
        db["meta"] = {"last": 1}
        db["list"] = ["a", "b"]
        assert db["meta"] == {"last": 1}
        assert "list" in db
        assert db.get("missing", 42) == 42
        del db["list"]
        assert "list" not in db
        with pytest.raises(KeyError):
            del db["list"]
        with pytest.raises(KeyError):
            _ = db["list"]


def test_kvstore_coerces_int_keys(tmp_path: Path):
    path = str(tmp_path / "intkeys.sqlite3")
    with KVStore(path) as db:
        db[0] = {"utterance": "hi", "label": "x"}
        assert db.get(0)["utterance"] == "hi"
        assert 0 in db


def test_kvstore_keys_materialised_for_concurrent_mutation(tmp_path: Path):
    path = str(tmp_path / "keys.sqlite3")
    with KVStore(path) as db:
        for i in range(5):
            db[f"k{i}"] = i
        seen = []
        for key in db.keys():
            seen.append(key)
            if key == "k0":
                db["k_new"] = 99
                del db["k4"]
        assert "k0" in seen
        assert "k_new" in db
        assert "k4" not in db


def test_kvstore_values_are_json_not_pickle(tmp_path: Path):
    path = tmp_path / "nopickle.sqlite3"
    with KVStore(str(path)) as db:
        db["obj"] = {"class_name_marker": "DefinitelyNotPickle"}
    raw = path.read_bytes()
    assert b"\x80\x05" not in raw
    assert b"DefinitelyNotPickle" in raw  # plain JSON text


def test_utterance_cache_float32_blob_round_trip(tmp_path: Path):
    path = str(tmp_path / "utt.sqlite3")
    vec = np.arange(8, dtype=np.float32)
    with UtteranceCacheStore(path) as store:
        store.upsert(
            "123",
            utterance="hello",
            command_mapping={"cmd": {"frequency": 1, "feedback_date": "t"}},
            embedding=vec,
        )
        entry = store.get("123")
        assert entry is not None
        assert entry["utterance"] == "hello"
        assert entry["command_mapping"]["cmd"]["frequency"] == 1
        np.testing.assert_array_equal(entry["embedding"], vec)

        entries = list(store.iter_entries())
        assert len(entries) == 1
        assert entries[0][0] == "123"


def test_utterance_cache_none_embedding_round_trip(tmp_path: Path):
    path = str(tmp_path / "utt_none.sqlite3")
    with UtteranceCacheStore(path) as store:
        store.upsert(
            "none-emb",
            utterance="hello-none",
            command_mapping={"cmd": {"frequency": 1, "feedback_date": "t"}},
            embedding=None,
        )
        entry = store.get("none-emb")
        assert entry is not None
        assert entry["utterance"] == "hello-none"
        assert entry["command_mapping"]["cmd"]["frequency"] == 1
        assert entry["embedding"] is None

        entries = list(store.iter_entries())
        assert len(entries) == 1
        assert entries[0][0] == "none-emb"
        assert entries[0][1]["embedding"] is None


def test_utterance_cache_empty_embedding_round_trip(tmp_path: Path):
    path = str(tmp_path / "utt_empty.sqlite3")
    empty_vec = np.array([], dtype=np.float32)
    with UtteranceCacheStore(path) as store:
        store.upsert(
            "empty-emb",
            utterance="hello-empty",
            command_mapping={"cmd": {"frequency": 1, "feedback_date": "t"}},
            embedding=empty_vec,
        )
        entry = store.get("empty-emb")
        assert entry is not None
        assert entry["utterance"] == "hello-empty"
        assert entry["command_mapping"]["cmd"]["frequency"] == 1
        # Empty vectors should round-trip to a None embedding, not an empty ndarray
        assert entry["embedding"] is None

        entries = list(store.iter_entries())
        assert len(entries) == 1
        assert entries[0][0] == "empty-emb"
        assert entries[0][1]["embedding"] is None


def test_kvstore_and_utterance_cache_share_file(tmp_path: Path):
    path = str(tmp_path / "shared.sqlite3")
    with KVStore(path) as kv:
        kv["suggested_commands"] = ["a", "b"]
    with UtteranceCacheStore(path) as utt:
        utt.upsert(
            "h1",
            utterance="x",
            command_mapping={"c": {"frequency": 1, "feedback_date": "t"}},
            embedding=np.ones(4, dtype=np.float32),
        )
    with KVStore(path) as kv:
        assert kv["suggested_commands"] == ["a", "b"]
    with UtteranceCacheStore(path) as utt:
        assert utt.get("h1")["utterance"] == "x"


def _mp_writer(path: str, worker_id: int, n: int, result_path: str) -> None:
    try:
        with KVStore(path, timeout=30.0) as db:
            for i in range(n):
                key = f"w{worker_id}:{i}"
                db[key] = {"worker": worker_id, "i": i}
                assert db[key]["worker"] == worker_id
        Path(result_path).write_text("ok", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — report failure to parent via file
        Path(result_path).write_text(f"fail:{exc!r}", encoding="utf-8")
        raise SystemExit(1) from exc


def test_kvstore_four_process_concurrency(tmp_path: Path):
    path = str(tmp_path / "concurrent.sqlite3")
    n_workers = 4
    n_ops = 50
    result_paths = [str(tmp_path / f"result_{i}.txt") for i in range(n_workers)]
    procs = [
        multiprocessing.Process(
            target=_mp_writer, args=(path, i, n_ops, result_paths[i])
        )
        for i in range(n_workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exit {p.exitcode}"

    for rp in result_paths:
        assert Path(rp).read_text(encoding="utf-8") == "ok"

    with KVStore(path) as db:
        keys = list(db.keys())
        assert len(keys) == n_workers * n_ops
        for key in keys:
            worker_s, i_s = key.split(":", 1)
            worker_id = int(worker_s[1:])
            i = int(i_s)
            assert db[key] == {"worker": worker_id, "i": i}
        for worker_id in range(n_workers):
            for i in range(n_ops):
                assert db[f"w{worker_id}:{i}"] == {"worker": worker_id, "i": i}
