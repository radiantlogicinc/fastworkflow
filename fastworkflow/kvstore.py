"""SQLite-backed key-value store.

Replaces speedict/RocksDB. RocksDB took a process-exclusive lock, which forced an
open/close cycle around every operation and still raced across processes; SQLite in
WAL mode supports concurrent readers alongside a writer. Values are JSON, not pickle,
so a writable store directory is not an arbitrary-code-execution primitive.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Iterator, Optional

import numpy as np


def _key_str(key: Any) -> str:
    """Coerce mapping keys to TEXT. Call sites historically used int keys with Rdict."""
    return key if isinstance(key, str) else str(key)


def _open_sqlite(path: str, *, timeout: float) -> sqlite3.Connection:
    """Open a WAL connection. ``timeout`` is enforced by sqlite3.connect (seconds)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # check_same_thread=False is safe here because sqlite3 serialises access
    # internally and every method below is a single self-contained statement.
    # Busy waiting uses connect(timeout=...); do not interpolate into PRAGMA SQL.
    conn = sqlite3.connect(path, timeout=timeout, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


class KVStore:
    """A durable dict[str, Any]. Values must be JSON-serialisable.

    Values are stored as JSON TEXT. That matches today's call sites (scalars,
    small dicts, conversation turns). Large binary or high-cardinality payloads
    should use a dedicated table with typed columns / BLOBs (see
    :class:`UtteranceCacheStore`) rather than stuffing them into ``v``.
    """

    def __init__(self, path: str, *, timeout: float = 30.0) -> None:
        self._conn = _open_sqlite(path, timeout=timeout)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
        )
        self._conn.commit()

    def __setitem__(self, key: Any, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (_key_str(key), json.dumps(value)),
        )
        self._conn.commit()

    def __getitem__(self, key: Any) -> Any:
        row = self._conn.execute(
            "SELECT v FROM kv WHERE k=?", (_key_str(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def __delitem__(self, key: Any) -> None:
        cur = self._conn.execute("DELETE FROM kv WHERE k=?", (_key_str(key),))
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM kv WHERE k=?", (_key_str(key),)
            ).fetchone()
            is not None
        )

    def get(self, key: Any, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT v FROM kv WHERE k=?", (_key_str(key),)
        ).fetchone()
        return default if row is None else json.loads(row[0])

    def keys(self) -> Iterator[str]:
        # Materialise first so callers may mutate the store while iterating
        # (the legacy conversation store historically did this against Rdict).
        return (r[0] for r in self._conn.execute("SELECT k FROM kv").fetchall())

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "KVStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class UtteranceCacheStore:
    """Per-utterance embedding cache with float32 BLOB vectors.

    The mechanical JSON-in-KVStore swap for this workload is 3–6x slower than
    speedict; one row per hash with a raw float32 column is ~132x faster.
    Shares a SQLite file safely with :class:`KVStore` (separate tables).
    """

    def __init__(self, path: str, *, timeout: float = 30.0) -> None:
        self._conn = _open_sqlite(path, timeout=timeout)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS utterance_cache (
                k TEXT PRIMARY KEY,
                meta TEXT NOT NULL,
                vec BLOB NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _pack_vec(embedding: Optional[Any]) -> bytes:
        if embedding is None:
            return b""
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.size == 0:
            return b""
        return arr.tobytes()

    @staticmethod
    def _unpack_vec(blob: bytes) -> Optional[np.ndarray]:
        if not blob:
            return None
        return np.frombuffer(blob, dtype=np.float32)

    def get(self, key: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT meta, vec FROM utterance_cache WHERE k=?", (key,)
        ).fetchone()
        if row is None:
            return None
        meta = json.loads(row[0])
        embedding = self._unpack_vec(row[1])
        return {
            "utterance": meta.get("utterance", ""),
            "command_mapping": meta.get("command_mapping", {}),
            "embedding": embedding,
        }

    def upsert(
        self,
        key: str,
        *,
        utterance: str,
        command_mapping: dict[str, Any],
        embedding: Optional[Any],
    ) -> None:
        meta = json.dumps(
            {"utterance": utterance, "command_mapping": command_mapping}
        )
        self._conn.execute(
            """
            INSERT INTO utterance_cache (k, meta, vec) VALUES (?, ?, ?)
            ON CONFLICT(k) DO UPDATE SET meta=excluded.meta, vec=excluded.vec
            """,
            (key, meta, self._pack_vec(embedding)),
        )
        self._conn.commit()

    def iter_entries(self) -> Iterator[tuple[str, dict[str, Any]]]:
        # Stream rows from the cursor — do not fetchall(); cache_match only
        # needs the best match and must not hold every embedding in memory.
        for key, meta_json, vec in self._conn.execute(
            "SELECT k, meta, vec FROM utterance_cache"
        ):
            meta = json.loads(meta_json)
            yield key, {
                "utterance": meta.get("utterance", ""),
                "command_mapping": meta.get("command_mapping", {}),
                "embedding": self._unpack_vec(vec),
            }

    def has_entries(self) -> bool:
        """True if the table has any rows. Cheap EXISTS check (no materialisation)."""
        return (
            self._conn.execute("SELECT 1 FROM utterance_cache LIMIT 1").fetchone()
            is not None
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "UtteranceCacheStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
