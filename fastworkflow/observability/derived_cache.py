"""Bounded in-process cache for per-turn derived markers and page stamps.

Opt-in: agent-facing `search_turns` callers pass nothing and behave as before.
The chatbot hands one of these in so a warm `/api/turns` avoids re-decoding
span attribute JSON when the underlying span set is unchanged.

Default capacity targets ~64 MB of typical entries (measured ~4 KB mean on the
ido store: markers.as_dict() + stamps JSON), not a small fixed count that a
complete scan would thrash through.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Optional

# Mean entry on the ido corpus was ~4 KB of JSON; 16_000 * 4 KB ≈ 64 MB.
DEFAULT_MAX_ENTRIES = 16_000
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class DerivedTurnEntry:
    """Cached span-basis markers plus optional page stamps."""

    markers: Any
    stamps: Mapping[str, Any]
    approx_bytes: int = 0


def estimate_entry_bytes(markers: Any, stamps: Mapping[str, Any]) -> int:
    """Approximate serialized size of one cache entry."""
    try:
        markers_payload = (
            markers.as_dict() if hasattr(markers, "as_dict") else markers
        )
        return len(json.dumps(markers_payload, default=str)) + len(
            json.dumps(dict(stamps), default=str)
        )
    except (TypeError, ValueError):
        return 4096


class DerivedTurnCache:
    """Thread-safe LRU of derived turn projections.

    Keys are opaque tuples built by the scan (stable store identity, turn key,
    span freshness, turn-row fields that affect markers, query parameters).
    Memory is bounded by entry count and approximate JSON bytes.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._lock = threading.RLock()
        self._entries: OrderedDict[Hashable, DerivedTurnEntry] = OrderedDict()
        self._approx_bytes = 0

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def approx_bytes(self) -> int:
        with self._lock:
            return self._approx_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._approx_bytes = 0

    def get(self, key: Hashable) -> Optional[DerivedTurnEntry]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: Hashable, entry: DerivedTurnEntry) -> None:
        with self._lock:
            sized = entry
            if sized.approx_bytes <= 0:
                sized = DerivedTurnEntry(
                    markers=entry.markers,
                    stamps=entry.stamps,
                    approx_bytes=estimate_entry_bytes(entry.markers, entry.stamps),
                )
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._approx_bytes = max(
                    0, self._approx_bytes - int(previous.approx_bytes or 0)
                )
            self._entries[key] = sized
            self._approx_bytes += int(sized.approx_bytes or 0)
            self._entries.move_to_end(key)
            while self._entries and (
                len(self._entries) > self._max_entries
                or self._approx_bytes > self._max_bytes
            ):
                _evicted_key, evicted = self._entries.popitem(last=False)
                self._approx_bytes = max(
                    0, self._approx_bytes - int(evicted.approx_bytes or 0)
                )
