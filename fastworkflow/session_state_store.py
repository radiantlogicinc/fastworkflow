"""
Pluggable persistence for suspended Topology-B agent sessions.

Disk backend (speeddict) suits local dev and single-node deployments.
Redis backend suits horizontal scale across workers/pods; workflow RocksDB
state still requires sticky routing per channel (one writer per channel).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import fastworkflow
from fastworkflow.utils.logging import logger

PENDING_STATE_KEY = "pending"
SCHEMA_VERSION = 1


class SessionStateStore(ABC):
    """Load/save/clear suspended-session blobs keyed by channel_id."""

    @abstractmethod
    def load(self, channel_id: str) -> Optional[dict[str, Any]]:
        """Return pending state dict or None."""

    @abstractmethod
    def save(self, channel_id: str, state: dict[str, Any]) -> None:
        """Persist pending state for channel_id."""

    @abstractmethod
    def clear(self, channel_id: str) -> None:
        """Remove pending state for channel_id."""

    @abstractmethod
    def exists(self, channel_id: str) -> bool:
        """True if pending state exists for channel_id."""


class DiskSessionStateStore(SessionStateStore):
    """One JSON file per channel under base_folder (portable, no pickle)."""

    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        os.makedirs(base_folder, exist_ok=True)

    def _json_path(self, channel_id: str) -> str:
        safe_id = channel_id.replace(os.sep, "_").replace("/", "_")
        return os.path.join(self.base_folder, f"{safe_id}_pending.json")

    def load(self, channel_id: str) -> Optional[dict[str, Any]]:
        path = self._json_path(channel_id)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save(self, channel_id: str, state: dict[str, Any]) -> None:
        path = self._json_path(channel_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str)

    def clear(self, channel_id: str) -> None:
        path = self._json_path(channel_id)
        if os.path.isfile(path):
            os.remove(path)

    def exists(self, channel_id: str) -> bool:
        return os.path.isfile(self._json_path(channel_id))


class RedisSessionStateStore(SessionStateStore):
    """
    Redis-backed pending state for multi-pod deployments.

    Requires redis package and REDIS_URL (or SESSION_STATE_REDIS_URL).
    Values are JSON-encoded strings.
    """

    def __init__(self, redis_url: str, key_prefix: str = "fw:session:pending:"):
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "Redis session store requires the 'redis' package"
            ) from exc
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._prefix = key_prefix

    def _key(self, channel_id: str) -> str:
        return f"{self._prefix}{channel_id}"

    def load(self, channel_id: str) -> Optional[dict[str, Any]]:
        raw = self._client.get(self._key(channel_id))
        if raw is None:
            return None
        return json.loads(raw)

    def save(self, channel_id: str, state: dict[str, Any]) -> None:
        self._client.set(self._key(channel_id), json.dumps(state))

    def clear(self, channel_id: str) -> None:
        self._client.delete(self._key(channel_id))

    def exists(self, channel_id: str) -> bool:
        return bool(self._client.exists(self._key(channel_id)))


def get_session_state_store(
    *,
    base_folder: Optional[str] = None,
) -> SessionStateStore:
    """
    Factory: SESSION_STATE_STORE=disk|redis (default disk).

    For disk, uses base_folder or SPEEDDICT_FOLDERNAME/channel_session_state.
    For redis, uses SESSION_STATE_REDIS_URL or REDIS_URL.
    """
    backend = str(
        fastworkflow.get_env_var("SESSION_STATE_STORE", default="disk")
    ).lower().strip()

    if backend == "redis":
        url = fastworkflow.get_env_var("SESSION_STATE_REDIS_URL", default=None)
        if not url:
            url = fastworkflow.get_env_var("REDIS_URL", default=None)
        if not url:
            raise ValueError(
                "SESSION_STATE_STORE=redis requires SESSION_STATE_REDIS_URL or REDIS_URL"
            )
        logger.info("Using RedisSessionStateStore for suspended sessions")
        return RedisSessionStateStore(url)

    if base_folder is None:
        speedict = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
        base_folder = os.path.join(speedict, "channel_session_state")
    os.makedirs(base_folder, exist_ok=True)
    logger.debug(f"Using DiskSessionStateStore at {base_folder}")
    return DiskSessionStateStore(base_folder)
