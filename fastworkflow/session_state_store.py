"""
Pluggable persistence for suspended Topology-B agent sessions.

Disk backend (speeddict) suits local dev and single-node deployments.
Redis backend suits horizontal scale across workers/pods; workflow RocksDB
state still requires sticky routing per channel (one writer per channel).
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import fastworkflow
from fastworkflow.state_serialization import encode_state
from fastworkflow.storage_keys import encode_path_component
from fastworkflow.utils.logging import logger

PENDING_STATE_KEY = "pending"
# 2: added the logical-turn accumulator and the CME continuation keys, without
# which a restored session started a new turn and dropped partially extracted
# parameters. A v1 blob is refused rather than migrated -- it is a suspended
# turn at most minutes old, and the fields it lacks are the ones that made
# restoring it wrong.
SCHEMA_VERSION = 2


class IncompatibleSessionState(Exception):
    """A pending blob was written at a schema version this build cannot read.

    Distinct from fastworkflow.UnsupportedStateVersion, which an author's
    from_state hook raises about their own state. This one is about the
    framework's blob format, so no author can act on it.

    Raised instead of applying what happens to parse, because the fields of a
    format this build does not know are not a subset of the fields it does: a
    partial restore produces a session that is neither the suspended one nor a
    clean one. Discarding loses the suspended turn, which is recoverable by
    asking again; a partial restore is not.
    """

    def __init__(self, found: Any, expected: int = SCHEMA_VERSION):
        self.found = found
        self.expected = expected
        super().__init__(
            f"pending session state is schema version {found!r}, "
            f"but this build reads version {expected}"
        )


SAVED_AT_KEY = "_saved_at"


@dataclass(frozen=True)
class PendingRetentionPolicy:
    """When an abandoned suspended session may be reclaimed. Stated, never hidden.

    A suspended session waits for a user who may never come back. Nothing else
    reclaims it: `/cancel_pending` needs a client that has already left, and
    completion needs an answer that is never given.

    * `max_age_seconds` — reclaim state nothing has written to for this long.
      **Not a bound on bytes**: steady state is
      `abandonment_rate x max_age_seconds x blob_size`, a plateau set by a rate
      the operator does not control. Defaults to 7 days because the thing being
      deleted is a user's half-finished conversation, and the cost of keeping it
      too long is disk while the cost of deleting it too early is their work.
    * `max_entries` — a hard count cap, oldest-first regardless of age. This is
      what makes size independent of abandonment rate, which is why it is on by
      default.

    The age window is also the safety net: the count cap reclaims the oldest
    entry whatever its age, so a caller that under-reports its live channels is
    protected only by `max_age_seconds` outlasting a session. Setting it to None
    hands that responsibility entirely to `protected_channel_ids`.
    """

    max_age_seconds: Optional[float] = 604_800.0
    max_entries: Optional[int] = 10_000

    def __post_init__(self) -> None:
        if self.max_age_seconds is not None and self.max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive or None")
        if self.max_entries is not None and self.max_entries < 0:
            raise ValueError("max_entries must be non-negative or None")


@dataclass(frozen=True)
class PendingReapOutcome:
    reclaimed: int = 0
    protected: int = 0
    scanned: int = 0
    unreadable: int = 0


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

    @abstractmethod
    def iter_entries(self) -> Iterable[tuple[str, Optional[float]]]:
        """Yield (channel_id, saved_at) for every stored blob.

        channel_id comes from inside the blob rather than from the storage key,
        because neither backend's key is guaranteed to be reversible.

        A saved_at of None means the age could not be established; such an entry
        is reported and left alone rather than reclaimed on a guess.
        """

    @staticmethod
    def _stamp(state: dict[str, Any]) -> dict[str, Any]:
        """Attach the save time as storage metadata.

        Deliberately not part of SCHEMA_VERSION: when a blob was written is a
        fact about the store, not about the session, and putting it in the
        session schema would make every retention change a schema migration.
        """
        return {**state, SAVED_AT_KEY: time.time()}

    def reap(
        self,
        policy: Optional[PendingRetentionPolicy] = None,
        *,
        protected_channel_ids: Iterable[str] = (),
        now: Optional[float] = None,
        dry_run: bool = False,
    ) -> PendingReapOutcome:
        """Reclaim abandoned suspended state under a stated policy.

        Invoked, never scheduled: a timer inside the store would be a second
        writer on a channel whose only writer is meant to be the process that
        owns it. The caller names what it holds live, and a protected channel is
        never reclaimed however old it is.
        """
        policy = policy or PendingRetentionPolicy()
        now = time.time() if now is None else now
        protected = set(protected_channel_ids)

        entries: list[tuple[str, float]] = []
        scanned = unreadable = 0
        for channel_id, saved_at in self.iter_entries():
            scanned += 1
            if saved_at is None:
                unreadable += 1
                continue
            entries.append((channel_id, saved_at))

        protected_hits = sum(c in protected for c, _ in entries)
        candidates = [(c, t) for c, t in entries if c not in protected]
        candidates.sort(key=lambda pair: pair[1])

        doomed: list[str] = []
        if policy.max_age_seconds is not None:
            cutoff = now - policy.max_age_seconds
            doomed = [c for c, t in candidates if t < cutoff]

        if policy.max_entries is not None and len(entries) > policy.max_entries:
            # Count against everything stored, including protected entries, or a
            # process holding many live sessions would silently raise the cap.
            over = len(entries) - policy.max_entries
            for channel_id, _ in candidates:
                if over <= 0:
                    break
                if channel_id not in doomed:
                    doomed.append(channel_id)
                    over -= 1

        if not dry_run:
            for channel_id in doomed:
                self.clear(channel_id)

        return PendingReapOutcome(
            reclaimed=len(doomed),
            protected=protected_hits,
            scanned=scanned,
            unreadable=unreadable,
        )


PENDING_SUFFIX = ".pending.json"

# What a blob written before fix-7hn is called. The old derivation folded
# separators -- `channel_id.replace(os.sep, "_").replace("/", "_")` -- so
# `tenant/user-1` and `tenant_user-1` addressed one file and whichever suspended
# second handed the other user its ReAct trajectory and its ask_user answers.
#
# The two suffixes differ in the character at position -13 (`.` against `_`), so
# no legacy name can ever equal a new one. That is load-bearing rather than
# cosmetic: a folded name like `tenant_a` is exactly what the new encoder emits
# for the channel id `tenant_a`, so had the suffix been kept, a new-form read
# could still land on a blob the old mapping had already conflated.
LEGACY_PENDING_SUFFIX = "_pending.json"


class DiskSessionStateStore(SessionStateStore):
    """One JSON file per channel under base_folder (portable, no pickle)."""

    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        os.makedirs(base_folder, exist_ok=True)

    def _json_path(self, channel_id: str) -> str:
        return os.path.join(
            self.base_folder,
            f"{encode_path_component(channel_id)}{PENDING_SUFFIX}",
        )

    def _legacy_json_path(self, channel_id: str) -> str:
        """The pre-fix-7hn name. Read and removed, never written -- transitional.

        A suspended session is a user's half-finished conversation, so an upgrade
        must not orphan what is already on disk. This whole fallback -- here, in
        `load`/`exists`, the second unlink in `clear` and the second suffix in
        `iter_entries` -- can go once every deployment has run longer since
        upgrading than `PendingRetentionPolicy.max_age_seconds`, because by then
        no legacy blob can still be a live suspension.
        """
        safe_id = channel_id.replace(os.sep, "_").replace("/", "_")
        return os.path.join(self.base_folder, f"{safe_id}{LEGACY_PENDING_SUFFIX}")

    def load(self, channel_id: str) -> Optional[dict[str, Any]]:
        path = self._json_path(channel_id)
        if not os.path.isfile(path):
            return self._load_legacy(channel_id)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_legacy(self, channel_id: str) -> Optional[dict[str, Any]]:
        """A blob still at its pre-fix-7hn name, but only if it is ours.

        The old name was shared by every channel id that folded onto it, so the
        channel_id inside the blob is the only evidence of ownership left. A
        mismatch reads as absent, and so does a blob too damaged to check:
        losing a suspended turn costs the user one re-ask, whereas serving one
        costs them a stranger's conversation. A blob carrying no channel_id at
        all cannot be checked either way and is served, as it was before.
        """
        path = self._legacy_json_path(channel_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(blob, dict):
            return None
        stored = blob.get("channel_id")
        if isinstance(stored, str) and stored != channel_id:
            return None
        return blob

    def save(self, channel_id: str, state: dict[str, Any]) -> None:
        path = self._json_path(channel_id)
        # Strictness is established upstream at serialize_state; encoding here
        # canonically keeps the two ends in agreement instead of re-coercing.
        with open(path, "w", encoding="utf-8") as f:
            f.write(encode_state(self._stamp(state)))
        # The legacy copy is superseded the instant that lands, and leaving it
        # would give one channel two entries in `iter_entries` -- one stale, one
        # fresh -- so the reaper could age out the stale entry and take the live
        # blob with it, since `clear` removes both names.
        legacy = self._legacy_json_path(channel_id)
        if os.path.isfile(legacy):
            os.remove(legacy)

    def clear(self, channel_id: str) -> None:
        # Both names, always: a session that can be resurrected from its legacy
        # file has not been cleared. Where two folded ids shared that file this
        # can drop the other one's blob, which is the lesser harm -- that blob
        # was already destined for whichever channel saved next.
        for path in (
            self._json_path(channel_id),
            self._legacy_json_path(channel_id),
        ):
            if os.path.isfile(path):
                os.remove(path)

    def exists(self, channel_id: str) -> bool:
        if os.path.isfile(self._json_path(channel_id)):
            return True
        # Agreeing with `load` is worth more than saving a read: a legacy blob at
        # a shared name belongs to this channel only if it says so.
        return self._load_legacy(channel_id) is not None

    def iter_entries(self) -> Iterable[tuple[str, Optional[float]]]:
        if not os.path.isdir(self.base_folder):
            return
        for entry in os.scandir(self.base_folder):
            if not entry.is_file() or not entry.name.endswith(
                (PENDING_SUFFIX, LEGACY_PENDING_SUFFIX)
            ):
                continue
            try:
                with open(entry.path, encoding="utf-8") as f:
                    blob = json.load(f)
                channel_id = blob["channel_id"]
                saved_at = blob.get(SAVED_AT_KEY)
                if saved_at is None:
                    # Written before this store stamped a save time. mtime is
                    # the same fact from the filesystem, so those blobs age out
                    # normally instead of being immortal for want of a field.
                    saved_at = entry.stat().st_mtime
            except (OSError, ValueError, KeyError, TypeError):
                # Genuinely unreadable: counted, never guessed at. Deleting a
                # blob whose age cannot be established would reclaim by
                # assumption, which is how a reaper eats live state.
                yield ("", None)
                continue
            yield (channel_id, float(saved_at))


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
        self._client.set(self._key(channel_id), encode_state(self._stamp(state)))

    def clear(self, channel_id: str) -> None:
        self._client.delete(self._key(channel_id))

    def exists(self, channel_id: str) -> bool:
        return bool(self._client.exists(self._key(channel_id)))

    def iter_entries(self) -> Iterable[tuple[str, Optional[float]]]:
        # scan_iter, not keys(): keys() blocks the server for the whole scan,
        # and this runs against a shared multi-pod instance. Redis has no
        # per-key write time, so unlike the disk store there is no fallback for
        # a blob written before save times were stamped.
        for key in self._client.scan_iter(match=f"{self._prefix}*"):
            raw = self._client.get(key)
            if raw is None:
                continue
            try:
                blob = json.loads(raw)
                channel_id = blob["channel_id"]
                saved_at = blob.get(SAVED_AT_KEY)
            except (ValueError, KeyError, TypeError):
                yield ("", None)
                continue
            if saved_at is None:
                yield ("", None)
                continue
            yield (channel_id, float(saved_at))


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
