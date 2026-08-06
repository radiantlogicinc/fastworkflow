"""Durable channel checkpoints: one generation, one commit, one identity.

This is a *second* record kind, deliberately not sharing a key space with
`session_state_store`'s pending namespace (invariant 16). It exists because the
pending store's disk mapping is non-injective — `_json_path` collapses
``tenant/a`` and ``tenant_a`` onto one file — and revision 3 of the memory-bounds
design makes the checkpoint authoritative. Making an authoritative record inherit
a collision does not inherit a defect, it enlarges one.

Three hazards drive nearly every decision here.

**Aliasing.** Two channels must never address one record. Encoding is injective
even after the filesystem has had its say: case folding on a case-insensitive
volume and NFD normalization on HFS+ are both silent aliasing, so the on-disk
name is restricted to lowercase ASCII (§`encode_path_component`). The raw channel
id also travels *inside* every record and is checked on read, so the one residual
aliasing route — a SHA-256 collision in the oversized-id tail hash — degrades to
quarantine rather than to serving one channel another channel's state.

**Torn multi-record writes.** A checkpoint spans four records. `os.replace()`
makes one file atomic; it does not make four files one transaction. Restoring
context from generation N against a continuation from N-1 replays or drops
effects, so the reader must be unable to see a partial generation at all
(§`ChannelCheckpointStore.publish`).

**Silent version skew.** A node that cannot read a record must not write around
it. That is why protocol version is *not* part of the directory path: a v1 node
must land on the v2 record, fail closed, and preserve it — not miss it and
publish a competing lineage beside it (invariant 31).

**Unbounded durable growth.** A store with no lifecycle is a leak with better
paperwork: the motivating workload issues a unique channel id per request, so
every checkpoint it writes is read by nobody and kept forever (§11.10, ≈700
MB/day). Reclamation is therefore part of this module rather than an operational
afterthought — but it is an *invoked* `reap`, never a background thread, because
decision 16 rules out a hidden TTL and because a sweeper that writes behind the
caller's back would break single-writer-per-channel (invariant 15). See
`RetentionPolicy` for the policy and `stats` for the measurement.

**What the lifecycle here does not bound**, stated because a bound nobody has
written the exceptions down for is a claim rather than a bound:

* Nothing is reclaimed unless somebody calls `reap`. The store has no timer. A
  deployment that never calls it grows exactly as it did before.
* Between passes, growth is unbounded. `max_channels` and `max_bytes` are
  enforced at reap time, so peak size is the cap plus whatever arrives before the
  next pass.
* One record's size is not bounded. A single channel with a gigabyte of context
  satisfies `max_channels=1000`.
* A channel a live process could still adopt is safe only if the caller names it
  in `protected_channel_ids`, or if `max_age_seconds` outlasts a session. The
  store cannot see runtimes, leases or in-flight publishes.
* Capacity caps reclaim young channels under pressure. A session that would have
  adopted a checkpoint can find it gone; that is the price of a bound that holds
  whatever the arrival rate is, and it is stated rather than hidden.
* The caps are per namespace, so a fleet with N deployment/fingerprint pairs is
  bounded at N times the cap, and a fingerprint that changes on every deploy
  creates a new namespace whose predecessor is only reclaimed by age.
* Other record kinds are out of scope: the pending suspended-session namespace
  is `fix-6b4`, and conversation records are neither.
* Physical-byte accounting is the filesystem's answer, not an exact one —
  compression, tail packing and block size all move it.

Nothing here logs payloads, context values, or credentials. A warning names the
channel, the offending field, the backend, and the exception class; never a
value.

Backend seam: key derivation, record construction and identity validation are
backend-independent module-level functions, and the composite-generation protocol
is expressed as "prepare invisibly, then flip one pointer" — which a Redis
backend implements with same-slot hash-tagged keys (`{channel_key}`) and a
single commit key. Only the publication primitives below are disk-specific. No
abstract base class is declared until there is a second backend to shape it.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Optional

from fastworkflow.state_serialization import (
    StateEncodingError,
    encode_state,
    state_digest,
)
from fastworkflow.utils.logging import logger

PROTOCOL_VERSION = 1

RECORD_TYPE = "channel_checkpoint"
MANIFEST_RECORD_TYPE = "channel_checkpoint_manifest"
COMMIT_RECORD_TYPE = "channel_checkpoint_commit"
ACCESS_RECORD_TYPE = "channel_checkpoint_access"
FLOOR_RECORD_TYPE = "channel_checkpoint_generation_floor"

BACKEND = "disk"

# The design's representative payload (§5.2). Used only to state a count cap's
# byte cost, because a count cap without one is unauditable `[R12]`.
REPRESENTATIVE_RECORD_BYTES = 450 * 1024

# The four records a checkpoint spans. Splitting them is not decoration: the
# design's crash schedule is exactly "context published, continuation not", so a
# store that cannot express more than one record per generation cannot be tested
# against the failure it exists to prevent.
PART_SECTIONS = ("context", "runtime", "startup", "launch_context")

_CHANNELS_DIRNAME = "channels"
_QUARANTINE_DIRNAME = "__quarantine__"
# Where a directory waits between "no longer visible" and "actually gone". Every
# destructive operation is a rename into here followed by a bulk delete, so an
# interrupted one leaves debris that no reader enumerates instead of a
# half-removed channel that reads as a partial generation.
_RECLAIM_DIRNAME = "__reclaim__"
_GENERATIONS_DIRNAME = "gen"
_COMMIT_FILENAME = "COMMIT"
_MANIFEST_FILENAME = "manifest.json"
# Liveness, deliberately not state: see `_touch_access`.
_ACCESS_FILENAME = "ACCESS"
# Survives `reset` so generation numbers never repeat for a channel id.
_FLOOR_FILENAME = "GENERATION_FLOOR"
_PENDING_PREFIX = ".pending-"
_TEMP_INFIX = ".tmp-"

_DIR_MODE = 0o700
_FILE_MODE = 0o600

# Stands in for the incarnation an adoption read does not have. It is never
# compared and never written; the value is self-describing so that if it ever
# does surface in a record or a log, the bug names itself.
_ADOPTION_PROBE_INCARNATION = "!adoption-probe-not-a-real-incarnation"

# NAME_MAX is 255 bytes almost everywhere; percent-encoding can triple a name.
# 200 leaves room for the ".{64 hex}" tail and for temp-name suffixes.
_MAX_NAME_LEN = 200

# The committed generation plus one recovery point, matching the convention the
# training artifacts already use.
_GENERATIONS_RETAINED = 2

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)

# Lowercase only. Uppercase is escaped because a case-insensitive volume folds
# `Tenant` onto `tenant`, and a fold is an alias the record-level identity check
# would only ever report *after* one channel had already overwritten the other.
# `.` is escaped too, which reserves it as an unambiguous delimiter for the
# oversized-id tail hash and rules out the `.` and `..` directory names.
_UNRESERVED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")


class CheckpointStoreError(RuntimeError):
    """Publication refused. The caller keeps its runtime live (invariant 8)."""


class QuarantineReason(Enum):
    """Why a record was set aside instead of applied."""

    RECORD_TYPE_MISMATCH = "record_type_mismatch"
    PROTOCOL_VERSION_UNREADABLE = "protocol_version_unreadable"
    DEPLOYMENT_MISMATCH = "deployment_mismatch"
    WORKFLOW_FINGERPRINT_MISMATCH = "workflow_fingerprint_mismatch"
    CHANNEL_KEY_MISMATCH = "channel_key_mismatch"
    CHANNEL_ID_MISMATCH = "channel_id_mismatch"
    SESSION_INCARNATION_MISMATCH = "session_incarnation_mismatch"
    GENERATION_MISMATCH = "generation_mismatch"
    INCOHERENT_GENERATION = "incoherent_generation"
    INCOMPLETE_GENERATION = "incomplete_generation"
    DIGEST_MISMATCH = "digest_mismatch"
    UNREADABLE_RECORD = "unreadable_record"
    OPERATOR_REQUEST = "operator_request"


def encode_path_component(raw: str) -> str:
    """Percent-encode ``raw`` into an injective, filesystem-safe ASCII name.

    Injective because decoding is unambiguous: `%` is itself always escaped, so
    every `%` in the output opens an escape, and every other output character
    stands for itself. Hence ``tenant/a`` -> ``tenant%2Fa``, ``tenant_a`` ->
    ``tenant_a`` and ``tenant%2Fa`` -> ``tenant%252Fa`` are three names.

    Injective *after case folding* as well, because escapes emit uppercase hex
    and nothing else emits a capital: two outputs that fold together must have
    identical `%` positions and therefore be identical.

    Oversized ids fall back to ``<prefix>.<sha256 of the raw id>``. That is
    collision-*resistant* rather than injective, which is why the raw id is also
    stored in the record and compared on read: an astronomically unlikely
    collision quarantines instead of cross-serving state.
    """
    if not isinstance(raw, str) or not raw:
        raise ValueError("path component must be a non-empty str")

    encoded = "".join(
        chr(byte) if chr(byte) in _UNRESERVED else f"%{byte:02X}"
        for byte in raw.encode("utf-8")
    )
    if len(encoded) <= _MAX_NAME_LEN:
        return encoded

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    prefix = _trim_partial_escape(encoded[: _MAX_NAME_LEN - len(digest) - 1])
    return f"{prefix}.{digest}"


def _trim_partial_escape(text: str) -> str:
    """Drop a truncated `%XX` so the readable prefix stays decodable."""
    if text.endswith("%"):
        return text[:-1]
    return text[:-2] if len(text) >= 2 and text[-2] == "%" else text


@dataclass(frozen=True)
class CheckpointIdentity:
    """What a record must prove it belongs to before any of it is applied."""

    deployment_id: str
    workflow_fingerprint: str
    channel_id: str
    session_incarnation: str

    def __post_init__(self) -> None:
        for name in (
            "deployment_id",
            "workflow_fingerprint",
            "channel_id",
            "session_incarnation",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")

    @property
    def channel_key(self) -> str:
        return encode_path_component(self.channel_id)

    def scope(self) -> dict[str, str]:
        """The fields that address a channel, without the session lifetime.

        These are the three a cold restore already knows. Spelling them out as a
        set makes ``store.load_for_adoption(**identity.scope())`` read as what it
        is, rather than as an identity check that happens to skip a field.
        """
        return {
            "deployment_id": self.deployment_id,
            "workflow_fingerprint": self.workflow_fingerprint,
            "channel_id": self.channel_id,
        }

    def as_fields(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "workflow_fingerprint": self.workflow_fingerprint,
            "channel_key": self.channel_key,
            "channel_id": self.channel_id,
            "session_incarnation": self.session_incarnation,
        }


@dataclass(frozen=True)
class CheckpointRecord:
    """The logical envelope, reassembled from the parts of one generation."""

    identity: CheckpointIdentity
    generation: int
    state_version: int
    context: dict[str, Any]
    runtime: dict[str, Any]
    startup: dict[str, Any]
    launch_context: dict[str, Any]
    protocol_version: int = PROTOCOL_VERSION

    def to_payload(self) -> dict[str, Any]:
        """The design's §11.3 shape, as one dictionary."""
        return {
            "protocol_version": self.protocol_version,
            "record_type": RECORD_TYPE,
            "generation": self.generation,
            **self.identity.as_fields(),
            "state_version": self.state_version,
            "context": self.context,
            "runtime": self.runtime,
            "startup": self.startup,
            "launch_context": self.launch_context,
        }


@dataclass(frozen=True)
class RetentionPolicy:
    """When a channel's durable bytes may be reclaimed. Stated, never hidden.

    Three knobs, because they bound different things and only one of them is
    actually a bound:

    * `max_age_seconds` — reclaim a channel nothing has published to for this
      long. **This is not a bound on bytes.** Steady-state size is
      `arrival_rate x max_age_seconds x record_size`, so at the design's
      motivating workload (65 unique channels/hour, 450 KB each) a 24-hour
      window plateaus at ≈700 MB. It plateaus — which is what §16.5 asks for —
      but the plateau is set by a rate the operator does not control.
    * `max_channels` — a hard count cap, enforced oldest-first regardless of
      age. This is what makes size independent of arrival rate, which is why it
      is on by default. `describe()` states its byte cost, because a count cap
      without one is unauditable `[R12]`.
    * `max_bytes` — a hard byte cap, the same bound stated in the units §16.5
      gates on. Off by default only because the count cap is the cheaper
      equivalent at a known payload size; set it when payload size varies.

    The age window is also the reaper's safety net. Capacity caps reclaim the
    oldest channel whatever its age, so a caller who under-reports its live
    channels is protected only by `max_age_seconds` being longer than a session
    lifetime. Setting it to None hands that responsibility entirely to
    `protected_channel_ids`.

    Quarantined records get their own two knobs. They are evidence, so they are
    kept longer and reclaimed on the same stated terms rather than forever —
    accumulating them without limit is just a slower version of the leak this
    policy exists to close.
    """

    max_age_seconds: Optional[float] = 86_400.0
    max_channels: Optional[int] = 1_000
    max_bytes: Optional[int] = None
    quarantine_max_age_seconds: Optional[float] = 604_800.0
    quarantine_max_entries: Optional[int] = 100

    def __post_init__(self) -> None:
        for name in ("max_age_seconds", "quarantine_max_age_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number or None")
            if value != value or value <= 0:  # NaN fails the first comparison
                raise ValueError(f"{name} must be > 0")
        for name in ("max_channels", "max_bytes", "quarantine_max_entries"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int or None")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")

    def worst_case_bytes(
        self, representative_record_bytes: int = REPRESENTATIVE_RECORD_BYTES
    ) -> Optional[int]:
        """Ceiling on one namespace's live bytes, or None if there is no ceiling.

        `max_age_seconds` contributes nothing here on purpose: it bounds age, and
        age times an unknown arrival rate is not a number.
        """
        candidates = [
            self.max_channels * representative_record_bytes * _GENERATIONS_RETAINED
            if self.max_channels is not None
            else None,
            self.max_bytes,
        ]
        return min((value for value in candidates if value is not None), default=None)

    def describe(
        self, representative_record_bytes: int = REPRESENTATIVE_RECORD_BYTES
    ) -> str:
        """One line for the §17 startup record, with the byte cost spelled out."""
        ceiling = self.worst_case_bytes(representative_record_bytes)
        return (
            "checkpoint retention: "
            f"max_age={_describe_seconds(self.max_age_seconds)}, "
            f"max_channels={self.max_channels if self.max_channels is not None else 'unbounded'}, "
            f"max_bytes={_human_bytes(self.max_bytes) if self.max_bytes is not None else 'unbounded'}, "
            f"quarantine_max_age={_describe_seconds(self.quarantine_max_age_seconds)}, "
            f"quarantine_max_entries="
            f"{self.quarantine_max_entries if self.quarantine_max_entries is not None else 'unbounded'}"
            f" -> live ceiling "
            f"{_human_bytes(ceiling) if ceiling is not None else 'UNBOUNDED'} per namespace "
            f"at {_human_bytes(representative_record_bytes)}/record x "
            f"{_GENERATIONS_RETAINED} retained generations"
        )


DEFAULT_RETENTION = RetentionPolicy()


@dataclass(frozen=True)
class ChannelInfo:
    """The `inspect` verb's answer: what is on disk for one channel.

    Identity fields are Optional because they are read *out of* the record. A
    channel whose COMMIT is missing or unparseable still has a directory, a size
    and an age — which is exactly when an operator most needs to see it — so
    inspection degrades to `channel_key` rather than raising. `channel_id` is
    unrecoverable from the path alone for an oversized id (the name carries a
    hash tail), so the record is the only source.
    """

    channel_key: str
    directory: str
    deployment_id: Optional[str] = None
    workflow_fingerprint: Optional[str] = None
    channel_id: Optional[str] = None
    session_incarnation: Optional[str] = None
    generation: Optional[int] = None
    generation_floor: int = 0
    published_at: Optional[float] = None
    last_seen_at: Optional[float] = None
    activity_at: float = 0.0
    generations_on_disk: int = 0
    file_count: int = 0
    bytes_apparent: int = 0
    bytes_physical: int = 0
    committed: bool = False

    def identity(self) -> Optional[CheckpointIdentity]:
        """A usable identity, or None when the record could not be read.

        Returning None rather than a partially-filled identity is deliberate: an
        identity is the thing every other operation validates against, so a
        guessed one would be worse than no answer.
        """
        if not all(
            (
                self.deployment_id,
                self.workflow_fingerprint,
                self.channel_id,
                self.session_incarnation,
            )
        ):
            return None
        return CheckpointIdentity(
            deployment_id=self.deployment_id,
            workflow_fingerprint=self.workflow_fingerprint,
            channel_id=self.channel_id,
            session_incarnation=self.session_incarnation,
        )

    def age_seconds(self, now: Optional[float] = None) -> float:
        return max(0.0, (time.time() if now is None else now) - self.activity_at)


@dataclass(frozen=True)
class QuarantineEntry:
    """One preserved quarantine directory, as `list_quarantine_entries` sees it."""

    path: str
    channel_key: str
    reason: Optional[str] = None
    quarantined_at: float = 0.0
    file_count: int = 0
    bytes_apparent: int = 0
    bytes_physical: int = 0

    def age_seconds(self, now: Optional[float] = None) -> float:
        return max(0.0, (time.time() if now is None else now) - self.quarantined_at)


@dataclass(frozen=True)
class NamespaceStats:
    """What §16.5's plateau gate and §17's metric both need.

    `total_bytes_physical` is the gated number. Apparent bytes are reported
    alongside it because they are exactly reproducible from file sizes and so
    can be asserted in a test, whereas physical bytes depend on the filesystem's
    block size, tail packing and compression.

    Debris is counted separately and included in the totals. Excluding it would
    let an interrupted reap look like reclaimed space that is still occupied.
    """

    deployment_id: Optional[str] = None
    workflow_fingerprint: Optional[str] = None
    namespaces: int = 0
    channels: int = 0
    committed_channels: int = 0
    generations: int = 0
    file_count: int = 0
    bytes_apparent: int = 0
    bytes_physical: int = 0
    quarantined_entries: int = 0
    quarantined_file_count: int = 0
    quarantined_bytes_apparent: int = 0
    quarantined_bytes_physical: int = 0
    reclaimable_entries: int = 0
    reclaimable_file_count: int = 0
    reclaimable_bytes_apparent: int = 0
    reclaimable_bytes_physical: int = 0

    @property
    def total_bytes_apparent(self) -> int:
        return (
            self.bytes_apparent
            + self.quarantined_bytes_apparent
            + self.reclaimable_bytes_apparent
        )

    @property
    def total_bytes_physical(self) -> int:
        return (
            self.bytes_physical
            + self.quarantined_bytes_physical
            + self.reclaimable_bytes_physical
        )

    @property
    def total_files(self) -> int:
        """Every file the namespace occupies, live or preserved or awaiting sweep."""
        return (
            self.file_count
            + self.quarantined_file_count
            + self.reclaimable_file_count
        )

    def describe(self) -> str:
        """One line for the §17 metric, in the units §16.5's plateau gate reads."""
        scope = (
            f"{self.deployment_id or '*'}/{self.workflow_fingerprint or '*'}"
        )
        return (
            f"checkpoint namespace {scope}: "
            f"channels={self.channels} (committed={self.committed_channels}) "
            f"generations={self.generations} files={self.total_files} "
            f"bytes_physical={_human_bytes(self.total_bytes_physical)} "
            f"bytes_apparent={_human_bytes(self.total_bytes_apparent)} "
            f"quarantined={self.quarantined_entries} "
            f"awaiting_sweep={self.reclaimable_entries}"
        )


@dataclass(frozen=True)
class ReapReport:
    """What one `reap` pass did, in the units an operator and a soak both need."""

    dry_run: bool = False
    scanned_channels: int = 0
    protected_channels: int = 0
    retained_channels: int = 0
    reclaimed_channels: int = 0
    reclaimed_bytes_apparent: int = 0
    reclaimed_bytes_physical: int = 0
    aged_out: int = 0
    over_capacity: int = 0
    reclaimed_quarantine_entries: int = 0
    reclaimed_quarantine_bytes: int = 0
    swept_debris_entries: int = 0
    swept_debris_bytes: int = 0
    failures: int = 0
    reclaimed_keys: tuple[str, ...] = ()

    def summary(self) -> str:
        return (
            f"checkpoint reap{' (dry run)' if self.dry_run else ''}: "
            f"scanned={self.scanned_channels} protected={self.protected_channels} "
            f"retained={self.retained_channels} reclaimed={self.reclaimed_channels} "
            f"(aged_out={self.aged_out} over_capacity={self.over_capacity}) "
            f"bytes={_human_bytes(self.reclaimed_bytes_apparent)} "
            f"quarantine_entries={self.reclaimed_quarantine_entries} "
            f"debris={self.swept_debris_entries} failures={self.failures}"
        )


@dataclass
class _Usage:
    """Mutable accumulator; only ever converted into a frozen report."""

    entries: int = 0
    files: int = 0
    apparent: int = 0
    physical: int = 0

    def add(self, files: int, apparent: int, physical: int) -> None:
        self.entries += 1
        self.files += files
        self.apparent += apparent
        self.physical += physical

    def add_tree(self, path: str) -> None:
        self.add(*_dir_usage(path))


class _Unreadable(Exception):
    """Internal: a record is present but must not be applied."""

    def __init__(self, reason: QuarantineReason, field: str):
        super().__init__(f"{reason.value}:{field}")
        self.reason = reason
        self.field = field


class ChannelCheckpointStore:
    """Disk-backed composite checkpoints, one directory per channel.

    Layout::

        <base>/channels/<dep>/<fingerprint>/<channel_key>/
            COMMIT                     <- the only thing a reader starts from
            ACCESS                     <- liveness for retention; no reader looks
            GENERATION_FLOOR           <- only after a reset; see `reset`
            gen/<N>/manifest.json
            gen/<N>/{context,runtime,startup,launch_context}.json
        <base>/__quarantine__/<dep>/<fingerprint>/<channel_key>/<reason>-<ts>-<uuid>/
        <base>/__reclaim__/<dep>/<fingerprint>/<flipped away, awaiting deletion>

    `protocol_version` is *not* a path element. Partitioning by version would let
    a v1 node miss a v2 record entirely and publish a competing lineage beside
    it, which is precisely what invariant 31 forbids. `session_incarnation` is
    not one either, for the same reason inverted: a recycled channel id must land
    on the existing record and be refused, not miss it and start a rival lineage.

    Reads come in two kinds because they answer different questions:

    * `load` — "I am session X; is this still my state?" A different stored
      incarnation means the channel id was reused, so it quarantines.
    * `load_for_adoption` — "which session was this channel?" A cold restore
      cannot assert an incarnation it has not read yet, so this discovers it.

    Writes only ever come in the strict kind: `publish` refuses to overwrite a
    record belonging to another incarnation.

    Lifecycle is the four operator verbs the design names — `inspect`,
    `quarantine`, `delete`, `reset` — plus `stats` to measure the namespace and
    `reap` to reclaim it under a stated `RetentionPolicy`. Every destructive
    operation among them is one rename into `__reclaim__/` followed by a bulk
    delete, which is `publish`'s prepare-then-flip run backwards.
    """

    def __init__(
        self,
        base_folder: str,
        *,
        protocol_version: int = PROTOCOL_VERSION,
        min_readable_protocol_version: int = PROTOCOL_VERSION,
    ):
        if protocol_version < 1:
            raise ValueError("protocol_version must be >= 1")
        if min_readable_protocol_version > protocol_version:
            raise ValueError(
                "min_readable_protocol_version cannot exceed the version this "
                "node writes; it would make the node unable to read itself"
            )
        self._base = os.path.abspath(base_folder)
        self._protocol_version = protocol_version
        self._min_readable = min_readable_protocol_version

    # -- paths ---------------------------------------------------------------

    @property
    def base_folder(self) -> str:
        return self._base

    def channel_directory(self, identity: CheckpointIdentity) -> str:
        """Where this identity's records live. Exposed for operator inspection."""
        return os.path.join(
            self._base,
            _CHANNELS_DIRNAME,
            encode_path_component(identity.deployment_id),
            encode_path_component(identity.workflow_fingerprint),
            identity.channel_key,
        )

    def quarantine_directory(self, identity: CheckpointIdentity) -> str:
        return os.path.join(
            self._base,
            _QUARANTINE_DIRNAME,
            encode_path_component(identity.deployment_id),
            encode_path_component(identity.workflow_fingerprint),
            identity.channel_key,
        )

    def _namespace_dir(self, root_name: str, dep_key: str, fp_key: str) -> str:
        return os.path.join(self._base, root_name, dep_key, fp_key)

    def _reclaim_dir(self, dep_key: str, fp_key: str) -> str:
        """Per-namespace so debris is attributable to the namespace it came from.

        A store-wide holding area would make `stats(deployment_id=...)` unable to
        say whether an interrupted reap's bytes belong to the namespace being
        measured, which is the one question §16.5 gates on.
        """
        return self._namespace_dir(_RECLAIM_DIRNAME, dep_key, fp_key)

    def _commit_path(self, identity: CheckpointIdentity) -> str:
        return os.path.join(self.channel_directory(identity), _COMMIT_FILENAME)

    def _generations_dir(self, identity: CheckpointIdentity) -> str:
        return os.path.join(self.channel_directory(identity), _GENERATIONS_DIRNAME)

    # -- public API ----------------------------------------------------------

    def exists(self, identity: CheckpointIdentity) -> bool:
        """True when a committed generation is present.

        Presence, not usability: a record that will quarantine on read still
        exists. `load` is the only authority on whether it can be applied.
        """
        return _is_regular_file(self._commit_path(identity))

    def load(self, identity: CheckpointIdentity) -> Optional[CheckpointRecord]:
        """Continuation read: the caller already knows which session it is.

        Returns the committed record, or None when absent or quarantined. None
        always means the same thing to the caller: start from launch
        configuration. Nothing is ever partially applied.

        A stored `session_incarnation` that differs from the caller's means this
        channel id has been reused for a different session lifetime, so the
        record quarantines. Use `load_for_adoption` when you do not yet know
        which incarnation you are.
        """
        return self._load(identity, adopt_incarnation=False)

    def load_for_adoption(
        self,
        *,
        deployment_id: str,
        workflow_fingerprint: str,
        channel_id: str,
    ) -> Optional[CheckpointRecord]:
        """Cold-restore read: take on whatever session lifetime the record names.

        A fresh process cannot know the stored incarnation before it reads it, so
        asserting one would quarantine every healthy record across every restart
        and make checkpoints worthless. This discovers the incarnation from
        COMMIT instead and returns it on `record.identity.session_incarnation`
        for the caller to adopt.

        It takes the three fields a cold restore actually has rather than a full
        `CheckpointIdentity` with one field ignored: a parameter that is not used
        is a lie in the signature, and it hides at the call site exactly which
        check was relaxed. `CheckpointIdentity.scope()` bridges the two.

        Only the comparison against the *caller* is relaxed. The records still
        have to agree with each other, because the discovered incarnation becomes
        the expected value for the manifest and every part — so a generation
        assembled from two different session lifetimes still fails closed.
        """
        probe = CheckpointIdentity(
            deployment_id=deployment_id,
            workflow_fingerprint=workflow_fingerprint,
            channel_id=channel_id,
            session_incarnation=_ADOPTION_PROBE_INCARNATION,
        )
        return self._load(probe, adopt_incarnation=True)

    def _load(
        self, identity: CheckpointIdentity, *, adopt_incarnation: bool
    ) -> Optional[CheckpointRecord]:
        try:
            return self._load_committed(
                identity, adopt_incarnation=adopt_incarnation
            )
        except _Unreadable as problem:
            self._quarantine(identity, problem.reason, field=problem.field)
            return None

    def publish(
        self,
        identity: CheckpointIdentity,
        *,
        context: dict[str, Any],
        runtime: dict[str, Any],
        startup: dict[str, Any],
        launch_context: dict[str, Any],
        state_version: int,
    ) -> int:
        """Publish one composite generation atomically; return its number.

        Crash safety rests on there being exactly one operation that makes the
        new generation visible — the `os.replace()` of COMMIT — and on every
        earlier write landing somewhere no reader looks:

        1. parts and manifest are written into ``gen/.pending-N-<uuid>/``;
           a crash here leaves an orphan directory the reader never enumerates.
        2. that directory is renamed to ``gen/N``; the rename is atomic, and
           even once it lands COMMIT still names the previous generation, so a
           complete-but-uncommitted generation is invisible.
        3. COMMIT is replaced. Before: the old generation, whole. After: the new
           one, whole, because step 2 had already finished. There is no
           in-between.
        4. superseded generations are reaped; a crash here loses only garbage.

        Encoding is validated before anything is created, so a state that cannot
        be encoded losslessly leaves no directory behind at all.

        The liveness marker is refreshed on every successful call, including the
        no-write fast path, so retention ages a channel from its last *use* and
        not from its last content change (see `_touch_access`).
        """
        now = time.time()
        if not isinstance(state_version, int) or isinstance(state_version, bool):
            raise ValueError("state_version must be an int")
        if state_version < 1:
            raise ValueError("state_version must be >= 1")

        sections = {
            "context": context,
            "runtime": runtime,
            "startup": startup,
            "launch_context": launch_context,
        }
        for name, value in sections.items():
            if not isinstance(value, dict):
                raise StateEncodingError(
                    f"{name} must be a dict, got {type(value).__name__}"
                )

        semantic = {
            "protocol_version": self._protocol_version,
            "record_type": RECORD_TYPE,
            **identity.as_fields(),
            "state_version": state_version,
            **sections,
        }
        # Digesting the whole envelope also validates it, and validating the
        # sections *together* is what catches a mutable container shared across
        # two of them — which would restore as two independent objects.
        content_digest = state_digest(semantic)

        committed = self._read_commit_for_write(identity)
        if (
            committed is not None
            and committed.get("content_digest") == content_digest
            and self._generation_is_structurally_present(
                identity, committed["generation"]
            )
        ):
            # Identity is inside the digested payload, so a digest match is also
            # an identity match. The residual gap is content-level bitrot in a
            # part file, which `load` still catches and quarantines.
            generation = int(committed["generation"])
            self._touch_access(identity, generation, now=now)
            return generation

        channel_dir = self._ensure_dir_chain(self.channel_directory(identity))
        generations_dir = self._ensure_dir_chain(self._generations_dir(identity))

        generation = self._next_generation(
            generations_dir,
            max(
                int(committed["generation"]) if committed else 0,
                self._generation_floor(identity),
            ),
        )

        pending_dir = os.path.join(
            generations_dir, f"{_PENDING_PREFIX}{generation}-{uuid.uuid4().hex}"
        )
        self._ensure_private_dir(pending_dir)

        part_digests: dict[str, str] = {}
        for section, value in sections.items():
            text = encode_state(
                {
                    "protocol_version": self._protocol_version,
                    "record_type": RECORD_TYPE,
                    "generation": generation,
                    **identity.as_fields(),
                    "state_version": state_version,
                    "section": section,
                    "payload": value,
                }
            )
            self._write_private_file(
                os.path.join(pending_dir, f"{section}.json"), text
            )
            part_digests[section] = _digest_text(text)

        manifest_text = encode_state(
            {
                "protocol_version": self._protocol_version,
                "record_type": MANIFEST_RECORD_TYPE,
                "generation": generation,
                **identity.as_fields(),
                "state_version": state_version,
                "content_digest": content_digest,
                "parts": part_digests,
            }
        )
        self._write_private_file(
            os.path.join(pending_dir, _MANIFEST_FILENAME), manifest_text
        )
        _fsync_dir(pending_dir)

        generation_dir = os.path.join(generations_dir, str(generation))
        _replace(pending_dir, generation_dir)
        _fsync_dir(generations_dir)

        commit_text = encode_state(
            {
                "protocol_version": self._protocol_version,
                "record_type": COMMIT_RECORD_TYPE,
                "generation": generation,
                **identity.as_fields(),
                "content_digest": content_digest,
                "manifest_digest": _digest_text(manifest_text),
                # Retention metadata, and deliberately only here: COMMIT holds
                # the digests of the other records and is itself covered by
                # none, so a field that changes on every publish costs nothing.
                # Inside a part it would change `content_digest` on every
                # retirement and force a full rewrite of an unchanged
                # several-hundred-kilobyte context (§11.4).
                "published_at": now,
            }
        )
        commit_tmp = os.path.join(
            channel_dir, f"{_COMMIT_FILENAME}{_TEMP_INFIX}{uuid.uuid4().hex}"
        )
        self._write_private_file(commit_tmp, commit_text)
        _replace(commit_tmp, self._commit_path(identity))
        _fsync_dir(channel_dir)

        self._prune_generations(
            generations_dir,
            keep={generation - offset for offset in range(_GENERATIONS_RETAINED)},
        )
        self._touch_access(identity, generation, now=now)
        return generation

    def quarantine(
        self, identity: CheckpointIdentity, reason: QuarantineReason
    ) -> None:
        """Set this channel's records aside, preserved, never deleted."""
        self._quarantine(identity, reason, field="")

    def list_quarantined(self, identity: CheckpointIdentity) -> list[str]:
        """Preserved quarantine directories for this channel, in name order.

        Names are `<reason>-<unix seconds>-<uuid>`, so the order is by reason
        then by time. Use `list_quarantine_entries` for sizes and ages.
        """
        root = self.quarantine_directory(identity)
        try:
            return sorted(
                os.path.join(root, name) for name in os.listdir(root)
            )
        except FileNotFoundError:
            return []

    # -- operator verbs: inspect, quarantine, delete, reset ------------------

    def inspect(self, identity: CheckpointIdentity) -> Optional[ChannelInfo]:
        """What is on disk for this channel, or None if there is no directory.

        Read-only and never quarantines: an operator asking what is wrong must
        not change the thing they are asking about. `load` remains the only
        authority on whether a record can be applied.
        """
        directory = self.channel_directory(identity)
        if not os.path.isdir(directory):
            return None
        return self._scan_channel(identity.channel_key, directory)

    def list_channels(
        self,
        *,
        deployment_id: Optional[str] = None,
        workflow_fingerprint: Optional[str] = None,
    ) -> list[ChannelInfo]:
        """Every channel in the namespace, oldest activity first.

        Scoped to one namespace when both arguments are given; otherwise the
        whole store. Ordering is total — activity time then channel key — so two
        channels published in the same clock tick still enumerate deterministically.
        """
        found: list[ChannelInfo] = []
        for dep_key, fp_key in self._namespace_keys(
            deployment_id, workflow_fingerprint
        ):
            found.extend(self._scan_namespace_channels(dep_key, fp_key))
        return sorted(found, key=lambda info: (info.activity_at, info.channel_key))

    def list_quarantine_entries(
        self,
        *,
        deployment_id: Optional[str] = None,
        workflow_fingerprint: Optional[str] = None,
    ) -> list[QuarantineEntry]:
        """Preserved records across the namespace, oldest first."""
        found: list[QuarantineEntry] = []
        for dep_key, fp_key in self._namespace_keys(
            deployment_id, workflow_fingerprint
        ):
            found.extend(self._scan_namespace_quarantine(dep_key, fp_key))
        return sorted(found, key=lambda entry: (entry.quarantined_at, entry.path))

    def delete(self, identity: CheckpointIdentity) -> bool:
        """Remove this channel as though it had never existed. True if it was there.

        One rename makes the whole channel invisible, then the bytes go. That
        ordering is the point: removing files in place would pass through states
        where COMMIT names a generation whose parts are gone, which reads as a
        damaged record and quarantines something the operator asked to delete.

        Quarantined copies are untouched — the reason they were set aside is that
        somebody needs to look at them — and so is nothing else in the namespace.

        A later session reusing this channel id starts a fresh lineage at
        generation 1 with its own incarnation, and cannot adopt the old one
        because the old one is not partially present anywhere a reader looks.
        Use `reset` instead when you want the numbering to keep going.
        """
        reclaimed, _apparent, _physical = self._retire_directory(
            self.channel_directory(identity),
            self._reclaim_dir(
                encode_path_component(identity.deployment_id),
                encode_path_component(identity.workflow_fingerprint),
            ),
        )
        return reclaimed

    def clear(self, identity: CheckpointIdentity) -> None:
        """Older name for `delete`, kept because callers already use it."""
        self.delete(identity)

    def reset(self, identity: CheckpointIdentity) -> int:
        """Forget this channel's state but remember its numbering. Returns the floor.

        The difference from `delete` is what a reused channel id sees afterwards.
        `delete` is amnesia: the next lineage starts at generation 1, which is
        correct but makes an old artifact and a new one numerically
        indistinguishable if one is ever restored from a backup beside the other.
        `reset` records the retired lineage's highest generation as a floor, so
        every future generation for this channel id is strictly greater than
        every past one and a number identifies a lineage.

        Crash-safe by ordering, in three steps whose failure directions all point
        the same way:

        1. write the floor. A crash here leaves the channel fully live with a
           floor recorded — harmless, because the floor only ever *raises* the
           next generation number.
        2. unlink COMMIT. This is the visibility flip, and it is atomic. Before
           it the record loads; after it the channel reads as absent, not as
           damaged, because a reader that finds no COMMIT looks no further.
        3. rename the generations away and delete them. A crash here leaves
           bytes that no reader enumerates and that the next `reap` sweeps.
        """
        directory = self.channel_directory(identity)
        if not os.path.isdir(directory):
            return 0

        floor = self._highest_generation(identity)
        if floor > 0:
            self._write_floor(identity, floor)

        _unlink(self._commit_path(identity))
        # The marker names a generation that is about to stop existing, and a
        # channel with no record has no last-publish time to report.
        _unlink(os.path.join(directory, _ACCESS_FILENAME))
        _fsync_dir(directory)

        self._retire_directory(
            self._generations_dir(identity),
            self._reclaim_dir(
                encode_path_component(identity.deployment_id),
                encode_path_component(identity.workflow_fingerprint),
            ),
        )
        return floor

    # -- measurement ---------------------------------------------------------

    def stats(
        self,
        *,
        deployment_id: Optional[str] = None,
        workflow_fingerprint: Optional[str] = None,
    ) -> NamespaceStats:
        """Total bytes and record counts, for §16.5's plateau gate and §17's metric.

        This is what the soak harness calls. `total_bytes_physical` is the number
        that must plateau; it walks real files rather than trusting a counter,
        because a counter that drifts is exactly how an unbounded store looks
        bounded.
        """
        channels = _Usage()
        quarantine = _Usage()
        reclaim = _Usage()
        committed = 0
        generations = 0

        namespaces = self._namespace_keys(deployment_id, workflow_fingerprint)
        for dep_key, fp_key in namespaces:
            for info in self._scan_namespace_channels(dep_key, fp_key):
                channels.add(
                    info.file_count, info.bytes_apparent, info.bytes_physical
                )
                generations += info.generations_on_disk
                if info.committed:
                    committed += 1
            for entry in self._scan_namespace_quarantine(dep_key, fp_key):
                quarantine.add(
                    entry.file_count, entry.bytes_apparent, entry.bytes_physical
                )
            for path in self._debris_paths(dep_key, fp_key):
                reclaim.add_tree(path)

        return NamespaceStats(
            deployment_id=deployment_id,
            workflow_fingerprint=workflow_fingerprint,
            namespaces=len(namespaces),
            channels=channels.entries,
            committed_channels=committed,
            generations=generations,
            file_count=channels.files,
            bytes_apparent=channels.apparent,
            bytes_physical=channels.physical,
            quarantined_entries=quarantine.entries,
            quarantined_file_count=quarantine.files,
            quarantined_bytes_apparent=quarantine.apparent,
            quarantined_bytes_physical=quarantine.physical,
            reclaimable_entries=reclaim.entries,
            reclaimable_file_count=reclaim.files,
            reclaimable_bytes_apparent=reclaim.apparent,
            reclaimable_bytes_physical=reclaim.physical,
        )

    # -- the reaper ----------------------------------------------------------

    def reap(
        self,
        policy: RetentionPolicy = DEFAULT_RETENTION,
        *,
        protected_channel_ids: Iterable[str] = (),
        deployment_id: Optional[str] = None,
        workflow_fingerprint: Optional[str] = None,
        now: Optional[float] = None,
        dry_run: bool = False,
    ) -> ReapReport:
        """Reclaim abandoned channels under `policy`. Returns what it did.

        Invoked, never scheduled. There is no thread and no timer here, for two
        reasons: decision 16 forbids a hidden durable-state TTL, and a background
        sweeper would be a second writer on a channel whose only writer is
        supposed to be the process that owns it (invariant 15).

        `protected_channel_ids` is how that invariant is honoured. The store
        cannot see live runtimes, leases or in-flight publishes, so the caller
        declares them and the reaper skips them — a required argument in spirit,
        defaulting to empty only so a dry run costs nothing. Under-report it and
        `policy.max_age_seconds` is the only thing standing between a live
        channel and its bytes.

        Crash safety is `publish`'s argument run backwards. Publication makes a
        complete generation visible with one rename; reclamation makes a complete
        channel invisible with one rename. Both directions have the same
        property: there is no instant at which a reader in the live namespace can
        observe a partially present generation. An interrupted reap leaves a
        directory in `__reclaim__/`, which no read path enumerates, and the next
        pass finishes it — which is why sweeping debris is the *first* thing this
        does rather than the last.

        Dry runs report exactly what a real pass would reclaim and touch nothing,
        so the policy can be argued about before it is applied.
        """
        moment = time.time() if now is None else now
        protected_keys = {
            encode_path_component(channel_id)
            for channel_id in protected_channel_ids
            if channel_id
        }

        totals = {
            "scanned": 0,
            "protected": 0,
            "retained": 0,
            "reclaimed": 0,
            "apparent": 0,
            "physical": 0,
            "aged": 0,
            "capacity": 0,
            "quarantine_entries": 0,
            "quarantine_bytes": 0,
            "debris_entries": 0,
            "debris_bytes": 0,
            "failures": 0,
        }
        reclaimed_keys: list[str] = []

        for dep_key, fp_key in self._namespace_keys(
            deployment_id, workflow_fingerprint
        ):
            # Debris first: it is already invisible, so finishing an interrupted
            # pass is pure reclamation and it keeps the byte figures the capacity
            # caps are compared against honest.
            for path in self._debris_paths(dep_key, fp_key):
                _files, apparent, physical = _dir_usage(path)
                totals["debris_entries"] += 1
                totals["debris_bytes"] += apparent
                if not dry_run:
                    _remove_tree(path)

            infos = self._scan_namespace_channels(dep_key, fp_key)
            totals["scanned"] += len(infos)

            candidates: list[ChannelInfo] = []
            for info in infos:
                if info.channel_key in protected_keys:
                    totals["protected"] += 1
                    continue
                candidates.append(info)

            aged, survivors = _partition_by_age(candidates, policy, moment)
            over_capacity, keepers = _partition_by_capacity(survivors, policy)

            totals["aged"] += len(aged)
            totals["capacity"] += len(over_capacity)
            totals["retained"] += len(keepers)

            reclaim_root = self._reclaim_dir(dep_key, fp_key)
            for info in aged + over_capacity:
                reclaimed_keys.append(info.channel_key)
                totals["reclaimed"] += 1
                totals["apparent"] += info.bytes_apparent
                totals["physical"] += info.bytes_physical
                if dry_run:
                    continue
                try:
                    self._retire_directory(info.directory, reclaim_root)
                except (CheckpointStoreError, OSError) as exc:
                    # One unreclaimable channel must not stop the pass; the
                    # namespace stays over its cap and says so in `failures`.
                    totals["failures"] += 1
                    logger.warning(
                        f"checkpoint reap could not reclaim a channel: "
                        f"channel_key={info.channel_key} backend={BACKEND} "
                        f"exception={type(exc).__name__}"
                    )

            entries, bytes_freed, failures = self._reap_quarantine(
                dep_key, fp_key, policy, moment, dry_run=dry_run
            )
            totals["quarantine_entries"] += entries
            totals["quarantine_bytes"] += bytes_freed
            totals["failures"] += failures

        report = ReapReport(
            dry_run=dry_run,
            scanned_channels=totals["scanned"],
            protected_channels=totals["protected"],
            retained_channels=totals["retained"],
            reclaimed_channels=totals["reclaimed"],
            reclaimed_bytes_apparent=totals["apparent"],
            reclaimed_bytes_physical=totals["physical"],
            aged_out=totals["aged"],
            over_capacity=totals["capacity"],
            reclaimed_quarantine_entries=totals["quarantine_entries"],
            reclaimed_quarantine_bytes=totals["quarantine_bytes"],
            swept_debris_entries=totals["debris_entries"],
            swept_debris_bytes=totals["debris_bytes"],
            failures=totals["failures"],
            reclaimed_keys=tuple(reclaimed_keys),
        )
        logger.debug(report.summary())
        return report

    def _reap_quarantine(
        self,
        dep_key: str,
        fp_key: str,
        policy: RetentionPolicy,
        now: float,
        *,
        dry_run: bool,
    ) -> tuple[int, int, int]:
        """Reclaim preserved records past the stated quarantine limits.

        Quarantine is evidence, so it is kept longer than live state and
        reclaimed oldest-first. Kept *forever*, though, it is the same leak with
        a different directory name — a quarantining failure that repeats every
        request accumulates as fast as the records themselves.
        """
        entries = self._scan_namespace_quarantine(dep_key, fp_key)
        expired = [
            entry
            for entry in entries
            if policy.quarantine_max_age_seconds is not None
            and entry.age_seconds(now) > policy.quarantine_max_age_seconds
        ]
        remaining = [entry for entry in entries if entry not in expired]
        if policy.quarantine_max_entries is not None:
            excess = len(remaining) - policy.quarantine_max_entries
            if excess > 0:
                expired.extend(remaining[:excess])

        reclaim_root = self._reclaim_dir(dep_key, fp_key)
        freed = 0
        failures = 0
        for entry in expired:
            freed += entry.bytes_apparent
            if dry_run:
                continue
            try:
                self._retire_directory(entry.path, reclaim_root)
                # An emptied per-channel quarantine directory is one inode per
                # channel that ever failed, which for a unique-channel workload
                # is unbounded growth in inodes rather than in bytes.
                _rmdir_if_empty(os.path.dirname(entry.path))
            except (CheckpointStoreError, OSError) as exc:
                failures += 1
                logger.warning(
                    f"checkpoint reap could not reclaim a quarantined record: "
                    f"channel_key={entry.channel_key} backend={BACKEND} "
                    f"exception={type(exc).__name__}"
                )
        return len(expired), freed, failures

    # -- read path -----------------------------------------------------------

    def _load_committed(
        self, identity: CheckpointIdentity, *, adopt_incarnation: bool
    ) -> Optional[CheckpointRecord]:
        commit_text = _read_text(self._commit_path(identity))
        if commit_text is None:
            return None

        commit = _parse(commit_text, field=_COMMIT_FILENAME)
        # Kind before identity, always: pulling a field out of a record whose
        # type and protocol version you have not checked is reading a value
        # whose meaning you are guessing at. That matters more here than in the
        # strict path, because adoption *trusts* the field it pulls out.
        self._validate_kind(commit, COMMIT_RECORD_TYPE)
        if adopt_incarnation:
            identity = _with_stored_incarnation(identity, commit)
        self._validate_identity(commit, identity)

        # Everything below runs against `identity` unchanged, so adoption is one
        # substitution rather than a second, weaker code path.
        generation = _require_generation(commit, field=_COMMIT_FILENAME)

        generation_dir = os.path.join(self._generations_dir(identity), str(generation))

        manifest_text = _read_text(os.path.join(generation_dir, _MANIFEST_FILENAME))
        if manifest_text is None:
            # COMMIT names a generation whose directory is gone. Older
            # generations may still be on disk; reading one of those would be
            # exactly the "context N with continuation N-1" merge this protocol
            # exists to prevent.
            raise _Unreadable(QuarantineReason.INCOMPLETE_GENERATION, _MANIFEST_FILENAME)
        if _digest_text(manifest_text) != commit.get("manifest_digest"):
            raise _Unreadable(QuarantineReason.DIGEST_MISMATCH, _MANIFEST_FILENAME)

        manifest = _parse(manifest_text, field=_MANIFEST_FILENAME)
        self._validate_envelope(manifest, identity, MANIFEST_RECORD_TYPE)
        if _require_generation(manifest, field=_MANIFEST_FILENAME) != generation:
            raise _Unreadable(QuarantineReason.GENERATION_MISMATCH, _MANIFEST_FILENAME)

        part_digests = manifest.get("parts")
        if not isinstance(part_digests, dict) or set(part_digests) != set(PART_SECTIONS):
            # An unexpected part set from a same-protocol writer is a bug, and
            # quietly using the subset this node understands is how a reader
            # ends up applying a checkpoint it did not fully read.
            raise _Unreadable(QuarantineReason.INCOMPLETE_GENERATION, "parts")

        state_version = manifest.get("state_version")
        if not isinstance(state_version, int) or isinstance(state_version, bool):
            raise _Unreadable(QuarantineReason.UNREADABLE_RECORD, "state_version")

        sections: dict[str, Any] = {}
        for section in PART_SECTIONS:
            path = os.path.join(generation_dir, f"{section}.json")
            text = _read_text(path)
            if text is None:
                raise _Unreadable(QuarantineReason.INCOMPLETE_GENERATION, section)
            if _digest_text(text) != part_digests[section]:
                raise _Unreadable(QuarantineReason.DIGEST_MISMATCH, section)

            part = _parse(text, field=section)
            self._validate_envelope(part, identity, RECORD_TYPE)
            if _require_generation(part, field=section) != generation:
                raise _Unreadable(QuarantineReason.GENERATION_MISMATCH, section)
            if part.get("state_version") != state_version:
                raise _Unreadable(QuarantineReason.INCOHERENT_GENERATION, section)
            if part.get("section") != section:
                raise _Unreadable(QuarantineReason.INCOHERENT_GENERATION, section)

            payload = part.get("payload")
            if not isinstance(payload, dict):
                raise _Unreadable(QuarantineReason.UNREADABLE_RECORD, section)
            sections[section] = payload

        record = CheckpointRecord(
            identity=identity,
            generation=generation,
            state_version=state_version,
            protocol_version=int(commit["protocol_version"]),
            **sections,
        )

        semantic = dict(record.to_payload())
        semantic.pop("generation")
        if state_digest(semantic) != commit.get("content_digest"):
            raise _Unreadable(QuarantineReason.DIGEST_MISMATCH, "content_digest")

        return record

    def _validate_envelope(
        self,
        payload: dict[str, Any],
        identity: CheckpointIdentity,
        expected_type: str,
    ) -> None:
        """Check kind then identity. A mismatch quarantines, never applies."""
        self._validate_kind(payload, expected_type)
        self._validate_identity(payload, identity)

    def _validate_kind(self, payload: dict[str, Any], expected_type: str) -> None:
        """Is this our kind of record, in a version this node can interpret?"""
        if payload.get("record_type") != expected_type:
            raise _Unreadable(QuarantineReason.RECORD_TYPE_MISMATCH, "record_type")

        version = payload.get("protocol_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise _Unreadable(
                QuarantineReason.PROTOCOL_VERSION_UNREADABLE, "protocol_version"
            )
        if not self._min_readable <= version <= self._protocol_version:
            raise _Unreadable(
                QuarantineReason.PROTOCOL_VERSION_UNREADABLE, "protocol_version"
            )

    def _validate_identity(
        self, payload: dict[str, Any], identity: CheckpointIdentity
    ) -> None:
        """Does this record belong to exactly this identity?"""
        expected = identity.as_fields()
        for field, reason in (
            ("deployment_id", QuarantineReason.DEPLOYMENT_MISMATCH),
            ("workflow_fingerprint", QuarantineReason.WORKFLOW_FINGERPRINT_MISMATCH),
            ("channel_key", QuarantineReason.CHANNEL_KEY_MISMATCH),
            ("channel_id", QuarantineReason.CHANNEL_ID_MISMATCH),
            ("session_incarnation", QuarantineReason.SESSION_INCARNATION_MISMATCH),
        ):
            if payload.get(field) != expected[field]:
                raise _Unreadable(reason, field)

    # -- write path helpers --------------------------------------------------

    def _read_commit_for_write(
        self, identity: CheckpointIdentity
    ) -> Optional[dict[str, Any]]:
        """Read COMMIT before publishing, quarantining what cannot be read.

        Publishing over a record this node cannot interpret is the "write around
        it" failure of invariant 31: the unreadable record would be reaped as a
        superseded generation and its lineage lost. Set it aside first, then
        start a fresh lineage at generation 1.

        A committed record under a *different* session incarnation is the one
        case that refuses instead. It is not corrupt and it is not misfiled — it
        is another session lifetime's valid state, and quarantine is for
        evidence of a fault, not for a live record that happens to be in the way.
        Setting it aside would also be the store deciding which session wins,
        which is the caller's decision and nobody else's. So the caller keeps its
        runtime live (invariant 8) and chooses: adopt the incarnation, clear the
        channel, or route elsewhere.
        """
        try:
            commit_text = _read_text(self._commit_path(identity))
            if commit_text is None:
                return None
            commit = _parse(commit_text, field=_COMMIT_FILENAME)
            self._validate_kind(commit, COMMIT_RECORD_TYPE)
            self._validate_identity(commit, identity)
            _require_generation(commit, field=_COMMIT_FILENAME)
        except _Unreadable as problem:
            if problem.reason is QuarantineReason.SESSION_INCARNATION_MISMATCH:
                # Neither incarnation is named: the message travels into logs the
                # store does not control, and the field name is what the operator
                # needs anyway.
                raise CheckpointStoreError(
                    f"refusing to publish for channel {identity.channel_id!r}: a "
                    f"committed checkpoint exists under a different "
                    f"session_incarnation. Adopt it with load_for_adoption(), or "
                    f"clear() the channel, but do not overwrite it."
                ) from problem
            self._quarantine(identity, problem.reason, field=problem.field)
            return None
        return commit

    def _generation_is_structurally_present(
        self, identity: CheckpointIdentity, generation: Any
    ) -> bool:
        """Cheap presence check guarding the no-write fast path.

        Skipping a write is only safe if there is something whole to skip it in
        favour of; stat-ing the parts costs nothing next to re-writing a
        several-hundred-kilobyte context.
        """
        if not isinstance(generation, int) or isinstance(generation, bool):
            return False
        generation_dir = os.path.join(self._generations_dir(identity), str(generation))
        if not _is_regular_file(os.path.join(generation_dir, _MANIFEST_FILENAME)):
            return False
        return all(
            _is_regular_file(os.path.join(generation_dir, f"{section}.json"))
            for section in PART_SECTIONS
        )

    def _next_generation(self, generations_dir: str, committed: int) -> int:
        """Strictly greater than anything already on disk.

        A crash can leave a complete-but-uncommitted ``gen/N`` above the
        committed number; reusing N would rename onto it.
        """
        highest = committed
        try:
            entries = os.listdir(generations_dir)
        except FileNotFoundError:
            entries = []
        for name in entries:
            if name.isascii() and name.isdigit():
                highest = max(highest, int(name))
        return highest + 1

    def _prune_generations(self, generations_dir: str, keep: set[int]) -> None:
        """Drop superseded generations and orphaned pending directories.

        Within-channel housekeeping during a publish, not the namespace reaper:
        this only ever removes generations the publisher has just superseded.

        Safe only under the single-writer-per-channel invariant (15); a
        concurrent reader of the same channel is out of scope by construction.
        """
        try:
            entries = os.listdir(generations_dir)
        except FileNotFoundError:
            return
        for name in entries:
            if name.isascii() and name.isdigit() and int(name) in keep:
                continue
            path = os.path.join(generations_dir, name)
            try:
                if os.path.islink(path) or os.path.isfile(path):
                    os.unlink(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError as exc:
                logger.debug(
                    f"checkpoint reap skipped an entry: backend={BACKEND} "
                    f"exception={type(exc).__name__}"
                )

    # -- liveness, floors, and reclamation primitives ------------------------

    def _touch_access(
        self, identity: CheckpointIdentity, generation: int, *, now: float
    ) -> None:
        """Record that this channel was published to, cheaply.

        Retention needs an age, and the only timestamp inside a record is the one
        in COMMIT — which does not advance when a digest-stable republish
        correctly skips the write (§11.4). Ageing on that alone would reclaim a
        channel that is being retired every few minutes with unchanged state,
        i.e. an actively used one. So liveness gets its own ~100-byte file,
        rewritten on every successful publish including the fast path, and the
        several-hundred-kilobyte context is still not rewritten.

        It is referenced by no digest and read by no `load` path, so it cannot
        make a record unreadable, and a failure to write it never fails a publish
        — losing an age hint is not worth losing a checkpoint. The cost of that
        choice is that the channel may age from its previous publish instead.
        """
        channel_dir = self.channel_directory(identity)
        destination = os.path.join(channel_dir, _ACCESS_FILENAME)
        temporary = f"{destination}{_TEMP_INFIX}{uuid.uuid4().hex}"
        try:
            self._write_private_file(
                temporary,
                encode_state(
                    {
                        "protocol_version": self._protocol_version,
                        "record_type": ACCESS_RECORD_TYPE,
                        "generation": generation,
                        "channel_key": identity.channel_key,
                        "last_publish_at": now,
                    }
                ),
            )
            _replace(temporary, destination)
        except (CheckpointStoreError, OSError, StateEncodingError) as exc:
            _unlink(temporary)
            logger.debug(
                f"checkpoint liveness marker not refreshed: backend={BACKEND} "
                f"exception={type(exc).__name__}"
            )

    def _generation_floor(self, identity: CheckpointIdentity) -> int:
        floor = _read_json(
            os.path.join(self.channel_directory(identity), _FLOOR_FILENAME)
        )
        value = floor.get("floor") if floor else None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return 0

    def _write_floor(self, identity: CheckpointIdentity, floor: int) -> None:
        channel_dir = self._ensure_dir_chain(self.channel_directory(identity))
        destination = os.path.join(channel_dir, _FLOOR_FILENAME)
        temporary = f"{destination}{_TEMP_INFIX}{uuid.uuid4().hex}"
        self._write_private_file(
            temporary,
            encode_state(
                {
                    "protocol_version": self._protocol_version,
                    "record_type": FLOOR_RECORD_TYPE,
                    "channel_key": identity.channel_key,
                    "floor": floor,
                }
            ),
        )
        _replace(temporary, destination)
        _fsync_dir(channel_dir)

    def _highest_generation(self, identity: CheckpointIdentity) -> int:
        """The largest generation number this channel id has ever been assigned.

        Every source is consulted — the committed pointer, the directories on
        disk, and any floor a previous reset left — because a floor that is not
        the maximum of all three would let a number repeat, which is the one
        thing it exists to prevent.
        """
        highest = self._generation_floor(identity)
        commit = _read_json(self._commit_path(identity))
        value = commit.get("generation") if commit else None
        if isinstance(value, int) and not isinstance(value, bool):
            highest = max(highest, value)
        for name in _listdir_names(self._generations_dir(identity)):
            if name.isascii() and name.isdigit():
                highest = max(highest, int(name))
        return highest

    def _retire_directory(
        self, path: str, reclaim_root: str
    ) -> tuple[bool, int, int]:
        """Make `path` invisible with one rename, then reclaim its bytes.

        The rename is the whole crash-safety argument. Deleting in place walks
        through states where some of a generation is present, and a reader that
        lands in one of them quarantines a record nobody asked to preserve.
        Renaming first means the live namespace only ever sees the whole
        directory or none of it, and whatever is left in `__reclaim__` is
        enumerated by no read path and swept by the next `reap`.

        Returns (was there, apparent bytes, physical bytes).
        """
        if os.path.islink(path):
            _unlink(path)
            return (True, 0, 0)
        if not os.path.isdir(path):
            return (False, 0, 0)

        _files, apparent, physical = _dir_usage(path)
        self._ensure_dir_chain(reclaim_root)
        destination = os.path.join(
            reclaim_root, f"{os.path.basename(path)}{_TEMP_INFIX}{uuid.uuid4().hex}"
        )
        _replace(path, destination)
        _fsync_dir(os.path.dirname(path))
        _remove_tree(destination)
        return (True, apparent, physical)

    # -- namespace scanning --------------------------------------------------

    def _namespace_keys(
        self,
        deployment_id: Optional[str],
        workflow_fingerprint: Optional[str],
    ) -> list[tuple[str, str]]:
        """Encoded (deployment, fingerprint) pairs present anywhere on disk.

        Quarantine and reclaim roots are included, so a namespace whose live
        channels are all gone still reports the bytes it is still occupying.
        """
        want_dep = (
            encode_path_component(deployment_id) if deployment_id else None
        )
        want_fp = (
            encode_path_component(workflow_fingerprint)
            if workflow_fingerprint
            else None
        )
        found: set[tuple[str, str]] = set()
        for root_name in (
            _CHANNELS_DIRNAME,
            _QUARANTINE_DIRNAME,
            _RECLAIM_DIRNAME,
        ):
            root = os.path.join(self._base, root_name)
            for dep_key in _listdir_dirs(root):
                if want_dep is not None and dep_key != want_dep:
                    continue
                for fp_key in _listdir_dirs(os.path.join(root, dep_key)):
                    if want_fp is not None and fp_key != want_fp:
                        continue
                    found.add((dep_key, fp_key))
        return sorted(found)

    def _scan_namespace_channels(
        self, dep_key: str, fp_key: str
    ) -> list[ChannelInfo]:
        root = self._namespace_dir(_CHANNELS_DIRNAME, dep_key, fp_key)
        return [
            self._scan_channel(name, os.path.join(root, name))
            for name in _listdir_dirs(root)
        ]

    def _scan_channel(self, channel_key: str, directory: str) -> ChannelInfo:
        """Describe one channel directory without trusting any of its contents.

        Nothing in here may raise on malformed data. Inspection, measurement and
        reaping all run over whatever is actually on disk, including records
        written by a future protocol or corrupted by a bad disk, and a scanner
        that dies on the first bad channel cannot report the namespace it was
        asked about.
        """
        files, apparent, physical = _dir_usage(directory)
        commit = _read_json(os.path.join(directory, _COMMIT_FILENAME))
        access = _read_json(os.path.join(directory, _ACCESS_FILENAME))
        floor = _read_json(os.path.join(directory, _FLOOR_FILENAME))

        published_at = _as_timestamp(commit.get("published_at") if commit else None)
        last_seen_at = _as_timestamp(
            access.get("last_publish_at") if access else None
        )
        generations = len(
            [
                name
                for name in _listdir_dirs(
                    os.path.join(directory, _GENERATIONS_DIRNAME)
                )
                if name.isascii() and name.isdigit()
            ]
        )

        # Directory mtime is the floor on age, not the authority. It can only ever
        # make a channel look *younger* than the record claims, which errs
        # towards keeping bytes rather than towards reclaiming a live channel —
        # and it is the only signal at all for a channel whose COMMIT is
        # unreadable or whose marker was never written.
        activity_at = max(
            published_at or 0.0, last_seen_at or 0.0, _mtime(directory)
        )
        floor_value = floor.get("floor") if floor else None

        return ChannelInfo(
            channel_key=channel_key,
            directory=directory,
            deployment_id=_as_text(commit.get("deployment_id") if commit else None),
            workflow_fingerprint=_as_text(
                commit.get("workflow_fingerprint") if commit else None
            ),
            channel_id=_as_text(commit.get("channel_id") if commit else None),
            session_incarnation=_as_text(
                commit.get("session_incarnation") if commit else None
            ),
            generation=_as_generation(commit.get("generation") if commit else None),
            generation_floor=floor_value
            if isinstance(floor_value, int) and not isinstance(floor_value, bool)
            else 0,
            published_at=published_at,
            last_seen_at=last_seen_at,
            activity_at=activity_at,
            generations_on_disk=generations,
            file_count=files,
            bytes_apparent=apparent,
            bytes_physical=physical,
            committed=_is_regular_file(
                os.path.join(directory, _COMMIT_FILENAME)
            ),
        )

    def _scan_namespace_quarantine(
        self, dep_key: str, fp_key: str
    ) -> list[QuarantineEntry]:
        root = self._namespace_dir(_QUARANTINE_DIRNAME, dep_key, fp_key)
        entries: list[QuarantineEntry] = []
        for channel_key in _listdir_dirs(root):
            channel_root = os.path.join(root, channel_key)
            for name in _listdir_dirs(channel_root):
                path = os.path.join(channel_root, name)
                files, apparent, physical = _dir_usage(path)
                reason, stamped_at = _parse_quarantine_name(name)
                entries.append(
                    QuarantineEntry(
                        path=path,
                        channel_key=channel_key,
                        reason=reason,
                        quarantined_at=stamped_at
                        if stamped_at is not None
                        else _mtime(path),
                        file_count=files,
                        bytes_apparent=apparent,
                        bytes_physical=physical,
                    )
                )
        return entries

    def _debris_paths(self, dep_key: str, fp_key: str) -> list[str]:
        root = self._reclaim_dir(dep_key, fp_key)
        return [os.path.join(root, name) for name in _listdir_names(root)]

    def _quarantine(
        self,
        identity: CheckpointIdentity,
        reason: QuarantineReason,
        *,
        field: str,
    ) -> None:
        channel_dir = self.channel_directory(identity)
        if not os.path.isdir(channel_dir):
            logger.warning(
                f"checkpoint quarantine requested for channel "
                f"{identity.channel_id!r} with nothing to preserve: "
                f"reason={reason.value} field={field} backend={BACKEND}"
            )
            return

        # `<reason>-<unix seconds>-<uuid>`: the reason leads because that is what
        # an operator greps for, and the timestamp is in the name rather than
        # only in mtime so that a copy, a restore or an rsync cannot make a
        # week-old piece of evidence look like it arrived this morning.
        destination = os.path.join(
            self.quarantine_directory(identity),
            f"{reason.value}-{int(time.time())}-{uuid.uuid4().hex}",
        )
        try:
            self._ensure_dir_chain(os.path.dirname(destination))
            os.replace(channel_dir, destination)
        except (OSError, CheckpointStoreError) as exc:
            # Preserving beats reclaiming: leave the record exactly where it is
            # rather than risk destroying the only copy of the evidence.
            logger.warning(
                f"checkpoint quarantine could not move records for channel "
                f"{identity.channel_id!r}: reason={reason.value} field={field} "
                f"backend={BACKEND} exception={type(exc).__name__}"
            )
            return

        logger.warning(
            f"checkpoint quarantined for channel {identity.channel_id!r}: "
            f"reason={reason.value} field={field} backend={BACKEND}; "
            f"starting from launch configuration"
        )

    # -- private-mode, symlink-refusing primitives ---------------------------

    def _ensure_dir_chain(self, path: str) -> str:
        """Create every level under the base with private modes, refusing links.

        Walking down from the base rather than calling `makedirs` is what makes
        the symlink refusal meaningful: a check on the leaf says nothing about
        an attacker-planted link three levels up.
        """
        relative = os.path.relpath(path, self._base)
        if relative.startswith(os.pardir):
            raise CheckpointStoreError("refusing to write outside the base folder")

        # The base's own parents are the operator's directories, not ours, so
        # they are created but never re-moded; privacy starts at the base.
        parent = os.path.dirname(self._base)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)

        current = self._base
        self._ensure_private_dir(current)
        if relative != os.curdir:
            for part in relative.split(os.sep):
                current = os.path.join(current, part)
                self._ensure_private_dir(current)
        return path

    def _ensure_private_dir(self, path: str) -> None:
        try:
            os.mkdir(path, _DIR_MODE)
        except FileExistsError:
            pass
        except FileNotFoundError as exc:
            raise CheckpointStoreError(
                f"cannot create checkpoint directory: exception={type(exc).__name__}"
            ) from exc

        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise CheckpointStoreError(
                "refusing to publish through a symlinked directory"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise CheckpointStoreError(
                "refusing to publish: path exists and is not a directory"
            )
        # mkdir's mode is masked by umask and can only ever be *more* private
        # than requested, so re-asserting it never widens access mid-flight.
        os.chmod(path, _DIR_MODE)

    def _write_private_file(self, path: str, text: str) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
        try:
            fd = os.open(path, flags, _FILE_MODE)
        except OSError as exc:
            raise CheckpointStoreError(
                f"cannot create checkpoint file: exception={type(exc).__name__}"
            ) from exc
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), _FILE_MODE)
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())


# -- module-level, backend-independent helpers -------------------------------


def _digest_text(text: str) -> str:
    """SHA-256 of the exact stored bytes.

    Hashing the bytes rather than the re-parsed value is what detects tampering
    that survives a JSON round trip. The bytes themselves still come only from
    `state_serialization.encode_state`; this is not a second encoder.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse(text: str, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise _Unreadable(QuarantineReason.UNREADABLE_RECORD, field) from exc
    if not isinstance(value, dict):
        raise _Unreadable(QuarantineReason.UNREADABLE_RECORD, field)
    return value


def _with_stored_incarnation(
    identity: CheckpointIdentity, commit: dict[str, Any]
) -> CheckpointIdentity:
    """Replace the probe incarnation with the one COMMIT names.

    This is the *entire* relaxation adoption buys. Everything downstream then
    validates against the result exactly as the strict path does, so a record
    whose parts disagree about the session lifetime still fails closed.
    """
    stored = commit.get("session_incarnation")
    if not isinstance(stored, str) or not stored:
        raise _Unreadable(
            QuarantineReason.SESSION_INCARNATION_MISMATCH, "session_incarnation"
        )
    return replace(identity, session_incarnation=stored)


def _require_generation(payload: dict[str, Any], *, field: str) -> int:
    value = payload.get("generation")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise _Unreadable(QuarantineReason.GENERATION_MISMATCH, field)
    return value


def _read_text(path: str) -> Optional[str]:
    """Read a regular file without following a symlink; None when absent."""
    try:
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            # A record replaced by a symlink is someone redirecting a read, not
            # a record.
            raise _Unreadable(
                QuarantineReason.UNREADABLE_RECORD, os.path.basename(path)
            ) from exc
        raise
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise _Unreadable(
                QuarantineReason.UNREADABLE_RECORD, os.path.basename(path)
            )
        raw = handle.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _Unreadable(
            QuarantineReason.UNREADABLE_RECORD, os.path.basename(path)
        ) from exc


def _is_regular_file(path: str) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _read_json(path: str) -> Optional[dict[str, Any]]:
    """Best-effort read for inspection and retention. Never raises, never quarantines.

    The read paths above are strict on purpose: a record they cannot understand
    must fail closed. This one is the opposite on purpose, because it serves
    `inspect`, `stats` and `reap`, and refusing to report a namespace because one
    channel in it is corrupt is how a store's growth becomes unmeasurable.
    """
    try:
        text = _read_text(path)
    except (_Unreadable, OSError):
        return None
    if text is None:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _as_text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _as_generation(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return None


def _as_timestamp(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    # NaN would compare false against every threshold, so a channel carrying one
    # would be immortal; infinity would make it immortal explicitly. Neither is a
    # timestamp, so neither is treated as one.
    if number != number or number in (float("inf"), float("-inf")) or number < 0:
        return None
    return number


def _parse_quarantine_name(name: str) -> tuple[Optional[str], Optional[float]]:
    """Split `<reason>-<unix seconds>-<uuid>`, tolerating anything else."""
    parts = name.rsplit("-", 2)
    if len(parts) != 3:
        return (None, None)
    reason, stamp, _uuid = parts
    if not (stamp.isascii() and stamp.isdigit()):
        return (reason or None, None)
    return (reason or None, float(stamp))


def _mtime(path: str) -> float:
    try:
        return os.lstat(path).st_mtime
    except OSError:
        return 0.0


def _listdir_names(path: str) -> list[str]:
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def _listdir_dirs(path: str) -> list[str]:
    """Immediate subdirectories, symlinks excluded.

    Following a symlink here would let a planted link make one channel's bytes
    count as another namespace's, or send a reclamation rename somewhere the
    store does not own.
    """
    names = []
    for name in _listdir_names(path):
        candidate = os.path.join(path, name)
        try:
            info = os.lstat(candidate)
        except OSError:
            continue
        if stat.S_ISDIR(info.st_mode):
            names.append(name)
    return names


def _dir_usage(path: str) -> tuple[int, int, int]:
    """(file count, apparent bytes, physical bytes) for a whole subtree.

    Physical bytes come from allocated blocks, because that is what fills a
    volume and what §16.5 gates on; apparent bytes are reported beside them
    because they are exactly reproducible from file sizes and therefore assertable.
    Where a platform has no block count, apparent stands in rather than reporting
    zero and making a full namespace look empty.
    """
    files = 0
    apparent = 0
    physical = 0
    if not os.path.isdir(path) or os.path.islink(path):
        try:
            info = os.lstat(path)
        except OSError:
            return (0, 0, 0)
        blocks = getattr(info, "st_blocks", None)
        return (
            1,
            info.st_size,
            int(blocks) * 512 if blocks is not None else info.st_size,
        )
    for directory, subdirs, names in os.walk(path, followlinks=False):
        for name in list(subdirs) + names:
            candidate = os.path.join(directory, name)
            try:
                info = os.lstat(candidate)
            except OSError:
                continue
            if stat.S_ISDIR(info.st_mode):
                blocks = getattr(info, "st_blocks", None)
                physical += int(blocks) * 512 if blocks else 0
                continue
            files += 1
            apparent += info.st_size
            blocks = getattr(info, "st_blocks", None)
            physical += int(blocks) * 512 if blocks is not None else info.st_size
    return (files, apparent, physical)


def _partition_by_age(
    candidates: list[ChannelInfo], policy: RetentionPolicy, now: float
) -> tuple[list[ChannelInfo], list[ChannelInfo]]:
    """(too old to keep, still within the window)."""
    if policy.max_age_seconds is None:
        return ([], list(candidates))
    expired = []
    survivors = []
    for info in candidates:
        target = expired if info.age_seconds(now) > policy.max_age_seconds else survivors
        target.append(info)
    return (expired, survivors)


def _partition_by_capacity(
    survivors: list[ChannelInfo], policy: RetentionPolicy
) -> tuple[list[ChannelInfo], list[ChannelInfo]]:
    """(evicted for capacity, kept), oldest evicted first.

    This is the part that is actually a bound. The age window plateaus at
    arrival-rate times window, which is a number nobody in the system controls;
    a count or byte cap holds whatever the arrival rate does. The cost is that a
    channel a future session would have adopted can be reclaimed while it is
    still young — stated rather than hidden, and the reason capacity caps are
    separate knobs from the window.

    Ordering is (activity, channel_key) so it is total: two channels published in
    the same clock tick still have a defined victim.
    """
    ordered = sorted(survivors, key=lambda info: (info.activity_at, info.channel_key))
    if policy.max_channels is None and policy.max_bytes is None:
        return ([], ordered)

    evicted: list[ChannelInfo] = []
    kept = list(ordered)
    live_bytes = sum(info.bytes_apparent for info in kept)
    index = 0
    while index < len(kept):
        remaining = len(kept) - index
        over_count = (
            policy.max_channels is not None and remaining > policy.max_channels
        )
        over_bytes = policy.max_bytes is not None and live_bytes > policy.max_bytes
        if not (over_count or over_bytes):
            break
        victim = kept[index]
        evicted.append(victim)
        live_bytes -= victim.bytes_apparent
        index += 1
    return (evicted, kept[index:])


def _replace(source: str, destination: str) -> None:
    """`os.replace`, with an interruption reported rather than raised as an OSError.

    Every atomic flip in this module goes through here. A rename that fails
    because the tree it was operating on disappeared underneath it means the
    single-writer-per-channel invariant was broken — most likely a reaper running
    against a channel the caller did not declare as live — and the caller needs to
    hear "publication refused" (invariant 8), not a bare OSError from a path.
    """
    try:
        os.replace(source, destination)
    except OSError as exc:
        raise CheckpointStoreError(
            f"checkpoint operation interrupted: the target path is no longer "
            f"where it was ({type(exc).__name__}); another writer or a reaper is "
            f"operating on this channel"
        ) from exc


def _unlink(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def _remove_tree(path: str) -> None:
    """Bulk removal of something already invisible to every read path."""
    if os.path.islink(path) or os.path.isfile(path):
        _unlink(path)
        return
    shutil.rmtree(path, ignore_errors=True)


def _rmdir_if_empty(path: str) -> None:
    with contextlib.suppress(OSError):
        os.rmdir(path)


def _human_bytes(count: Optional[int]) -> str:
    if count is None:
        return "unbounded"
    value = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _describe_seconds(seconds: Optional[float]) -> str:
    return "unbounded" if seconds is None else f"{seconds:g}s"


def _fsync_dir(path: str) -> None:
    """Flush a directory entry so a rename survives power loss.

    Best effort: some filesystems reject it, and failing the publish over a
    durability hint would trade a real record for a theoretical one.
    """
    try:
        fd = os.open(path, os.O_RDONLY | _O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
