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

import errno
import hashlib
import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional

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

BACKEND = "disk"

# The four records a checkpoint spans. Splitting them is not decoration: the
# design's crash schedule is exactly "context published, continuation not", so a
# store that cannot express more than one record per generation cannot be tested
# against the failure it exists to prevent.
PART_SECTIONS = ("context", "runtime", "startup", "launch_context")

_CHANNELS_DIRNAME = "channels"
_QUARANTINE_DIRNAME = "__quarantine__"
_GENERATIONS_DIRNAME = "gen"
_COMMIT_FILENAME = "COMMIT"
_MANIFEST_FILENAME = "manifest.json"
_PENDING_PREFIX = ".pending-"

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
            gen/<N>/manifest.json
            gen/<N>/{context,runtime,startup,launch_context}.json
        <base>/__quarantine__/<dep>/<fingerprint>/<channel_key>/<reason>-<uuid>/

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
        """
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
            return int(committed["generation"])

        channel_dir = self._ensure_dir_chain(self.channel_directory(identity))
        generations_dir = self._ensure_dir_chain(self._generations_dir(identity))

        generation = self._next_generation(
            generations_dir,
            int(committed["generation"]) if committed else 0,
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
        os.replace(pending_dir, generation_dir)
        _fsync_dir(generations_dir)

        commit_text = encode_state(
            {
                "protocol_version": self._protocol_version,
                "record_type": COMMIT_RECORD_TYPE,
                "generation": generation,
                **identity.as_fields(),
                "content_digest": content_digest,
                "manifest_digest": _digest_text(manifest_text),
            }
        )
        commit_tmp = os.path.join(
            channel_dir, f"{_COMMIT_FILENAME}.tmp-{uuid.uuid4().hex}"
        )
        self._write_private_file(commit_tmp, commit_text)
        os.replace(commit_tmp, self._commit_path(identity))
        _fsync_dir(channel_dir)

        self._reap(
            generations_dir,
            keep={generation - offset for offset in range(_GENERATIONS_RETAINED)},
        )
        return generation

    def quarantine(
        self, identity: CheckpointIdentity, reason: QuarantineReason
    ) -> None:
        """Set this channel's records aside, preserved, never deleted."""
        self._quarantine(identity, reason, field="")

    def list_quarantined(self, identity: CheckpointIdentity) -> list[str]:
        """Preserved quarantine directories for this channel, oldest name first."""
        root = self.quarantine_directory(identity)
        try:
            return sorted(
                os.path.join(root, name) for name in os.listdir(root)
            )
        except FileNotFoundError:
            return []

    def clear(self, identity: CheckpointIdentity) -> None:
        """Delete this channel's live records.

        Deliberately destructive and deliberately not what quarantine does:
        quarantined copies are untouched, because the reason they were set aside
        is that somebody needs to look at them.
        """
        shutil.rmtree(self.channel_directory(identity), ignore_errors=True)

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

    def _reap(self, generations_dir: str, keep: set[int]) -> None:
        """Drop superseded generations and orphaned pending directories.

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

        destination = os.path.join(
            self.quarantine_directory(identity),
            f"{reason.value}-{uuid.uuid4().hex}",
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
