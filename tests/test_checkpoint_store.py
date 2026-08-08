"""Integration tests for the durable channel checkpoint store.

Real temp directories, real files, real crashes simulated by mutating the bytes
on disk. No mocks: the properties under test — injectivity of a filesystem name,
atomicity of a rename, what survives a torn write — are properties of the
filesystem, and a mock has none of them.

The fault injections mirror the design's crash schedule (§11.8): every write
boundary in `publish` is interrupted in turn, and the reader must either see the
previous generation whole or refuse, never a mixture of the two.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import time
from datetime import datetime

import pytest

from fastworkflow.checkpoint_store import (
    COMMIT_RECORD_TYPE,
    DEFAULT_RETENTION,
    PART_SECTIONS,
    PROTOCOL_VERSION,
    REPRESENTATIVE_RECORD_BYTES,
    ChannelCheckpointStore,
    CheckpointIdentity,
    CheckpointRecord,
    CheckpointStoreError,
    QuarantineReason,
    RetentionPolicy,
    encode_path_component,
)
from fastworkflow.session_state_store import DiskSessionStateStore
from fastworkflow.state_serialization import StateEncodingError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def base(tmp_path):
    return str(tmp_path / "checkpoints")


@pytest.fixture
def store(base):
    return ChannelCheckpointStore(base)


def identity_for(channel: str, **overrides) -> CheckpointIdentity:
    """The first argument names the channel; overrides replace any field by name."""
    fields = {
        "deployment_id": "deploy-blue",
        "workflow_fingerprint": "wf-abc123",
        "channel_id": channel,
        "session_incarnation": "inc-1",
    }
    fields.update(overrides)
    return CheckpointIdentity(**fields)


def sample_sections(marker: str = "one") -> dict:
    return {
        "context": {"api_base": "https://example.invalid", "marker": marker},
        "runtime": {
            "active_conversation_id": 7,
            "stream_format": "ndjson",
            "is_complete": False,
        },
        "startup": {"state": "succeeded", "idempotency_key": "idem-1", "epoch": 3},
        "launch_context": {"prior_projection": {"tenant": "t1"}, "digest": "d1"},
    }


def publish(store, identity, marker: str = "one", state_version: int = 2) -> int:
    return store.publish(
        identity, **sample_sections(marker), state_version=state_version
    )


def snapshot_tree(root: str) -> dict:
    """Every path with its size, mtime and bytes, so "no write" is provable."""
    seen = {}
    for directory, subdirs, files in os.walk(root):
        subdirs.sort()
        for name in sorted(files):
            path = os.path.join(directory, name)
            info = os.lstat(path)
            with open(path, "rb") as handle:
                seen[os.path.relpath(path, root)] = (
                    info.st_size,
                    info.st_mtime_ns,
                    handle.read(),
                )
    return seen


def snapshot_records(root: str) -> dict:
    """`snapshot_tree` without the liveness marker.

    The marker is retention metadata that is refreshed on every publish by
    design, so a snapshot including it cannot express "the record was not
    rewritten" — which is the property §11.4 actually cares about, because the
    record is the several-hundred-kilobyte write and the marker is ~100 bytes.
    """
    return {
        path: value
        for path, value in snapshot_tree(root).items()
        if os.path.basename(path) != "ACCESS"
    }


def access_marker(store, identity) -> dict:
    with open(
        os.path.join(store.channel_directory(identity), "ACCESS"), encoding="utf-8"
    ) as handle:
        return json.load(handle)


def generation_dir(store, identity, generation: int) -> str:
    return os.path.join(
        store.channel_directory(identity), "gen", str(generation)
    )


def rewrite_part(path: str, **changes) -> None:
    """Tamper with one stored record, keeping it valid JSON."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.update(changes)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def rechain(store, identity, generation: int = 1) -> None:
    """Recompute the manifest and commit digests over whatever is on disk.

    Lets a test tamper *coherently* — repairing the digest chain the way a buggy
    migration or a determined hand would — so an identity check is exercised on
    its own instead of being shadowed by a digest mismatch that would have fired
    first. Digests are computed here with plain `hashlib` rather than the
    module's helper, so the test pins the on-disk format independently.
    """
    directory = generation_dir(store, identity, generation)
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    for section in PART_SECTIONS:
        with open(os.path.join(directory, f"{section}.json"), "rb") as handle:
            manifest["parts"][section] = hashlib.sha256(handle.read()).hexdigest()

    manifest_bytes = json.dumps(manifest).encode("utf-8")
    with open(manifest_path, "wb") as handle:
        handle.write(manifest_bytes)

    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    with open(commit_path, encoding="utf-8") as handle:
        commit = json.load(handle)
    commit["manifest_digest"] = hashlib.sha256(manifest_bytes).hexdigest()
    with open(commit_path, "w", encoding="utf-8") as handle:
        json.dump(commit, handle)


@pytest.fixture
def warnings_logged():
    """Collect real records off the fastWorkflow logger (it does not propagate)."""
    collected: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record):
            collected.append(record)

    handler = _Collector(level=logging.WARNING)
    logger = logging.getLogger("fastWorkflow")
    previous_level = logger.level
    logger.setLevel(min(previous_level, logging.WARNING))
    logger.addHandler(handler)
    try:
        yield collected
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


# ---------------------------------------------------------------------------
# 1. Round trip
# ---------------------------------------------------------------------------


def test_publish_then_load_returns_an_equal_record(store):
    identity = identity_for("channel-1")
    generation = publish(store, identity)

    assert generation == 1
    assert store.exists(identity) is True

    expected = CheckpointRecord(
        identity=identity,
        generation=generation,
        state_version=2,
        protocol_version=PROTOCOL_VERSION,
        **sample_sections(),
    )
    assert store.load(identity) == expected


def test_the_payload_has_the_designed_shape(store):
    identity = identity_for("channel-shape")
    publish(store, identity)

    payload = store.load(identity).to_payload()

    assert set(payload) == {
        "protocol_version",
        "record_type",
        "generation",
        "deployment_id",
        "workflow_fingerprint",
        "channel_key",
        "channel_id",
        "session_incarnation",
        "state_version",
        "context",
        "runtime",
        "startup",
        "launch_context",
    }
    assert payload["record_type"] == "channel_checkpoint"
    assert payload["protocol_version"] == 1
    # The raw id is kept so a key collision can be detected rather than served.
    assert payload["channel_id"] == "channel-shape"


def test_absent_and_empty_are_different(store):
    identity = identity_for("channel-empty")
    assert store.load(identity) is None
    assert store.exists(identity) is False

    store.publish(
        identity,
        context={},
        runtime={},
        startup={},
        launch_context={},
        state_version=1,
    )

    record = store.load(identity)
    assert record is not None
    assert record.context == {}


def test_republishing_new_content_supersedes_the_old_generation(store):
    identity = identity_for("channel-super")
    assert publish(store, identity, marker="one") == 1
    assert publish(store, identity, marker="two") == 2

    record = store.load(identity)
    assert record.generation == 2
    assert record.context["marker"] == "two"


def test_clear_removes_the_record_but_not_quarantined_copies(store):
    identity = identity_for("channel-clear")
    publish(store, identity)
    store.quarantine(identity, QuarantineReason.OPERATOR_REQUEST)
    preserved = store.list_quarantined(identity)

    publish(store, identity)
    store.clear(identity)

    assert store.load(identity) is None
    assert store.exists(identity) is False
    assert store.list_quarantined(identity) == preserved
    assert os.path.isdir(preserved[0])


# ---------------------------------------------------------------------------
# 2. Injectivity (invariant 27)
# ---------------------------------------------------------------------------

ALIASING_IDS = [
    "tenant/a",
    "tenant_a",
    "tenant%2Fa",
    "tenant\\a",
    "tenant a",
    "tenant.a",
    "Tenant_a",
    "TENANT_A",
    "tenant\na",
    "tenant\ta",
    "tenant\x00a",
    "tenant\x7fa",
    "tenant/日本",
    "tenant_日本",
    "tenant/\u00e9",
    "tenant/e\u0301",
    ".",
    "..",
    "../../etc/passwd",
    "a" * 300,
    "a" * 300 + "b",
    "b" + "a" * 300,
]


def test_the_encoding_is_injective_over_the_aliasing_corpus():
    encoded = [encode_path_component(raw) for raw in ALIASING_IDS]
    assert len(set(encoded)) == len(ALIASING_IDS)

    # A case-insensitive volume folds names before the record-level check can
    # run, so folded names must be distinct too.
    assert len({name.casefold() for name in encoded}) == len(ALIASING_IDS)


def test_encoded_names_are_usable_filenames():
    for raw in ALIASING_IDS:
        name = encode_path_component(raw)
        assert name and name.isascii()
        assert len(name.encode("utf-8")) <= 200
        assert os.sep not in name and "/" not in name and "\\" not in name
        assert "\x00" not in name
        assert name not in (os.curdir, os.pardir)
        assert not name.startswith(".")


def test_separator_aliases_are_three_records_that_never_read_each_other(store):
    aliases = ["tenant/a", "tenant_a", "tenant%2Fa"]
    for index, channel_id in enumerate(aliases):
        publish(store, identity_for(channel_id), marker=f"marker-{index}")

    directories = {
        store.channel_directory(identity_for(channel_id)) for channel_id in aliases
    }
    assert len(directories) == 3

    for index, channel_id in enumerate(aliases):
        record = store.load(identity_for(channel_id))
        assert record.context["marker"] == f"marker-{index}"
        assert record.identity.channel_id == channel_id


@pytest.mark.parametrize(
    "left,right",
    [
        ("tenant/日本", "tenant_日本"),
        ("ctrl\na", "ctrl_a"),
        ("ctrl\x00a", "ctrl%00a"),
        ("Case", "case"),
        ("nfc-\u00e9", "nfd-e\u0301"),
        ("x" * 300, "x" * 300 + "y"),
        ("z" * 400, "z" * 401),
    ],
)
def test_hostile_id_pairs_stay_separate_on_disk(store, left, right):
    publish(store, identity_for(left), marker="left")
    publish(store, identity_for(right), marker="right")

    assert store.load(identity_for(left)).context["marker"] == "left"
    assert store.load(identity_for(right)).context["marker"] == "right"


def test_an_oversized_id_round_trips_and_keeps_its_raw_form(store):
    channel_id = "über/" + "x" * 400
    assert len(channel_id.encode("utf-8")) > 255

    identity = identity_for(channel_id)
    publish(store, identity)

    record = store.load(identity)
    assert record.identity.channel_id == channel_id
    assert len(os.path.basename(store.channel_directory(identity))) <= 200


def test_deployment_and_fingerprint_namespaces_are_separate(store):
    channel_id = "shared-channel"
    publish(store, identity_for(channel_id, deployment_id="blue"), marker="blue")
    publish(store, identity_for(channel_id, deployment_id="green"), marker="green")
    publish(
        store,
        identity_for(channel_id, workflow_fingerprint="wf-other"),
        marker="other-wf",
    )

    assert (
        store.load(identity_for(channel_id, deployment_id="blue")).context["marker"]
        == "blue"
    )
    assert (
        store.load(identity_for(channel_id, deployment_id="green")).context["marker"]
        == "green"
    )
    assert (
        store.load(
            identity_for(channel_id, workflow_fingerprint="wf-other")
        ).context["marker"]
        == "other-wf"
    )


def test_the_checkpoint_namespace_does_not_collide_with_pending_state(base):
    """Invariant 16: two record kinds, one channel id, no shared key space."""
    os.makedirs(base, exist_ok=True)
    pending = DiskSessionStateStore(base)
    checkpoints = ChannelCheckpointStore(base)
    identity = identity_for("dual-kind")

    pending.save("dual-kind", {"awaiting_user": True})
    publish(checkpoints, identity)

    pending.clear("dual-kind")

    assert pending.exists("dual-kind") is False
    assert checkpoints.load(identity) is not None


# ---------------------------------------------------------------------------
# 3 & 4. Identity mismatch quarantines, and quarantine preserves
# ---------------------------------------------------------------------------


def relocate(store, source: CheckpointIdentity, target: CheckpointIdentity) -> None:
    """Move one channel's records to where another channel's reader will look.

    This is the exact hazard a non-injective key produced for free: a record
    that is physically in the right place and semantically the wrong channel's.

    Fields outside the key path — session incarnation above all — already put the
    record exactly where the reader looks, which is deliberate: a recycled
    channel id must be *found* and refused, not missed.
    """
    destination = store.channel_directory(target)
    origin = store.channel_directory(source)
    if os.path.realpath(origin) == os.path.realpath(destination):
        return
    shutil.rmtree(destination, ignore_errors=True)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.move(origin, destination)


@pytest.mark.parametrize(
    "overrides,expected_field",
    [
        ({"deployment_id": "deploy-green"}, "deployment_id"),
        ({"workflow_fingerprint": "wf-other"}, "workflow_fingerprint"),
        ({"session_incarnation": "inc-2"}, "session_incarnation"),
        ({"channel_id": "channel-victim"}, "channel_key"),
    ],
)
def test_an_identity_mismatch_quarantines_instead_of_applying(
    store, warnings_logged, overrides, expected_field
):
    writer = identity_for("channel-writer")
    reader = identity_for("channel-writer", **overrides)
    publish(store, writer, marker="not-yours")

    relocate(store, writer, reader)
    assert store.exists(reader) is True

    assert store.load(reader) is None
    assert store.exists(reader) is False

    preserved = store.list_quarantined(reader)
    assert len(preserved) == 1
    assert os.path.isfile(os.path.join(preserved[0], "COMMIT"))

    message = "\n".join(record.getMessage() for record in warnings_logged)
    assert expected_field in message
    assert reader.channel_id in message


def test_two_channels_with_swapped_records_both_quarantine(store):
    left = identity_for("channel-left")
    right = identity_for("channel-right")
    publish(store, left, marker="left")
    publish(store, right, marker="right")

    left_dir = store.channel_directory(left)
    right_dir = store.channel_directory(right)
    staging = f"{left_dir}.staging"
    shutil.move(left_dir, staging)
    shutil.move(right_dir, left_dir)
    shutil.move(staging, right_dir)

    assert store.load(left) is None
    assert store.load(right) is None
    assert len(store.list_quarantined(left)) == 1
    assert len(store.list_quarantined(right)) == 1


def test_swapping_a_single_part_between_channels_quarantines(store):
    left = identity_for("part-left")
    right = identity_for("part-right")
    publish(store, left, marker="left")
    publish(store, right, marker="right")

    shutil.copy(
        os.path.join(generation_dir(store, right, 1), "context.json"),
        os.path.join(generation_dir(store, left, 1), "context.json"),
    )

    assert store.load(left) is None
    assert len(store.list_quarantined(left)) == 1
    # The other channel is untouched: quarantine is per channel.
    assert store.load(right).context["marker"] == "right"


def test_channel_id_reuse_across_incarnations_quarantines(store):
    first = identity_for("recycled", session_incarnation="inc-1")
    second = identity_for("recycled", session_incarnation="inc-2")
    publish(store, first)
    relocate(store, first, second)

    assert store.load(second) is None
    assert len(store.list_quarantined(second)) == 1


def test_a_quarantine_warning_names_the_channel_but_never_a_value(
    store, warnings_logged
):
    """A record set aside because it is suspect is not a licence to log it."""
    secret = "sk-live-51H9zzzTOPSECRET"
    identity = identity_for("channel-hygiene")
    store.publish(
        identity,
        context={"api_key": secret, "api_base": "https://vendor.invalid"},
        runtime={"stream_format": "ndjson"},
        startup={"state": "succeeded", "idempotency_key": "idem-secret-9"},
        launch_context={"prior_projection": {"tenant": "acme-holdings"}},
        state_version=1,
    )

    relocate(store, identity, identity_for("channel-hygiene", deployment_id="other"))
    assert store.load(identity_for("channel-hygiene", deployment_id="other")) is None

    logged = "\n".join(record.getMessage() for record in warnings_logged)
    assert "channel-hygiene" in logged
    assert "deployment_id" in logged
    assert "disk" in logged
    for value in (secret, "vendor.invalid", "idem-secret-9", "acme-holdings", "ndjson"):
        assert value not in logged


def test_quarantine_preserves_every_participating_record(store):
    identity = identity_for("channel-preserve")
    publish(store, identity, marker="keepme")

    store.quarantine(identity, QuarantineReason.OPERATOR_REQUEST)

    assert store.load(identity) is None
    assert store.exists(identity) is False

    preserved = store.list_quarantined(identity)
    assert len(preserved) == 1
    assert "operator_request" in os.path.basename(preserved[0])

    generation_path = os.path.join(preserved[0], "gen", "1")
    for section in PART_SECTIONS:
        assert os.path.isfile(os.path.join(generation_path, f"{section}.json"))
    with open(os.path.join(generation_path, "context.json"), encoding="utf-8") as f:
        assert json.load(f)["payload"]["marker"] == "keepme"


def test_repeated_quarantine_never_overwrites_an_earlier_one(store):
    identity = identity_for("channel-twice")
    publish(store, identity, marker="first")
    store.quarantine(identity, QuarantineReason.OPERATOR_REQUEST)
    publish(store, identity, marker="second")
    store.quarantine(identity, QuarantineReason.OPERATOR_REQUEST)

    preserved = store.list_quarantined(identity)
    assert len(preserved) == 2

    markers = set()
    for directory in preserved:
        with open(
            os.path.join(directory, "gen", "1", "context.json"), encoding="utf-8"
        ) as handle:
            markers.add(json.load(handle)["payload"]["marker"])
    assert markers == {"first", "second"}


def test_publishing_after_a_quarantine_starts_a_fresh_lineage(store):
    identity = identity_for("channel-restart")
    publish(store, identity, marker="old")
    store.quarantine(identity, QuarantineReason.OPERATOR_REQUEST)

    assert publish(store, identity, marker="new") == 1
    assert store.load(identity).context["marker"] == "new"
    assert len(store.list_quarantined(identity)) == 1


# ---------------------------------------------------------------------------
# 5. Protocol version floor (invariant 31)
# ---------------------------------------------------------------------------


def test_a_record_above_the_readable_ceiling_quarantines(base, warnings_logged):
    identity = identity_for("channel-v2")
    future_node = ChannelCheckpointStore(base, protocol_version=2)
    publish(future_node, identity, marker="from-v2")

    old_node = ChannelCheckpointStore(base)
    assert old_node.load(identity) is None

    preserved = old_node.list_quarantined(identity)
    assert len(preserved) == 1
    message = "\n".join(record.getMessage() for record in warnings_logged)
    assert "protocol_version" in message


def test_a_record_below_the_declared_floor_quarantines(base):
    identity = identity_for("channel-floor")
    publish(ChannelCheckpointStore(base), identity)

    strict_node = ChannelCheckpointStore(
        base, protocol_version=2, min_readable_protocol_version=2
    )
    assert strict_node.load(identity) is None
    assert len(strict_node.list_quarantined(identity)) == 1


def test_an_unreadable_record_is_never_written_around(base):
    """Invariant 31: an old node must not publish a competing lineage beside it."""
    identity = identity_for("channel-skew")
    future_node = ChannelCheckpointStore(base, protocol_version=2)
    publish(future_node, identity, marker="from-v2")

    old_node = ChannelCheckpointStore(base)
    assert old_node.publish(
        identity, **sample_sections("from-v1"), state_version=1
    ) == 1

    preserved = old_node.list_quarantined(identity)
    assert len(preserved) == 1
    with open(
        os.path.join(preserved[0], "gen", "1", "context.json"), encoding="utf-8"
    ) as handle:
        assert json.load(handle)["payload"]["marker"] == "from-v2"


def test_the_protocol_version_is_not_part_of_the_path(base):
    """Version-partitioned paths are how a node misses a record it cannot read."""
    identity = identity_for("channel-path")
    v1 = ChannelCheckpointStore(base)
    v2 = ChannelCheckpointStore(base, protocol_version=2)

    assert v1.channel_directory(identity) == v2.channel_directory(identity)


def test_a_node_cannot_declare_a_floor_it_cannot_itself_read(base):
    with pytest.raises(ValueError, match="unable to read itself"):
        ChannelCheckpointStore(base, min_readable_protocol_version=2)


# ---------------------------------------------------------------------------
# 6. Generation atomicity (invariant 26)
# ---------------------------------------------------------------------------


def test_a_deleted_participating_record_fails_closed(store):
    identity = identity_for("gen-delete")
    publish(store, identity, marker="one")
    publish(store, identity, marker="two")

    os.remove(os.path.join(generation_dir(store, identity, 2), "runtime.json"))

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_a_truncated_participating_record_fails_closed(store):
    identity = identity_for("gen-truncate")
    publish(store, identity, marker="one")
    publish(store, identity, marker="two")

    path = os.path.join(generation_dir(store, identity, 2), "context.json")
    with open(path, "r+", encoding="utf-8") as handle:
        handle.truncate(12)

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_a_missing_generation_never_falls_back_to_the_previous_one(store):
    """The merge this protocol exists to prevent: context N, continuation N-1."""
    identity = identity_for("gen-missing")
    publish(store, identity, marker="one")
    publish(store, identity, marker="two")

    assert os.path.isdir(generation_dir(store, identity, 1))
    shutil.rmtree(generation_dir(store, identity, 2))

    assert store.load(identity) is None

    preserved = store.list_quarantined(identity)
    assert len(preserved) == 1
    # Generation 1 was intact and right there; it was still not served.
    assert os.path.isdir(os.path.join(preserved[0], "gen", "1"))


def test_a_part_stamped_with_the_wrong_generation_fails_closed(store):
    identity = identity_for("gen-stamp")
    publish(store, identity, marker="one")
    publish(store, identity, marker="two")

    source = os.path.join(generation_dir(store, identity, 1), "context.json")
    target = os.path.join(generation_dir(store, identity, 2), "context.json")
    shutil.copy(source, target)

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_a_tampered_part_payload_fails_closed(store):
    """Valid JSON, correct identity, correct generation — and different content."""
    identity = identity_for("gen-tamper-payload")
    publish(store, identity, marker="one")

    path = os.path.join(generation_dir(store, identity, 1), "context.json")
    with open(path, encoding="utf-8") as handle:
        part = json.load(handle)
    part["payload"]["marker"] = "rewritten"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(part, handle)

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_a_byte_identical_rewrite_of_a_part_still_fails_closed(store):
    """The manifest pins exact bytes, so even a semantics-preserving rewrite
    quarantines. Strict is the point: an authoritative record nobody wrote is
    a record somebody else wrote, and quarantine preserves it either way."""
    identity = identity_for("gen-reformat")
    publish(store, identity, marker="one")

    path = os.path.join(generation_dir(store, identity, 1), "context.json")
    with open(path, encoding="utf-8") as handle:
        part = json.load(handle)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(part, handle, indent=2)

    with open(path, encoding="utf-8") as handle:
        assert json.load(handle) == part

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_a_part_with_a_disagreeing_state_version_fails_closed(store):
    identity = identity_for("gen-stateversion")
    publish(store, identity, state_version=2)

    rewrite_part(
        os.path.join(generation_dir(store, identity, 1), "startup.json"),
        state_version=3,
    )

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_a_crash_between_the_generation_rename_and_the_commit_keeps_the_old_record(
    store,
):
    """Boundary: parts of N are whole on disk, COMMIT still names N-1."""
    identity = identity_for("gen-precommit")
    publish(store, identity, marker="one")

    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    with open(commit_path, "rb") as handle:
        commit_at_generation_one = handle.read()

    publish(store, identity, marker="two")
    assert os.path.isdir(generation_dir(store, identity, 2))

    with open(commit_path, "wb") as handle:
        handle.write(commit_at_generation_one)

    record = store.load(identity)
    assert record is not None
    assert record.generation == 1
    assert record.context["marker"] == "one"
    assert store.list_quarantined(identity) == []


def test_a_crash_while_writing_parts_leaves_the_committed_record_readable(store):
    """Boundary: a pending directory the reader must never enumerate."""
    identity = identity_for("gen-pending")
    publish(store, identity, marker="one")

    pending = os.path.join(
        store.channel_directory(identity), "gen", ".pending-2-deadbeef"
    )
    os.makedirs(pending)
    with open(os.path.join(pending, "context.json"), "w", encoding="utf-8") as handle:
        handle.write('{"protocol_version": 1, "record')

    record = store.load(identity)
    assert record.generation == 1
    assert record.context["marker"] == "one"

    # The next publish steps over the orphan rather than colliding with it.
    assert publish(store, identity, marker="two") == 2
    assert not os.path.isdir(pending)


def test_a_crash_during_the_generation_rename_is_invisible(store):
    """Boundary: `gen/N` exists but was never committed."""
    identity = identity_for("gen-halfrename")
    publish(store, identity, marker="one")

    orphan = generation_dir(store, identity, 5)
    shutil.copytree(generation_dir(store, identity, 1), orphan)

    record = store.load(identity)
    assert record.generation == 1

    # Monotonicity: the next generation clears the orphan's number.
    assert publish(store, identity, marker="two") == 6


def test_a_tampered_commit_fails_closed(store):
    identity = identity_for("gen-commit-tamper")
    publish(store, identity, marker="one")

    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    rewrite_part(commit_path, manifest_digest="0" * 64)

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_a_commit_naming_a_generation_that_never_existed_fails_closed(store):
    identity = identity_for("gen-phantom")
    publish(store, identity, marker="one")

    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    rewrite_part(commit_path, generation=99)

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_an_unparseable_commit_fails_closed(store):
    identity = identity_for("gen-garbage")
    publish(store, identity, marker="one")

    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    with open(commit_path, "w", encoding="utf-8") as handle:
        handle.write("{not json")

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_a_manifest_missing_a_part_entry_fails_closed(store):
    identity = identity_for("gen-shortmanifest")
    publish(store, identity, marker="one")

    manifest_path = os.path.join(generation_dir(store, identity, 1), "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["parts"].pop("startup")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_the_committed_generation_is_reconstructible_in_a_fresh_process(store):
    """The design asks for reconstruction in a fresh process, not a fresh object."""
    identity = identity_for("gen-freshproc")
    publish(store, identity, marker="cold")

    script = textwrap.dedent(
        """
        import json, sys
        from fastworkflow.checkpoint_store import (
            ChannelCheckpointStore, CheckpointIdentity,
        )

        store = ChannelCheckpointStore(sys.argv[1])
        record = store.load(CheckpointIdentity(*json.loads(sys.argv[2])))
        print(json.dumps(None if record is None else record.to_payload()))
        """
    )
    arguments = json.dumps(
        [
            identity.deployment_id,
            identity.workflow_fingerprint,
            identity.channel_id,
            identity.session_incarnation,
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, store.base_folder, arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["generation"] == 1
    assert payload["context"]["marker"] == "cold"
    assert payload["channel_id"] == identity.channel_id


# ---------------------------------------------------------------------------
# 7. Digest stability
# ---------------------------------------------------------------------------


def test_an_unchanged_republish_rewrites_no_record(store):
    identity = identity_for("channel-stable")
    first = publish(store, identity, marker="same")
    before = snapshot_records(store.base_folder)

    second = publish(store, identity, marker="same")

    assert second == first
    assert snapshot_records(store.base_folder) == before


def test_an_unchanged_republish_still_counts_as_activity(store):
    """Otherwise retention would reap the channel that is being used most."""
    identity = identity_for("channel-stable-age")
    publish(store, identity, marker="same")
    first_seen = access_marker(store, identity)["last_publish_at"]

    publish(store, identity, marker="same")

    second_seen = access_marker(store, identity)["last_publish_at"]
    assert second_seen > first_seen
    # The timestamp that would have cost a full rewrite did not move with it.
    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    with open(commit_path, encoding="utf-8") as handle:
        assert json.load(handle)["published_at"] < second_seen


def test_key_order_does_not_count_as_a_change(store):
    """An unstable encoding would force a write on every retirement."""
    identity = identity_for("channel-order")
    store.publish(
        identity,
        context={"a": 1, "b": {"c": 2, "d": 3}},
        runtime={},
        startup={},
        launch_context={},
        state_version=1,
    )
    before = snapshot_records(store.base_folder)

    generation = store.publish(
        identity,
        context={"b": {"d": 3, "c": 2}, "a": 1},
        runtime={},
        startup={},
        launch_context={},
        state_version=1,
    )

    assert generation == 1
    assert snapshot_records(store.base_folder) == before


def test_a_changed_section_does_produce_a_write(store):
    identity = identity_for("channel-changed")
    publish(store, identity, marker="one")
    before = snapshot_tree(store.base_folder)

    assert publish(store, identity, marker="two") == 2
    assert snapshot_tree(store.base_folder) != before


def test_a_changed_state_version_alone_produces_a_write(store):
    identity = identity_for("channel-version-bump")
    publish(store, identity, marker="same", state_version=1)

    assert publish(store, identity, marker="same", state_version=2) == 2
    assert store.load(identity).state_version == 2


def test_the_fast_path_does_not_skip_over_a_damaged_generation(store):
    """Skipping a write is only safe if there is something whole to keep."""
    identity = identity_for("channel-damaged")
    publish(store, identity, marker="same")
    os.remove(os.path.join(generation_dir(store, identity, 1), "startup.json"))

    assert publish(store, identity, marker="same") == 2
    assert store.load(identity).context["marker"] == "same"


# ---------------------------------------------------------------------------
# 8. Atomic publication
# ---------------------------------------------------------------------------


def test_a_truncated_temp_file_never_becomes_the_live_record(store):
    identity = identity_for("channel-tmp")
    publish(store, identity, marker="live")

    channel_dir = store.channel_directory(identity)
    leftover = os.path.join(channel_dir, "COMMIT.tmp-abandoned")
    with open(leftover, "w", encoding="utf-8") as handle:
        handle.write('{"protocol_version": 1, "generation": 99')

    record = store.load(identity)
    assert record.generation == 1
    assert record.context["marker"] == "live"

    assert publish(store, identity, marker="next") == 2
    assert store.load(identity).context["marker"] == "next"


def test_the_commit_file_is_replaced_whole(store):
    identity = identity_for("channel-wholecommit")
    publish(store, identity, marker="one")
    publish(store, identity, marker="two")

    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    with open(commit_path, encoding="utf-8") as handle:
        commit = json.load(handle)

    assert commit["record_type"] == COMMIT_RECORD_TYPE
    assert commit["generation"] == 2
    assert commit["channel_id"] == identity.channel_id


def test_only_the_retained_generations_survive_a_publish(store):
    identity = identity_for("channel-retention")
    for index in range(5):
        publish(store, identity, marker=f"m{index}")

    generations = sorted(
        int(name)
        for name in os.listdir(os.path.join(store.channel_directory(identity), "gen"))
        if name.isdigit()
    )
    assert generations == [4, 5]
    assert store.load(identity).generation == 5


def test_publication_refuses_to_follow_a_symlinked_directory(store, tmp_path):
    identity = identity_for("channel-symlink")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    victim = store.channel_directory(identity)
    os.makedirs(os.path.dirname(victim), exist_ok=True)
    os.symlink(str(elsewhere), victim, target_is_directory=True)

    with pytest.raises(CheckpointStoreError, match="symlink"):
        publish(store, identity)

    assert not any(elsewhere.iterdir())


def test_a_record_replaced_by_a_symlink_is_not_read_through(store, tmp_path):
    identity = identity_for("channel-symlink-read")
    publish(store, identity, marker="real")

    decoy = tmp_path / "decoy.json"
    decoy.write_text("{}", encoding="utf-8")
    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    os.remove(commit_path)
    os.symlink(str(decoy), commit_path)

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


# ---------------------------------------------------------------------------
# 9. Private modes
# ---------------------------------------------------------------------------


def test_every_directory_and_file_is_private(store):
    publish(store, identity_for("channel-modes"))
    store.quarantine(identity_for("channel-modes"), QuarantineReason.OPERATOR_REQUEST)
    publish(store, identity_for("channel-modes"))

    checked = 0
    for directory, subdirs, files in os.walk(store.base_folder):
        assert stat.S_IMODE(os.lstat(directory).st_mode) == 0o700, directory
        checked += 1
        for name in files:
            path = os.path.join(directory, name)
            assert stat.S_IMODE(os.lstat(path).st_mode) == 0o600, path
            checked += 1

    assert stat.S_IMODE(os.lstat(store.base_folder).st_mode) == 0o700
    assert checked > 5


def test_private_modes_hold_under_a_permissive_umask(store):
    previous = os.umask(0o000)
    try:
        publish(store, identity_for("channel-umask"))
    finally:
        os.umask(previous)

    for directory, _subdirs, files in os.walk(store.base_folder):
        assert stat.S_IMODE(os.lstat(directory).st_mode) == 0o700
        for name in files:
            path = os.path.join(directory, name)
            assert stat.S_IMODE(os.lstat(path).st_mode) == 0o600


# ---------------------------------------------------------------------------
# 10. Strictness, through state_serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section",
    ["context", "runtime", "startup", "launch_context"],
)
def test_a_non_json_native_value_is_rejected_and_nothing_is_written(store, section):
    identity = identity_for(f"strict-{section}")
    sections = sample_sections()
    sections[section] = {"deadline": datetime(2026, 8, 6)}

    with pytest.raises(StateEncodingError) as failure:
        store.publish(identity, **sections, state_version=1)

    assert "datetime" in str(failure.value)
    assert section in str(failure.value)
    assert not os.path.exists(store.channel_directory(identity))


def test_a_non_string_key_is_rejected_and_nothing_is_written(store):
    identity = identity_for("strict-key")
    sections = sample_sections()
    sections["context"] = {1: "first"}

    with pytest.raises(StateEncodingError, match="not str"):
        store.publish(identity, **sections, state_version=1)

    assert not os.path.exists(store.channel_directory(identity))


@pytest.mark.parametrize("value", [float("nan"), math.inf, -math.inf])
def test_non_finite_floats_are_rejected(store, value):
    identity = identity_for("strict-float")
    sections = sample_sections()
    sections["runtime"] = {"ratio": value}

    with pytest.raises(StateEncodingError, match="no JSON representation"):
        store.publish(identity, **sections, state_version=1)

    assert not os.path.exists(store.channel_directory(identity))


def test_a_cycle_is_rejected(store):
    identity = identity_for("strict-cycle")
    sections = sample_sections()
    loop: dict = {}
    loop["self"] = loop
    sections["context"] = loop

    with pytest.raises(StateEncodingError):
        store.publish(identity, **sections, state_version=1)

    assert not os.path.exists(store.channel_directory(identity))


def test_a_container_shared_across_two_sections_is_rejected(store):
    """Restoring it as two objects silently breaks anything relying on sharing."""
    identity = identity_for("strict-shared")
    shared = ["x"]
    sections = sample_sections()
    sections["context"] = {"items": shared}
    sections["runtime"] = {"items": shared}

    with pytest.raises(StateEncodingError, match="two independent copies"):
        store.publish(identity, **sections, state_version=1)

    assert not os.path.exists(store.channel_directory(identity))


def test_a_failed_publish_leaves_a_prior_record_intact(store):
    identity = identity_for("strict-preserve")
    publish(store, identity, marker="good")
    before = snapshot_tree(store.base_folder)

    sections = sample_sections("bad")
    sections["context"] = {"deadline": datetime(2026, 8, 6)}
    with pytest.raises(StateEncodingError):
        store.publish(identity, **sections, state_version=1)

    assert snapshot_tree(store.base_folder) == before
    assert store.load(identity).context["marker"] == "good"


@pytest.mark.parametrize("section", list(PART_SECTIONS))
def test_a_non_dict_section_is_rejected(store, section):
    identity = identity_for(f"strict-shape-{section}")
    sections = sample_sections()
    sections[section] = ["not", "a", "dict"]

    with pytest.raises(StateEncodingError, match="must be a dict"):
        store.publish(identity, **sections, state_version=1)

    assert not os.path.exists(store.channel_directory(identity))


@pytest.mark.parametrize("bad_version", [0, -1, "2", 1.0, True])
def test_an_invalid_state_version_is_rejected(store, bad_version):
    identity = identity_for("strict-version")

    with pytest.raises(ValueError):
        store.publish(
            identity, **sample_sections(), state_version=bad_version
        )

    assert not os.path.exists(store.channel_directory(identity))


@pytest.mark.parametrize(
    "overrides",
    [
        {"deployment_id": ""},
        {"workflow_fingerprint": ""},
        {"channel_id": ""},
        {"session_incarnation": ""},
        {"channel_id": None},
        {"channel_id": 42},
    ],
)
def test_an_empty_identity_field_is_refused(overrides):
    with pytest.raises(ValueError, match="non-empty str"):
        identity_for("channel-ok", **overrides)


# ---------------------------------------------------------------------------
# 11. Adoption: the cold restore that cannot know its own incarnation
# ---------------------------------------------------------------------------


def test_scope_names_the_three_fields_a_cold_restore_already_has():
    identity = identity_for("scope-check", session_incarnation="inc-9")

    assert identity.scope() == {
        "deployment_id": "deploy-blue",
        "workflow_fingerprint": "wf-abc123",
        "channel_id": "scope-check",
    }
    assert "session_incarnation" not in identity.scope()


def test_a_fresh_incarnation_adopts_cleanly_where_a_strict_read_refuses(store):
    """The gap: minting an incarnation and reading strictly bins a good record."""
    stored = identity_for("adopt-contrast", session_incarnation="inc-stored")
    publish(store, stored, marker="durable")

    adopted = store.load_for_adoption(**stored.scope())
    assert adopted is not None
    assert adopted.identity.session_incarnation == "inc-stored"
    assert adopted.context["marker"] == "durable"
    assert adopted.generation == 1
    assert store.list_quarantined(stored) == []

    fresh = identity_for("adopt-contrast", session_incarnation="inc-fresh")
    assert store.load(fresh) is None
    assert len(store.list_quarantined(fresh)) == 1
    assert store.exists(stored) is False


def test_adoption_reports_the_stored_incarnation_verbatim(store):
    identity = identity_for("adopt-verbatim", session_incarnation="inc-\u00e9-42/x")
    publish(store, identity)

    adopted = store.load_for_adoption(**identity.scope())

    assert adopted.identity.session_incarnation == "inc-\u00e9-42/x"
    assert adopted.identity == identity
    assert adopted.to_payload()["session_incarnation"] == "inc-\u00e9-42/x"


def test_adoption_of_an_absent_record_returns_none_and_quarantines_nothing(store):
    identity = identity_for("adopt-absent")

    assert store.load_for_adoption(**identity.scope()) is None
    assert store.list_quarantined(identity) == []
    assert not os.path.exists(store.channel_directory(identity))


def test_publish_adopt_publish_keeps_the_incarnation_and_advances_the_generation(
    store,
):
    original = identity_for("adopt-cycle", session_incarnation="inc-original")
    assert publish(store, original, marker="before") == 1

    adopted = store.load_for_adoption(**original.scope())
    assert adopted.identity.session_incarnation == "inc-original"

    assert publish(store, adopted.identity, marker="after") == 2

    again = store.load_for_adoption(**original.scope())
    assert again.identity.session_incarnation == "inc-original"
    assert again.generation == 2
    assert again.context["marker"] == "after"
    assert store.list_quarantined(original) == []

    # Having adopted, the caller knows who it is, so the strict read works too.
    assert store.load(adopted.identity).generation == 2


def test_the_adoption_probe_never_reaches_disk(store):
    """A placeholder that leaked into a record would be a fabricated identity."""
    identity = identity_for("adopt-probe")
    publish(store, identity)
    assert store.load_for_adoption(**identity.scope()) is not None

    displaced = identity_for("adopt-probe", deployment_id="other")
    relocate(store, identity, displaced)
    assert store.load_for_adoption(**displaced.scope()) is None

    for directory, _subdirs, files in os.walk(store.base_folder):
        assert "adoption-probe" not in directory
        for name in files:
            assert "adoption-probe" not in name
            with open(os.path.join(directory, name), "rb") as handle:
                assert b"adoption-probe" not in handle.read()


# -- relaxing the incarnation must not relax anything else ------------------


@pytest.mark.parametrize(
    "overrides,expected_field",
    [
        ({"deployment_id": "deploy-green"}, "deployment_id"),
        ({"workflow_fingerprint": "wf-other"}, "workflow_fingerprint"),
        ({"channel_id": "channel-victim"}, "channel_key"),
    ],
)
def test_adoption_still_quarantines_on_a_non_incarnation_mismatch(
    store, warnings_logged, overrides, expected_field
):
    writer = identity_for("adopt-mismatch")
    reader = identity_for("adopt-mismatch", **overrides)
    publish(store, writer, marker="not-yours")
    relocate(store, writer, reader)

    assert store.load_for_adoption(**reader.scope()) is None
    assert len(store.list_quarantined(reader)) == 1

    message = "\n".join(record.getMessage() for record in warnings_logged)
    assert expected_field in message


def test_adoption_still_quarantines_an_unreadable_protocol_version(base):
    """Kind is checked before the incarnation is trusted, not after."""
    identity = identity_for("adopt-protocol")
    publish(ChannelCheckpointStore(base, protocol_version=2), identity)

    old_node = ChannelCheckpointStore(base)
    assert old_node.load_for_adoption(**identity.scope()) is None
    assert len(old_node.list_quarantined(identity)) == 1


def test_adoption_still_quarantines_below_the_declared_floor(base):
    identity = identity_for("adopt-floor")
    publish(ChannelCheckpointStore(base), identity)

    strict_node = ChannelCheckpointStore(
        base, protocol_version=2, min_readable_protocol_version=2
    )
    assert strict_node.load_for_adoption(**identity.scope()) is None
    assert len(strict_node.list_quarantined(identity)) == 1


def test_adoption_refuses_a_commit_with_no_usable_incarnation(store):
    identity = identity_for("adopt-noincarnation")
    publish(store, identity)

    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    rewrite_part(commit_path, session_incarnation="")

    assert store.load_for_adoption(**identity.scope()) is None
    assert len(store.list_quarantined(identity)) == 1


def test_an_unreadable_version_is_reported_before_the_session_is_trusted(store):
    """Ordering discipline, made observable through the quarantine reason.

    Adoption *trusts* the incarnation it reads out of COMMIT, so the version
    that gives that field its meaning has to be checked first. Break both and
    the version has to be the reason.
    """
    identity = identity_for("adopt-order")
    publish(store, identity)

    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    with open(commit_path, encoding="utf-8") as handle:
        commit = json.load(handle)
    commit["protocol_version"] = 99
    commit.pop("session_incarnation")
    with open(commit_path, "w", encoding="utf-8") as handle:
        json.dump(commit, handle)

    assert store.load_for_adoption(**identity.scope()) is None

    preserved = store.list_quarantined(identity)
    assert len(preserved) == 1
    assert os.path.basename(preserved[0]).startswith("protocol_version_unreadable")


def test_rechaining_alone_leaves_the_record_readable(store):
    """Control for the coherent-tamper tests below: repairing digests is a no-op."""
    identity = identity_for("adopt-rechain-control")
    publish(store, identity, marker="one")

    rechain(store, identity)

    assert store.load(identity).context["marker"] == "one"
    assert store.load_for_adoption(**identity.scope()) is not None


def test_adoption_refuses_a_generation_whose_parts_disagree_about_the_session(store):
    """The discovered incarnation binds the parts; it does not excuse them.

    Digests are repaired so this is not caught by the digest chain — it has to
    be caught by validating each part against the incarnation COMMIT named.
    """
    identity = identity_for("adopt-incoherent")
    publish(store, identity, marker="one")

    rewrite_part(
        os.path.join(generation_dir(store, identity, 1), "context.json"),
        session_incarnation="inc-somebody-else",
    )
    rechain(store, identity)

    assert store.load_for_adoption(**identity.scope()) is None
    assert len(store.list_quarantined(identity)) == 1


def test_adoption_refuses_a_manifest_from_another_session(store):
    identity = identity_for("adopt-manifest-session")
    publish(store, identity, marker="one")

    rewrite_part(
        os.path.join(generation_dir(store, identity, 1), "manifest.json"),
        session_incarnation="inc-somebody-else",
    )
    rechain(store, identity)

    assert store.load_for_adoption(**identity.scope()) is None
    assert len(store.list_quarantined(identity)) == 1


# Every generation and digest fault from section 6, replayed through adoption.


def _break_deleted_part(store, identity):
    publish(store, identity, marker="one")
    publish(store, identity, marker="two")
    os.remove(os.path.join(generation_dir(store, identity, 2), "runtime.json"))


def _break_truncated_part(store, identity):
    publish(store, identity, marker="one")
    path = os.path.join(generation_dir(store, identity, 1), "context.json")
    with open(path, "r+", encoding="utf-8") as handle:
        handle.truncate(12)


def _break_missing_generation(store, identity):
    publish(store, identity, marker="one")
    publish(store, identity, marker="two")
    shutil.rmtree(generation_dir(store, identity, 2))


def _break_back_dated_part(store, identity):
    publish(store, identity, marker="one")
    publish(store, identity, marker="two")
    shutil.copy(
        os.path.join(generation_dir(store, identity, 1), "context.json"),
        os.path.join(generation_dir(store, identity, 2), "context.json"),
    )


def _break_tampered_payload(store, identity):
    publish(store, identity, marker="one")
    path = os.path.join(generation_dir(store, identity, 1), "context.json")
    with open(path, encoding="utf-8") as handle:
        part = json.load(handle)
    part["payload"]["marker"] = "rewritten"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(part, handle)


def _break_reformatted_part(store, identity):
    publish(store, identity, marker="one")
    path = os.path.join(generation_dir(store, identity, 1), "context.json")
    with open(path, encoding="utf-8") as handle:
        part = json.load(handle)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(part, handle, indent=2)


def _break_state_version_disagreement(store, identity):
    publish(store, identity, state_version=2)
    rewrite_part(
        os.path.join(generation_dir(store, identity, 1), "startup.json"),
        state_version=3,
    )


def _break_manifest_digest(store, identity):
    publish(store, identity, marker="one")
    rewrite_part(
        os.path.join(store.channel_directory(identity), "COMMIT"),
        manifest_digest="0" * 64,
    )


def _break_phantom_generation(store, identity):
    publish(store, identity, marker="one")
    rewrite_part(
        os.path.join(store.channel_directory(identity), "COMMIT"), generation=99
    )


def _break_unparseable_commit(store, identity):
    publish(store, identity, marker="one")
    with open(
        os.path.join(store.channel_directory(identity), "COMMIT"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("{not json")


def _break_short_manifest(store, identity):
    publish(store, identity, marker="one")
    manifest_path = os.path.join(generation_dir(store, identity, 1), "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["parts"].pop("startup")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


TAMPERINGS = [
    ("deleted_part", _break_deleted_part),
    ("truncated_part", _break_truncated_part),
    ("missing_generation", _break_missing_generation),
    ("back_dated_part", _break_back_dated_part),
    ("tampered_payload", _break_tampered_payload),
    ("reformatted_part", _break_reformatted_part),
    ("state_version_disagreement", _break_state_version_disagreement),
    ("manifest_digest", _break_manifest_digest),
    ("phantom_generation", _break_phantom_generation),
    ("unparseable_commit", _break_unparseable_commit),
    ("short_manifest", _break_short_manifest),
]


@pytest.mark.parametrize(
    "break_it", [fault for _name, fault in TAMPERINGS], ids=[n for n, _ in TAMPERINGS]
)
def test_adoption_fails_closed_on_every_generation_and_digest_fault(store, break_it):
    identity = identity_for("adopt-fault")
    break_it(store, identity)

    assert store.load_for_adoption(**identity.scope()) is None
    assert len(store.list_quarantined(identity)) == 1


@pytest.mark.parametrize(
    "break_it", [fault for _name, fault in TAMPERINGS], ids=[n for n, _ in TAMPERINGS]
)
def test_the_strict_read_fails_closed_on_the_same_faults(store, break_it):
    """The matrix is only evidence about adoption if it also holds for `load`."""
    identity = identity_for("strict-fault")
    break_it(store, identity)

    assert store.load(identity) is None
    assert len(store.list_quarantined(identity)) == 1


def test_an_untampered_record_survives_the_matrix_setup(store):
    """Guards the matrix against passing because the setup wrote nothing."""
    identity = identity_for("adopt-control")
    publish(store, identity, marker="one")
    publish(store, identity, marker="two")

    assert store.load_for_adoption(**identity.scope()).context["marker"] == "two"
    assert store.load(identity).context["marker"] == "two"
    assert store.list_quarantined(identity) == []


# ---------------------------------------------------------------------------
# 12. The publish reuse guard
# ---------------------------------------------------------------------------


def test_publishing_under_a_different_incarnation_refuses(store):
    """Another session lifetime's valid state is not ours to displace."""
    owner = identity_for("reuse", session_incarnation="inc-owner")
    publish(store, owner, marker="owned")
    before = snapshot_tree(store.base_folder)

    intruder = identity_for("reuse", session_incarnation="inc-intruder")
    with pytest.raises(CheckpointStoreError, match="session_incarnation"):
        publish(store, intruder, marker="stolen")

    # Refused, not quarantined: the record stays exactly where it was, whole.
    assert snapshot_tree(store.base_folder) == before
    assert store.list_quarantined(owner) == []
    assert store.exists(owner) is True

    record = store.load(owner)
    assert record.generation == 1
    assert record.context["marker"] == "owned"


def test_the_reuse_refusal_names_the_channel_and_no_incarnation(store):
    owner = identity_for("reuse-hygiene", session_incarnation="inc-secret-owner")
    publish(store, owner)
    intruder = identity_for(
        "reuse-hygiene", session_incarnation="inc-secret-intruder"
    )

    with pytest.raises(CheckpointStoreError) as failure:
        publish(store, intruder)

    message = str(failure.value)
    assert "reuse-hygiene" in message
    assert "session_incarnation" in message
    assert "load_for_adoption" in message
    assert "inc-secret-owner" not in message
    assert "inc-secret-intruder" not in message


def test_the_refused_publisher_can_adopt_the_stored_lifetime(store):
    owner = identity_for("reuse-adopt", session_incarnation="inc-owner")
    publish(store, owner, marker="owned")
    intruder = identity_for("reuse-adopt", session_incarnation="inc-intruder")

    with pytest.raises(CheckpointStoreError):
        publish(store, intruder, marker="stolen")

    adopted = store.load_for_adoption(**intruder.scope())
    assert adopted.identity.session_incarnation == "inc-owner"
    assert publish(store, adopted.identity, marker="continued") == 2
    assert store.load(owner).context["marker"] == "continued"


def test_the_refused_publisher_can_clear_and_take_the_channel(store):
    owner = identity_for("reuse-clear", session_incarnation="inc-owner")
    publish(store, owner, marker="owned")
    intruder = identity_for("reuse-clear", session_incarnation="inc-intruder")

    with pytest.raises(CheckpointStoreError):
        publish(store, intruder, marker="stolen")

    store.clear(intruder)

    assert publish(store, intruder, marker="mine-now") == 1
    assert store.load(intruder).context["marker"] == "mine-now"


def test_the_guard_does_not_fire_for_the_owning_incarnation(store):
    """A guard that refuses the ordinary case is an outage, not a guard."""
    owner = identity_for("reuse-noop", session_incarnation="inc-owner")

    assert publish(store, owner, marker="one") == 1
    assert publish(store, owner, marker="two") == 2
    assert publish(store, owner, marker="two") == 2
    assert store.load(owner).context["marker"] == "two"


def test_a_non_incarnation_fault_still_quarantines_on_publish(store, base):
    """The guard narrows the refusal to reuse; it does not blanket the write path."""
    identity = identity_for("reuse-vs-protocol")
    publish(ChannelCheckpointStore(base, protocol_version=2), identity)

    assert publish(store, identity, marker="fresh") == 1
    assert len(store.list_quarantined(identity)) == 1


def test_reuse_is_refused_before_anything_is_created(tmp_path):
    """Nothing on disk, not even a directory, from a refused publish."""
    base = str(tmp_path / "fresh")
    store = ChannelCheckpointStore(base)
    owner = identity_for("reuse-nothing", session_incarnation="inc-owner")
    publish(store, owner)
    before = snapshot_tree(base)

    intruder = identity_for("reuse-nothing", session_incarnation="inc-intruder")
    with pytest.raises(CheckpointStoreError):
        publish(store, intruder)

    assert snapshot_tree(base) == before


# ---------------------------------------------------------------------------
# 13. Namespace lifecycle: retention, reaping, operator verbs, measurement
# ---------------------------------------------------------------------------

DAY = 86_400.0

# Every knob off. Lets one test exercise one rule instead of discovering which
# of three defaults happened to fire first.
NO_RETENTION = RetentionPolicy(
    max_age_seconds=None,
    max_channels=None,
    max_bytes=None,
    quarantine_max_age_seconds=None,
    quarantine_max_entries=None,
)


def reclaim_root(store, identity) -> str:
    return os.path.join(
        store.base_folder,
        "__reclaim__",
        encode_path_component(identity.deployment_id),
        encode_path_component(identity.workflow_fingerprint),
    )


def interrupt_flip(store, source: str, identity) -> str:
    """Do the rename half of a destructive operation and stop, as a crash would.

    This is the state the store is in between "no longer visible" and "actually
    gone" — the only intermediate state any of its destructive operations has.
    Reproducing it by hand rather than by patching keeps the fault injection at
    the filesystem, where the atomicity being relied on actually lives.
    """
    root = reclaim_root(store, identity)
    os.makedirs(root, mode=0o700, exist_ok=True)
    destination = os.path.join(
        root, f"{os.path.basename(source)}.tmp-interrupted"
    )
    os.replace(source, destination)
    return destination


def block_reclamation(store, identity) -> str:
    """Make retiring any channel in this namespace fail, as a crash would not.

    A plain file where the reclaim holding area belongs: the rename that makes a
    channel invisible has nowhere to land, so `_retire_directory` refuses and the
    channel stays live and loadable. Injected at the filesystem rather than by
    revoking write permission on a parent, which a suite running as root would
    not even notice.
    """
    root = reclaim_root(store, identity)
    os.makedirs(os.path.dirname(root), mode=0o700, exist_ok=True)
    with open(root, "w", encoding="utf-8") as handle:
        handle.write("not a directory")
    return root


def write_floor(store, identity, floor: int) -> None:
    """The first step of `reset`, on its own, so a crash after it can be tested."""
    path = os.path.join(store.channel_directory(identity), "GENERATION_FLOOR")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol_version": PROTOCOL_VERSION,
                "record_type": "channel_checkpoint_generation_floor",
                "channel_key": identity.channel_key,
                "floor": floor,
            },
            handle,
        )


def walk_files(root: str) -> tuple[int, int]:
    """(file count, total apparent bytes) straight off the filesystem."""
    count = 0
    total = 0
    for directory, _subdirs, names in os.walk(root):
        for name in names:
            count += 1
            total += os.lstat(os.path.join(directory, name)).st_size
    return (count, total)


# -- the stated policy -------------------------------------------------------


def test_the_policy_states_its_byte_ceiling():
    """A count cap without its byte cost is unauditable."""
    policy = RetentionPolicy(max_channels=1_000, max_bytes=None)
    described = policy.describe()

    assert "max_channels=1000" in described
    assert "max_age=86400s" in described
    assert policy.worst_case_bytes(REPRESENTATIVE_RECORD_BYTES) == 1_000 * 450 * 1024 * 2
    # The ceiling is stated in bytes rather than left for the reader to multiply.
    assert "878.9 MB" in described


def test_a_policy_with_no_capacity_cap_says_it_is_unbounded():
    """Age alone plateaus at a rate nobody in the system controls."""
    policy = RetentionPolicy(max_age_seconds=DAY, max_channels=None, max_bytes=None)

    assert policy.worst_case_bytes() is None
    assert "UNBOUNDED" in policy.describe()


def test_the_default_policy_is_bounded():
    assert DEFAULT_RETENTION.worst_case_bytes() is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_age_seconds": 0},
        {"max_age_seconds": -1},
        {"max_age_seconds": float("nan")},
        {"max_channels": -1},
        {"max_bytes": -1},
        {"quarantine_max_entries": -1},
        {"max_channels": 1.5},
        {"max_channels": True},
    ],
)
def test_a_nonsensical_policy_is_refused_at_construction(kwargs):
    with pytest.raises(ValueError):
        RetentionPolicy(**kwargs)


# -- reaping an abandoned channel -------------------------------------------


def test_reaping_an_abandoned_channel_reclaims_its_bytes(store):
    identity = identity_for("abandoned")
    publish(store, identity, marker="forgotten")
    before = store.stats()
    assert before.total_bytes_apparent > 0

    report = store.reap(
        RetentionPolicy(max_age_seconds=DAY, max_channels=None),
        now=time.time() + 2 * DAY,
    )

    assert report.reclaimed_channels == 1
    assert report.aged_out == 1
    assert report.reclaimed_bytes_apparent == before.bytes_apparent
    assert store.load(identity) is None
    assert store.exists(identity) is False
    assert store.stats().total_bytes_apparent == 0
    # Reclaimed, not merely hidden: nothing is left waiting in the holding area.
    assert store.stats().reclaimable_entries == 0
    assert walk_files(store.base_folder)[1] == 0


def test_reaping_does_not_quarantine_what_it_reclaims(store):
    """Quarantine is for evidence of a fault; retention is not a fault."""
    identity = identity_for("abandoned-quiet")
    publish(store, identity)

    store.reap(now=time.time() + 400 * DAY)

    assert store.list_quarantined(identity) == []
    assert store.list_quarantine_entries() == []


def test_a_channel_inside_the_window_is_never_reaped(store):
    identity = identity_for("recent")
    publish(store, identity, marker="fresh")

    report = store.reap(RetentionPolicy(max_age_seconds=DAY, max_channels=None))

    assert report.reclaimed_channels == 0
    assert report.retained_channels == 1
    assert store.load(identity).context["marker"] == "fresh"


def test_a_channel_that_is_still_adoptable_is_never_reaped(store):
    """The live process holds it, so the store must be told and must comply."""
    live = identity_for("live-channel")
    dead = identity_for("dead-channel")
    publish(store, live, marker="in-use")
    publish(store, dead, marker="gone")

    report = store.reap(
        RetentionPolicy(max_age_seconds=DAY, max_channels=None),
        protected_channel_ids=[live.channel_id],
        now=time.time() + 90 * DAY,
    )

    assert report.protected_channels == 1
    assert report.reclaimed_channels == 1
    adopted = store.load_for_adoption(**live.scope())
    assert adopted is not None
    assert adopted.context["marker"] == "in-use"
    assert store.load_for_adoption(**dead.scope()) is None


def test_protection_survives_an_oversized_channel_id(store):
    """The protected set is raw ids; the reaper walks encoded names."""
    live = identity_for("z" * 400)
    publish(store, live)

    report = store.reap(
        protected_channel_ids=[live.channel_id], now=time.time() + 90 * DAY
    )

    assert report.protected_channels == 1
    assert report.reclaimed_channels == 0
    assert store.load(live) is not None


def test_capacity_reclaims_the_oldest_first_regardless_of_age(store):
    """The only bound that holds whatever the arrival rate does."""
    identities = [identity_for(f"cap-{index}") for index in range(5)]
    for identity in identities:
        publish(store, identity, marker=f"m-{identity.channel_id}")

    report = store.reap(
        RetentionPolicy(max_age_seconds=None, max_channels=2), now=time.time()
    )

    assert report.over_capacity == 3
    assert report.aged_out == 0
    survivors = {
        info.channel_id for info in store.list_channels() if info.committed
    }
    assert survivors == {"cap-3", "cap-4"}


def test_a_byte_cap_reclaims_until_the_namespace_fits(store):
    identities = [identity_for(f"bytes-{index}") for index in range(4)]
    for identity in identities:
        publish(store, identity)
    # Sized off the largest channel so the cap admits exactly two whichever two
    # survive; channels differ by a few bytes because a timestamp's decimal
    # expansion does.
    largest = max(info.bytes_apparent for info in store.list_channels())
    cap = 2 * largest + 1

    store.reap(
        RetentionPolicy(max_age_seconds=None, max_channels=None, max_bytes=cap)
    )

    assert store.stats().channels == 2
    assert store.stats().bytes_apparent <= cap


def test_a_byte_cap_of_zero_empties_the_namespace(store):
    for index in range(3):
        publish(store, identity_for(f"zero-{index}"))

    store.reap(
        RetentionPolicy(max_age_seconds=None, max_channels=None, max_bytes=0)
    )

    assert store.stats().total_bytes_apparent == 0


def test_the_capacity_bound_holds_across_repeated_unique_channels(store):
    """The motivating workload: a unique channel per request, none read again."""
    policy = RetentionPolicy(max_age_seconds=None, max_channels=3)
    sizes = []
    for index in range(30):
        publish(store, identity_for(f"unique-{index:03d}"))
        store.reap(policy)
        sizes.append(store.stats().total_bytes_apparent)

    assert store.stats().channels == 3
    one_channel = max(info.bytes_apparent for info in store.list_channels())

    # A plateau, not a slope. The samples are not bit-identical because a
    # timestamp's decimal expansion varies by a byte or two, so the claim is that
    # 20 further requests add nothing of record size — not that they add nothing.
    plateau = sizes[9:]
    assert max(plateau) - min(plateau) < one_channel
    assert max(plateau) < 4 * one_channel
    assert sizes[-1] <= sizes[9] + one_channel


def test_a_dry_run_reports_what_it_would_do_and_touches_nothing(store):
    identity = identity_for("dry")
    publish(store, identity)
    before = snapshot_tree(store.base_folder)

    report = store.reap(now=time.time() + 400 * DAY, dry_run=True)

    assert report.dry_run is True
    assert report.reclaimed_channels == 1
    assert report.reclaimed_bytes_apparent > 0
    assert snapshot_tree(store.base_folder) == before
    assert store.load(identity) is not None


def test_reaping_an_empty_store_is_a_no_op(store):
    report = store.reap()

    assert report.reclaimed_channels == 0
    assert report.scanned_channels == 0
    assert store.stats().total_bytes_apparent == 0


def test_reaping_is_scoped_to_one_namespace_when_asked(store):
    mine = identity_for("scoped", deployment_id="deploy-green")
    theirs = identity_for("scoped", deployment_id="deploy-blue")
    publish(store, mine)
    publish(store, theirs)

    store.reap(
        deployment_id="deploy-green",
        workflow_fingerprint="wf-abc123",
        now=time.time() + 400 * DAY,
    )

    assert store.load(mine) is None
    assert store.load(theirs) is not None


def test_an_uncommitted_channel_is_reclaimed_like_any_other(store):
    """A crashed first publish leaves bytes no reader can ever use."""
    identity = identity_for("half-born")
    publish(store, identity)
    os.remove(os.path.join(store.channel_directory(identity), "COMMIT"))
    assert store.stats().committed_channels == 0
    assert store.stats().bytes_apparent > 0

    store.reap(now=time.time() + 400 * DAY)

    assert store.stats().total_bytes_apparent == 0


# -- crash safety of reclamation --------------------------------------------


def test_a_reap_interrupted_before_the_flip_leaves_the_channel_whole(store):
    identity = identity_for("crash-before")
    publish(store, identity, marker="intact")

    store.reap(NO_RETENTION, now=time.time() + 400 * DAY)

    assert store.load(identity).context["marker"] == "intact"
    assert store.list_quarantined(identity) == []


def test_a_reap_interrupted_after_the_flip_reads_as_absent_not_damaged(store):
    identity = identity_for("crash-after-flip")
    publish(store, identity, marker="gone")
    debris = interrupt_flip(store, store.channel_directory(identity), identity)

    assert os.path.isdir(debris)
    assert store.load(identity) is None
    assert store.load_for_adoption(**identity.scope()) is None
    # Absent, not quarantined: nothing partial was ever visible to a reader.
    assert store.list_quarantined(identity) == []
    # And the bytes are still counted, so an interrupted pass cannot look like
    # reclaimed space that is in fact still occupied.
    assert store.stats().reclaimable_entries == 1
    assert store.stats().total_bytes_apparent > 0


def test_the_next_pass_finishes_an_interrupted_reap(store):
    identity = identity_for("crash-resume")
    publish(store, identity)
    interrupt_flip(store, store.channel_directory(identity), identity)

    report = store.reap(NO_RETENTION)

    assert report.swept_debris_entries == 1
    assert report.swept_debris_bytes > 0
    assert store.stats().total_bytes_apparent == 0
    assert walk_files(store.base_folder)[0] == 0


def test_partially_deleted_debris_is_still_swept(store):
    """A crash during the bulk delete, not just during the rename."""
    identity = identity_for("crash-mid-delete")
    publish(store, identity)
    debris = interrupt_flip(store, store.channel_directory(identity), identity)
    for directory, _subdirs, names in os.walk(debris):
        for name in names:
            if name.endswith("context.json"):
                os.remove(os.path.join(directory, name))

    store.reap(NO_RETENTION)

    assert store.stats().total_bytes_apparent == 0


def test_a_channel_republishes_cleanly_after_an_interrupted_reap(store):
    """Reclamation must not leave the channel id unusable."""
    identity = identity_for("crash-republish")
    publish(store, identity, marker="old")
    interrupt_flip(store, store.channel_directory(identity), identity)

    assert publish(store, identity, marker="new") == 1
    assert store.load(identity).context["marker"] == "new"


def test_reaping_survives_a_channel_it_cannot_reclaim(store, tmp_path):
    """One stuck channel must not abort the pass."""
    stuck = identity_for("stuck")
    ordinary = identity_for("ordinary")
    publish(store, stuck)
    publish(store, ordinary)

    # A directory replaced by a symlink is not a channel and cannot be renamed
    # away as one; the pass must report it and carry on.
    shutil.rmtree(store.channel_directory(stuck))
    elsewhere = tmp_path / "not-a-channel"
    elsewhere.mkdir()
    os.symlink(str(elsewhere), store.channel_directory(stuck))

    report = store.reap(now=time.time() + 400 * DAY)

    assert report.reclaimed_channels >= 1
    assert store.load(ordinary) is None


def test_a_channel_that_could_not_be_retired_is_not_counted_as_reclaimed(store):
    """`reclaimed` is confirmations, as PendingReapOutcome's is (fix-3lm).

    The two stores' reap reports share this field name across namespaces and an
    operator compares them to judge whether retention is working. Counting the
    selection here made the answer optimistic in exactly the case that matters:
    when something failed.
    """
    stuck = identity_for("stuck-count")
    publish(store, stuck)
    block_reclamation(store, stuck)

    report = store.reap(now=time.time() + 400 * DAY)

    assert report.aged_out == 1
    assert report.failures == 1
    assert report.reclaimed_channels == 0
    # The bytes are still occupied, so reporting them as freed would make an
    # unshrinking namespace look like one that shrank and refilled.
    assert report.reclaimed_bytes_apparent == 0
    assert report.reclaimed_keys == ()
    assert store.load(stuck) is not None
    assert store.stats().bytes_apparent > 0


# -- quarantine reclamation -------------------------------------------------


def test_quarantined_records_are_reclaimable(store):
    identity = identity_for("q-reclaim")
    publish(store, identity)
    store.quarantine(identity, QuarantineReason.OPERATOR_REQUEST)
    assert store.stats().quarantined_entries == 1
    assert store.stats().quarantined_bytes_apparent > 0

    report = store.reap(
        RetentionPolicy(quarantine_max_age_seconds=DAY, quarantine_max_entries=None),
        now=time.time() + 2 * DAY,
    )

    assert report.reclaimed_quarantine_entries == 1
    assert report.reclaimed_quarantine_bytes > 0
    assert store.list_quarantined(identity) == []
    assert store.stats().total_bytes_apparent == 0


def test_recent_evidence_is_kept_longer_than_live_state(store):
    identity = identity_for("q-recent")
    publish(store, identity)
    store.quarantine(identity, QuarantineReason.DIGEST_MISMATCH)

    report = store.reap(
        RetentionPolicy(
            max_age_seconds=DAY, quarantine_max_age_seconds=7 * DAY
        ),
        now=time.time() + 2 * DAY,
    )

    assert report.reclaimed_quarantine_entries == 0
    assert len(store.list_quarantined(identity)) == 1


def test_a_quarantine_count_cap_keeps_the_newest(store):
    """Preserving forever is the same leak under a different directory name."""
    for index in range(6):
        identity = identity_for(f"q-many-{index}")
        publish(store, identity)
        store.quarantine(identity, QuarantineReason.OPERATOR_REQUEST)
    assert store.stats().quarantined_entries == 6

    store.reap(
        RetentionPolicy(
            max_age_seconds=None,
            max_channels=None,
            quarantine_max_age_seconds=None,
            quarantine_max_entries=2,
        )
    )

    assert store.stats().quarantined_entries == 2


def test_reclaiming_evidence_does_not_leave_an_empty_directory_per_channel(store):
    """One inode per channel that ever failed is unbounded growth in inodes."""
    identity = identity_for("q-inode")
    publish(store, identity)
    store.quarantine(identity, QuarantineReason.OPERATOR_REQUEST)

    store.reap(now=time.time() + 400 * DAY)

    assert not os.path.isdir(store.quarantine_directory(identity))


def test_a_quarantine_entry_reports_its_reason_and_age(store):
    identity = identity_for("q-inspect")
    publish(store, identity)
    store.quarantine(identity, QuarantineReason.CHANNEL_ID_MISMATCH)

    entries = store.list_quarantine_entries()

    assert len(entries) == 1
    assert entries[0].reason == "channel_id_mismatch"
    assert entries[0].channel_key == identity.channel_key
    assert entries[0].bytes_apparent > 0
    assert entries[0].age_seconds() < 60


# -- delete, reset, and generation-safe channel reuse -----------------------


def test_delete_removes_the_channel_whole(store):
    identity = identity_for("del")
    publish(store, identity)

    assert store.delete(identity) is True
    assert store.load(identity) is None
    assert store.exists(identity) is False
    assert store.list_quarantined(identity) == []
    assert store.stats().total_bytes_apparent == 0


def test_deleting_a_channel_that_is_not_there_says_so(store):
    assert store.delete(identity_for("never-existed")) is False


def test_delete_never_passes_through_a_damaged_state(store):
    """Removing files in place would quarantine what was asked to be deleted."""
    identity = identity_for("del-atomic")
    publish(store, identity)
    publish(store, identity, marker="two")

    store.delete(identity)

    assert store.load(identity) is None
    assert store.load_for_adoption(**identity.scope()) is None
    assert store.list_quarantined(identity) == []


def test_a_reused_channel_id_after_delete_cannot_adopt_the_old_lineage(store):
    old = identity_for("recycled", session_incarnation="inc-old")
    publish(store, old, marker="old-secret")
    store.delete(old)

    assert store.load_for_adoption(**old.scope()) is None

    new = identity_for("recycled", session_incarnation="inc-new")
    assert publish(store, new, marker="new-state") == 1
    adopted = store.load_for_adoption(**new.scope())
    assert adopted.identity.session_incarnation == "inc-new"
    assert adopted.context["marker"] == "new-state"
    # And the refusal that protects a live owner does not fire against a
    # channel id whose previous owner was deleted.
    assert store.load(new) is not None


def test_a_reused_channel_id_after_delete_cannot_collide_with_old_generations(store):
    old = identity_for("recycled-gen", session_incarnation="inc-old")
    for index in range(4):
        publish(store, old, marker=f"old-{index}")
    store.delete(old)

    new = identity_for("recycled-gen", session_incarnation="inc-new")
    assert publish(store, new, marker="new-1") == 1
    assert publish(store, new, marker="new-2") == 2
    # Nothing from the retired lineage is left for the new numbering to land on.
    generations = sorted(
        name
        for name in os.listdir(
            os.path.join(store.channel_directory(new), "gen")
        )
        if name.isdigit()
    )
    assert generations == ["1", "2"]
    assert store.load(new).context["marker"] == "new-2"


def test_reset_forgets_the_state_but_keeps_the_numbering(store):
    identity = identity_for("reset-me")
    for index in range(3):
        publish(store, identity, marker=f"m-{index}")

    assert store.reset(identity) == 3
    assert store.load(identity) is None
    assert store.exists(identity) is False
    assert store.list_quarantined(identity) == []

    fresh = identity_for("reset-me", session_incarnation="inc-after-reset")
    # Strictly greater than every generation the retired lineage ever used, so a
    # generation number identifies a lineage rather than a position.
    assert publish(store, fresh, marker="after") == 4
    assert store.load(fresh).generation == 4


def test_reset_reclaims_the_bytes_it_forgets(store):
    identity = identity_for("reset-bytes")
    publish(store, identity)
    before = store.stats().bytes_apparent

    store.reset(identity)

    after = store.stats()
    assert after.total_bytes_apparent < before
    assert after.generations == 0
    assert after.committed_channels == 0


def test_a_reset_channel_cannot_adopt_its_old_lineage(store):
    identity = identity_for("reset-adopt", session_incarnation="inc-before")
    publish(store, identity, marker="secret")

    store.reset(identity)

    assert store.load_for_adoption(**identity.scope()) is None


def test_repeated_resets_keep_raising_the_floor(store):
    identity = identity_for("reset-twice")
    publish(store, identity, marker="one")
    assert store.reset(identity) == 1

    assert publish(store, identity, marker="two") == 2
    assert store.reset(identity) == 2
    assert publish(store, identity, marker="three") == 3


def test_resetting_a_channel_that_is_not_there_says_so(store):
    assert store.reset(identity_for("reset-absent")) == 0


def test_delete_after_reset_takes_the_floor_with_it(store):
    """Delete means "as though it never existed", floor included."""
    identity = identity_for("reset-then-delete")
    publish(store, identity)
    publish(store, identity, marker="two")
    store.reset(identity)

    store.delete(identity)

    assert publish(store, identity, marker="brand-new") == 1
    assert store.stats().channels == 1


def test_a_reset_interrupted_after_the_floor_leaves_the_record_readable(store):
    identity = identity_for("reset-crash-floor")
    publish(store, identity, marker="still-here")
    publish(store, identity, marker="still-here-2")

    write_floor(store, identity, 2)

    assert store.load(identity).context["marker"] == "still-here-2"
    assert store.list_quarantined(identity) == []
    # And the floor cannot lower a number, only raise it.
    assert publish(store, identity, marker="next") == 3


def test_a_reset_interrupted_after_the_commit_unlink_reads_as_absent(store):
    identity = identity_for("reset-crash-commit")
    publish(store, identity)
    publish(store, identity, marker="two")

    write_floor(store, identity, 2)
    os.remove(os.path.join(store.channel_directory(identity), "COMMIT"))

    assert store.load(identity) is None
    assert store.load_for_adoption(**identity.scope()) is None
    assert store.list_quarantined(identity) == []
    # Generation directories are still on disk, and the next lineage must clear
    # both them and the floor.
    assert publish(store, identity, marker="after-crash") == 3


def test_a_reset_interrupted_before_the_bulk_delete_leaves_a_readable_store(store):
    identity = identity_for("reset-crash-flip")
    publish(store, identity)
    publish(store, identity, marker="two")
    write_floor(store, identity, 2)
    os.remove(os.path.join(store.channel_directory(identity), "COMMIT"))
    interrupt_flip(
        store,
        os.path.join(store.channel_directory(identity), "gen"),
        identity,
    )

    assert store.load(identity) is None
    assert publish(store, identity, marker="after") == 3
    assert store.load(identity).context["marker"] == "after"

    report = store.reap(NO_RETENTION)
    assert report.swept_debris_entries == 1
    assert store.load(identity).context["marker"] == "after"


def test_reset_leaves_other_channels_and_evidence_alone(store):
    victim = identity_for("reset-blast-radius")
    neighbour = identity_for("reset-neighbour")
    publish(store, victim)
    publish(store, neighbour, marker="untouched")
    store.quarantine(victim, QuarantineReason.OPERATOR_REQUEST)
    preserved = store.list_quarantined(victim)
    publish(store, victim)

    store.reset(victim)

    assert store.load(neighbour).context["marker"] == "untouched"
    assert store.list_quarantined(victim) == preserved
    assert os.path.isdir(preserved[0])


# -- inspect ----------------------------------------------------------------


def test_inspect_reports_the_record_it_finds(store):
    identity = identity_for("inspected")
    publish(store, identity, marker="visible")
    publish(store, identity, marker="visible-2")

    info = store.inspect(identity)

    assert info.channel_id == identity.channel_id
    assert info.deployment_id == identity.deployment_id
    assert info.workflow_fingerprint == identity.workflow_fingerprint
    assert info.session_incarnation == identity.session_incarnation
    assert info.generation == 2
    assert info.committed is True
    assert info.generations_on_disk == 2
    assert info.bytes_apparent > 0
    assert info.age_seconds() < 60
    assert info.identity() == identity


def test_inspect_is_none_for_a_channel_that_was_never_written(store):
    assert store.inspect(identity_for("unwritten")) is None


def test_inspect_never_quarantines_what_it_looks_at(store):
    """An operator asking what is wrong must not change it."""
    identity = identity_for("inspect-broken")
    publish(store, identity)
    commit_path = os.path.join(store.channel_directory(identity), "COMMIT")
    with open(commit_path, "w", encoding="utf-8") as handle:
        handle.write("{not json")

    info = store.inspect(identity)

    assert info is not None
    assert info.channel_id is None
    assert info.identity() is None
    assert info.bytes_apparent > 0
    assert store.list_quarantined(identity) == []
    assert store.exists(identity) is True


def test_inspect_still_measures_a_channel_whose_id_is_not_in_its_path(store):
    """An oversized id is a hash tail on disk; only the record has the raw form."""
    identity = identity_for("y" * 400)
    publish(store, identity)

    info = store.inspect(identity)

    assert info.channel_key != identity.channel_id
    assert info.channel_id == identity.channel_id


def test_listing_channels_orders_oldest_activity_first(store):
    for index in range(4):
        publish(store, identity_for(f"order-{index}"))

    listed = store.list_channels()

    assert [info.channel_id for info in listed] == [
        "order-0",
        "order-1",
        "order-2",
        "order-3",
    ]


def test_listing_channels_can_be_scoped_to_one_namespace(store):
    publish(store, identity_for("ns", deployment_id="deploy-green"))
    publish(store, identity_for("ns", deployment_id="deploy-blue"))

    scoped = store.list_channels(
        deployment_id="deploy-green", workflow_fingerprint="wf-abc123"
    )

    assert [info.deployment_id for info in scoped] == ["deploy-green"]
    assert len(store.list_channels()) == 2


# -- measurement ------------------------------------------------------------


def test_the_measurement_matches_the_files_on_disk(store):
    for index in range(3):
        publish(store, identity_for(f"measure-{index}"))

    stats = store.stats()
    files, total = walk_files(store.base_folder)

    assert stats.channels == 3
    assert stats.committed_channels == 3
    assert stats.generations == 3
    assert stats.total_files == files
    assert stats.total_bytes_apparent == total
    assert stats.bytes_physical > 0
    assert stats.bytes_physical % 512 == 0


def test_the_measurement_accounts_for_every_bucket(store):
    live = identity_for("bucket-live")
    quarantined = identity_for("bucket-q")
    debris = identity_for("bucket-debris")
    publish(store, live)
    publish(store, quarantined)
    store.quarantine(quarantined, QuarantineReason.OPERATOR_REQUEST)
    publish(store, debris)
    interrupt_flip(store, store.channel_directory(debris), debris)

    stats = store.stats()

    assert stats.channels == 1
    assert stats.quarantined_entries == 1
    assert stats.reclaimable_entries == 1
    assert stats.bytes_apparent > 0
    assert stats.quarantined_bytes_apparent > 0
    assert stats.reclaimable_bytes_apparent > 0
    assert stats.total_bytes_apparent == walk_files(store.base_folder)[1]
    assert stats.describe().startswith("checkpoint namespace")


def test_the_measurement_can_be_scoped_to_one_namespace(store):
    green = identity_for("m-scope", deployment_id="deploy-green")
    blue = identity_for("m-scope", deployment_id="deploy-blue")
    publish(store, green)
    publish(store, blue)

    scoped = store.stats(
        deployment_id="deploy-green", workflow_fingerprint="wf-abc123"
    )

    assert scoped.namespaces == 1
    assert scoped.channels == 1
    assert scoped.total_bytes_apparent < store.stats().total_bytes_apparent
    assert store.stats().namespaces == 2


def test_the_measurement_tracks_a_growing_and_then_reaped_namespace(store):
    """What the soak harness reads: a number that must stop going up."""
    assert store.stats().total_bytes_apparent == 0
    growing = []
    for index in range(4):
        publish(store, identity_for(f"soak-{index}"))
        growing.append(store.stats().total_bytes_apparent)

    assert growing == sorted(growing)
    assert growing[0] < growing[-1]

    store.reap(now=time.time() + 400 * DAY)

    assert store.stats().total_bytes_apparent == 0


def test_a_measurement_of_an_empty_store_is_zero(store):
    stats = store.stats()

    assert stats.channels == 0
    assert stats.total_files == 0
    assert stats.total_bytes_apparent == 0
    assert stats.total_bytes_physical == 0


def test_lifecycle_files_are_as_private_as_records(store):
    identity = identity_for("lifecycle-modes")
    publish(store, identity)
    publish(store, identity, marker="two")
    store.reset(identity)
    publish(store, identity, marker="three")
    store.quarantine(identity, QuarantineReason.OPERATOR_REQUEST)
    publish(store, identity, marker="four")
    store.reap(NO_RETENTION)

    for directory, _subdirs, files in os.walk(store.base_folder):
        assert stat.S_IMODE(os.lstat(directory).st_mode) == 0o700, directory
        for name in files:
            path = os.path.join(directory, name)
            assert stat.S_IMODE(os.lstat(path).st_mode) == 0o600, path


# -- ordering, pinned by stopping mid-operation -----------------------------


class InterruptedStore(ChannelCheckpointStore):
    """A store that stops dead at a named boundary, the way a killed process does.

    The crash-schedule tests above reproduce *states* on disk; these reproduce
    the *sequence*. Both are needed, because a store could respond correctly to
    every hand-built partial state while still writing them in an order that
    produces a different one. `KeyboardInterrupt` is used deliberately: nothing
    in the module catches a `BaseException`, so the stop is a stop.
    """

    def __init__(self, base_folder: str, *, stop_after: str):
        super().__init__(base_folder)
        self._stop_after = stop_after

    def _write_floor(self, identity, floor):
        super()._write_floor(identity, floor)
        if self._stop_after == "floor":
            raise KeyboardInterrupt("interrupted after recording the floor")

    def _retire_directory(self, path, reclaim_root):
        if self._stop_after == "unlink":
            raise KeyboardInterrupt("interrupted before the flip")
        return super()._retire_directory(path, reclaim_root)


def test_a_reset_stopped_at_the_floor_leaves_the_record_loadable(base):
    """The floor must be written before anything becomes invisible."""
    identity = identity_for("stop-at-floor")
    writer = ChannelCheckpointStore(base)
    publish(writer, identity, marker="one")
    publish(writer, identity, marker="two")

    with pytest.raises(KeyboardInterrupt):
        InterruptedStore(base, stop_after="floor").reset(identity)

    reader = ChannelCheckpointStore(base)
    assert reader.load(identity).context["marker"] == "two"
    assert reader.list_quarantined(identity) == []
    assert publish(reader, identity, marker="three") == 3


def test_a_reset_stopped_before_the_flip_reads_as_absent(base):
    """COMMIT must be gone before the generations are, or a reader sees damage."""
    identity = identity_for("stop-before-flip")
    writer = ChannelCheckpointStore(base)
    publish(writer, identity, marker="one")
    publish(writer, identity, marker="two")

    with pytest.raises(KeyboardInterrupt):
        InterruptedStore(base, stop_after="unlink").reset(identity)

    reader = ChannelCheckpointStore(base)
    assert reader.load(identity) is None
    assert reader.load_for_adoption(**identity.scope()) is None
    assert reader.list_quarantined(identity) == []
    assert publish(reader, identity, marker="three") == 3


def test_a_delete_stopped_before_the_flip_changes_nothing(base):
    """Delete is one rename, so an interruption before it is a no-op."""
    identity = identity_for("stop-delete")
    writer = ChannelCheckpointStore(base)
    publish(writer, identity, marker="intact")
    before = snapshot_records(base)

    with pytest.raises(KeyboardInterrupt):
        InterruptedStore(base, stop_after="unlink").delete(identity)

    reader = ChannelCheckpointStore(base)
    assert reader.load(identity).context["marker"] == "intact"
    assert snapshot_records(base) == before


# -- the liveness marker, isolated from every other age signal --------------


def backdate(store, identity, seconds_ago: float, *, keep_marker: bool) -> None:
    """Make a channel look old in every way the store can tell.

    `published_at` lives in COMMIT and in no digest, which is why it can be
    rewritten here without the record failing closed — and is exactly why it is
    kept there rather than inside a part.
    """
    directory = store.channel_directory(identity)
    old = time.time() - seconds_ago

    rewrite_part(os.path.join(directory, "COMMIT"), published_at=old)

    marker_path = os.path.join(directory, "ACCESS")
    if keep_marker:
        rewrite_part(marker_path, last_publish_at=old)
    else:
        os.remove(marker_path)

    for path in (directory, os.path.join(directory, "gen")):
        os.utime(path, (old, old))


def test_a_channel_old_by_every_signal_is_reaped(store):
    identity = identity_for("aged-all")
    publish(store, identity)
    backdate(store, identity, 10 * DAY, keep_marker=True)

    report = store.reap(RetentionPolicy(max_age_seconds=DAY, max_channels=None))

    assert report.aged_out == 1
    assert store.load(identity) is None


def test_a_fresh_liveness_marker_alone_keeps_a_channel(store):
    """The whole reason the marker exists: content age is not use age."""
    identity = identity_for("aged-but-used")
    publish(store, identity)
    backdate(store, identity, 10 * DAY, keep_marker=True)
    # A digest-stable republish: the several-hundred-kilobyte record is correctly
    # not rewritten, so only the marker records that the channel is in use.
    publish(store, identity)

    directory = store.channel_directory(identity)
    with open(os.path.join(directory, "COMMIT"), encoding="utf-8") as handle:
        assert json.load(handle)["published_at"] < time.time() - 5 * DAY
    # Directory mtime is stripped too, so the marker is the only fresh signal left.
    os.utime(directory, (time.time() - 10 * DAY,) * 2)

    report = store.reap(RetentionPolicy(max_age_seconds=DAY, max_channels=None))

    assert report.aged_out == 0
    assert store.load(identity) is not None


def test_a_channel_with_no_marker_still_has_a_measurable_age(store):
    """A missing marker must not make a channel immortal."""
    identity = identity_for("aged-no-marker")
    publish(store, identity)
    backdate(store, identity, 10 * DAY, keep_marker=False)

    report = store.reap(RetentionPolicy(max_age_seconds=DAY, max_channels=None))

    assert report.aged_out == 1
    assert store.load(identity) is None


@pytest.mark.parametrize("planted", ["not-a-number", None, float("inf")])
def test_a_nonsense_timestamp_does_not_make_a_channel_immortal(store, planted):
    """NaN and infinity compare false against every threshold."""
    identity = identity_for(f"aged-nonsense-{planted}")
    publish(store, identity)
    backdate(store, identity, 10 * DAY, keep_marker=False)

    marker_path = os.path.join(store.channel_directory(identity), "ACCESS")
    with open(marker_path, "w", encoding="utf-8") as handle:
        json.dump({"last_publish_at": planted}, handle)
    os.utime(
        store.channel_directory(identity), (time.time() - 10 * DAY,) * 2
    )

    report = store.reap(RetentionPolicy(max_age_seconds=DAY, max_channels=None))

    assert report.aged_out == 1


def test_a_corrupt_marker_never_makes_a_record_unreadable(store):
    """It is retention metadata, not state, and no digest covers it."""
    identity = identity_for("marker-corrupt")
    publish(store, identity, marker="fine")
    with open(
        os.path.join(store.channel_directory(identity), "ACCESS"), "w",
        encoding="utf-8",
    ) as handle:
        handle.write("{not json at all")

    assert store.load(identity).context["marker"] == "fine"
    assert store.load_for_adoption(**identity.scope()) is not None
    assert store.list_quarantined(identity) == []
    assert store.inspect(identity).last_seen_at is None
