"""Retention for abandoned suspended sessions (fix-6b4).

A suspended session waits on a user who may never return, and nothing else
reclaims it: /cancel_pending needs a client that has already left, and
completion needs an answer never given. Since v2.28.0 removed the pin such a
session can also be evicted, which makes the blob the *only* copy of that
conversation -- so every test here is as much about what the reaper must not
touch as about what it must.
"""

from __future__ import annotations

import json
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from fastworkflow.session_state_store import (
    SAVED_AT_KEY,
    DiskSessionStateStore,
    PendingRetentionPolicy,
)

DAY = 86_400.0


@pytest.fixture
def store(tmp_path) -> DiskSessionStateStore:
    return DiskSessionStateStore(str(tmp_path / "pending"))


def _save(store: DiskSessionStateStore, channel_id: str, **extra) -> None:
    store.save(channel_id, {"channel_id": channel_id, "awaiting_user": True, **extra})


def _age(store: DiskSessionStateStore, channel_id: str, seconds: float) -> None:
    """Backdate a blob's stored save time."""
    path = store._json_path(channel_id)
    blob = json.loads(Path(path).read_text())
    blob[SAVED_AT_KEY] = time.time() - seconds
    Path(path).write_text(json.dumps(blob))


def test_save_stamps_a_time_without_touching_the_session_schema(store):
    """When a blob was written is a fact about the store, not the session.

    Putting it in SCHEMA_VERSION would make every retention change a schema
    migration, and would make v2 blobs written one commit apart differ.
    """
    _save(store, "c1")
    blob = store.load("c1")

    assert blob[SAVED_AT_KEY] <= time.time()
    assert blob["awaiting_user"] is True
    assert "schema_version" not in blob, "the fixture writes no schema; save adds none"


def test_abandoned_state_is_reclaimed_after_the_age_window(store):
    _save(store, "abandoned")
    _age(store, "abandoned", 8 * DAY)

    outcome = store.reap(PendingRetentionPolicy(max_age_seconds=7 * DAY))

    assert outcome.reclaimed == 1
    assert not store.exists("abandoned")


def test_recent_state_is_left_alone(store):
    _save(store, "recent")
    _age(store, "recent", 1 * DAY)

    outcome = store.reap(PendingRetentionPolicy(max_age_seconds=7 * DAY))

    assert outcome.reclaimed == 0
    assert store.exists("recent")


def test_a_live_channel_is_never_reclaimed_however_old(store):
    """The protection guard is the whole safety story.

    Since the pin was removed, an evicted-but-live session's blob is the only
    copy of its conversation. Age alone must not be enough to delete it.
    """
    _save(store, "live")
    _age(store, "live", 400 * DAY)

    outcome = store.reap(
        PendingRetentionPolicy(max_age_seconds=1.0),
        protected_channel_ids={"live"},
    )

    assert outcome.reclaimed == 0
    assert outcome.protected == 1
    assert store.exists("live"), "a live session's only copy was deleted"


def test_the_count_cap_reclaims_oldest_first(store):
    for i in range(5):
        _save(store, f"c{i}")
        _age(store, f"c{i}", (10 - i) * DAY)

    outcome = store.reap(
        PendingRetentionPolicy(max_age_seconds=None, max_entries=2)
    )

    assert outcome.reclaimed == 3
    # c0 is oldest (10 days), c4 newest (6 days).
    assert not store.exists("c0")
    assert not store.exists("c1")
    assert not store.exists("c2")
    assert store.exists("c3")
    assert store.exists("c4")


def test_the_count_cap_counts_protected_entries_but_never_deletes_them(store):
    """Otherwise a process holding many live sessions silently raises the cap.

    The cap is a bound on stored bytes, and a protected entry occupies bytes
    just the same. It must contribute to the count while being ineligible to
    satisfy it.
    """
    for i in range(4):
        _save(store, f"c{i}")
        _age(store, f"c{i}", (10 - i) * DAY)

    outcome = store.reap(
        PendingRetentionPolicy(max_age_seconds=None, max_entries=2),
        protected_channel_ids={"c0", "c1"},
    )

    # 4 stored, cap 2, so 2 must go -- and only c2/c3 are eligible.
    assert outcome.reclaimed == 2
    assert store.exists("c0") and store.exists("c1")
    assert not store.exists("c2") and not store.exists("c3")


def test_a_blob_written_before_save_times_falls_back_to_mtime(store, tmp_path):
    """The transitional population must not become immortal.

    Blobs written before the store stamped a save time have no _saved_at. On
    disk, mtime is the same fact, so they age out normally instead of being
    kept forever for want of a field.
    """
    _save(store, "legacy")
    path = Path(store._json_path("legacy"))
    blob = json.loads(path.read_text())
    del blob[SAVED_AT_KEY]
    path.write_text(json.dumps(blob))

    old = time.time() - 30 * DAY
    import os

    os.utime(path, (old, old))

    outcome = store.reap(PendingRetentionPolicy(max_age_seconds=7 * DAY))

    assert outcome.unreadable == 0, "mtime should have supplied the age"
    assert outcome.reclaimed == 1
    assert not store.exists("legacy")


def test_an_unreadable_blob_is_reported_and_left_in_place(store, tmp_path):
    """Reclaiming on a guess is how a reaper eats live state."""
    corrupt = Path(store.base_folder) / "broken_pending.json"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{not json at all")

    outcome = store.reap(PendingRetentionPolicy(max_age_seconds=1.0))

    assert outcome.unreadable == 1
    assert outcome.reclaimed == 0
    assert corrupt.exists()


def test_dry_run_reports_without_deleting(store):
    _save(store, "abandoned")
    _age(store, "abandoned", 30 * DAY)

    outcome = store.reap(
        PendingRetentionPolicy(max_age_seconds=7 * DAY), dry_run=True
    )

    assert outcome.reclaimed == 1
    assert store.exists("abandoned"), "dry_run deleted something"


def test_channel_id_comes_from_the_blob_not_the_filename(store):
    """The disk key is not reversible, so the filename cannot identify a channel.

    _json_path used to fold separators, so 'a/b' and 'a_b' produced the same
    name; fix-7hn made the mapping injective, and it is still not reversible
    because an oversized id carries a hash tail. The reaper compares against
    protected ids, and comparing a mangled name would fail to protect a live
    channel.
    """
    _save(store, "tenant/user-1")

    entries = list(store.iter_entries())

    assert [entry.channel_id for entry in entries] == ["tenant/user-1"]
    assert entries[0].saved_at == pytest.approx(time.time(), abs=10)
    # The name on disk is not that id, which is why the entry carries it too:
    # `reap` removes this path rather than one derived from the id (fix-xm1).
    assert entries[0].storage_key == store._json_path("tenant/user-1")
    assert "tenant/user-1" not in entries[0].storage_key


def test_policy_rejects_nonsense_values():
    with pytest.raises(ValueError):
        PendingRetentionPolicy(max_age_seconds=0)
    with pytest.raises(ValueError):
        PendingRetentionPolicy(max_age_seconds=-1)
    with pytest.raises(ValueError):
        PendingRetentionPolicy(max_entries=-1)


def test_retention_off_keeps_everything(store):
    _save(store, "ancient")
    _age(store, "ancient", 3650 * DAY)

    outcome = store.reap(
        PendingRetentionPolicy(max_age_seconds=None, max_entries=None)
    )

    assert outcome.reclaimed == 0
    assert store.exists("ancient")


# ---------------------------------------------------------------------------
# The reaper's own reporting must name fields that exist
# ---------------------------------------------------------------------------


def test_the_periodic_reaper_reads_only_fields_its_outcomes_actually_have():
    """A log line that names a missing field reports silence, not an error.

    The checkpoint branch read `outcome.channels_reclaimed` through a
    getattr default; the field is `reclaimed_channels`, so every pass since
    retention shipped logged nothing and looked like "nothing was reclaimed".
    A plain attribute access would have raised on the first run.

    Asserted structurally rather than by driving the loop, because the defect
    was a NAME, and a test that drove the loop with a real store would have
    passed just as happily against the wrong one.
    """
    import ast
    from dataclasses import fields as dataclass_fields
    from pathlib import Path

    from fastworkflow.checkpoint_store import ReapReport
    from fastworkflow.session_state_store import PendingReapOutcome

    known = {f.name for f in dataclass_fields(ReapReport)}
    known |= {f.name for f in dataclass_fields(PendingReapOutcome)}

    # Parsed from the file rather than imported: importing __main__ runs its
    # argparse and exits the interpreter.
    server_path = (
        Path(__file__).parent.parent
        / "fastworkflow" / "run_fastapi_mcp" / "__main__.py"
    )
    tree = ast.parse(server_path.read_text(encoding="utf-8"))

    # Both spellings, because the defect used the second one and a getattr
    # default is the only form that can hide a bad name at runtime. Checking
    # only direct attribute access would miss the very bug this pins.
    outcome_names = {"outcome", "pending"}
    read = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in outcome_names
    }
    read |= {
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in outcome_names
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }

    assert read, "found no reap-outcome attribute reads to check"
    unknown = read - known
    assert not unknown, (
        f"the reaper reads fields no reap outcome defines: {sorted(unknown)}. "
        f"Known fields: {sorted(known)}"
    )
