"""The reaper removes the blob it enumerated, and counts only what it removed (fix-xm1).

`iter_entries` reads `channel_id` from *inside* each blob, deliberately: the
storage key is not reversible. `reap` then used to call `clear(channel_id)`,
which derives the path back *from that id* -- so a blob sitting at any other
name was never the blob that got removed. `clear` unlinked nothing, `reclaimed`
was incremented regardless, and the blob was enumerated forever while the metric
said retention was working.

Such a blob does not come from this store: it was hand-copied, restored from a
backup under a different name, or written by an older tool. That is exactly why
the tests below write files by hand -- real files, at real names, in a real
directory, because a name is a property of the filesystem and the whole defect
lives in the mapping between a name and a blob's opinion of itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fastworkflow.session_state_store import (
    PENDING_SUFFIX,
    SAVED_AT_KEY,
    DiskSessionStateStore,
    PendingRetentionPolicy,
    RedisSessionStateStore,
    SessionStateStore,
)

DAY = 86_400.0


@pytest.fixture
def store(tmp_path) -> DiskSessionStateStore:
    return DiskSessionStateStore(str(tmp_path / "pending"))


def _blobs_on_disk(store: DiskSessionStateStore) -> set[str]:
    return {path.name for path in Path(store.base_folder).iterdir()}


def _write_stray(
    store: DiskSessionStateStore,
    filename: str,
    channel_id: str,
    *,
    age_days: float,
) -> Path:
    """Write a blob at *filename* claiming *channel_id*, as a copy or restore would.

    The suffix is kept because that is what `iter_entries` scans for; everything
    before it is a name this store would never have derived for that id.
    """
    path = Path(store.base_folder) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "channel_id": channel_id,
                "awaiting_user": True,
                SAVED_AT_KEY: time.time() - age_days * DAY,
            }
        )
    )
    return path


def test_a_blob_whose_id_does_not_derive_its_own_name_is_actually_removed(store):
    """The defect itself: `clear(channel_id)` could not reach this file.

    The id inside the blob derives two names -- the current one and the legacy
    one -- and this blob is at neither, so the old reaper removed nothing, said
    it had reclaimed one, and enumerated the same blob on every pass thereafter.
    """
    channel_id = "tenant/user-1"
    stray = _write_stray(
        store, f"restored-from-backup{PENDING_SUFFIX}", channel_id, age_days=400
    )

    assert not Path(store._json_path(channel_id)).exists()
    assert not Path(store._legacy_json_path(channel_id)).exists()
    assert stray.name not in (
        Path(store._json_path(channel_id)).name,
        Path(store._legacy_json_path(channel_id)).name,
    ), "the fixture no longer reproduces the mismatch it is testing"

    outcome = store.reap(PendingRetentionPolicy(max_age_seconds=7 * DAY))

    assert not stray.exists(), "the blob is immortal: enumerated forever, never removed"
    assert outcome.reclaimed == 1
    assert outcome.failures == 0
    assert list(store.iter_entries()) == [], "a reaped namespace still enumerates it"


def test_reclaimed_equals_the_number_of_blobs_that_left_the_store(store):
    """`reclaimed` is a claim about the namespace shrinking, so it must be one.

    Two blobs, one claimed channel: a hand-copied duplicate alongside the real
    thing. Selecting by id put that id in the doomed list twice, `clear` removed
    at most the derived name once, and the count said two.
    """
    channel_id = "tenant/user-1"
    store.save(channel_id, {"channel_id": channel_id, "awaiting_user": True})
    real = Path(store._json_path(channel_id))
    blob = json.loads(real.read_text())
    blob[SAVED_AT_KEY] = time.time() - 400 * DAY
    real.write_text(json.dumps(blob))

    duplicate = _write_stray(
        store, f"hand-copy{PENDING_SUFFIX}", channel_id, age_days=401
    )

    before = _blobs_on_disk(store)
    assert len(before) == 2

    outcome = store.reap(PendingRetentionPolicy(max_age_seconds=7 * DAY))

    after = _blobs_on_disk(store)
    assert outcome.reclaimed == len(before) - len(after)
    assert outcome.reclaimed == 2
    assert not real.exists() and not duplicate.exists()


def test_a_blob_that_cannot_be_removed_is_reported_rather_than_counted(store):
    """A removal that was refused is not a reclamation, and must not end the pass.

    A read-only folder is the cheapest real version of "this will never go away"
    -- the blob is readable, so it is enumerated and selected, and the unlink
    fails. `reclaimed` must stay honest and `failures` must say what happened.
    """
    store.save("abandoned", {"channel_id": "abandoned", "awaiting_user": True})
    path = Path(store._json_path("abandoned"))
    blob = json.loads(path.read_text())
    blob[SAVED_AT_KEY] = time.time() - 400 * DAY
    path.write_text(json.dumps(blob))

    folder = Path(store.base_folder)
    folder.chmod(0o500)
    try:
        outcome = store.reap(PendingRetentionPolicy(max_age_seconds=7 * DAY))
    finally:
        folder.chmod(0o700)

    assert outcome.reclaimed == 0, "counted a blob it did not remove"
    assert outcome.failures == 1
    assert outcome.scanned == 1
    assert store.exists("abandoned")


def test_a_stray_blob_claiming_a_protected_channel_is_left_alone(store):
    """Removal moved to the storage key; selection did not.

    `protected_channel_ids` is expressed in channel ids, and the id inside the
    blob is the only thing that can be compared against it. A copy claiming a
    live channel is therefore protected like the original -- the conservative
    direction, since a protected blob may be the only copy of that conversation.
    """
    stray = _write_stray(
        store, f"copy-of-live{PENDING_SUFFIX}", "tenant/live-1", age_days=400
    )

    outcome = store.reap(
        PendingRetentionPolicy(max_age_seconds=1.0),
        protected_channel_ids={"tenant/live-1"},
    )

    assert outcome.reclaimed == 0
    assert outcome.protected == 1
    assert stray.exists()


def test_the_count_cap_removes_each_selected_blob_once(store):
    """The cap's de-duplication is per blob, because the cap counts blobs.

    De-duplicating by channel id would let one channel's two blobs satisfy the
    cap once and leave the namespace over it.
    """
    real = "tenant/user-1"
    store.save(real, {"channel_id": real, "awaiting_user": True})
    _write_stray(store, f"copy-a{PENDING_SUFFIX}", real, age_days=9)
    _write_stray(store, f"copy-b{PENDING_SUFFIX}", real, age_days=10)

    outcome = store.reap(
        PendingRetentionPolicy(max_age_seconds=None, max_entries=1)
    )

    assert outcome.scanned == 3
    assert outcome.reclaimed == 2
    assert len(_blobs_on_disk(store)) == 1
    assert store.exists(real), "the newest blob is the one that survives"


def test_remove_at_refuses_a_key_from_outside_the_store(store, tmp_path):
    """It unlinks whatever it is handed, so what it is handed is checked.

    Every legitimate key reaches it from `iter_entries` or from `clear`, both of
    which produce a pending blob directly inside `base_folder`. Anything else is
    a caller bug, and taking it on trust would unlink an arbitrary file.
    """
    outsider = tmp_path / f"someone-elses{PENDING_SUFFIX}"
    outsider.write_text("{}")
    unrelated = Path(store.base_folder) / "notes.txt"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text("keep me")

    for path in (outsider, unrelated):
        with pytest.raises(ValueError):
            store.remove_at(str(path))
        assert path.exists()

    # Nested one level down is still not "directly inside", because nothing this
    # store writes is, and a subdirectory is somebody else's data.
    nested_dir = Path(store.base_folder) / "sub"
    nested_dir.mkdir()
    nested = nested_dir / f"c1{PENDING_SUFFIX}"
    nested.write_text("{}")
    with pytest.raises(ValueError):
        store.remove_at(str(nested))
    assert nested.exists()


class _RacedStore(DiskSessionStateStore):
    """A store whose blobs are removed by somebody else just before it gets to them.

    Two processes running retention over one folder is the real version of this --
    the server's reaper and a second one in an ops script, say. The subclass only
    fixes the interleaving so the assertion is deterministic: the competing
    removal is a real unlink of a real file through a second store object on the
    same folder, which is exactly what the other process would issue.
    """

    def __init__(self, base_folder: str):
        super().__init__(base_folder)
        self.competitor = DiskSessionStateStore(base_folder)

    def remove_at(self, storage_key: str) -> bool:
        self.competitor.remove_at(storage_key)
        return super().remove_at(storage_key)


def test_a_blob_another_pass_removed_first_is_not_counted_as_reclaimed(tmp_path):
    """Two passes must not each claim the same reclamation.

    The namespace did shrink, so nothing failed and nothing is left behind -- but
    it shrank because of the other pass. An operator adding up `reclaimed` across
    processes would otherwise count these blobs twice and conclude retention is
    reclaiming more than exists.
    """
    store = _RacedStore(str(tmp_path / "pending"))
    for channel_id in ("c1", "c2"):
        _write_stray(
            store, f"{channel_id}{PENDING_SUFFIX}", channel_id, age_days=400
        )

    outcome = store.reap(PendingRetentionPolicy(max_age_seconds=7 * DAY))

    assert outcome.scanned == 2
    assert outcome.reclaimed == 0, "counted a removal another pass performed"
    assert outcome.failures == 0, "being beaten to it is not a failure"
    assert _blobs_on_disk(store) == set()


def test_remove_at_reports_absence_rather_than_raising(store):
    """Being already gone is a normal outcome of a concurrent pass, not an error.

    `reap` distinguishes the two: a False means some other pass reclaimed the
    blob, so this one did not, while a raise means the blob is still there.
    """
    store.save("c1", {"channel_id": "c1", "awaiting_user": True})
    path = store._json_path("c1")

    assert store.remove_at(path) is True
    assert store.remove_at(path) is False


def test_clear_still_removes_both_names_through_the_same_unlink(store):
    """`clear` is now expressed in `remove_at`, so there is one unlink, not two.

    Its contract is unchanged and load-bearing: a session that can be
    resurrected from its legacy file has not been cleared (fix-7hn).
    """
    channel_id = "tenant/user-1"
    store.save(channel_id, {"channel_id": channel_id, "awaiting_user": True})
    legacy = Path(store._legacy_json_path(channel_id))
    legacy.write_text(json.dumps({"channel_id": channel_id, "awaiting_user": True}))

    store.clear(channel_id)

    assert not Path(store._json_path(channel_id)).exists()
    assert not legacy.exists()
    assert store.load(channel_id) is None


def test_both_backends_must_implement_removal_by_storage_key():
    """What keeps the two backends consistent is the ABC, so that is asserted.

    A backend that inherited a channel-keyed removal would reintroduce the defect
    for its own storage, and no disk test would notice. There is no Redis server
    here to exercise the Redis path end to end, which is precisely why the
    contract is pinned rather than assumed.
    """
    assert "remove_at" in SessionStateStore.__abstractmethods__
    for backend in (DiskSessionStateStore, RedisSessionStateStore):
        assert "remove_at" in vars(backend), f"{backend.__name__} inherits removal"
        assert "iter_entries" in vars(backend)


def test_the_redis_backend_refuses_a_key_outside_its_prefix():
    """The same guard as the disk store's, on a shared multi-pod instance.

    No server is contacted: `redis.from_url` builds a client lazily and the guard
    fires before any command is issued.
    """
    pytest.importorskip("redis")
    store = RedisSessionStateStore(
        "redis://127.0.0.1:6379/0", key_prefix="fw:test:pending:"
    )

    with pytest.raises(ValueError):
        store.remove_at("fw:other:pending:c1")
