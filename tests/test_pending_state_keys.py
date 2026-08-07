"""One suspended session per storage key (fix-7hn).

`DiskSessionStateStore._json_path` used to fold separators, so `tenant/user-1`
and `tenant_user-1` named one file. Whichever session suspended second overwrote
the first, and the survivor was then served to *both* channels -- so a user could
resume a stranger's conversation, complete with its ReAct trajectory and whatever
had been typed into an `ask_user` prompt. Path-shaped channel ids (tenant/user,
org/team/user) are an ordinary multi-tenant shape, so this was reachable rather
than theoretical.

Real temp directories and real files throughout, because injectivity of a
filesystem name is a property of the filesystem. The one property that cannot be
observed on a case-sensitive Linux volume -- that two names stay apart under case
folding -- is asserted on the names themselves instead.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from fastworkflow import checkpoint_store, session_state_store
from fastworkflow.checkpoint_store import CheckpointIdentity
from fastworkflow.session_state_store import (
    PENDING_SUFFIX,
    SAVED_AT_KEY,
    DiskSessionStateStore,
    PendingRetentionPolicy,
)
from fastworkflow.storage_keys import encode_path_component

DAY = 86_400.0


@pytest.fixture
def store(tmp_path) -> DiskSessionStateStore:
    return DiskSessionStateStore(str(tmp_path / "pending"))


def _blob(channel_id: str, answer: str) -> dict:
    """The shape a real suspension has: the channel id travels inside the blob."""
    return {
        "channel_id": channel_id,
        "awaiting_user": True,
        "pending_clarification_request": answer,
    }


def _answer(store: DiskSessionStateStore, channel_id: str) -> str:
    return store.load(channel_id)["pending_clarification_request"]


def _stored_names(store: DiskSessionStateStore) -> set[str]:
    return {path.name for path in Path(store.base_folder).iterdir()}


def test_a_path_shaped_id_and_its_folded_twin_are_never_one_blob(store):
    """The bug itself: `tenant/user-1` and `tenant_user-1` folded to one name.

    Overwriting was the visible symptom. The harm was that the surviving blob
    answered to both channels, so one user resumed the other's suspended turn.
    """
    store.save("tenant/user-1", _blob("tenant/user-1", "alice's answer"))
    store.save("tenant_user-1", _blob("tenant_user-1", "bob's answer"))

    assert store._json_path("tenant/user-1") != store._json_path("tenant_user-1")
    assert len(_stored_names(store)) == 2

    assert _answer(store, "tenant/user-1") == "alice's answer"
    assert _answer(store, "tenant_user-1") == "bob's answer"

    # Re-suspending one channel must not disturb the other's blob either.
    store.save("tenant/user-1", _blob("tenant/user-1", "alice's second answer"))
    assert _answer(store, "tenant_user-1") == "bob's answer"

    store.clear("tenant_user-1")
    assert store.exists("tenant/user-1"), "clearing one channel took the other's state"
    assert _answer(store, "tenant/user-1") == "alice's second answer"


def test_ids_differing_only_by_case_are_never_one_blob(store):
    """A case-insensitive volume folds `Tenant` onto `tenant` before any read.

    Linux keeps these apart on the raw bytes alone, so the property is asserted
    on the names: on APFS or NTFS two names that differ only in case are one
    file, and the collision would surface there and nowhere else.
    """
    upper, lower = "Tenant/user", "tenant/user"
    store.save(upper, _blob(upper, "upper"))
    store.save(lower, _blob(lower, "lower"))

    names = {os.path.basename(store._json_path(cid)) for cid in (upper, lower)}
    assert len(names) == 2
    assert (
        len({name.casefold() for name in names}) == 2
    ), "these two names are one file on a case-insensitive volume"

    assert _answer(store, upper) == "upper"
    assert _answer(store, lower) == "lower"


def test_no_legacy_name_can_ever_equal_a_new_name(store):
    """What makes the read-both-shapes window safe.

    A folded name like `tenant_a` is exactly what the encoder emits for the
    channel id `tenant_a`. If the two shapes shared a suffix, a new-form read
    could still land on a blob the old mapping had conflated, and a save could
    unlink a live neighbour's blob as though it were its own legacy copy.
    """
    hazards = [
        "tenant/a",
        "tenant_a",
        "tenant/user-1",
        "tenant_user-1",
        "a",
        "A/b",
        "org/team/user",
    ]
    new_names = {store._json_path(cid) for cid in hazards}
    legacy_names = {store._legacy_json_path(cid) for cid in hazards}

    assert not (new_names & legacy_names)


def test_a_legacy_named_blob_is_still_readable_and_clear_removes_it(store):
    """An upgrade must not orphan a suspension that is already on disk.

    The blob is written by hand at the pre-fix name, which is what an existing
    deployment's disk looks like at the moment it upgrades.
    """
    channel_id = "tenant/user-1"
    legacy = Path(store._legacy_json_path(channel_id))
    legacy.write_text(json.dumps(_blob(channel_id, "half-finished")))

    assert store.exists(channel_id)
    assert _answer(store, channel_id) == "half-finished"

    store.clear(channel_id)

    assert (
        not legacy.exists()
    ), "a cleared session can still be resurrected from its legacy file"
    assert store.exists(channel_id) is False
    assert store.load(channel_id) is None


def test_a_legacy_blob_is_not_served_to_the_channel_it_collided_with(store):
    """On a shared legacy name, the blob's own channel_id decides ownership.

    `tenant/user-1` folded onto `tenant_user-1_pending.json`, which is precisely
    the name the old rule gave `tenant_user-1` as well. Serving it to whoever
    asks would carry the original defect straight through the migration window.
    """
    owner, neighbour = "tenant/user-1", "tenant_user-1"
    legacy = Path(store._legacy_json_path(owner))
    assert legacy == Path(
        store._legacy_json_path(neighbour)
    ), "the fixture no longer reproduces the collision it is testing"
    legacy.write_text(json.dumps(_blob(owner, "alice's answer")))

    assert _answer(store, owner) == "alice's answer"
    assert (
        store.load(neighbour) is None
    ), "served another channel's suspended conversation"
    assert store.exists(neighbour) is False


def test_saving_supersedes_the_legacy_copy(store):
    """Two files for one channel would let the reaper delete a live suspension.

    `iter_entries` would report the channel twice, once with the stale blob's
    age. The stale entry ages out, and `clear` -- which must remove both names or
    a cleared session comes back -- takes the fresh blob with it.
    """
    channel_id = "tenant/user-1"
    legacy = Path(store._legacy_json_path(channel_id))
    legacy.write_text(json.dumps({**_blob(channel_id, "stale"), SAVED_AT_KEY: 1.0}))

    store.save(channel_id, _blob(channel_id, "fresh"))

    assert not legacy.exists()
    assert [cid for cid, _ in store.iter_entries()] == [channel_id]

    outcome = store.reap(PendingRetentionPolicy(max_age_seconds=DAY))

    assert outcome.reclaimed == 0
    assert _answer(store, channel_id) == "fresh"


def test_iter_entries_reports_both_shapes(store):
    """A legacy blob must be enumerated or it is immortal.

    Nothing else would ever offer it to `clear`, so the reaper is the only way it
    is reclaimed -- and the reaper matches protected ids, which is why the id
    still comes from inside the blob rather than from the filename.
    """
    store.save("new/shape", _blob("new/shape", "fresh"))
    Path(store._legacy_json_path("old/shape")).write_text(
        json.dumps({**_blob("old/shape", "stale"), SAVED_AT_KEY: 1.0})
    )

    found = dict(store.iter_entries())

    assert set(found) == {"new/shape", "old/shape"}
    assert found["old/shape"] == 1.0


def test_oversized_ids_round_trip_and_stay_distinct(store):
    """Past the filesystem name limit the name is a readable prefix plus a hash.

    Truncating alone would put every id sharing a long prefix on one file, which
    is the original defect with more characters in front of it.
    """
    long_a = "tenant/" + "u" * 400
    long_b = f"{long_a}x"
    store.save(long_a, _blob(long_a, "a"))
    store.save(long_b, _blob(long_b, "b"))

    for channel_id in (long_a, long_b):
        name = os.path.basename(store._json_path(channel_id))
        assert len(name.encode("utf-8")) <= 255

    assert len(_stored_names(store)) == 2
    assert store.load(long_a)["channel_id"] == long_a
    assert store.load(long_b)["channel_id"] == long_b
    assert _answer(store, long_a) == "a"
    assert _answer(store, long_b) == "b"


@pytest.mark.parametrize(
    "channel_id",
    [
        "tenant/user-1",
        "org/team/user-9",
        "Tenant/User",
        "tenant\\windows-user",
        "tenant%2Fuser-1",
        "tenant.user-1",
        "tenant/日本",
        "..",
    ],
)
def test_save_load_exists_clear_round_trip(store, channel_id):
    assert store.exists(channel_id) is False
    assert store.load(channel_id) is None

    store.save(channel_id, _blob(channel_id, "answer"))

    assert store.exists(channel_id) is True
    loaded = store.load(channel_id)
    assert loaded["channel_id"] == channel_id
    assert loaded["pending_clarification_request"] == "answer"
    assert loaded[SAVED_AT_KEY] <= time.time()

    name = os.path.basename(store._json_path(channel_id))
    assert name.isascii()
    assert os.sep not in name and "/" not in name and "\\" not in name
    assert name not in (os.curdir, os.pardir) and not name.startswith(".")

    store.clear(channel_id)

    assert store.exists(channel_id) is False
    assert store.load(channel_id) is None


def test_both_stores_derive_their_key_with_the_one_shared_encoder(store):
    """Two encoders are two chances to fold two channels onto one path.

    The identity assertions are the point: a second copy differing only in its
    character class would pass every behavioural test in this file and then
    reintroduce fix-7hn the first time a channel id grew a capital letter.
    """
    assert session_state_store.encode_path_component is encode_path_component
    assert checkpoint_store.encode_path_component is encode_path_component

    channel_id = "tenant/user-1"
    assert (
        os.path.basename(store._json_path(channel_id))
        == encode_path_component(channel_id) + PENDING_SUFFIX
    )
    assert (
        CheckpointIdentity(
            deployment_id="deploy-blue",
            workflow_fingerprint="wf-abc123",
            channel_id=channel_id,
            session_incarnation="inc-1",
        ).channel_key
        == encode_path_component(channel_id)
    )
