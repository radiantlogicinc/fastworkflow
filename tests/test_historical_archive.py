"""Archiving a store nobody owns must leave the evidence byte-identical.

Every source here is built by this file: a real observability store, seeded
through the real writer, then copied away from that writer mid-flight so the
copy carries a committed but un-checkpointed WAL. That is the shape of the
historical stores on disk, and the shape under which the old archive idiom
silently rewrites the evidence it is sealing — so it is the shape the
regression has to use. No store outside `tmp_path` is read or written.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.observability import store as obs
from fastworkflow.observability import archive as archive_module
from fastworkflow.observability.archive import (
    archive_historical_store,
    source_file_state,
)

WAL_ONLY_KEY = "wal_evidence"
WAL_ONLY_VALUE = "committed-in-wal"


def _file_bytes(path: Path) -> dict[str, str | None]:
    """Presence and digest of a store and its sidecars, read as plain files."""
    state: dict[str, str | None] = {}
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        try:
            state[suffix or "main"] = hashlib.sha256(
                candidate.read_bytes()
            ).hexdigest()
        except FileNotFoundError:
            state[suffix or "main"] = None
    return state


def _copy_away_from_the_writer(live: Path, destination: Path) -> Path:
    """Copy a store while its writer holds it, so the copy keeps the WAL.

    A clean close checkpoints; this is how the historical stores on disk came
    to carry committed rows in a `-wal` with nobody holding the file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{live}{suffix}")
        if candidate.exists():
            shutil.copy2(candidate, f"{destination}{suffix}")
    assert os.path.getsize(f"{destination}-wal") > 0
    return destination


def _seed_pending_wal(connection) -> None:
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute(
        "INSERT INTO diagnostics(key,value,updated_at) VALUES(?,?,?)",
        (WAL_ONLY_KEY, WAL_ONLY_VALUE, "2026-09-20T00:00:00+00:00"),
    )
    connection.commit()


@pytest.fixture
def historical_store(tmp_path, monkeypatch) -> Path:
    """A store with committed rows sitting in a WAL and no writer to its name."""
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("FASTWORKFLOW_WORKFLOW_ID", "historical_archive_test")
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    live = Path(state_paths.observability_db(str(workflow)))
    assert live.resolve().is_relative_to(tmp_path.resolve()), live

    conn = obs.ObservabilityStore(str(live))._connect()
    try:
        _seed_pending_wal(conn)
        return _copy_away_from_the_writer(
            live, tmp_path / "historical" / "observability.sqlite3"
        )
    finally:
        conn.close()


def test_archiving_historical_evidence_changes_none_of_its_bytes(
    historical_store, tmp_path
):
    before = _file_bytes(historical_store)
    destination = tmp_path / "sealed" / "evidence.sqlite3"

    result = archive_historical_store(str(historical_store), str(destination))

    assert _file_bytes(historical_store) == before
    assert result["source_bytes_verified_unchanged"] is True
    assert result["source_baseline_taken_before_any_open"] is True
    assert result["source_wal_bytes"] == os.path.getsize(
        f"{historical_store}-wal"
    )
    assert result["source_path"] == os.path.abspath(str(historical_store))
    assert result["source_baseline"] == source_file_state(str(historical_store))


def test_the_archive_carries_the_rows_that_were_only_in_the_wal(
    historical_store, tmp_path
):
    destination = tmp_path / "evidence.sqlite3"

    result = archive_historical_store(str(historical_store), str(destination))

    assert result["sealed"] is True
    assert result["read_only"] is True
    assert result["sidecar_free"] is True
    assert not os.path.exists(f"{destination}-wal")
    assert not os.path.exists(f"{destination}-shm")
    assert destination.stat().st_mode & 0o777 == 0o444
    assert result["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    with sqlite3.connect(
        f"{destination.resolve().as_uri()}?mode=ro", uri=True
    ) as conn:
        row = conn.execute(
            "SELECT value FROM diagnostics WHERE key=?", (WAL_ONLY_KEY,)
        ).fetchone()
    assert row == (WAL_ONLY_VALUE,)


def test_nothing_is_left_beside_the_archive_to_be_mistaken_for_evidence(
    historical_store, tmp_path
):
    destination = tmp_path / "sealed" / "evidence.sqlite3"

    archive_historical_store(str(historical_store), str(destination))

    assert sorted(p.name for p in destination.parent.iterdir()) == [
        destination.name
    ]


def test_the_writer_construction_this_entrypoint_exists_to_avoid_still_damages(
    historical_store, tmp_path
):
    """Why the entrypoint is not `ObservabilityStore(path).archive_to(...)`.

    The old idiom is measured here rather than described, so that a future
    attempt to route historical archival back through a writer construction
    has to argue with a green test instead of a comment. `archive_to`'s own
    unchanged-bytes claim survives the damage, because its baseline is taken
    after construction — which is the whole defect.
    """
    before = _file_bytes(historical_store)

    result = obs.ObservabilityStore(
        str(historical_store), migrate=False
    ).archive_to(str(tmp_path / "old-idiom.sqlite3"))

    after = _file_bytes(historical_store)
    # The WAL is folded into main and then emptied or removed, so both the
    # database and the log read differently than they did before the open.
    assert after["main"] != before["main"]
    assert after["-wal"] != before["-wal"]
    assert result["source_bytes_verified_unchanged"] is True


def test_a_store_from_an_older_build_is_refused_and_left_as_it_was(
    tmp_path,
):
    """Nothing migrates, so an older store is not archivable by this build.

    The copy is read through the current-schema reader, which refuses it; the
    source keeps every byte and no half-made archive is left behind.
    """
    live = tmp_path / "older" / "observability.sqlite3"
    obs.ObservabilityStore(str(live))
    with sqlite3.connect(live) as conn:
        conn.execute(f"PRAGMA user_version = {obs.SCHEMA_VERSION - 1}")
    before = _file_bytes(live)
    destination = tmp_path / "sealed" / "older.sqlite3"

    with pytest.raises(obs.IncompatibleObservabilityDB, match="carries no migration"):
        archive_historical_store(str(live), str(destination))

    assert _file_bytes(live) == before
    assert not destination.exists()


def test_an_existing_archive_is_never_overwritten(historical_store, tmp_path):
    destination = tmp_path / "already-there.sqlite3"
    destination.write_bytes(b"an earlier seal")

    with pytest.raises(FileExistsError):
        archive_historical_store(str(historical_store), str(destination))

    assert destination.read_bytes() == b"an earlier seal"


def test_a_source_this_process_is_recording_into_is_refused(
    tmp_path, monkeypatch
):
    """An owned live writer belongs to `archive_to`, which can hold it still.

    Copying it instead would seal a file that is moving underneath the copy.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("FASTWORKFLOW_WORKFLOW_ID", "historical_archive_live")
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    sink = obs.get_observability_sink(str(workflow))
    assert sink is not None
    live = Path(state_paths.observability_db(str(workflow)))
    assert live.resolve().is_relative_to(tmp_path.resolve()), live
    try:
        with pytest.raises(obs.WriterStillOpen):
            archive_historical_store(str(live), str(tmp_path / "live.sqlite3"))
    finally:
        sink.close()
    assert not (tmp_path / "live.sqlite3").exists()


def test_a_source_that_really_moves_mid_archive_aborts_and_cleans_up(
    historical_store, tmp_path, monkeypatch
):
    """The guard the pre-open baseline makes reachable at all.

    With the baseline taken after the archiver's own open, every comparison is
    between two post-open readings and no change can register. Here a real
    write lands on the real source file while the archive is being taken —
    the digests are computed normally, nothing is faked — and the finished
    archive goes away rather than standing as a seal on bytes that moved.
    """
    source = historical_store

    class WritesToTheSourceMidArchive(archive_module.ReadOnlyObservabilityStore):
        """The real reader, with a real concurrent writer behind it."""

        def snapshot_settled_source_to(self, destination):
            result = super().snapshot_settled_source_to(destination)
            with open(source, "r+b") as handle:
                handle.seek(0, os.SEEK_END)
                handle.write(b"a concurrent write")
            return result

    monkeypatch.setattr(
        archive_module, "ReadOnlyObservabilityStore", WritesToTheSourceMidArchive
    )
    before = _file_bytes(source)
    destination = tmp_path / "aborted.sqlite3"

    with pytest.raises(obs.SourceChangedDuringArchive):
        archive_historical_store(str(source), str(destination))

    assert not destination.exists()
    assert _file_bytes(source) != before
