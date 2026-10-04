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
import importlib.util
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
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
REPO_ROOT = Path(__file__).resolve().parents[1]


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


@pytest.fixture(scope="session")
def v6_store_module():
    """The real v6 store code, loaded out of this repository's own history.

    Same construction as the legacy fixture in `test_task_feedback.py`: the
    newest commit whose `store.py` still says `SCHEMA_VERSION = 6` IS the
    build that wrote every v6 corpus in existence, so a database it creates
    has the authentic old shape instead of a v7 file with its version pragma
    edited. No v6 database is committed and no schema is restated here.
    """
    log = subprocess.run(
        ["git", "log", "--format=%H", "--", "fastworkflow/observability/store.py"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if log.returncode != 0:
        pytest.skip("not a git checkout; the v6 baseline comes from history")
    for commit in log.stdout.split():
        shown = subprocess.run(
            ["git", "show", f"{commit}:fastworkflow/observability/store.py"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if shown.returncode == 0 and "\nSCHEMA_VERSION = 6\n" in shown.stdout:
            break
    else:
        pytest.skip("no v6 store.py in history to build a legacy fixture from")
    directory = tempfile.mkdtemp(prefix="fw-v6-archive-baseline-")
    source = Path(directory) / "store_v6.py"
    source.write_text(shown.stdout)
    spec = importlib.util.spec_from_file_location(
        "fastworkflow_store_v6_archive", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fastworkflow_store_v6_archive"] = module
    spec.loader.exec_module(module)
    assert module.SCHEMA_VERSION == obs.MIN_READABLE_SCHEMA_VERSION
    return module


@pytest.fixture
def historical_v6_store(tmp_path, v6_store_module) -> Path:
    """A real v6 database, written by the real v6 writer, WAL still pending."""
    live = tmp_path / "v6-live" / "observability.sqlite3"
    live.parent.mkdir()
    conn = v6_store_module.ObservabilityStore(str(live))._connect()
    try:
        _seed_pending_wal(conn)
        return _copy_away_from_the_writer(
            live, tmp_path / "v6-historical" / "observability.sqlite3"
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


def test_a_real_v6_store_is_archived_as_v6_and_reported_as_v6(
    historical_v6_store, tmp_path
):
    """v6 is readable, so v6 is archivable — and stays v6 either way.

    The archive of a v6 database is a v6 database, so the metadata has to say
    six. Reporting the archiving build's `SCHEMA_VERSION` instead would claim
    a migration that this build explicitly does not perform, about a file
    whose own pragma disagrees.
    """
    before = _file_bytes(historical_v6_store)
    destination = tmp_path / "v6.sqlite3"

    result = archive_historical_store(str(historical_v6_store), str(destination))

    assert _file_bytes(historical_v6_store) == before
    assert result["schema_version"] == obs.MIN_READABLE_SCHEMA_VERSION
    assert obs.SCHEMA_VERSION != obs.MIN_READABLE_SCHEMA_VERSION
    with sqlite3.connect(
        f"{destination.resolve().as_uri()}?mode=ro", uri=True
    ) as conn:
        assert (
            conn.execute("PRAGMA user_version").fetchone()[0]
            == obs.MIN_READABLE_SCHEMA_VERSION
        )
        # The v7 feedback columns are the visible half of the version gap, so
        # their absence is what "nothing was migrated" looks like on disk.
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(human_feedback)")
        }
        assert "comment" in columns and "category" not in columns
        assert conn.execute(
            "SELECT value FROM diagnostics WHERE key=?", (WAL_ONLY_KEY,)
        ).fetchone() == (WAL_ONLY_VALUE,)


def test_the_shared_archive_path_also_reports_the_version_it_produced(
    historical_v6_store, tmp_path
):
    """The same claim, made by `archive_to` rather than by the new helper.

    `_snapshot_to` used to return this build's `SCHEMA_VERSION` for every
    archive it took, so a v6 file came back labelled v7. Nothing in either
    path migrates, so the number has to come out of the archive.
    """
    destination = tmp_path / "shared-path.sqlite3"

    result = obs.ReadOnlyObservabilityStore(
        str(historical_v6_store)
    ).archive_to(str(destination))

    assert result["schema_version"] == obs.MIN_READABLE_SCHEMA_VERSION
    with sqlite3.connect(
        f"{destination.resolve().as_uri()}?mode=ro", uri=True
    ) as conn:
        assert (
            conn.execute("PRAGMA user_version").fetchone()[0]
            == result["schema_version"]
        )


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
