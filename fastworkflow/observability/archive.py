"""Archiving evidence this process does not own, without touching a byte of it.

`ObservabilityStore.archive_to` takes an honest snapshot. What it is not is a
read-only way to REACH one: getting to the method means constructing a store on
the source, and construction opens the database through `_connect`, whose
journal-mode pragma is write-capable. On a database carrying a pending WAL —
which is the shape of every observability store copied away from a running
server — that open alone checkpoints: main rewritten, `-wal` and `-shm`
removed. `archive_to` then samples its "before" bytes and reports
`source_bytes_verified_unchanged`, truthfully, about a baseline the archiving
process had already moved. Every `SourceChangedDuringArchive` guard compares
post-checkpoint bytes with post-checkpoint bytes and cannot fire.

So the baseline has to be taken before anything opens the file, and nothing may
open the file afterwards either. `archive_historical_store` digests the source
and its sidecars with plain file reads, copies them byte for byte, re-digests
the source to prove the copy is of a file that held still, and then opens only
the copy — read-only, so the copy's own WAL survives into the snapshot rather
than being folded away first. The source is digested once more at the end, so
the unchanged-bytes claim this returns is a claim about the historical
evidence, not about a moment after the archiver got there.

Presence counts as bytes here. A `-wal` that existed before and is gone after
is exactly the damage this module exists to detect, so the baseline records
each file as present-with-digest or absent, and both halves must match.

WHAT THIS DOES NOT CHANGE. Ordinary writer construction is untouched and is
still a writer; `archive_to` is still the entrypoint for a store this process
owns and is recording into, where the writer is held still instead of copied
around. Nothing here migrates, and a v6 database is archived exactly as it is
read — through `ReadOnlyObservabilityStore`, which accepts v6 and v7 alike.
The returned `schema_version` says so, and says it for every archive rather
than only for this path: `_snapshot_to` reads the pragma out of the file it
produced instead of reporting the archiving build's `SCHEMA_VERSION`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastworkflow.observability.store import (
    ReadOnlyObservabilityStore,
    SourceChangedDuringArchive,
    WriterStillOpen,
    sink_for_db_path,
)

# The files SQLite keeps beside a WAL-mode database. Both are part of the
# evidence: the committed rows can be in `-wal`, and `-shm` is the index that
# makes them readable.
SIDECAR_SUFFIXES = ("-wal", "-shm")


def _file_state(path: str) -> Optional[dict[str, Any]]:
    """Size and digest of a file, or None when it is not there."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        size = os.stat(path).st_size
    except FileNotFoundError:
        return None
    return {"size_bytes": size, "sha256": digest.hexdigest()}


def source_file_state(source: str) -> dict[str, Optional[dict[str, Any]]]:
    """Digest a store and its sidecars by reading them, never by opening them.

    Callers hold this against the same call made after the archive: equal
    mappings mean every file is still present, absent, and byte-identical.
    """
    base = os.path.abspath(source)
    paths = [base] + [f"{base}{suffix}" for suffix in SIDECAR_SUFFIXES]
    return {path: _file_state(path) for path in paths}


def archive_historical_store(
    source: str,
    destination: str,
    *,
    scratch_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Seal a store this process does not own, leaving its bytes untouched.

    `scratch_dir` is where the private copy is taken; it defaults to the
    destination's directory, which is the one place the caller has already
    chosen to have room for a copy of the evidence.

    Raises `WriterStillOpen` when this process is itself recording into the
    source (use `archive_to`, which holds that writer still), and
    `SourceChangedDuringArchive` when the source moved while being copied —
    which makes the copy a snapshot of no single moment and the archive a
    claim about bytes that were not still.
    """
    absolute_source = os.path.abspath(source)
    if not os.path.exists(absolute_source):
        raise FileNotFoundError(f"no evidence database at {absolute_source!r}")

    live_sink = sink_for_db_path(absolute_source)
    if live_sink is not None and not live_sink._closed:
        raise WriterStillOpen(
            f"refusing to copy {absolute_source!r} while this process is "
            "recording into it; archive_to holds an owned writer still"
        )

    target = Path(destination)
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing evidence archive: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    baseline = source_file_state(absolute_source)
    scratch = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.copy.",
            dir=scratch_dir or str(target.parent),
        )
    )
    try:
        copy = scratch / Path(absolute_source).name
        for suffix in ("",) + SIDECAR_SUFFIXES:
            candidate = f"{absolute_source}{suffix}"
            if os.path.exists(candidate):
                shutil.copy2(candidate, f"{copy}{suffix}")
        copied_from = source_file_state(absolute_source)
        if copied_from != baseline:
            raise SourceChangedDuringArchive(
                f"{absolute_source!r} changed while it was being copied; the "
                "copy is not a snapshot of any one moment"
            )

        # Only the copy is ever opened, and only read-only: its WAL is read
        # into the snapshot rather than checkpointed away ahead of it.
        reader = ReadOnlyObservabilityStore(str(copy))
        result = dict(reader.snapshot_settled_source_to(str(target)))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    finished = source_file_state(absolute_source)
    if finished != baseline:
        os.chmod(target, 0o644)
        target.unlink(missing_ok=True)
        raise SourceChangedDuringArchive(
            f"{absolute_source!r} changed while it was being archived"
        )

    wal_state = baseline.get(f"{absolute_source}{SIDECAR_SUFFIXES[0]}")
    result.update(
        {
            "source_path": absolute_source,
            "source_baseline": baseline,
            "source_baseline_taken_before_any_open": True,
            "source_wal_bytes": (wal_state or {}).get("size_bytes", 0),
            # Restated against the pre-open baseline. The inner snapshot
            # verified the private copy; this verified the evidence.
            "source_bytes_verified_unchanged": True,
        }
    )
    return result
