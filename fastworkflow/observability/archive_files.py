"""Archived experiments live only as files in the workflow's archives folder.

Nothing in the live database names them. A visit that lists them opens every
archive file and reads the experiment row inside it.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def archives_directory(workflow_path: str) -> str:
    """The archives folder for a workflow. Does not create it."""
    from fastworkflow import state_paths

    return os.path.join(
        os.path.dirname(state_paths.observability_db(workflow_path)),
        "archives",
    )


def list_archived_experiments(directory: str) -> list[dict[str, Any]]:
    """Open every archive file and return the experiment it holds.

    A file with no experiment row (a conversation archive, a scratch file)
    is skipped. ``archived_at`` is the file's modification time, which is
    when the copy was written.
    """
    root = Path(directory)
    if not root.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix != ".sqlite3" or path.name.startswith("."):
            continue
        row = read_archived_experiment(path)
        if row is not None:
            found.append(row)
    found.sort(key=lambda row: row.get("archived_at") or "", reverse=True)
    return found


def read_archived_experiment(path: Path | str) -> Optional[dict[str, Any]]:
    """The single experiment row in one archive file, or None."""
    file = Path(path)
    try:
        uri = file.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT experiment_id, description, status, invalid_reason,
                          benchmark_id, benchmark_version, created_at, completed_at
                     FROM experiments"""
            ).fetchall()
    except sqlite3.Error:
        return None
    if len(rows) != 1:
        return None
    row = dict(rows[0])
    stat = file.stat()
    row.update(
        path=str(file),
        size_bytes=stat.st_size,
        archived_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        readable=True,
    )
    return row


def archive_file_for(directory: str, experiment_id: str) -> Optional[str]:
    """The archive file whose name starts with this experiment id, if it holds it.

    The seal writes ``<experiment id>-<stamp>.sqlite3``. The id is checked
    inside the file so a prefix collision cannot pick the wrong copy.
    """
    root = Path(directory)
    if not experiment_id or not root.is_dir():
        return None
    matches = [
        path for path in root.glob(f"{experiment_id}-*.sqlite3")
        if path.is_file() and not path.name.startswith(".")
    ]
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in matches:
        row = read_archived_experiment(path)
        if row is not None and row["experiment_id"] == experiment_id:
            return str(path)
    return None


def list_archived_conversations(directory: str) -> list[dict[str, Any]]:
    """Open every archive file and return the standalone conversations it holds."""
    root = Path(directory)
    if not root.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix != ".sqlite3" or path.name.startswith("."):
            continue
        row = read_archived_conversation(path)
        if row is not None:
            found.append(row)
    found.sort(key=lambda row: row.get("archived_at") or "", reverse=True)
    return found


def read_archived_conversation(path: Path | str) -> Optional[dict[str, Any]]:
    """The one standalone conversation in an archive file, or None."""
    file = Path(path)
    try:
        uri = file.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT channel_id, conversation_id, topic, experiment_id
                   FROM conversations"""
            ).fetchall()
            if len(rows) != 1 or rows[0]["experiment_id"] is not None:
                return None
            row = dict(rows[0])
            row["turn_count"] = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    except sqlite3.Error:
        return None
    stat = file.stat()
    row.update(
        path=str(file),
        size_bytes=stat.st_size,
        archived_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        readable=True,
    )
    return row


def archived_conversation_file_for(
    directory: str, channel_id: str, conversation_id: int
) -> Optional[str]:
    """Find a standalone conversation archive by opening the folder's files."""
    for row in list_archived_conversations(directory):
        if (
            row["channel_id"] == channel_id
            and int(row["conversation_id"]) == int(conversation_id)
        ):
            return str(row["path"])
    return None


def max_archived_conversation_id(directory: str, channel_id: str) -> int:
    """The highest archived id for one channel, used to prevent id reuse."""
    return max(
        (
            int(row["conversation_id"])
            for row in list_archived_conversations(directory)
            if row["channel_id"] == channel_id
        ),
        default=0,
    )
