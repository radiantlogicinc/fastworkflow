"""Archive one recorded conversation the way an experiment is archived.

`GET /api/archived-conversations` lists the archived ones and the live ones
ready to archive. `POST` copies one conversation's
turns into a read-only file under the workflow's archives directory and
removes those rows from the live database. `DELETE
/api/archived-conversations/<channel>/<id>` removes that file. `GET` of the
same path reads the turns back from the file.

A conversation that belongs to an experiment is refused: those turns are
that experiment's evidence, and archiving the experiment takes them.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

from fastworkflow import state_paths
from fastworkflow.observability.archive_files import (
    archives_directory,
    list_archived_conversations,
)
from fastworkflow.observability.store import (
    ConversationArchiveRefused,
    ConversationNotArchived,
    ConversationNotFound,
    IncompatibleObservabilityDB,
    open_live_store,
)
from fastworkflow.storage_keys import encode_path_component

CONVERSATIONS_PREFIX = "/api/archived-conversations"


def conversation_archive_key(path: str) -> Optional[tuple[str, int]]:
    """The channel and conversation id a delete or detail path names."""
    prefix = CONVERSATIONS_PREFIX + "/"
    if not path.startswith(prefix):
        return None
    channel, _sep, raw_id = path[len(prefix):].partition("/")
    if not channel or not raw_id or "/" in raw_id or not raw_id.isdigit():
        return None
    return unquote(channel), int(raw_id)


def conversation_delete_path(path: str) -> bool:
    return conversation_archive_key(path) is not None


def _archived_rows(workflow_path: str) -> list[dict[str, Any]]:
    """Archived conversations are discovered from files, not an index."""
    return list_archived_conversations(archives_directory(workflow_path))


def _ready_rows(store: Any) -> list[dict[str, Any]]:
    """Live conversations that can be archived on their own.

    A conversation labeled with an experiment, or one whose turns are, stays
    with that experiment. Archiving the experiment takes those turns.
    """
    conversations: list[dict[str, Any]] = []
    while True:
        batch = store.list_conversations(limit=200, offset=len(conversations))
        conversations.extend(batch)
        if len(batch) < 200:
            break
    with store._connect() as conn:
        counts = {
            (row[0], row[1]): row[2]
            for row in conn.execute(
                """SELECT channel_id, conversation_id, COUNT(*)
                     FROM turns GROUP BY channel_id, conversation_id"""
            )
        }
        owned = {
            (row[0], row[1])
            for row in conn.execute(
                """SELECT DISTINCT channel_id, conversation_id FROM turns
                    WHERE experiment_id IS NOT NULL"""
            )
        }
    ready = []
    for row in conversations:
        key = (row.get("channel_id"), row.get("conversation_id"))
        if row.get("experiment_id") or key in owned:
            continue
        ready.append({
            "channel_id": row.get("channel_id"),
            "conversation_id": row.get("conversation_id"),
            "topic": row.get("topic"),
            "turn_count": counts.get(key, 0),
            "started_at": row.get("started_at"),
            "last_turn_at": row.get("last_turn_at"),
        })
    return ready


def _turns(path: str) -> list[dict[str, Any]]:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [
            {
                "turn_key": row["turn_key"],
                "ordinal": row["ordinal"],
                "user_message": row["user_message"],
                "answer": row["answer"],
                "status": row["status"],
                "started_at": row["started_at"],
            }
            for row in conn.execute(
                """SELECT turn_key, ordinal, user_message, answer, status, started_at
                     FROM turns ORDER BY ordinal, turn_key"""
            )
        ]


class _ConversationArchiveRoutes:
    def _handle_archived_conversations(self, path: str) -> None:
        """List archived conversations, or one conversation's recorded turns."""
        try:
            store = self.chatbot.open_store()
        except IncompatibleObservabilityDB as exc:
            self._error(409, str(exc))
            return
        key = conversation_archive_key(path)
        if store is None:
            if key is None and path == CONVERSATIONS_PREFIX:
                self._send_json({"archived": [], "ready": []})
                return
            self._error(404, "observability DB not found")
            return
        if key is None:
            if path != CONVERSATIONS_PREFIX:
                self._error(404, "no archived conversation with that id")
                return
            self._send_json({
                "archived": _archived_rows(self.chatbot.workflow_path or ""),
                "ready": _ready_rows(store),
            })
            return
        channel_id, conversation_id = key
        match = next(
            (
                row for row in _archived_rows(self.chatbot.workflow_path or "")
                if row["channel_id"] == channel_id
                and row["conversation_id"] == conversation_id
            ),
            None,
        )
        if match is None or not match["readable"]:
            self._error(404, "no archived conversation with that id")
            return
        match["turns"] = _turns(match["path"])
        self._send_json(match)

    def _post_archive_conversation(
        self, path: str, body: Any, query: dict[str, list[str]]
    ) -> None:
        """Copy one conversation out of the live database."""
        workflow_path = (self.chatbot.workflow_path or "").strip()
        channel_id = body.get("channel_id")
        conversation_id = body.get("conversation_id")
        if not isinstance(channel_id, str) or not channel_id:
            self._error(400, "channel_id is required")
            return
        if type(conversation_id) is not int:
            self._error(400, "conversation_id is required")
            return
        if not workflow_path:
            self._error(409, "select a workflow before archiving a conversation")
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = os.path.join(
            state_paths.archives_dir(workflow_path),
            "conversation-"
            + encode_path_component(f"{channel_id}-{conversation_id}")
            + f"-{stamp}.sqlite3",
        )
        try:
            store = open_live_store(self.chatbot.db_path or "", write=True)
            archived = store.archive_conversation(
                channel_id, conversation_id, destination
            )
        except ConversationNotFound:
            self._error(404, "conversation not found")
            return
        except ConversationArchiveRefused as exc:
            self._error(409, str(exc))
            return
        except (IncompatibleObservabilityDB, FileExistsError) as exc:
            self._error(409, str(exc))
            return
        except sqlite3.OperationalError as exc:
            self._write_refused(exc)
            return
        self._send_json(archived, status=201)

    def _delete_archived_conversation(
        self, path: str, body: Any, query: dict[str, list[str]]
    ) -> None:
        """Remove an archived conversation and its file."""
        key = conversation_archive_key(path)
        if key is None:
            self._error(404, "no archived conversation with that id")
            return
        channel_id, conversation_id = key
        try:
            store = open_live_store(self.chatbot.db_path or "", write=True)
            result = store.delete_archived_conversation(channel_id, conversation_id)
        except ConversationNotArchived:
            self._error(404, "no archived conversation with that id")
            return
        except IncompatibleObservabilityDB as exc:
            self._error(409, str(exc))
            return
        except sqlite3.OperationalError as exc:
            self._write_refused(exc)
            return
        self._send_json(result)
