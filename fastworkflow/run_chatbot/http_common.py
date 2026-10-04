"""Shared names for chatbot handler mixins.

These live outside ``server.py`` so the mixins can use them without importing
that module.
"""

from __future__ import annotations

import os

from fastworkflow import state_paths
from fastworkflow.observability.store import ObservabilityStore

# The one sentence that explains an unreadable evidence store, shared by the
# two payloads that report it: the navigation warning band -- the sidebar is
# where a reader meets the failure -- and the /experiments payload.
STORE_UNAVAILABLE = "Recorded experiments are unavailable in the selected evidence store: "


def run_clear_conversations(db_path: str, workflow_path: str = "") -> dict[str, int]:
    """Erase all conversation/turn observability for one workflow.

    Including every offload evidence row -- the archived execute responses
    of the conversations being cleared, and of experiment runs, whose
    experiment records are cleared too. ``clear_conversations`` deletes them
    in the same transaction as the turn records.
    """
    deleted = ObservabilityStore(db_path).clear_conversations()
    if workflow_path:
        legacy_dir = state_paths.conversations_dir(workflow_path)
        removed = 0
        try:
            names = os.listdir(legacy_dir)
        except FileNotFoundError:
            names = []
        for name in names:
            if not name.endswith((".sqlite3", ".sqlite3-wal", ".sqlite3-shm")):
                continue
            path = os.path.join(legacy_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                pass
        deleted["legacy_conversation_db_files"] = removed
    return deleted
