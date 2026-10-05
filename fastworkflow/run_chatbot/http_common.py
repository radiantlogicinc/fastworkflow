"""Shared names for chatbot handler mixins.

These live outside ``server.py`` so the mixins can use them without importing
that module.
"""

from __future__ import annotations

from fastworkflow.observability.store import ObservabilityStore

# The one sentence that explains an unreadable evidence store, shared by the
# two payloads that report it: the navigation warning band -- the sidebar is
# where a reader meets the failure -- and the /experiments payload.
STORE_UNAVAILABLE = "Recorded experiments are unavailable in the selected evidence store: "


def run_clear_conversations(db_path: str) -> dict[str, int]:
    """Erase all conversation/turn observability for one workflow.

    Including every offload evidence row -- the archived execute responses
    of the conversations being cleared, and of experiment runs, whose
    experiment records are cleared too. ``clear_conversations`` deletes them
    in the same transaction as the turn records.
    """
    return ObservabilityStore(db_path).clear_conversations()
