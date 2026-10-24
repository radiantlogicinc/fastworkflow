"""Shared names for chatbot handler mixins.

These live outside ``server.py`` so the mixins can use them without importing
that module.
"""

from __future__ import annotations

# The one sentence that explains an unreadable evidence store, shared by the
# two payloads that report it: the navigation warning band -- the sidebar is
# where a reader meets the failure -- and the /experiments payload.
STORE_UNAVAILABLE = "Recorded experiments are unavailable in the selected evidence store: "
