"""Framework-internal ``json_schema_extra`` keys on skill/command Input fields."""
from __future__ import annotations

from typing import Mapping

#: Preserved when a hand-written ``available_from`` is stripped from ``json_schema_extra`` so
#: preview ``--check`` and later ``resolve_field_sources`` runs still report the error after
#: the imported ``Signature`` class is mutated (module stays in ``sys.modules``).
HAND_WRITTEN_AVAILABLE_FROM_KEY = "_hand_written_available_from"

SKILL_JSON_SCHEMA_EXTRA_INTERNAL_KEYS = frozenset({HAND_WRITTEN_AVAILABLE_FROM_KEY})


def omit_internal_json_schema_extra(extra: Mapping | None) -> dict | None:
    """Drop framework-internal keys from *extra* for metadata / DSPy / model-facing consumers."""
    if extra is None:
        return None
    if not isinstance(extra, Mapping):
        return dict(extra)
    filtered = {k: v for k, v in extra.items() if k not in SKILL_JSON_SCHEMA_EXTRA_INTERNAL_KEYS}
    return filtered or None
