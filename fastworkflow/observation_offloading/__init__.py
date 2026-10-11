"""Observation offloading: compact, archive and search_memory.

This is how fastWorkflow runs a tool agent; there is no flag to turn it off.
``build_tool_agent`` always returns the plain ReAct, execute
observations always carry their canonical ``O{step_index}`` alias, compaction
always swaps an observation that is no longer worth its residency for its own
label, and the text behind every label stays reachable through ``search_memory``
and through answer-time rehydration. See ``docs/observation_search.md``.
"""
from __future__ import annotations

from fastworkflow.observation_offloading.agent import (
    DEFAULT_MAX_ITERS,
    build_tool_agent,
    open_handle_archive,
)
from fastworkflow.observation_offloading.archive import (
    PersistenceError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
    UnavailableHandleArchive,
)
from fastworkflow.observation_offloading.compact import (
    MIN_OFFLOAD_SAVING_BYTES,
    PACKED_TARGET_BYTES,
    RECENT_OBSERVATIONS_PROTECTED,
    archive_step,
    compact_trajectory,
    execute_step_indexes,
    min_offload_saving_bytes_from_env,
)
from fastworkflow.observation_offloading.labels import (
    alias_line,
    annotated_observation,
    context_clause,
    offload_label,
    offload_saving_bytes,
    printed_alias,
    printed_context,
    strip_alias_line,
)
from fastworkflow.observation_offloading.manifest import (
    classify_against_steps,
    install_span_policy,
    uninstall_span_policy,
)
from fastworkflow.observation_offloading.state import (
    archive_for_path,
    durable_archive,
)

__all__ = [
    "DEFAULT_MAX_ITERS",
    "MIN_OFFLOAD_SAVING_BYTES",
    "PACKED_TARGET_BYTES",
    "PersistenceError",
    "RECENT_OBSERVATIONS_PROTECTED",
    "RuntimeHandleArchive",
    "RuntimeHandleScope",
    "UnavailableHandleArchive",
    "alias_line",
    "archive_for_path",
    "annotated_observation",
    "context_clause",
    "durable_archive",
    "archive_step",
    "build_tool_agent",
    "classify_against_steps",
    "compact_trajectory",
    "execute_step_indexes",
    "install_span_policy",
    "min_offload_saving_bytes_from_env",
    "offload_label",
    "offload_saving_bytes",
    "open_handle_archive",
    "printed_alias",
    "printed_context",
    "strip_alias_line",
    "uninstall_span_policy",
]
