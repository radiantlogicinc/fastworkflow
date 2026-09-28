"""Compatibility imports for trajectory-manifest observability enrichment."""
from fastworkflow.observability.enrichment import (
    TRAJECTORY_MANIFEST_CONTRACT as CONTRACT,
    classify_against_steps,
    install_trajectory_manifest_enrichment as install_span_policy,
    manifest_from_messages_json,
    observation_row,
    uninstall_trajectory_manifest_enrichment as uninstall_span_policy,
)

__all__ = [
    "CONTRACT",
    "classify_against_steps",
    "install_span_policy",
    "manifest_from_messages_json",
    "observation_row",
    "uninstall_span_policy",
]
