"""Checkpointing a live channel runtime: eligibility, retirement, restore.

This is the seam between the server's live-session cache and the durable
checkpoint record. Three rules govern everything here:

* A session that cannot be checkpointed safely is **not evicted**. The cache may
  sit over target indefinitely. Never trade an allocation problem for silent
  application-state loss.
* The runtime is popped only **after** the write succeeds. "Best effort, then
  evict" is how state disappears.
* Launch-time configuration is **reconciled**, not silently outranked by a
  snapshot and not blindly merged over one.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import fastworkflow
from fastworkflow import serialization_hooks
from fastworkflow.checkpoint_store import (
    PROTOCOL_VERSION,
    CheckpointIdentity,
    CheckpointRecord,
    CheckpointStoreError,
    ChannelCheckpointStore,
)
from fastworkflow.serialization_hooks import ProjectionOutcome
from fastworkflow.state_serialization import StateEncodingError
from fastworkflow.utils.logging import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .utils import ChannelRuntime

# Startup states (design 11.5). A boolean cannot express "attempted and failed"
# or "suspended half way", and restart behaviour differs for each.
STARTUP_NOT_ATTEMPTED = "not_attempted"
STARTUP_IN_PROGRESS = "in_progress"
STARTUP_SUSPENDED = "suspended"
STARTUP_SUCCEEDED = "succeeded"
STARTUP_FAILED = "failed"

# Request-scoped, never durable: it belongs to the request that presented it and
# is re-supplied on the next one (see turns.installed_credential).
NEVER_PERSISTED_CONTEXT_KEYS = frozenset({"http_bearer_token"})

_pin_warned: set[tuple[str, str]] = set()


def new_session_incarnation() -> str:
    return uuid.uuid4().hex


def fleet_protocol_floor() -> int:
    """The oldest checkpoint protocol every node in this fleet can read.

    During a mixed-version rollout the old binary is the constraint: if it
    cannot read what the new one writes, it will start from launch
    configuration, mutate application state, and leave the newer record stale
    for whoever reads it next. Declaring the floor makes that a refusal rather
    than a silent divergence, which is why there is no "ignore and continue"
    branch anywhere below.
    """
    floor = int(
        fastworkflow.get_env_var(
            "FASTWORKFLOW_CHECKPOINT_PROTOCOL_FLOOR", int, default=PROTOCOL_VERSION
        )
    )
    if floor > PROTOCOL_VERSION:
        raise ValueError(
            f"FASTWORKFLOW_CHECKPOINT_PROTOCOL_FLOOR={floor} is newer than this "
            f"build can write (protocol {PROTOCOL_VERSION}). This node cannot "
            "join that fleet; deploy a newer build or lower the floor."
        )
    return floor


def get_checkpoint_dir() -> str:
    speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
    path = os.path.join(speedict_foldername, "channel_checkpoints")
    os.makedirs(path, exist_ok=True)
    return path


def deployment_id() -> str:
    """Namespace component so two deployments sharing a backend cannot collide."""
    return str(
        fastworkflow.get_env_var("FASTWORKFLOW_DEPLOYMENT_ID", default="default")
    )


def workflow_fingerprint(workflow_path: str) -> str:
    """Namespace component that changes when the workflow's commands change."""
    from fastworkflow.command_directory import compute_commands_source_fingerprint

    try:
        return compute_commands_source_fingerprint(workflow_path)
    except Exception:  # a workflow we cannot fingerprint gets its own namespace
        return "unfingerprinted"


def identity_for(runtime: "ChannelRuntime") -> CheckpointIdentity:
    return CheckpointIdentity(
        deployment_id=deployment_id(),
        workflow_fingerprint=workflow_fingerprint(runtime.workflow_path),
        channel_id=runtime.channel_id,
        session_incarnation=runtime.session_incarnation,
    )


@dataclass
class Eligibility:
    """Whether this runtime may be evicted, and if not, why."""

    evictable: bool
    reason: str
    projection: Optional[serialization_hooks.ContextProjection] = None


def assess(runtime: "ChannelRuntime") -> Eligibility:
    """Decide whether a runtime can be checkpointed, so it can be evicted.

    Pinning is a first-class outcome with a metric, not an error.
    """
    ctx = runtime.execution_context

    if ctx.awaiting_user:
        # The suspended snapshot does not carry the logical-turn accumulator or
        # the CME continuation keys, so restoring one loses the pre-suspension
        # outputs. Pin rather than make a known defect routine.
        return Eligibility(False, "session is awaiting_user")

    workflow = ctx.app_workflow
    if workflow is None:
        return Eligibility(False, "no app workflow bound")

    projection = serialization_hooks.project_command_contexts(workflow)
    if not projection.is_checkpointable:
        serialization_hooks.warn_if_pinned(workflow, projection)
        return Eligibility(False, projection.reason or "context is not projectable",
                           projection)

    return Eligibility(True, "checkpointable", projection)


def durable_context(workflow: "fastworkflow.Workflow") -> dict[str, Any]:
    """The workflow context as it should be persisted.

    Everything JSON-native persists by default; the caller's credential never
    does, because it is request-scoped and would be stale the moment it landed.
    """
    context = dict(workflow.context or {})
    for key in NEVER_PERSISTED_CONTEXT_KEYS:
        context.pop(key, None)
    return context


def runtime_projection(runtime: "ChannelRuntime") -> dict[str, Any]:
    """Runtime fields the framework knows how to restore.

    Enumerated deliberately: a field outside this list is not silently dropped,
    it is a reason to pin (which is why adding one here is a decision, not a
    convenience).
    """
    workflow = runtime.execution_context.app_workflow
    return {
        "active_conversation_id": runtime.active_conversation_id,
        "stream_format": runtime.stream_format,
        "is_complete": bool(workflow.is_complete) if workflow else False,
        "durable_turn_count": runtime.durable_turn_count,
        "startup_ran": runtime.startup_ran,
    }


def startup_projection(runtime: "ChannelRuntime") -> dict[str, Any]:
    return {
        "state": runtime.startup_state,
        "idempotency_key": runtime.startup_idempotency_key,
        "epoch": runtime.startup_epoch,
    }


def launch_projection(launch_context: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The launch configuration as it stood when this record was written.

    Stored in full rather than as a digest: a digest can detect that launch
    configuration changed but cannot say which key, so it cannot support a real
    three-way merge — only a guess.
    """
    return {"prior_projection": dict(launch_context or {})}


def publish(
    store: ChannelCheckpointStore,
    runtime: "ChannelRuntime",
    eligibility: Eligibility,
    launch_context: Optional[dict[str, Any]],
) -> int:
    """Write the checkpoint. Raises rather than returning a partial success."""
    workflow = runtime.execution_context.app_workflow
    projection = eligibility.projection
    return store.publish(
        identity_for(runtime),
        context={
            "workflow_context": durable_context(workflow),
            "command_contexts": projection.as_record() if projection else {},
        },
        runtime=runtime_projection(runtime),
        startup=startup_projection(runtime),
        launch_context=launch_projection(launch_context),
        state_version=(projection.state_version if projection else 1) or 1,
    )


def reconcile_launch_context(
    *,
    prior_launch: dict[str, Any],
    saved_context: dict[str, Any],
    current_launch: dict[str, Any],
    channel_id: str,
) -> dict[str, Any]:
    """Three-way merge of (prior launch, saved application, current launch).

    A snapshot must not silently outrank a redeployed configuration, and a
    redeployed configuration must not silently overwrite what the application
    has since written. Attribution needs all three inputs: with only the saved
    state and the current launch you cannot tell an operator's change from an
    application's.
    """
    merged = dict(saved_context)
    changed: list[str] = []
    conflicts: list[str] = []

    for key, current_value in current_launch.items():
        in_prior = key in prior_launch
        prior_value = prior_launch.get(key)
        launch_changed = (not in_prior) or prior_value != current_value

        if not launch_changed:
            continue

        app_changed = key in saved_context and saved_context[key] != prior_value
        if app_changed and in_prior:
            # Both sides moved. The operator wins, because launch configuration
            # is how a deployment is corrected, but this is worth counting.
            conflicts.append(key)
        merged[key] = current_value
        if in_prior:
            changed.append(key)

    removed = [
        key
        for key in prior_launch
        if key not in current_launch
        # Only remove what the application has not since taken ownership of.
        and saved_context.get(key) == prior_launch[key]
    ]
    for key in removed:
        merged.pop(key, None)

    if changed or conflicts or removed:
        logger.warning(
            f"Launch-context reconciliation for channel_id {channel_id}: "
            f"changed={sorted(changed)}, removed={sorted(removed)}, "
            f"conflicts_resolved_to_launch={sorted(conflicts)}"
        )
    return merged


def restore(
    runtime_workflow: "fastworkflow.Workflow",
    record: CheckpointRecord,
    *,
    current_launch: dict[str, Any],
    channel_id: str,
) -> dict[str, Any]:
    """Apply a checkpoint to a freshly built workflow.

    Returns the runtime section so the caller can restore its own fields. Raises
    if the command contexts cannot be rebuilt, so the caller quarantines rather
    than continuing with half a session.
    """
    context_section = record.context or {}
    saved_context = dict(context_section.get("workflow_context") or {})
    prior_launch = dict((record.launch_context or {}).get("prior_projection") or {})

    merged = reconcile_launch_context(
        prior_launch=prior_launch,
        saved_context=saved_context,
        current_launch=current_launch,
        channel_id=channel_id,
    )
    # Assigning rather than updating: an empty saved context is a real snapshot,
    # and must not resurrect keys the application deleted.
    runtime_workflow.context = merged

    serialization_hooks.restore_command_contexts(
        runtime_workflow, context_section.get("command_contexts") or {}
    )
    return dict(record.runtime or {})


def warn_pinned_once(channel_id: str, workflow_path: str, reason: str) -> None:
    """One line per (workflow, reason). Pinned is a steady state, not an event."""
    key = (workflow_path, reason)
    if key in _pin_warned:
        return
    _pin_warned.add(key)
    logger.warning(
        f"Channel {channel_id} cannot be checkpointed and will not be evicted: "
        f"{reason}"
    )


def reset_warnings() -> None:
    _pin_warned.clear()
    serialization_hooks.reset_warnings()
