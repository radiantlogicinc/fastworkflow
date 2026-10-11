"""Build the workflow tool agent. Observation offloading is how it is built."""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import fastworkflow
from fastworkflow import context_budget
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    UnavailableHandleArchive,
)
from fastworkflow.observation_offloading.compact import compact_trajectory
from fastworkflow.observation_offloading.manifest import install_span_policy
from fastworkflow.observability.prompt_slots import install_prompt_slot_enrichment
from fastworkflow.observation_offloading.state import (
    archive_for_path,
    observability_db_path,
    prune_once,
    record_event,
    scope_for_host,
)
from fastworkflow.utils.react import fastWorkflowReAct

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERS = 25


def _trajectory_target_bytes(chat_session: Any) -> int:
    """The trajectory's share of the executor prompt budget.

    The budget bounds the whole ReAct prompt, so the rest of the prompt (the
    instructions, inputs and command list) is taken off it first.
    """
    budget = context_budget.trajectory_max_bytes()
    agent = getattr(chat_session, "workflow_tool_agent", None)
    if agent is None:
        return budget
    try:
        overhead = agent.prompt_overhead_bytes(agent.inputs)
    except Exception as error:  # noqa: BLE001
        logger.debug("prompt overhead unavailable, using the whole budget: %s", error)
        return budget
    return max(budget - overhead, 0)


def build_compacting_step(
    chat_session: Any,
    *,
    selected_archive: RuntimeHandleArchive,
    on_step_complete: Optional[Callable[[int, dict[str, Any]], bool]] = None,
    describe_output: Optional[Callable[[str, str], str]] = None,
) -> Callable[[int, dict[str, Any]], bool]:
    """The ReAct on_step_complete hook: compact, then defer to the caller's hook.

    ``_run_loop`` invokes this with no try/except of its own, so anything raised
    here would turn a successful tool call into a full-turn abort. Offloading is
    an optimisation; a failure to compact is logged and recorded, and the step
    continues with its observation left inline.
    """

    def compacting_step(idx: int, trajectory: dict[str, Any]) -> bool:
        scope = scope_for_host(chat_session)
        try:
            compact_trajectory(
                trajectory,
                step_index=idx,
                scope=scope,
                selected_archive=selected_archive,
                describe_output=describe_output,
                packed_target_bytes=_trajectory_target_bytes(chat_session),
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "observation offloading skipped compaction at step %d: %s: %s",
                idx, type(error).__name__, error,
            )
            record_event(
                {
                    "kind": "compaction_failed",
                    "step_index": idx,
                    "error": type(error).__name__,
                    "detail": str(error)[:300],
                },
                scope=scope, store=selected_archive,
            )
        if on_step_complete is not None and not on_step_complete(idx, trajectory):
            return False
        return True

    return compacting_step


def open_handle_archive(archive_path: str) -> Any:
    """The turn archive, or an inert stand-in and one event saying why.

    Opening or creating the observability database is the FIRST thing agent construction does
    that touches the disk, and it used to be the only one allowed to fail the
    turn: a read-only state root, a permission bit or a file that is not a
    database raised out of ``RuntimeHandleArchive`` and no agent was built at
    all, so the persist-before-label recovery -- the design's answer to exactly
    this class of failure -- never ran.

    Evidence storage is an optimisation, so an initialisation failure is
    degraded through the policy the WRITES already have rather than a second one
    invented here: the observation stays inline, the refusal is recorded, and
    the turn proceeds. Reported once, at the seam that failed; the per-alias
    ``archive_refused`` event that follows is the
    ordinary write-degradation record and says the same thing per observation.
    """
    try:
        opened = archive_for_path(archive_path)
    except Exception as error:  # noqa: BLE001
        unavailable = UnavailableHandleArchive(archive_path, error)
        logger.warning(
            "observation offloading has no archive at %s: %s: %s; "
            "observations stay inline for this agent",
            unavailable.db_path, type(error).__name__, error,
        )
        record_event(
            {
                "kind": "archive_unavailable",
                "db_path": unavailable.db_path,
                "reason": "initialization_failed_observations_inline",
                "error": type(error).__name__,
                "detail": str(error)[:300],
            }
        )
        return unavailable
    prune_once(opened.db_path)
    return opened


def describe_command_output(chat_session: Any, command: str, response: str) -> str:
    """Resolve authored output fields from the command that actually produced this text."""
    core = getattr(chat_session, "_core", chat_session)
    records = getattr(core, "action_log", [])
    record = next((r for r in reversed(records)
                   if r.get("command") == command and r.get("response") == response), None)
    if record is None:
        return ""
    try:
        workflow = chat_session.get_active_workflow()
        routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        metadata = CommandMetadataAPI._extract_signature_info(
            record["command_name"], routing, routing)
        fields = metadata.get("outputs", [])
        return "; ".join(f"{field['name']}: {field['description']}"
                         for field in fields if field.get("description"))
    except Exception:
        # Missing metadata must never prevent persistence or turn success.
        return ""


def build_tool_agent(
    chat_session: Any,
    signature: Any,
    tools: list[Callable[..., Any]],
    *,
    max_iters: int,
    on_step_complete=None,
) -> Any:
    """Construct the ReAct agent once from ``tools``.

    The DSPy signature build (tool wrapping, instruction assembly, the react and
    extract predictors) happens exactly once. Every observation-offloading step
    derives its turn scope from *chat_session* when it runs, so a suspended turn
    resumed in this or another process reads and writes the same handles.
    """
    install_span_policy()
    install_prompt_slot_enrichment()
    scope = scope_for_host(chat_session)
    # In the workflow's own observability database, so the evidence a turn
    # can be replayed from lives, and is erased, where the turn's record is.
    archive_path = observability_db_path(chat_session)
    # An archive that cannot be opened degrades; it does not stop the agent
    # being built.
    selected_archive = open_handle_archive(archive_path)

    compacting_step = build_compacting_step(
        chat_session,
        selected_archive=selected_archive,
        on_step_complete=on_step_complete,
        describe_output=lambda command, response: describe_command_output(chat_session, command, response),
    )
    agent = fastWorkflowReAct(
        signature,
        tools=tools,
        max_iters=int(max_iters or DEFAULT_MAX_ITERS),
        on_step_complete=compacting_step,
    )
    agent.describe_output = lambda command, response: describe_command_output(chat_session, command, response)
    record_event(
        {
            "kind": "agent_installed",
            "max_iters": agent.max_iters,
            "tools": sorted(agent.tools),
        },
        scope=scope, store=selected_archive,
    )
    return agent
