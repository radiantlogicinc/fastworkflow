"""Build the workflow tool agent. Observation offloading is how it is built."""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import fastworkflow
from fastworkflow import state_paths
from fastworkflow.agent_runtime import build_turn_runtime
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
    UnavailableHandleArchive,
)
from fastworkflow.observation_offloading.compact import compact_trajectory
from fastworkflow.observation_offloading.continuation import (
    DEFAULT_MAX_ITERS,
    MAX_FORCED_REPLANS,
    StructuredContinuationReAct,
)
from fastworkflow.observation_offloading.manifest import install_span_policy
from fastworkflow.observation_offloading.search import search_memory
from fastworkflow.observation_offloading.search_router import router_for_workflow
from fastworkflow.observation_offloading.state import (
    prune_once,
    record_event,
    scope_for_host,
)

logger = logging.getLogger(__name__)


def _scope_for_session(chat_session: Any) -> RuntimeHandleScope:
    """The turn scope for this session. One implementation, in state.py, so a
    handle stored from a command's frame and one stored from the ReAct loop are
    stored under the same scope."""
    return scope_for_host(chat_session)


def build_compacting_step(
    agent_ref: Callable[[], Any],
    *,
    fallback_scope: RuntimeHandleScope,
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
        agent = agent_ref()
        scope = getattr(agent, "continuation_scope", None) or fallback_scope
        # ido-7qd: the agent's ledger, not a recount of this trajectory. The
        # alias printed and archived here is then the same one the command
        # inside the step already declared and stamped under, in a process that
        # resumed the turn as much as in the one that started it.
        pairs = getattr(agent, "execute_ordinal_pairs", None)
        try:
            compact_trajectory(
                trajectory,
                scope=scope,
                selected_archive=selected_archive,
                ordinal_offset=int(getattr(agent, "truncated_execute_steps", 0) or 0),
                executes=pairs(trajectory) if callable(pairs) else None,
                describe_output=describe_output,
            )
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "observation offloading skipped compaction at step %d: %s: %s",
                idx, type(error).__name__, error,
            )
            record_event(
                {
                    "kind": "compaction_failed",
                    "scope_id": scope.scope_id,
                    "step_index": idx,
                    "error": type(error).__name__,
                    "detail": str(error)[:300],
                }
            )
        if on_step_complete is not None and not on_step_complete(idx, trajectory):
            return False
        return True

    return compacting_step


def open_handle_archive(
    archive_path: str, *, scope: Optional[RuntimeHandleScope] = None
) -> Any:
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
    ``archive_refused`` / ``offload_refused`` events that follow are the
    ordinary write-degradation record and say the same thing per observation.
    """
    try:
        opened = RuntimeHandleArchive(archive_path)
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
                "scope_id": getattr(scope, "scope_id", None),
                "db_path": unavailable.db_path,
                "reason": "initialization_failed_observations_inline",
                "error": type(error).__name__,
                "detail": str(error)[:300],
            }
        )
        return unavailable
    prune_once(opened.db_path)
    return opened


def current_search_reasoning(agent: Any) -> str:
    """Read the current search step, not an earlier completed tool's thought."""
    trajectory = agent.current_trajectory
    indexes = [int(key.removeprefix("tool_name_")) for key in trajectory
               if key.startswith("tool_name_") and key.removeprefix("tool_name_").isdigit()]
    if not indexes:
        return ""
    index = max(indexes)
    if trajectory.get(f"tool_name_{index}") != "search_memory":
        return ""
    return str(trajectory.get(f"thought_{index}") or "")


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


def describe_command_inputs(chat_session: Any, command: str) -> list[dict[str, Any]]:
    """Declared inputs of the command most recently run under the name *command*.

    *command* is the bare command word the archive recorded (``show_holders``);
    the action log maps it to the qualified command that actually ran.
    """
    core = getattr(chat_session, "_core", chat_session)
    records = getattr(core, "action_log", [])
    record = next((r for r in reversed(records)
                   if str(r.get("command_name") or "").rsplit("/", 1)[-1] == command), None)
    if record is None:
        return []
    try:
        workflow = chat_session.get_active_workflow()
        routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        metadata = CommandMetadataAPI._extract_signature_info(
            record["command_name"], routing, routing)
        return list(metadata.get("inputs", []))
    except Exception:
        # Missing metadata must never prevent a search.
        return []


def build_tool_agent(
    chat_session: Any,
    signature: Any,
    tools: list[Callable[..., Any]],
    *,
    max_iters: int,
    on_step_complete=None,
) -> Any:
    """Construct the ReAct agent once: a StructuredContinuationReAct.

    The DSPy signature build (tool wrapping, instruction assembly, the react and
    extract predictors) happens exactly once, with ``search_memory`` appended to
    ``tools``. Observation offloading is how fastWorkflow runs a tool agent, not
    a mode it can be put into.
    """
    install_span_policy()
    # The scope is re-resolved by the agent at every forward(), so the turn_key
    # it carries is the turn actually running. This one is only the fallback
    # for a step that fires before the first forward() bound a scope.
    scope = _scope_for_session(chat_session)
    # In the workflow's own observability database, so the evidence a turn
    # can be replayed from lives, and is erased, where the turn's record is.
    getter = getattr(chat_session, "get_active_workflow", None)
    active_workflow = getter() if callable(getter) else None
    workflow_path = str(getattr(active_workflow, "folderpath", "") or "")
    archive_path = state_paths.observability_db(workflow_path)
    # An archive that cannot be opened degrades; it does not stop the agent
    # being built (ido-t5x). ``build_compacting_step`` catches compaction
    # failures, and this is the one storage failure that used to happen too
    # early for it to catch.
    selected_archive = open_handle_archive(archive_path, scope=scope)
    router = router_for_workflow(workflow_path)
    turn_runtime = build_turn_runtime(scope, archive=selected_archive)
    agent: Any = None

    compacting_step = build_compacting_step(
        lambda: agent,
        fallback_scope=scope,
        selected_archive=selected_archive,
        on_step_complete=on_step_complete,
        describe_output=lambda command, response: describe_command_output(chat_session, command, response),
    )

    def scoped_search_memory(question: str, alias: str) -> str:
        """Answer a question inside ONE earlier execute_workflow_query observation.

        alias is the O-number printed on that observation's first line
        ("Observation O42 (execute_workflow_query)", or
        "Observation O42 (execute_workflow_query, in Account 28c5aeb5... Alan
        Cooper)" when the command ran inside a context) or named in its offload
        label. Pass only the O-number. Any printed O-number works, whether its
        result is still shown in full or was replaced by a label. Never pass a
        step number. An alias that was never printed is a miss, not another
        observation.

        The "in <Context> <instance>" part of that line says WHICH instance the
        observation is about: a listing produced inside an account belongs to
        that account even though its rows do not repeat the account's id.

        The search also knows WHICH context instance the framework recorded for
        that observation, and is told it separately from the evidence, so a
        question about that subject can be answered from rows that never repeat
        its id. It will not adopt a subject your question assumes: when no
        subject was recorded it says so instead of guessing one.

        Two different bounds apply, and the right response to each is the
        opposite of the other.

        The READ is bounded: at most a budget of the observation's LEADING
        UTF-8 bytes reaches the search model, derived from that model's own
        context window (a quarter of it: 131,072 bytes at the reference window). A long
        observation is therefore searched as a prefix, not in full. An answer
        produced from a partial read says so and states the bytes it did not
        read; nothing missing from it is thereby absent from the observation.
        Every search of an observation starts at byte zero, so re-asking with a
        narrower question reads the same bytes and cannot reach the rest. To
        reach the rest, re-run the command that produced the observation with a
        narrower filter or a smaller page and search the NEW observation. If
        the search model refuses even that bounded prompt as too long, the call
        says so and returns no evidence; repeating it sends the same bytes, so
        do not retry it unchanged.

        The ANSWER is bounded separately: a long answer is cut to fit the
        trajectory, says so, and reports how many bytes it left out. That one
        IS worth asking again on the same observation with a narrower question,
        because the evidence was read and only its presentation was cut.

        Do not search to collect rows for the final answer. Every observation
        of this turn, offloaded or not, is restored in full when the final
        answer is written, so a table you only need to REPORT needs no search.
        Search for the specific values you need to choose your NEXT step: a uid
        to open, whether a named item is present, a count, one field. A request
        for a whole table is usually declined or cut and costs a step.

        An observation short enough to print whole is returned verbatim instead
        of searched, together with the observations in this turn that match the
        question better.
        """

        current = getattr(agent, "continuation_scope", None) or scope
        return search_memory(
            question, alias, reasoning=current_search_reasoning(agent),
            scope=current, selected_archive=selected_archive,
            router=router,
            describe_inputs=lambda command: describe_command_inputs(chat_session, command),
            trace_host=chat_session,
        )

    scoped_search_memory.__name__ = "search_memory"
    agent = StructuredContinuationReAct(
        signature,
        tools=[*tools, scoped_search_memory],
        max_iters=int(max_iters or DEFAULT_MAX_ITERS),
        on_step_complete=compacting_step,
        scope_factory=lambda: _scope_for_session(chat_session),
        turn_runtime=turn_runtime,
    )
    agent.continuation_scope_id = scope.scope_id
    agent.observation_archive = selected_archive
    agent.turn_runtime = turn_runtime
    agent.describe_output = lambda command, response: describe_command_output(chat_session, command, response)
    if agent.evaluation_control_overrides:
        record_event(
            {
                "kind": "evaluation_controls",
                "scope_id": scope.scope_id,
                "finish_reminders_enabled": agent.finish_reminders_enabled,
                "overrides": dict(agent.evaluation_control_overrides),
            }
        )
    record_event(
        {
            "kind": "agent_installed",
            "max_iters": agent.max_iters,
            "max_forced_replans": MAX_FORCED_REPLANS,
            "tools": sorted(agent.tools),
            "scope_id": scope.scope_id,
        }
    )
    return agent
