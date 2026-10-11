"""
Agent integration module for fastWorkflow.
Provides workflow tool agent functionality for intelligent tool selection.
"""

import functools
import os
import re
import time
import traceback
from datetime import datetime, timezone

import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
from dspy.adapters.types.tool import Tool
from dspy.experimental import Choice, TypeSafe

import fastworkflow
from fastworkflow import tracing
from fastworkflow.utils.logging import logger
from fastworkflow.workflow_execution_context import CommandCancelledError
from fastworkflow.utils import dspy_utils
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_executor import AlreadyInContextError, CommandNotFoundError
from fastworkflow.context_navigation import (
    enters_current_context, render_already_in_context, unavailable_command_message)
from fastworkflow.utils.react import AskUserSuspend
from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter
from fastworkflow.observation_offloading.agent import build_tool_agent
from fastworkflow.observation_offloading.search import search_memory as search_observation
from fastworkflow.observation_offloading.labels import annotated_observation
from fastworkflow.observation_offloading.state import (
    archive_for_path,
    current_execute_alias,
    observability_db_path,
    scope_for_host,
)
from fastworkflow.context_identity import context_clause_for

_FINAL_ANSWER_DESC = "The answer from the evidence gathered: one line for each part of the request, giving what was found with its names and uids, or saying plainly that it was not found or not reached. Never claim work the trajectory does not show, and do not say the task is complete."

class WorkflowAgentSignature(dspy.Signature):
    """
    Carefully review the user request, then execute the next steps using available tools for building the final answer.
    Address every part of the request before finishing.
    A command output offloaded to memory is not lost: every observation of this turn is normally restored in full
    when the final answer is written. If the answer's evidence limit is reached, the oldest observations are not
    restored and the answer names them. If you need a value from an offloaded observation to choose your next
    step, run its command again; never re-run commands just to collect rows for the final answer.
    """
    user_query = dspy.InputField(desc="The natural language user query.")
    final_answer = dspy.OutputField(desc=_FINAL_ANSWER_DESC)

def run_progress_check(chat_session_obj, trigger: str, *, done_only: bool = False) -> str:
    """What's done and what remains, from the agent's current view of its trajectory.

    done_only: return just the "what's done" summary, without the remaining steps.
    """
    return build_query_with_next_steps(
        "",
        chat_session_obj, with_agent_inputs_and_trajectory=True,
        planner_lm=getattr(chat_session_obj, '_current_planner_lm', None),
        trace_trigger=trigger,
        done_only=done_only,
    )


def _append_action_record(chat_session_obj, record: dict) -> None:
    """Append to session-scoped action log (WEC or ChatSession delegating to core).

    Duck-typed like _append_turn_output; no-ops gracefully if neither exposes the
    action log. The cwd action.jsonl fallback has been retired — the
    in-process log is the only sink, and the observability DB is the post-mortem
    record.
    """
    if hasattr(chat_session_obj, "append_action_log"):
        chat_session_obj.append_action_log(record)
        return
    core = getattr(chat_session_obj, "_core", None)
    if core is not None and hasattr(core, "append_action_log"):
        core.append_action_log(record)


def _append_turn_output(chat_session_obj, command_output) -> None:
    """Append a CommandOutput to the turn accumulator (WEC or ChatSession delegating to core).

    Duck-typed like _append_action_record; no-ops gracefully if the accumulator
    methods are not available yet.
    """
    if hasattr(chat_session_obj, "append_turn_output"):
        chat_session_obj.append_turn_output(command_output)
        return
    core = getattr(chat_session_obj, "_core", None)
    if core is not None and hasattr(core, "append_turn_output"):
        core.append_turn_output(command_output)


def _append_ask_user_entry(chat_session_obj, question: str):
    """Append an unanswered ask_user entry to the turn accumulator (duck-typed).

    Only called on the Topology-A blocking path (_ask_user_tool). Topology-B
    suspension (AskUserSuspend) does NOT call this — the WEC core appends the
    unanswered entry itself at suspend time, so this split avoids double-appends.
    No-ops gracefully if the accumulator methods are not available yet.
    """
    if hasattr(chat_session_obj, "append_ask_user_entry"):
        return chat_session_obj.append_ask_user_entry(question)
    core = getattr(chat_session_obj, "_core", None)
    if core is not None and hasattr(core, "append_ask_user_entry"):
        return core.append_ask_user_entry(question)
    return None


def _complete_ask_user_entry(chat_session_obj, answer: str) -> None:
    """Record the user's answer on the pending ask_user entry (duck-typed).

    No-ops gracefully if the accumulator methods are not available yet.
    """
    if hasattr(chat_session_obj, "complete_ask_user_entry"):
        chat_session_obj.complete_ask_user_entry(answer)
        return
    core = getattr(chat_session_obj, "_core", None)
    if core is not None and hasattr(core, "complete_ask_user_entry"):
        core.complete_ask_user_entry(answer)


def _what_can_i_do(chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Returns a list of available commands, including their names and parameters.
    """
    return _executor_commands_text(chat_session_obj.get_active_workflow())


def _executor_commands_text(current_workflow, exclude: frozenset[str] = frozenset()) -> str:
    """The active context's commands in full (minus *exclude*), then one line per other context."""
    commands = CommandMetadataAPI.get_command_display_text(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name=current_workflow.current_command_context_name,
        exclude=exclude,
    )
    other_contexts = CommandMetadataAPI.get_other_contexts_text(
        subject_workflow_path=current_workflow.folderpath,
        active_context_name=current_workflow.current_command_context_name,
        navigation_workflow=current_workflow,
        exclude=exclude,
    )
    return f"{commands}\n\n{other_contexts}" if other_contexts else commands

def _refresh_agent_available_commands(host, exclude: frozenset[str] = frozenset()) -> None:
    """Re-scope the active ReAct agent's ``available_commands`` to the CURRENT context.

    Invoked by the workflow's context-change observer (registered in WEC agent init), so it
    fires only on an actual context switch. This is the EXECUTOR's scoped view
    (_what_can_i_do); it never touches the planner's full map. No-ops if there is no active
    agent or it was invoked without available_commands.

    ``host`` is duck-typed for ChatSession and WorkflowExecutionContext: resolve the agent
    via the public ``workflow_tool_agent`` property first, then ``_workflow_tool_agent``.
    Resolve the workflow via ``get_active_workflow()`` with an ``app_workflow`` /
    ``_app_workflow`` fallback so a missing contextvar stack does not skip the refresh.
    """
    agent = getattr(host, "workflow_tool_agent", None)
    if agent is None:
        agent = getattr(host, "_workflow_tool_agent", None)
    inputs = getattr(agent, "inputs", None)
    if not (isinstance(inputs, dict) and "available_commands" in inputs):
        return

    current_workflow = None
    getter = getattr(host, "get_active_workflow", None)
    if callable(getter):
        current_workflow = getter()
    if current_workflow is None:
        current_workflow = getattr(host, "app_workflow", None)
    if current_workflow is None:
        current_workflow = getattr(host, "_app_workflow", None)
    if current_workflow is None:
        return

    inputs["available_commands"] = _executor_commands_text(current_workflow, exclude)


_REPEAT_REASON = "it just returned the same result again"


class _RepeatGuard:
    """One executor run's repeat memory; lives in the agent's ``run_state``.

    ``pending`` is set by a repeat in the step now running and becomes
    ``blocked_command`` (name, reason) at the end of that step, so it governs
    exactly the next step.
    """

    def __init__(self) -> None:
        self.last_results: dict[tuple[str, str], str] = {}
        self.pending: tuple[str, str] | None = None
        self.blocked_command: tuple[str, str] | None = None


def _repeat_guard(chat_session_obj) -> _RepeatGuard | None:
    run_state = getattr(getattr(chat_session_obj, "workflow_tool_agent", None), "run_state", None)
    if run_state is None:
        return None
    return run_state.setdefault("repeat", _RepeatGuard())


def _set_blocked_command(chat_session_obj, guard: _RepeatGuard, block: tuple[str, str] | None) -> None:
    guard.blocked_command = block
    _refresh_agent_available_commands(chat_session_obj, exclude=frozenset({block[0]}) if block else frozenset())


def _end_repeat_step(chat_session_obj) -> None:
    """End-of-step hook: a repeat in the step that just ran blocks the NEXT step only."""
    guard = _repeat_guard(chat_session_obj)
    if guard is None:
        return
    upcoming, guard.pending = guard.pending, None
    if upcoming != guard.blocked_command:
        _set_blocked_command(chat_session_obj, guard, upcoming)


def _clear_blocked_command(chat_session_obj) -> None:
    """Drop the block now: ask_user suspends the run, so the end-of-step hook would not run."""
    guard = _repeat_guard(chat_session_obj)
    if guard is not None and guard.blocked_command is not None:
        _set_blocked_command(chat_session_obj, guard, None)


def _display_context_name(context: str) -> str:
    return "global" if context == "*" else context


def _command_name_matches(token: str, name: str) -> bool:
    """Whether an agent's command token names *name* (qualified or short form)."""
    return token.lower() == (name if "/" in token else name.split("/")[-1]).lower()


def core_command_names() -> frozenset[str]:
    """The framework's core command names, as the routing registry lists them (qualified
    by context): the IntentDetection and ErrorCorrection contexts of the internal
    command_metadata_extraction workflow.

    Read from the routing registry on every call rather than memoised: the registry
    can be cleared and rebuilt, and a stale answer would outlive it.
    """
    cme = fastworkflow.RoutingRegistry.get_definition(
        fastworkflow.get_internal_workflow_path("command_metadata_extraction"))
    return frozenset(name
                     for context in ("IntentDetection", "ErrorCorrection")
                     for name in cme.get_command_names(context))


def _explicit_agent_command(command: str, workflow) -> str:
    """Resolve the agent tool's command token against its current command surface.

    This tool takes command names, not natural-language intents. Never let a
    command name that is unavailable here fall through to fuzzy/cache/classifier
    substitution: name the contexts that do have it, and the commands that
    reach one of them from here, instead. A token that names no command in any
    context of the workflow is not a command name, so it goes to the assistant
    NLU unchanged. Parameter extraction still runs as before.
    """
    parts = command.strip().split(maxsplit=1)
    token = parts[0].lstrip("/") if parts else ""
    app = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
    current_context = workflow.current_command_context_name
    available = set(app.get_command_names(current_context)) | core_command_names()
    matches = [name for name in available if _command_name_matches(token, name)]
    if not matches:
        home_contexts = {
            context for context, names in app.contexts.items()
            if any(_command_name_matches(token, name) for name in names)}
        if not home_contexts:
            return command
        # A context another home context inherits from is a base, not
        # somewhere to navigate to; name only the most specific ones.
        inherited = set().union(*(app.context_model.inherited_base_contexts(c)
                                  for c in home_contexts))
        home_contexts = sorted(home_contexts - inherited)
        if enters_current_context(workflow, token, current_context):
            raise AlreadyInContextError(
                render_already_in_context(token, current_context, workflow))
        raise CommandNotFoundError(unavailable_command_message(
            workflow, token, current_context, home_contexts))
    if len(matches) != 1:
        raise CommandNotFoundError(
            f"Command {token!r} is ambiguous in context "
            f"{_display_context_name(current_context)!r}; it matches "
            f"{', '.join(repr(m) for m in sorted(matches))}. Use the qualified name.")
    # The CME exact-prefix matcher consumes short names. Do not permit a
    # qualified token to collapse onto a different command with the same tail.
    short = matches[0].split("/")[-1]
    if sum(name.split("/")[-1].lower() == short.lower() for name in available) != 1:
        raise CommandNotFoundError(f"Ambiguous command name {short!r} in current context")
    return short + (" " + parts[1] if len(parts) > 1 else "")


def _commands_in(command: str, names: set[str]) -> list[str]:
    """The lines of *command* that start with a known command name (short, lowercased)."""
    found = []
    for line in (raw.strip() for raw in command.splitlines()):
        token = re.split(r"[\s<]", line, maxsplit=1)[0].lstrip("/")
        if line and token.split("/")[-1].lower() in names:
            found.append(line)
    return found


def _execute_workflow_query(command: str, chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    Executes the command and returns either a response, or a clarification request.
    Use the "what_can_i_do" tool to get details on available commands, including their names and parameters. Fyi, values in the 'examples' field are fake and for illustration purposes only.
    Commands must be formatted using plain text for command name followed by XML tags enclosing parameter values (if any) as follows: command_name <param1_name>param1_value</param1_name> <param2_name>param2_value</param2_name> ...
    Don't use this tool to respond to a clarification requests in PARAMETER EXTRACTION ERROR state
    """
    # Refuse a bundle of several commands before anything runs: intent detection
    # would execute only the first one and silently drop the rest.
    active = chat_session_obj.get_active_workflow()
    if active is not None:
        app = fastworkflow.RoutingRegistry.get_definition(active.folderpath)
        known = {name.split("/")[-1].lower()
                 for names in app.contexts.values() for name in names}
        known |= {name.split("/")[-1].lower() for name in core_command_names()}
        known.discard("wildcard")
        commands = _commands_in(command, known)
        if len(commands) >= 2:
            return (f"You sent {len(commands)} commands in one call; nothing ran. "
                    f"Send one command per call: {commands[0]} first.")

    # A command blocked for this step (it just repeated its own result) is refused unrun.
    guard = _repeat_guard(chat_session_obj)
    token = (command.split() or [""])[0].lstrip("/").split("/")[-1].lower()
    if guard is not None and guard.blocked_command and token == guard.blocked_command[0].lower():
        return (f"{guard.blocked_command[0]} is unavailable for this step: "
                f"{guard.blocked_command[1]}. Use what it returned, or do something else.")
    ran_in = active.current_command_context_name if active is not None else None

    # Emit trace event before execution
    if chat_session_obj.command_trace_queue is not None:
        chat_session_obj.command_trace_queue.put(fastworkflow.CommandTraceEvent(
        direction=fastworkflow.CommandTraceEventDirection.AGENT_TO_WORKFLOW,
        raw_command=command,
        command_name=None,
        parameters=None,
        response_text=None,
        success=None,
        timestamp_ms=int(time.time() * 1000),
        turn_key=tracing.get_turn_key(chat_session_obj),
        ))

    # fw.agent.tool_call span — deliberately OUTSIDE the trace-queue guard:
    # the sink is reached via the WEC/ChatSession core, not the transport-queue
    # contract, so queue-less embedders still trace [R28].
    span = tracing.start_span(
        chat_session_obj,
        tracing.SPAN_AGENT_TOOL_CALL,
        kind=tracing.KIND_TOOL,
        attributes={"raw_command": command},
    )

    # §12.1.1's shared capture for the agent-tool path (arch §12.0 deltas 1-2/4).
    # This was the one fw.agent.tool_call emitter it had not reached, which is why
    # the agent-tool, resumed-turn and distillation rows of the conformance matrix
    # could not claim P0 — and why a single contract version for this span name
    # would have described three different attribute sets.
    #
    # The projection is `tracing`'s, not a copy of the WorkflowExecutionContext
    # methods: the same record written three ways is three things to drift.
    # Resolving the workflow is gated on a span having opened, so with
    # observability off this seam costs a `start_span` that already declined. One
    # workflow reference serves both context reads, matching the sibling sites —
    # the type is re-read after execution, so a context the command MOVED is
    # still recorded as a transition.
    workflow = tracing.active_workflow(chat_session_obj) if span is not None else None
    context_before = tracing.context_before(span, workflow)

    # Directly invoke the command without going through queues
    # This allows the agent to synchronously call workflow tools
    from fastworkflow.command_executor import CommandExecutor, _annotation
    started = datetime.now(timezone.utc)
    try:
        resolved_command = _explicit_agent_command(command, chat_session_obj.get_active_workflow())
        command_output = CommandExecutor.invoke_command(chat_session_obj, resolved_command)
    except BaseException as e:
        # BaseException, not CommandCancelledError. This arm used to name only
        # that one exception, and the comment beside it even observed that
        # AskUserSuspend "subclasses BaseException, so `except Exception` below
        # never catches it either" — and then nothing caught it here at all. So
        # an agent-mode ask_user unwound straight through this frame to the
        # react loop and the fw.agent.tool_call span opened above was NEVER
        # closed: it leaked onto the parenting stack, every later span in the
        # turn nested under it, and it never got an end time or a status.
        # fix-ajv.21.
        if tracing.is_control_signal(e):
            tracing.end_span(
                chat_session_obj,
                span,
                status=tracing.status_for_dispatch_exception(e),
            )
            raise
        if not isinstance(e, Exception):
            # Any other BaseException (KeyboardInterrupt, SystemExit): close the
            # span so it does not leak, but do not build a failure CommandOutput
            # for it — that is reserved for real command failures, and this
            # frame must not dress an interpreter-level signal as one.
            tracing.end_span(
                chat_session_obj,
                span,
                status=tracing.STATUS_ERROR,
                attributes={"error_type": type(e).__name__},
            )
            raise
        if isinstance(e, CommandNotFoundError):
            # Routing guidance, not a command failure: no command ran, and the
            # caller hands the agent the message as its observation. Recording
            # it as a failed CommandOutput put a traceback artifact in the
            # turn's answer and marked the turn unsuccessful.
            tracing.end_span(
                chat_session_obj,
                span,
                status=tracing.STATUS_ERROR,
                attributes={"error_type": type(e).__name__},
            )
            raise
        # Capture the failed tool call as a CommandOutput(success=False).
        # This block must never mask the original exception.
        try:
            try:
                error_message = str(e)[:1000]
            except Exception:
                error_message = repr(e)[:1000]
            # Routing has almost always completed by the time a command raises,
            # and command_executor stamps what it had resolved onto the
            # exception. These stay empty only when the failure really did
            # pre-empt routing, or when invoke_command was replaced wholesale
            # (tests/test_turn_result_capture.py does), hence the annotation
            # defaults rather than a declared attribute. fix-ajv.16 FW-2.
            failure_output = fastworkflow.CommandOutput(
                command_name=_annotation(e, "_fw_command_name") or "",
                workflow_name=_annotation(e, "_fw_workflow_name") or "",
                context=_annotation(e, "_fw_context") or "",
                command_call_id=_annotation(e, "_fw_call_id"),
                # Explicitly False, not left to the name: this is the output
                # whose routed name could collide with `ask_user`. fix-ajv.17.
                ask_user_entry=False,
                command_response=
                    fastworkflow.CommandResponse(
                        response=f"Execution error: {e!r}"[:500],
                        success=False,
                        artifacts={
                            "error_type": type(e).__name__,
                            "error_message": error_message,
                            "traceback": traceback.format_exc()[:4000],
                        },
                    ),
                started_at=started,
                duration_ms=int(
                    (datetime.now(timezone.utc) - started).total_seconds() * 1000
                ),
            )
            _append_turn_output(chat_session_obj, failure_output)
        except Exception:
            pass  # capture failed — swallow, never mask the original exception
        tracing.end_span(
            chat_session_obj,
            span,
            status=tracing.STATUS_ERROR,
            attributes={"error_type": type(e).__name__},
        )
        raise

    command_output.started_at = started
    command_output.duration_ms = int(
        (datetime.now(timezone.utc) - started).total_seconds() * 1000
    )
    _append_turn_output(chat_session_obj, command_output)

    # Emit trace event after execution
    # Extract command name and parameters from command_output
    name = command_output.command_name
    params = command_output.command_parameters

    # Live path: typed Pydantic instance. After cold-rehydrate of turn outputs,
    # command_parameters is the dumped dict [A10]. Accept both.
    if params is None:
        params_dict = None
    elif hasattr(params, "model_dump"):
        params_dict = params.model_dump()
    elif isinstance(params, dict):
        params_dict = params
    else:
        params_dict = None

    # Extract response text
    response_text = ""
    if command_output.command_response.response:
        response_text = command_output.command_response.response
    else:
        response_text = "Command executed successfully but produced no output."

    # Same command text, same context, same response as this run's previous call: block it next step.
    if guard is not None:
        key = (ran_in, " ".join(command.split()))
        if guard.last_results.get(key) == response_text:
            guard.pending = ((name or token).split("/")[-1], _REPEAT_REASON)
        guard.last_results[key] = response_text

    tracing.end_span(
        chat_session_obj,
        span,
        status=tracing.STATUS_OK if command_output.success else tracing.STATUS_ERROR,
        command_name=name or None,
        context=command_output.context or None,
        attributes={
            "response_text": response_text,
            "success": bool(command_output.success),
            **tracing.capture_attributes(
                span, command_output, context_before, workflow
            ),
        },
    )

    if chat_session_obj.command_trace_queue is not None:
        chat_session_obj.command_trace_queue.put(fastworkflow.CommandTraceEvent(
            direction=fastworkflow.CommandTraceEventDirection.WORKFLOW_TO_AGENT,
            raw_command=None,
            command_name=name,
            parameters=params_dict,
            response_text=response_text,
            success=bool(command_output.success),
            timestamp_ms=int(time.time() * 1000),
            turn_key=tracing.get_turn_key(chat_session_obj),
        ))

    # Append executed action to the session action log for external consumers
    # (agent mode only): distillation compares teacher/student passes off it and
    # final-answer synthesis summarizes it.
    record = {
        "command": command,
        "command_name": name,
        "parameters": params_dict,
        "response": response_text
    }
    _append_action_record(chat_session_obj, record)

    # Check workflow context to determine if we're in an error state that needs specialized handling
    cme_workflow = chat_session_obj.cme_workflow
    nlu_stage = cme_workflow.context.get("NLU_Pipeline_Stage")

    # Intent ambiguity / misunderstanding: the NLU's reply already lists the
    # candidate commands (ambiguity) or this context's commands (misunderstanding),
    # so the agent chooses from it. Abort first so its next command is routed
    # fresh rather than read as an answer to the clarification prompt.
    if nlu_stage in (
        fastworkflow.NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
        fastworkflow.NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION,
    ):
        abort_confirmation = _execute_workflow_query('abort', chat_session_obj=chat_session_obj)
        return f'{response_text}\n{abort_confirmation}'
    # Handle parameter extraction errors with abort
    if nlu_stage == fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION:
        abort_confirmation = _execute_workflow_query('abort', chat_session_obj=chat_session_obj)
        # Thread the active planner LM so replanning uses the same planner LM
        # as the current turn (critical for distillation: otherwise
        # replans silently fall back to LLM_PLANNER instead of the teacher/student LM).
        planner_lm = getattr(chat_session_obj, '_current_planner_lm', None)
        return build_query_with_next_steps(
            f'{response_text}\n{abort_confirmation}',
            chat_session_obj, with_agent_inputs_and_trajectory=True,
            planner_lm=planner_lm,
            trace_trigger="parameter_extraction_error",
        )

    # Clean up the context flag after command execution
    workflow = chat_session_obj.get_active_workflow()
    if "is_user_command" in workflow.context:
        del workflow.context["is_user_command"]

    return response_text


def _apply_reply_intent(agent, intent: str) -> None:
    """Take ask_user away after a reply that stops the asking; leave only finish after an abort."""
    if intent == "stop_asking":
        agent.disabled_tools = agent.disabled_tools | {"ask_user"}
    elif intent == "abort":
        agent.disabled_tools = set(agent.tools) - {"finish"}


def _post_ask_user_response(
    clarification_request: str,
    user_response: str,
    chat_session_obj,
) -> str:
    """
    Steps 2-4 after the user answers an ask_user clarification (Topology-A parity).

    Appends the dialog to the session action log, sets raw_user_message, and replans.
    Returns the observation string for the ReAct loop.
    """
    _append_action_record(
        chat_session_obj,
        {
            "agent_query": clarification_request,
            "user_response": user_response,
        },
    )
    # Complete the pending unanswered ask_user entry with the user's answer.
    # Both topologies route resumes through here.
    _complete_ask_user_entry(chat_session_obj, user_response)

    workflow = chat_session_obj.get_active_workflow()
    if workflow:
        workflow.context["raw_user_message"] = user_response

    _apply_reply_intent(
        chat_session_obj.workflow_tool_agent,
        classify_user_reply(user_response, clarification_request),
    )

    planner_lm = getattr(chat_session_obj, '_current_planner_lm', None)
    return build_query_with_next_steps(
        user_response,
        chat_session_obj,
        with_agent_inputs_and_trajectory=True,
        planner_lm=planner_lm,
        trace_trigger="ask_user_response",
    )


def _ask_user_tool(clarification_request: str, chat_session_obj: fastworkflow.ChatSession) -> str:
    """
    If the missing_information_guidance_tool does not help and only as the last resort, request clarification for missing information from the human user. 
    The clarification_request must be plain text without any formatting.
    Note that using the wrong command name can produce missing information errors. Double-check with the missing_information_guidance_tool to verify that the correct command name is being used 
    """
    user_queue = chat_session_obj.user_message_queue
    output_queue = chat_session_obj.command_output_queue
    if user_queue is None:
        raise CommandCancelledError("ask_user requires a user_message_queue (not available in this context)")

    active = chat_session_obj.get_active_workflow()
    workflow_name = active.folderpath.split('/')[-1] if active else ""
    command_output = fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(response=clarification_request),
        workflow_name=workflow_name,
    )
    if output_queue is not None:
        output_queue.put(command_output)
        # Preserve the transport contract: the output must be visible before the
        # trace sentinel releases the CLI to read it. In v3.0 the queued payload
        # becomes a partial TurnOutput, but the pairing and ordering stay the same.
        trace_queue = chat_session_obj.command_trace_queue
        if trace_queue is not None:
            trace_queue.put(None)

    # Topology A: append the unanswered ask_user entry before blocking.
    # (Topology B appends it inside the WEC core at AskUserSuspend time.)
    _append_ask_user_entry(chat_session_obj, clarification_request)

    # Topology A (blocking): a persistent worker thread runs the agent and a human
    # is expected to answer, so we block indefinitely. (Topology B has no queue and
    # suspends via AskUserSuspend instead — see the ask_user tool closure.)
    user_query = user_queue.get()

    return _post_ask_user_response(
        clarification_request, user_query, chat_session_obj
    )

def initialize_workflow_tool_agent(chat_session: fastworkflow.ChatSession, max_iters: int = 25,
                                   on_step_complete=None):
    """
    Initialize and return a DSPy ReAct agent that exposes individual MCP tools.
    Each tool expects a single query string for its specific tool.

    Args:
        chat_session: fastworkflow.ChatSession instance
        max_iters: Maximum iterations for the ReAct agent
        on_step_complete: Optional callback(step_idx, trajectory) -> bool for
            step-by-step interception (distillation). Return False to stop early.

    Returns:
        DSPy ReAct agent configured with workflow tools
    """
    chat_session_obj = chat_session
    if not chat_session_obj:
        raise ValueError("chat session cannot be null")

    AgentSignature = WorkflowAgentSignature

    # **_ignored on the zero-argument tools: a small model often passes one
    # anyway, and that should not cost it a traceback for an observation.
    # Wrapped below with an empty args schema, so the model is shown none.
    def what_can_i_do(**_ignored) -> str:
        """
        Returns a list of available commands, including their names and parameters
        """
        return _what_can_i_do(chat_session_obj=chat_session_obj)

    def run_execute_workflow_query(command: str) -> str:
        # Check if this command originated from user input (iteration_counter == 0)
        # Set flag in workflow context so validate_extracted_parameters can access it
        is_user_command = chat_session_obj.workflow_tool_agent.iteration_counter <= 0 
        workflow = chat_session_obj.get_active_workflow()
        if workflow:
            workflow.context["is_user_command"] = is_user_command
        
        # Retry logic for workflow execution
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return _execute_workflow_query(command, chat_session_obj=chat_session_obj)
            except AlreadyInContextError as e:
                return reenter_from_parent(command, e)
            except CommandNotFoundError as e:
                # Deterministic and recoverable: the message says where to go.
                return str(e)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    message = f"Terminate immediately! Exception processing {command}: {str(e)}"
                    logger.critical(message)                    
                    return message
                # Continue to next attempt
                logger.warning(f"Attempt {attempt + 1} failed for command '{command}': {str(e)}")

    def reenter_from_parent(command: str, refusal: AlreadyInContextError) -> str:
        """Open another instance of the context the agent is in: go_up, then the command.

        A weak model told "go_up first" retries the same command in place many
        times over. Re-opening the SAME instance is refused as before, since
        going up and back in would only repeat it.
        """
        workflow = chat_session_obj.get_active_workflow()
        name, _, instance = context_clause_for(workflow).partition(" ")
        uid = instance.split(" ")[0]
        if uid and uid in command:
            return str(refusal)
        _execute_workflow_query("go_up", chat_session_obj=chat_session_obj)
        try:
            return (f"(Left the current {name} with go_up first.)\n"
                    + _execute_workflow_query(command, chat_session_obj=chat_session_obj))
        except CommandNotFoundError as again:
            return str(again)

    def execute_workflow_query(command: str) -> str:
        """
        Takes just a single argument called 'command'.
        Executes the command and returns either a response, or a clarification request.
        Use the "what_can_i_do" tool to get details on available commands, including their names and parameters. Fyi, values in the 'examples' field are fake and for illustration purposes only.
        Commands must be formatted using plain text for command name followed by XML tags enclosing parameter values (if any) as follows: command_name <param1_name>param1_value</param1_name> <param2_name>param2_value</param2_name> ...
        Don't use this tool to respond to a clarification requests in PARAMETER EXTRACTION ERROR state
        """
        # The context this command RUNS IN, taken before dispatch: a command that
        # moves the context is evidence about the context it ran in.
        alias = current_execute_alias(chat_session_obj.workflow_tool_agent)
        workflow = chat_session_obj.get_active_workflow()
        clause = context_clause_for(workflow)
        context_before = getattr(workflow, "current_command_context", None)
        response = run_execute_workflow_query(command)
        if alias is None:
            return response
        workflow_after = chat_session_obj.get_active_workflow()
        context_changed = getattr(workflow_after, "current_command_context", None) is not context_before
        # Name where the command left the agent: a weak model otherwise reads
        # the context the command ran in as the one it is now in.
        now_in = context_clause_for(workflow_after) if context_changed else None
        return annotated_observation(alias, clause, response,
                                     context_changed=context_changed, now_in=now_in)

    def ask_user(clarification_request: str) -> str:
        """
        Only as the last resort, request clarification for missing information from the human user. Ask at most once about the same thing.
        The clarification_request must be plain text without any formatting.
        Note that using the wrong command name can produce missing information errors. Double-check with the what_can_i_do tool to verify that the correct command name is being used 
        """
        _clear_blocked_command(chat_session_obj)
        # reset iteration counter, everytime we ask the user
        # reset to -1, because we are dual purposing (iteration_counter <= 0) to check
        # if command passed to execute_workflow_query() originated either:
        # externally or inside agent loop immediately after an ask_user()_call 
        chat_session_obj.workflow_tool_agent.iteration_counter = -1
        if chat_session_obj.user_message_queue is not None:
            return _ask_user_tool(clarification_request, chat_session_obj=chat_session_obj)
        raise AskUserSuspend(clarification_request)

    def search_memory(question: str, alias: str) -> str:
        """
        Answers a question about ONE earlier execute_workflow_query observation.
        alias is the O-number printed on that observation's first line ("Observation O8 (...)") or in its offload label, e.g. O8.
        Goal is to locate values needed to choose the next step: a uid, a count, one field, whether an item is present.
        Offloaded observations are restored for the final answer, so do not search just to collect rows to report.
        question must be clear, relevant and complete sentence
        """
        return search_observation(
            question, alias,
            scope=scope_for_host(chat_session_obj),
            selected_archive=archive_for_path(observability_db_path(chat_session_obj)),
        )

    tools = [
        Tool(what_can_i_do, args={}),
        execute_workflow_query,
        ask_user,
        # search_memory disabled 2026-10-09: in live runs it retrieved nothing useful; kept for later.
        # search_memory,
    ]

    agent = build_tool_agent(
        chat_session_obj,
        AgentSignature,
        tools,
        max_iters=max_iters,
        on_step_complete=on_step_complete,
    )
    agent.on_step_end = lambda: _end_repeat_step(chat_session_obj)
    return agent


def _previous_turn_summaries_for_planner(chat_session_obj: fastworkflow.ChatSession) -> str:
    """Every completed turn summary for the initial planner (not mid-turn replans)."""
    history = getattr(chat_session_obj, "conversation_history", None)
    messages = getattr(history, "messages", None) if history is not None else None
    if not messages:
        return ""
    lines: list[str] = []
    for turn_number, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            continue
        summary = message.get("conversation summary") or message.get("conversation_summary")
        if summary:
            lines.append(f"Turn {turn_number}: {summary}")
    return "\n".join(lines)


#: More sub-tasks than this is a list too long to fan out: run the request whole.
MAX_SUBTASKS = 10

_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s+(.*\S)\s*$")
#: Typographic hyphens a model writes into names it was asked to copy exactly.
_HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})


class SplitIntoSubtasks(dspy.Signature):
    """
    Split the user's request into the separate requests it bundles together.
    A sub-task is a request the user could have sent on its own, with its own answer. It is NOT a step: never break one request into the steps needed to carry it out.
    - job_types lists kinds of job a request in this workflow may contain, and how each one splits. Use it only to decide how to split: every sub-task is made from the user's own words, never from job_types text, and a job type the user did not ask for is not a sub-task.
    - Sentences that only give background (why the user needs it, what is happening to a group) or announce what follows are not sub-tasks.
    - Copy the user's own words, and every name exactly as written. A name that contains "and" is one name: keep it whole. Change only what a sub-task needs to make sense on its own: replace words that point to another part of the request ("it", "both", "each of them", "the same treatment", "that group", he/she) with what they refer to, using names.
    - When the same work is asked for several named items, make one sub-task per item. What is asked about one item stays together in that item's sub-task.
    - Work asked for a group as a whole (every member of a group, a whole list) is one sub-task.
    - An item together with the things it refers to is one sub-task, even when the request says it in two sentences.
    - Never add, drop or reinterpret any work the user asked for.
    - A request that asks for one thing is returned unchanged, as a single sub-task.

    Example 1
    user_query: The job on item Alpha and the job on item Beta both fail: check each item and the thing it feeds. Three people need a new account: Pat Doe, Lee Kim and Sam Roe. Go through the whole Group North list too, every member's record. The case 'Queue stalls at night' is still open, so work it and hand it to one of the owners it names. Max Ong is moving teams, so a new locker for Max Ong.
    subtasks:
    1. The job on item Alpha fails: check item Alpha and the thing it feeds.
    2. The job on item Beta fails: check item Beta and the thing it feeds.
    3. Pat Doe needs a new account.
    4. Lee Kim needs a new account.
    5. Sam Roe needs a new account.
    6. Go through the whole Group North list, every member's record.
    7. The case 'Queue stalls at night' is still open: work it and hand it to one of the owners it names.
    8. Max Ong is moving teams: a new locker for Max Ong.

    Example 2
    user_query: Show me Pat Doe's record, its owner and the items attached to it.
    subtasks:
    1. Show me Pat Doe's record, its owner and the items attached to it.

    Example 3
    user_query: Find item Alpha and tell me who uses it, then list the cases it has.
    subtasks:
    1. Find item Alpha and tell me who uses it.
    2. List the cases item Alpha has.
    """
    job_types: str = dspy.InputField(desc="kinds of job a request may contain, and how each one splits")
    user_query: str = dspy.InputField()
    subtasks: str = dspy.OutputField(desc="the sub-tasks as a numbered list, one per line")


@functools.lru_cache(maxsize=None)
def load_skills(workflow_folderpath: str) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """``(name, description, steps)`` for each ``_skills/<name>/SKILL.md`` of the workflow.

    A skill file is plain text: its first line describes the job, every other
    non-empty line is one step. The folder name is the skill's name.
    """
    root = os.path.join(workflow_folderpath, "_skills")
    skills = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else ():
        path = os.path.join(root, name, "SKILL.md")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            if lines:
                skills.append((name, lines[0], tuple(lines[1:])))
    return tuple(skills)


@functools.lru_cache(maxsize=None)
def _skill_file_text(workflow_folderpath: str, filename: str) -> str | None:
    """Text of the optional ``_skills/<filename>`` of the workflow, or None when absent."""
    path = os.path.join(workflow_folderpath, "_skills", filename)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read()


@functools.lru_cache(maxsize=None)
def _split_signature(workflow_folderpath: str) -> type[dspy.Signature]:
    """``SplitIntoSubtasks``, instructed by the workflow's ``_skills/split_instructions.md`` when it has one."""
    instructions = _skill_file_text(workflow_folderpath, "split_instructions.md")
    if instructions is None:
        return SplitIntoSubtasks
    return SplitIntoSubtasks.with_instructions(instructions)


def _job_types_for(workflow_folderpath: str | None) -> str:
    """One ``- name: description`` line per skill of the workflow; "(none)" without skills."""
    skills = load_skills(workflow_folderpath) if workflow_folderpath else ()
    if not skills:
        return "(none)"
    return "\n".join(f"- {name}: {description}" for name, description, _ in skills)


#: A matched skill's steps are offered to the planner only at or above this Jev
#: probability (measured offline: at 0.5, 96.6% of sub-tasks get a template, 1.1% a wrong one).
SKILL_MATCH_THRESHOLD = 0.5


@functools.lru_cache(maxsize=None)
def _jev_lm():
    """The Jev (dspy.experimental TypeSafe) LM, or None without a JEV_API_KEY."""
    key = fastworkflow.get_env_var("JEV_API_KEY", default=None)
    return TypeSafe(api_key=key, cache=True) if key else None


def _choice_value_and_probability(choice) -> tuple[str, float]:
    value = str(choice.value)
    return value, float((choice.probabilities or {}).get(value, choice.confidence))


@functools.lru_cache(maxsize=None)
def _skill_matcher(workflow_folderpath: str):
    """``(predictor, jev_lm)`` choosing a sub-task's skill, or None without skills or a Jev key."""
    skills = load_skills(workflow_folderpath)
    jev_lm = _jev_lm()
    if not skills or jev_lm is None:
        return None
    options = tuple((name, description) for name, description, _ in skills) + (
        ("none", "None of the job types fits this sub-task."),)

    class MatchSkill(dspy.Signature):
        """Decide which job type a sub-task asks for."""

        subtask: str = dspy.InputField(desc="one sub-task from a user request")
        job_types: str = dspy.InputField(desc="the available job types, one per line as '- name: description'")
        job: Choice[options] = dspy.OutputField(
            desc="Which job type does this sub-task ask for? 'none' if none of them fits.")

    return dspy.Predict(MatchSkill), jev_lm


def match_skill(text: str, workflow_folderpath: str) -> tuple[str, tuple[str, ...]] | None:
    """``(name, steps)`` of the skill ``text`` asks for, when Jev is confident; None otherwise.

    Never raises: a failed match only means the planner gets no template.
    """
    try:
        matcher = _skill_matcher(workflow_folderpath)
        if matcher is None:
            return None
        predictor, jev_lm = matcher
        skills = load_skills(workflow_folderpath)
        job_types = "\n".join(f"- {name}: {description}" for name, description, _ in skills)
        with dspy.context(lm=jev_lm):
            choice = predictor(subtask=text, job_types=job_types).job
        value, probability = _choice_value_and_probability(choice)
        logger.info(f"skill match for {text!r}: {value} (confidence {probability:.2f})")
        steps = {name: steps for name, _, steps in skills}
        if value in steps and probability >= SKILL_MATCH_THRESHOLD:
            return value, steps[value]
        return None
    except Exception:  # noqa: BLE001 - a template is an optimisation; never fail the turn
        logger.warning("could not match the sub-task to a skill", exc_info=True)
        return None


def plan_template_for(text: str, workflow_folderpath: str) -> str:
    """The matched skill's steps as a planner template for ``text``; "" when no skill matches."""
    matched = match_skill(text, workflow_folderpath)
    if matched is None:
        return ""
    _name, steps = matched
    lines = "\n".join(f"- {step}" for step in steps)
    return (f"A job like this is usually done as:\n{lines}\n"
            "Adapt this to the request: drop steps it does not need and add steps it asks for.")


#: A reply's non-answer intent applies only at or above this Jev probability.
REPLY_INTENT_THRESHOLD = 0.5
REPLY_INTENTS = (
    ("answer", "The reply answers the question or gives direction for the work to go on."),
    ("stop_asking", "The reply declines to answer or says not to ask again, but does not ask to stop the work."),
    ("abort", "The reply asks to stop, cancel or abort the work."),
)


class ClassifyReply(dspy.Signature):
    """Decide what a user's reply to the agent's clarification request does."""

    question: str = dspy.InputField(desc="the clarification request the agent asked the user")
    reply: str = dspy.InputField(desc="the user's reply")
    intent: Choice[REPLY_INTENTS] = dspy.OutputField(desc="What does the reply do?")


def classify_user_reply(reply: str, question: str) -> str:
    """The reply's intent: "abort", "stop_asking", or "answer" (the default).

    Any other intent needs Jev to be confident. Never raises: without a Jev key,
    or on a failure, the reply is taken as an answer.
    """
    jev_lm = _jev_lm()
    if jev_lm is None:
        return "answer"
    try:
        with dspy.context(lm=jev_lm):
            choice = dspy.Predict(ClassifyReply)(question=question, reply=reply).intent
        value, probability = _choice_value_and_probability(choice)
        logger.info(f"reply intent for {reply!r}: {value} (confidence {probability:.2f})")
        if value != "answer" and probability >= REPLY_INTENT_THRESHOLD:
            return value
        return "answer"
    except Exception:  # noqa: BLE001 - the reply is then taken as an answer; never fail the turn
        logger.warning("could not classify the reply intent", exc_info=True)
        return "answer"


#: Phrases that point at other parts of the request instead of naming them; a sub-task with one is broken.
_UNRESOLVED_REFERENCE = re.compile(
    r"\b(?:each of them|both of them|all of them|the same treatment|the same look)\b", re.IGNORECASE)


def _visibly_broken(subtasks: list[str]) -> bool:
    """Two identical sub-tasks, or one that still refers to the rest of the request ("each of them")."""
    return len(set(subtasks)) < len(subtasks) or any(_UNRESOLVED_REFERENCE.search(t) for t in subtasks)


def split_into_subtasks(user_query: str, planner_lm=None,
                        workflow_folderpath: str | None = None) -> list[str]:
    """The request as independent sub-tasks, in order; ``[user_query]`` when it is not split.

    One call on the planner LM, with no command catalog: the split is about the
    request, not about how to carry it out. It is shown the workflow's skills
    (``_skills``), as the kinds of job it may contain.
    Any failure, an empty answer, or more than MAX_SUBTASKS sub-tasks leaves the
    request whole.
    """
    if planner_lm is None:
        planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
    job_types = _job_types_for(workflow_folderpath)
    # Temperature 0, so the same request splits the same way; a split that is
    # visibly broken is retried once at 0.7, which also misses the LM cache.
    # A split whose only flaw is a leftover reference ("each of them") is still
    # better than none, so it is kept when the retry does not fix it.
    usable = None
    for temperature in (0.0, 0.7):
        try:
            with dspy.context(lm=planner_lm.copy(temperature=temperature),
                              adapter=CommandsSystemPreludeAdapter()):
                signature = _split_signature(workflow_folderpath) if workflow_folderpath else SplitIntoSubtasks
                text = dspy.ChainOfThought(signature)(
                    job_types=job_types, user_query=user_query).subtasks or ""
        except Exception:  # noqa: BLE001 - splitting is an optimisation; never fail the turn
            logger.warning("could not split the request; running it whole", exc_info=True)
            return [user_query]
        subtasks = [match.group(1).translate(_HYPHENS) for line in text.splitlines()
                    if (match := _NUMBERED_LINE.match(line))]
        if not subtasks or len(subtasks) > MAX_SUBTASKS:
            return [user_query]
        if not _visibly_broken(subtasks):
            return subtasks
        if usable is None and len(set(subtasks)) == len(subtasks):
            usable = subtasks
    return usable or [user_query]


def build_query_with_next_steps(user_query: str,
    chat_session_obj: fastworkflow.ChatSession, with_agent_inputs_and_trajectory: bool = False,
    planner_lm = None,
    trace_trigger: str | None = None,
    planner_user_query: str | None = None,
    plan_template: str | None = None,
    done_only: bool = False) -> str:
    """
    Generate a todo list.
    Return a string that combine the user query and todo list

    Args:
        user_query: The user's natural language query
        chat_session_obj: The active chat session
        with_agent_inputs_and_trajectory: Whether to include agent trajectory for replanning
        planner_lm: Optional planner LM to use (if None, uses LLM_PLANNER from env)
        trace_trigger: What re-triggered planning mid-turn (e.g.
            "ask_user_response", "parameter_extraction_error"). None means the
            turn's initial plan; set, it marks the span fw.planner.replan.
        planner_user_query: Current-turn text for the planner LLM only. When prior
            turns are supplied via ``previous_turn_summaries``, pass the raw user
            message here so ``user_query`` (often a refined string that already
            embeds recent summaries) is not duplicated in the planner prompt.
        plan_template: Optional skill steps for the planner only, appended to the
            request in the planner prompt (not in the returned string). Used on
            the turn's initial plan, not on a replan.
        done_only: With a trajectory, return only the "what's done" summary.
    """
    current_workflow = chat_session_obj.get_active_workflow()
    base_docstring = """
    Carefully review the user_query and generate a next steps sequence based only on available commands.
    Walk the graph of commands based on the 'available_from' hints to build the most appropriate command sequence.
    Use names and values exactly as written in the user_query; never abbreviate or reword them.

    IMPORTANT: 9 times out of 10 information can be found via available commands. However, when generating the plan:
    - If required information is missing and cannot be found via commands, explicitly specify in the plan that the user needs to be consulted
    - If confirmation is needed before proceeding, explicitly specify in the plan that user confirmation is required
    """
    # The plain-text planner.
    class TaskPlannerTextSignature(dspy.Signature):
        __doc__ = base_docstring
        user_query: str = dspy.InputField()
        next_steps: str = dspy.OutputField(desc="task descriptions as a numbered list of short sentences separated by line breaks")

    class TaskPlannerTextWithTrajectoryAndAgentInputsSignature(dspy.Signature):
        __doc__ = base_docstring
        agent_inputs: dict = dspy.InputField()
        agent_trajectory: dict = dspy.InputField()
        user_response: str = dspy.InputField(desc="the user's latest response, empty for a progress check")
        whats_done: str = dspy.OutputField(desc="concise summary of what the agent_trajectory has accomplished so far, including key values found")
        next_steps: str = dspy.OutputField(desc="the remaining tasks of the original plan, revised for what is done, as a numbered list of short sentences separated by line breaks")

    class TaskPlannerTextWithPreviousTurnSummariesSignature(dspy.Signature):
        __doc__ = base_docstring
        user_query: str = dspy.InputField()
        previous_turn_summaries: str = dspy.InputField(
            desc="Summaries of every completed turn before this one, oldest first"
        )
        next_steps: str = dspy.OutputField(desc="task descriptions as a numbered list of short sentences separated by line breaks")

    available_commands = CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=current_workflow.folderpath,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name=current_workflow.current_command_context_name,
        navigation_workflow=current_workflow,
    )

    # Use provided planner_lm if available (distillation mode), else build from env
    if planner_lm is None:
        planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
    agent_adapter = CommandsSystemPreludeAdapter()

    # fw.planner.plan for the turn's initial plan, fw.planner.replan for
    # mid-turn re-planning (trace_trigger names what re-triggered it).
    span = tracing.start_span(
        chat_session_obj,
        tracing.SPAN_PLANNER_REPLAN if trace_trigger else tracing.SPAN_PLANNER_PLAN,
        kind=tracing.KIND_LLM,
        attributes={
            "model": getattr(planner_lm, "model", None),
            "replan_trigger": trace_trigger,
        },
    )

    def with_template(text: str) -> str:
        return f"{text}\n\n{plan_template}" if plan_template else text

    def plan_with():
        if with_agent_inputs_and_trajectory:
            workflow_tool_agent = chat_session_obj.workflow_tool_agent
            task_planner_func = dspy.ChainOfThought(TaskPlannerTextWithTrajectoryAndAgentInputsSignature)
            agent_inputs, agent_trajectory = workflow_tool_agent.planner_view()
            cleaned_agent_inputs = {k: v for k, v in agent_inputs.items() if k != "available_commands"}
            return task_planner_func(
                agent_inputs = cleaned_agent_inputs,
                agent_trajectory = agent_trajectory,
                user_response = user_query,
                available_commands=available_commands) # Note that this is not part of the signature. It is extra metadata that will be picked up by the CommandsSystemPreludeAdapter
        previous_turn_summaries = (
            _previous_turn_summaries_for_planner(chat_session_obj)
            if trace_trigger is None
            else ""
        )
        if previous_turn_summaries:
            task_planner_func = dspy.ChainOfThought(
                TaskPlannerTextWithPreviousTurnSummariesSignature
            )
            planner_query = (
                planner_user_query
                if planner_user_query is not None
                else user_query
            )
            return task_planner_func(
                user_query=with_template(planner_query),
                previous_turn_summaries=previous_turn_summaries,
                available_commands=available_commands,
            )
        task_planner_func = dspy.ChainOfThought(TaskPlannerTextSignature)
        return task_planner_func(
            user_query=with_template(user_query),
            available_commands=available_commands) # Note that this is not part of the signature. It is extra metadata that will be picked up by the CommandsSystemPreludeAdapter

    plan_text = ""
    try:
        with dspy.context(lm=planner_lm, adapter=agent_adapter):
            prediction = plan_with()
            plan_text = prediction.next_steps or ""
    except BaseException:
        tracing.end_span(chat_session_obj, span, status=tracing.STATUS_ERROR)
        raise
    tracing.end_span(
        chat_session_obj,
        span,
        attributes={
            "plan": plan_text,
            "plan_source": "text" if plan_text else "none",
        },
    )

    # A mid-turn replan with nothing remaining still reports what's done and says to finish.
    if not plan_text and not with_agent_inputs_and_trajectory:
        return user_query

    generated_plan = plan_text.split()
    # Capture the generated plan for distillation when a capture list is present
    # on the session (set only during a DistillationSession planning pass).
    planning_capture = getattr(chat_session_obj, '_planning_steps_capture', None)
    if planning_capture is not None:
        from fastworkflow.distillation import PlanningStep
        planning_capture.append(PlanningStep(
            step_number=len(planning_capture),
            user_query=user_query,
            generated_plan=generated_plan,
            reasoning=getattr(prediction, 'reasoning', ''),
        ))

    steps_formatted = " ".join(generated_plan)
    if with_agent_inputs_and_trajectory:
        whats_done = getattr(prediction, "whats_done", "") or ""
        if done_only:
            return whats_done
        sections = [f"User Query:\n{user_query}"] if user_query else []
        sections.append(f"What's done:\n{whats_done}")
        sections.append(
            f"What remains (execute these next steps):\n{steps_formatted}"
            if steps_formatted else
            "Nothing left to do. Call finish now."
        )
        return "\n\n".join(sections)
    return f"{user_query}\n\nExecute these next steps:\n{steps_formatted}"
