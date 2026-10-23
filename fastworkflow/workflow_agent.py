"""
Agent integration module for fastWorkflow.
Provides workflow tool agent functionality for intelligent tool selection.
"""

import dataclasses
import functools
import json
import os
import re
import threading
import time
import traceback
from datetime import datetime, timezone

import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
from dspy.adapters.types.tool import Tool
from dspy.experimental import Choice, TypeSafe

import fastworkflow
from fastworkflow import tracing
from fastworkflow import skill_commands
from fastworkflow.skill_commands import SkillCommand, SkillFormatError, plan_skill_call
from fastworkflow.split_merge import definite_descriptions, merge_subtasks, unnamed_subjects, with_referent
from fastworkflow.utils.logging import logger
from fastworkflow.workflow_execution_context import CommandCancelledError
from fastworkflow.utils import dspy_utils
from fastworkflow.command_metadata_api import CommandMetadataAPI
from fastworkflow.command_executor import AlreadyInContextError, CommandNotFoundError
from fastworkflow.context_navigation import (
    enters_current_context, render_already_in_context, unavailable_command_message)
from fastworkflow.context_navigation import home_contexts as context_homes
from fastworkflow.utils.react import AskUserSuspend
from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter
from fastworkflow.utils.dspy_logger import watch_cut_off_replies
from fastworkflow.observation_offloading.agent import build_tool_agent
from fastworkflow.observation_offloading.search import search_memory as search_observation
from fastworkflow.observation_offloading.labels import annotated_observation
from fastworkflow.observation_offloading.archive import command_run_record
from fastworkflow.observation_offloading.state import (
    archive_for_path,
    current_execute_alias,
    note_command_run,
    note_folded_command,
    observability_db_path,
    scope_for_host,
)
from fastworkflow.context_identity import context_clause_for, context_identity

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


def _hold_command_run(chat_session_obj, command_output, context_clause,
                      context_name=None) -> None:
    """Hold this command's run for the in-flight execute step's archive row.

    Its name, success, context, parameters and ``Signature.Output``
    (``CommandResponse.output``) are stored beside the step's text when the
    step is archived; the observation the agent reads is not touched. Outside
    an agent execute step there is no row to store it in, and nothing is held.
    Never raises: this record must not be able to fail a command.
    """
    try:
        name = command_output.command_name
        if not name:
            return
        alias = current_execute_alias(getattr(chat_session_obj, "workflow_tool_agent", None))
        if alias is None:
            return
        note_command_run(scope_for_host(chat_session_obj), alias, command_run_record(
            command_name=name,
            success=bool(command_output.success),
            context_clause=context_clause,
            context_name=context_name,
            parameters=command_output.command_parameters,
            output=command_output.command_response.output,
        ))
    except Exception:  # noqa: BLE001
        logger.debug("could not hold the command run for its archive row", exc_info=True)


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
    ``blocked_command`` (name, reason, command text) at the end of that step, so it
    governs exactly the next step. Only that same command text is blocked: the
    command with other arguments (its next page) still runs.
    """

    def __init__(self) -> None:
        self.last_results: dict[tuple[str, str], str] = {}
        self.pending: tuple[str, str, str] | None = None
        self.blocked_command: tuple[str, str, str] | None = None


def _repeat_guard(chat_session_obj) -> _RepeatGuard | None:
    run_state = getattr(getattr(chat_session_obj, "workflow_tool_agent", None), "run_state", None)
    if run_state is None:
        return None
    return run_state.setdefault("repeat", _RepeatGuard())


def _set_blocked_command(chat_session_obj, guard: _RepeatGuard, block: tuple[str, str, str] | None) -> None:
    guard.blocked_command = block


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
        home_contexts = context_homes(workflow, token)
        if not home_contexts:
            return command
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


# Keys a model nests a command's parameters under, and keys it puts the command
# name under when ``command`` itself is missing (fix-5jss).
_NESTED_ARG_KEYS = ("args", "arguments", "params", "parameters", "kwargs")
_COMMAND_NAME_KEYS = ("command_name", "cmd", "action", "function", "tool")
_COMMAND_NAME_TOKEN = re.compile(r"[A-Za-z_][\w./-]*")
_XML_TAG = re.compile(r"</?[A-Za-z_][\w.-]*>")


def _arg_text(value) -> str:
    """A tool argument as command text: strings as they are, everything else JSON."""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _key_value_text(key: str, value: str) -> str:
    """``key=value``, quoted only where the key=value parser would otherwise cut
    the value: it strips a trailing ``,``/``;`` and a trailing ``(note)``. A
    `` word=`` inside a value ends it whatever the quoting, so ``=`` (or a
    comma inside the value) is no reason to quote."""
    if value[-1] in ",;" or re.search(r"\([^()]*\)\.?$", value):
        quote = '"' if '"' not in value else "'"
        if quote not in value:
            return f"{key}={quote}{value}{quote}"
    return f"{key}={value}"


def fold_tool_kwargs(command, extra: dict,
                     fields: frozenset[str] | None = None) -> tuple[str, list[str]]:
    """Fold keyword arguments the model passed beside ``command`` into the command text.

    The command tool takes one ``command`` string, but small models often pass a
    command's parameters as tool arguments of their own
    (``{"command": "find_item", "name": "Ann Lee"}``), and the tool's arg check
    refused every such call with a traceback the model then repeated (fix-5jss).
    Each extra argument is appended as `` key=value`` (as ``<key>value</key>``
    when the command text already uses XML tags), so the usual parsers bind it
    or the usual refusals say why not:

    - a dict under ``args``/``arguments``/``params``/``parameters``/``kwargs`` is
      unpacked into its keys; any other value there is a bare positional value;
    - with no command text, the name may come from ``command_name``, ``cmd``,
      ``action``, ``function`` or ``tool`` when that value is one token;
    - lists and dicts become JSON text; ``None`` and empty strings carry no value;
    - given the command's parameter *fields*, a lone argument whose key is none
      of them is sent as a bare value (``uid=ab12`` for a ``item_uid`` field
      becomes ``ab12``), for the positional parser and its guards.

    Returns the folded command, and one note per argument that carried no value
    so that nothing is dropped unrecorded.
    """
    text = command.strip() if isinstance(command, str) else _arg_text(command) if command else ""
    pairs: list[tuple[str | None, object]] = []
    for key, value in extra.items():
        if key in _NESTED_ARG_KEYS and isinstance(value, dict):
            pairs.extend(value.items())
        elif key in _NESTED_ARG_KEYS:
            pairs.append((None, value))
        else:
            pairs.append((key, value))
    if not text:
        for i, (key, value) in enumerate(pairs):
            if (key in _COMMAND_NAME_KEYS and isinstance(value, str)
                    and _COMMAND_NAME_TOKEN.fullmatch(value.strip())):
                text = value.strip()
                del pairs[i]
                break
    if fields is not None and len(pairs) == 1 and pairs[0][0] not in fields:
        pairs = [(None, pairs[0][1])]
    xml = bool(_XML_TAG.search(text))
    parts = [text] if text else []
    empty: list[str] = []
    for key, value in pairs:
        value_text = "" if value is None else _arg_text(value)
        if not value_text:
            empty.append(f"{key or 'args'} had no value")
        elif key is None:
            parts.append(value_text)
        elif xml:
            parts.append(f"<{key}>{value_text}</{key}>")
        else:
            parts.append(_key_value_text(key, value_text))
    return " ".join(parts), empty


def _parameter_fields(command: str, workflow) -> frozenset[str] | None:
    """The parameter field names of the command *command* starts with, if it is one
    command of the current context; None when that is not certain."""
    token = (command.split() or [""])[0].lstrip("/")
    if workflow is None or not _COMMAND_NAME_TOKEN.fullmatch(token):
        return None
    try:
        app = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        names = [name for name in app.get_command_names(workflow.current_command_context_name)
                 if _command_name_matches(token, name)]
        if len(names) != 1:
            return None
        parameters = app.get_command_class(names[0], fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
    except Exception:  # noqa: BLE001 - a lookup aid, never a reason to fail the call
        return None
    return frozenset(parameters.model_fields) if parameters is not None else frozenset()


def _execute_workflow_query(command: str, chat_session_obj: fastworkflow.ChatSession,
                            *, internal: bool = False) -> str:
    """
    Executes the command and returns either a response, or a clarification request.
    Use the "what_can_i_do" tool to get details on available commands, including their names and parameters. Fyi, values in the 'examples' field are fake and for illustration purposes only.
    Commands must be formatted using plain text for command name followed by XML tags enclosing parameter values (if any) as follows: command_name <param1_name>param1_value</param1_name> <param2_name>param2_value</param2_name> ...
    Don't use this tool to respond to a clarification requests in PARAMETER EXTRACTION ERROR state

    ``internal`` marks the framework's own calls (the abort that ends an error
    state): the repeat guard neither refuses nor records them, since a refused
    abort leaves the session stuck in a stale parameter round (fix-hia1.13).
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
    if internal:
        guard = None
    if guard is not None and guard.blocked_command and " ".join(command.split()) == guard.blocked_command[2]:
        return (f"{guard.blocked_command[0]} with these arguments is unavailable for this step: "
                f"{guard.blocked_command[1]}. Use what it returned, or do something else.")
    ran_in = active.current_command_context_name if active is not None else None
    try:
        ran_in_clause = context_clause_for(active) if active is not None else ""
        ran_in_name = context_identity(active)[0]
    except Exception:  # noqa: BLE001 - a label hook must not stop the command
        ran_in_clause = ran_in_name = None

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
            guard.pending = ((name or token).split("/")[-1], _REPEAT_REASON, key[1])
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
    _hold_command_run(chat_session_obj, command_output, ran_in_clause, ran_in_name)

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
        abort_confirmation = _execute_workflow_query(
            'abort', chat_session_obj=chat_session_obj, internal=True)
        return f'{response_text}\n{abort_confirmation}'
    # Handle parameter extraction errors with abort
    if nlu_stage == fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION:
        abort_confirmation = _execute_workflow_query(
            'abort', chat_session_obj=chat_session_obj, internal=True)
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

    # ``command`` defaults and ``**extra`` collects anything else, so a call with
    # parameters as tool arguments is folded instead of refused (fix-5jss). The
    # Tool below still shows the model the one ``command`` argument.
    def execute_workflow_query(command: str = "", **extra) -> str:
        """
        Takes just a single argument called 'command'.
        Executes the command and returns either a response, or a clarification request.
        Use the "what_can_i_do" tool to get details on available commands, including their names and parameters. Fyi, values in the 'examples' field are fake and for illustration purposes only.
        Commands must be formatted using plain text for command name followed by XML tags enclosing parameter values (if any) as follows: command_name <param1_name>param1_value</param1_name> <param2_name>param2_value</param2_name> ...
        Don't use this tool to respond to a clarification requests in PARAMETER EXTRACTION ERROR state
        """
        folded_note = ""
        if extra or not isinstance(command, str):
            name = command if isinstance(command, str) else ""
            for key in _COMMAND_NAME_KEYS:
                name = name or (extra[key] if isinstance(extra.get(key), str) else "")
            command, empty = fold_tool_kwargs(
                command, extra, _parameter_fields(name, chat_session_obj.get_active_workflow()))
            # Read by the ReAct loop into the step span's `folded_command`.
            run_state = getattr(chat_session_obj.workflow_tool_agent, "run_state", None)
            if isinstance(run_state, dict):
                run_state["folded_command"] = command
            # Parameters with no command name: refused here, since intent
            # detection would guess a command from ``key=value`` text.
            if not _COMMAND_NAME_TOKEN.fullmatch((command.split() or [""])[0].lstrip("/")):
                return (f"No command name was given, so nothing ran. Put the command name "
                        f"first in the command string, then its parameters: "
                        f"<command_name> {command}".rstrip())
            # The step's archive row keeps this text as its command: the finish
            # check reads what ran, not the bare ``command`` the model sent.
            folded_alias = current_execute_alias(chat_session_obj.workflow_tool_agent)
            if folded_alias is not None:
                note_folded_command(scope_for_host(chat_session_obj), folded_alias, command)
            folded_note =(f"\n(Ran as: {command} -- parameters go inside the command string."
                           + (f" Not used: {'; '.join(empty)}." if empty else "") + ")")
        # The context this command RUNS IN, taken before dispatch: a command that
        # moves the context is evidence about the context it ran in.
        alias = current_execute_alias(chat_session_obj.workflow_tool_agent)
        workflow = chat_session_obj.get_active_workflow()
        clause = context_clause_for(workflow)
        context_before = getattr(workflow, "current_command_context", None)
        response = run_execute_workflow_query(command) + folded_note
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
        Tool(execute_workflow_query, args={"command": {"type": "string"}}),
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

#: Output cap (tokens) of one splitter call; ``LLM_SPLIT_MAX_TOKENS`` overrides it. Without a cap
#: the splitter's reasoning can loop for 100k+ characters, past the LM timeout. A capped answer
#: that does not parse gets DSPy's JSON retry (capped too). A reply cut off at the cap is never
#: used as the split, even when it parses: ``split_into_subtasks`` retries once, then leaves the
#: request whole (fix-rj86). Set from the largest stored splitter output that
#: did not loop: 17.6k characters, about 4,700 tokens at the fewest characters per token measured
#: on long splitter outputs (3.77; fix-ajir.1).
SPLIT_MAX_TOKENS = 4700


def split_max_tokens() -> int:
    """The splitter's output cap in tokens: ``LLM_SPLIT_MAX_TOKENS``, else ``SPLIT_MAX_TOKENS``."""
    return int(fastworkflow.get_env_var("LLM_SPLIT_MAX_TOKENS", int, default=SPLIT_MAX_TOKENS))


def _split_lm(planner_lm, temperature: float):
    """The planner LM as the splitter uses it: ``temperature``, output capped at ``split_max_tokens()``."""
    return planner_lm.copy(temperature=temperature, max_tokens=split_max_tokens())


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


def _read_skill_file(name: str, path: str, workflow_folderpath: str):
    """The ``SkillFile`` of ``_skills/<name>.py``.

    The file is imported as a command file is (``python_utils.get_module`` under the
    workflow's package, so package-relative imports work: ``from ._helpers import ...`` reads
    ``_skills/_helpers.py``). Raises SkillFormatError for a file name that is not a Python
    identifier or a malformed declaration, and the import's own error for a file that does
    not import.
    """
    from fastworkflow.utils import python_utils

    if not name.isidentifier():
        raise SkillFormatError(f"{path}: a skill's name is its file name, a Python identifier "
                               f"(snake_case): rename it to {re.sub(r'[^0-9A-Za-z_]', '_', name)}.py")
    module = python_utils.get_module(path, workflow_folderpath)
    return skill_commands.command_from_module(name, module, path)


def _load_skill_file(name: str, path: str, workflow_folderpath: str):
    """The ``SkillFile`` of ``_skills/<name>.py`` (``_read_skill_file``), or None when it
    cannot be read: a broken skill file costs that skill, not the workflow. The error is
    logged, and ``fastworkflow skill preview`` reports the skill as not loaded.
    """
    try:
        return _read_skill_file(name, path, workflow_folderpath)
    except SkillFormatError as error:
        logger.error(f"skipping skill {name!r}: {error}")
    except Exception:  # noqa: BLE001 - workflow code
        logger.error(f"skipping skill {name!r}: could not import {path}", exc_info=True)
    return None


def _skill_files(root: str) -> list[tuple[str, str]]:
    """``(name, path)`` of each skill file of a workflow's ``_skills`` folder, by name.

    As for command files at the root of ``_commands``: a skill is a file ``_skills/<name>.py``
    and its name is the file stem, a Python identifier (snake_case; ``_read_skill_file``
    refuses another). A ``.py`` file whose name starts with ``_`` is a helper
    the skill files may import, not a skill; ``_split.py`` is the workflow's split settings
    (``split_settings``). Other files are not read. A subfolder is reserved for a skill context (as a
    ``_commands/<Context>/`` folder is): its files are not loaded yet, and it is logged and
    skipped (one starting with ``_`` or ``.``, such as ``__pycache__``, silently).
    """
    out = []
    for entry in sorted(os.listdir(root)) if os.path.isdir(root) else ():
        path = os.path.join(root, entry)
        if os.path.isdir(path):
            if not entry.startswith(("_", ".")):
                logger.warning(f"{path}: skill contexts (subfolders of _skills) are not loaded yet; skipped")
            continue
        stem, ext = os.path.splitext(entry)
        if ext != ".py" or stem.startswith("_"):
            continue
        out.append((stem, path))
    return out


#: Held while a workflow's skills are read, so threads that start together build them once.
_parsed_skills_lock = threading.RLock()


def _parsed_skills(workflow_folderpath: str) -> tuple[
        tuple[str, str, tuple[str, ...], "SkillCommand | None", tuple[str, ...]], ...]:
    """``(name, description, template steps, command or None, plain_utterances)`` per skill;
    see ``load_skills``. Built once per process (``_build_parsed_skills``): a skill file that
    cannot be read is skipped (``_load_skill_file``), so it is not read again each turn."""
    with _parsed_skills_lock:
        return _build_parsed_skills(workflow_folderpath)


@functools.lru_cache(maxsize=None)
def _build_parsed_skills(workflow_folderpath: str) -> tuple[
        tuple[str, str, tuple[str, ...], "SkillCommand | None", tuple[str, ...]], ...]:
    root = os.path.join(workflow_folderpath, "_skills")
    identifiers = identifier_patterns(workflow_folderpath)
    files = []   # (name, SkillFile)
    for name, path in _skill_files(root):
        loaded = _load_skill_file(name, path, workflow_folderpath)
        if loaded is not None:
            files.append((name, loaded))
    out = []
    for name, loaded in files:
        if loaded.command is None:
            out.append((name, loaded.description, loaded.steps, None, loaded.plain_utterances))
            continue
        command = dataclasses.replace(loaded.command, identifiers=identifiers)
        problems = step_command_problems(command, workflow_folderpath)
        if problems:
            logger.error(f"skipping skill {name!r}: " + "; ".join(problems))
            continue
        if command.phase_outputs:
            _warn_phase_sources(command, workflow_folderpath)
        out.append((name, loaded.description, command.template, command, loaded.plain_utterances))
    return tuple(out)


def _warn_phase_sources(command: SkillCommand, workflow_folderpath: str) -> None:
    """Log a warning for each phase output field whose ``"from"`` names no command field.

    Not a load failure: the workflow may add the command's ``Output`` later
    (``skill_commands.check_phase_sources``). A phase whose field is still missing
    when it runs fails then, naming the field; ``fastworkflow skill preview --check``
    fails on it (``phase_source_problems``).
    """
    for warning in phase_source_problems(command, workflow_folderpath):
        logger.warning(f"skill {command.name!r}: phase output source not found (loaded anyway): {warning}")


def step_command_problems(command: SkillCommand, workflow_folderpath: str) -> list[str]:
    """``skill_commands.check_step_commands`` of *command* against the workflow's command
    signatures: a step naming a command the workflow lacks, or a parameter its
    ``Signature.Input`` lacks. A load error (the skill is skipped); empty, with a logged
    warning, when the signatures cannot be read (``fastworkflow skill preview --check``
    still checks)."""
    from fastworkflow.observation_offloading.facts import workflow_signatures

    try:
        signatures = workflow_signatures(workflow_folderpath)
    except Exception as error:  # noqa: BLE001 - checked again by the preview's --check
        logger.warning(f"skill {command.name!r}: could not read the workflow's command signatures "
                       f"to check its steps: {type(error).__name__}: {error}")
        return []
    return skill_commands.check_step_commands(command, signatures)


def phase_source_problems(command: SkillCommand, workflow_folderpath: str) -> list[str]:
    """``skill_commands.check_phase_sources`` of *command* against the workflow's command
    signatures; empty, with a logged warning, when they cannot be read."""
    from fastworkflow.observation_offloading.facts import workflow_signatures

    try:
        signatures = workflow_signatures(workflow_folderpath)
    except Exception as error:  # noqa: BLE001 - the check is advice; the skill still loads
        logger.warning(f"skill {command.name!r}: could not read the workflow's command signatures "
                       f"to check its phase outputs: {type(error).__name__}: {error}")
        return []
    return skill_commands.check_phase_sources(command, signatures)


def load_skills(workflow_folderpath: str) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """``(name, description, steps)`` for each skill of the workflow (``_skills/<name>.py``).

    A skill is defined by the file ``_skills/<name>.py`` (format: ``fastworkflow.skill_commands``;
    which files are skills: ``_skill_files``); the file stem is the skill's name and the
    ``Signature`` docstring its description. A
    skill without parameters or phases gives its fixed ``steps()``; a skill with parameters
    or phases gives its template text, and ``load_skill_commands`` holds its parsed form. A skill
    file that cannot be imported or is malformed is logged as an error and skipped (the
    other skills still load; ``fastworkflow skill preview`` fails on it); a malformed split
    settings file raises SkillFormatError. ``_skills`` also holds the workflow's
    split settings, ``_split.py`` (``split_settings``), and helper modules named ``_<name>.py``.
    """
    return tuple((name, description, steps) for name, description, steps, *_ in _parsed_skills(workflow_folderpath))


def load_skill_commands(workflow_folderpath: str) -> dict[str, SkillCommand]:
    """The workflow's skills called with typed arguments (those with params or phases), by
    name."""
    return {name: command for name, _, _, command, _ in _parsed_skills(workflow_folderpath) if command is not None}


def skill_utterances(workflow_folderpath: str) -> dict[str, tuple[str, ...]]:
    """Each skill's ``Signature.plain_utterances`` (empty when it declares none), by name."""
    return {name: utterances for name, _, _, _, utterances in _parsed_skills(workflow_folderpath)}


def warm_skills(workflow_folderpath: str) -> None:
    """Read the workflow's skills and its commands' utterances once, at startup, so the first
    request does not pay for them (``load_skills``; ``skill_commands.command_utterances``,
    which ``plan_skill_call`` reaches through ``command_texts``). The skill cache is keyed by
    the path string: pass the one the turn path uses, the app workflow's ``folderpath``.
    A workflow without a ``_skills`` folder has nothing to warm. Never raises: a failure is
    logged and the turn path meets it again."""
    if not os.path.isdir(os.path.join(workflow_folderpath, "_skills")):
        return
    for step in (load_skills, skill_commands.command_utterances):
        try:
            step(workflow_folderpath)
        except Exception as error:  # noqa: BLE001 - warm-up must not stop startup
            logger.warning(f"skill warm-up: {step.__name__} failed for {workflow_folderpath}: "
                           f"{type(error).__name__}: {error}")


#: The workflow's split settings file, ``_skills/_split.py`` (see ``split_settings``).
SPLIT_FILE = "_split.py"
#: The fields of ``SplitIntoSubtasks`` a split settings file may describe, as ``Signature.Input``
#: and ``Signature.Output``.
_SPLIT_INPUTS, _SPLIT_OUTPUTS = frozenset({"job_types", "user_query"}), frozenset({"subtasks"})


@dataclasses.dataclass(frozen=True)
class SplitSettings:
    """What a workflow's ``_skills/_split.py`` sets for the split stage; the defaults when absent."""

    #: Replaces ``SplitIntoSubtasks``'s instructions (``Signature``'s docstring), or None.
    instructions: str | None = None
    #: ``(field, description)`` for each splitter field the file describes.
    field_descs: tuple[tuple[str, str], ...] = ()
    #: The workflow's identifier formats (``identifier_patterns``).
    identifier_patterns: tuple[re.Pattern, ...] = ()
    #: What the workflow counts as a record, for the referent step (``resolve_referents``), or None.
    subject_hints: str | None = None


def _split_field_descs(signature, path: str) -> tuple[tuple[str, str], ...]:
    """``(field, description)`` of the ``Signature.Input`` / ``Signature.Output`` fields that have one."""
    import inspect

    from pydantic import BaseModel

    out = []
    for block, allowed in (("Input", _SPLIT_INPUTS), ("Output", _SPLIT_OUTPUTS)):
        model = getattr(signature, block, None)
        if model is None:
            continue
        if not (inspect.isclass(model) and issubclass(model, BaseModel)):
            raise SkillFormatError(f"{path}: Signature.{block} must be a pydantic BaseModel")
        for name, info in model.model_fields.items():
            if name not in allowed:
                raise SkillFormatError(f"{path}: Signature.{block}.{name} is not a splitter "
                                       f"{block.lower()} (one of {', '.join(sorted(allowed))})")
            if info.description is not None:
                out.append((name, info.description))
    return tuple(out)


def split_settings_from_module(module, path: str) -> SplitSettings:
    """The ``SplitSettings`` a split settings file (already imported) declares.

    Every attribute is optional. Raises SkillFormatError for a malformed one (see ``split_settings``).
    """
    import inspect

    instructions, field_descs = None, ()
    signature = getattr(module, "Signature", None)
    if signature is not None:
        if not inspect.isclass(signature):
            raise SkillFormatError(f"{path}: Signature must be a class")
        instructions = inspect.cleandoc(signature.__doc__ or "") or None
        field_descs = _split_field_descs(signature, path)
    patterns = getattr(module, "identifier_patterns", ())
    if isinstance(patterns, (str, re.Pattern)) or not isinstance(patterns, (list, tuple)):
        raise SkillFormatError(f"{path}: identifier_patterns must be a list of compiled patterns (re.compile)")
    for pattern in patterns:
        if not isinstance(pattern, re.Pattern):
            raise SkillFormatError(f"{path}: identifier_patterns holds {pattern!r}; write re.compile({pattern!r})")
    hints = getattr(module, "subject_hints", None)
    if hints is not None and not isinstance(hints, str):
        raise SkillFormatError(f"{path}: subject_hints must be a str")
    return SplitSettings(instructions=instructions, field_descs=field_descs,
                         identifier_patterns=tuple(patterns), subject_hints=hints)


@functools.lru_cache(maxsize=None)
def split_settings(workflow_folderpath: str) -> SplitSettings:
    """The workflow's split settings, from its optional ``_skills/_split.py``; the defaults without one.

    The file configures the split stage (the splitter, the split merge and the referent step); it
    is not a skill (its leading ``_`` keeps ``_skill_files`` from loading it as one). It is
    written as a command file is, and every attribute is optional:

    - ``class Signature``: its docstring replaces ``SplitIntoSubtasks``'s instructions (as
      ``inspect.cleandoc`` gives it). ``Signature.Input`` (fields ``job_types``, ``user_query``)
      and ``Signature.Output`` (field ``subtasks``) are optional pydantic models; a field's
      ``description`` replaces that splitter field's description. A field the splitter does
      not have fails the load.
    - ``identifier_patterns``: a list of compiled regular expressions (``re.compile``), the
      workflow's identifier formats (``identifier_patterns``).
    - ``subject_hints``: a str telling the referent step what the workflow counts as a record
      (``resolve_referents``).

    The file is imported as a skill file is (``python_utils.get_module`` under the workflow's
    package, so package-relative imports work). It is the whole split configuration, so a file
    that cannot be imported (an invalid pattern raises in ``re.compile``) is not skipped, as a
    skill file would be, but fails the load as a malformed skill file does: the error is logged
    and SkillFormatError raised naming the file.
    """
    from fastworkflow.utils import python_utils

    path = os.path.join(workflow_folderpath, "_skills", SPLIT_FILE)
    if not os.path.isfile(path):
        return SplitSettings()
    try:
        module = python_utils.get_module(path, workflow_folderpath)
    except Exception as exc:  # noqa: BLE001 - reported as the file's error
        logger.error(f"could not import the split settings {path}", exc_info=True)
        raise SkillFormatError(f"{path}: could not import the split settings: {exc}") from exc
    return split_settings_from_module(module, path)


def identifier_patterns(workflow_folderpath: str) -> tuple[re.Pattern, ...]:
    """The workflow's identifier patterns (``identifier_patterns`` of its optional ``_skills/_split.py``).

    Each match in a request is a subject for the split merge (``split_merge.merge_subtasks``),
    like a quoted name: the framework knows no workflow's identifier formats.
    """
    return split_settings(workflow_folderpath).identifier_patterns


@functools.lru_cache(maxsize=None)
def _split_signature(workflow_folderpath: str) -> type[dspy.Signature]:
    """``SplitIntoSubtasks``, with the instructions and field descriptions of the workflow's
    ``_skills/_split.py`` when it sets them."""
    settings = split_settings(workflow_folderpath)
    signature = SplitIntoSubtasks
    if settings.instructions is not None:
        signature = signature.with_instructions(settings.instructions)
    for name, desc in settings.field_descs:
        signature = signature.with_updated_fields(name, desc=desc)
    return signature


def _job_types_for(workflow_folderpath: str | None) -> str:
    """One ``- name: description`` line per skill of the workflow; "(none)" without skills."""
    skills = load_skills(workflow_folderpath) if workflow_folderpath else ()
    if not skills:
        return "(none)"
    return "\n".join(f"- {name}: {description}" for name, description, _ in skills)


#: The skill matcher shows at most this many of a skill's ``plain_utterances`` (its first
#: ones) after its description, to bound the prompt.
MATCHER_EXAMPLES = 9


def _matcher_job_types(workflow_folderpath: str, examples: int = MATCHER_EXAMPLES) -> str:
    """The skill matcher's job types: one ``- name: description`` line per skill, followed by
    ``e.g. "...", "..."`` when the skill declares ``plain_utterances`` (at most ``examples``)."""
    utterances = skill_utterances(workflow_folderpath)
    lines = []
    for name, description, _ in load_skills(workflow_folderpath):
        shown = [" ".join(u.split()) for u in utterances.get(name, ())[:max(0, examples)]]
        shown = [u for u in shown if u]
        lines.append(f"- {name}: {description}" + (
            f" e.g. {', '.join(json.dumps(u, ensure_ascii=False) for u in shown)}" if shown else ""))
    return "\n".join(lines)


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
    listing = "the available job types, one per line as '- name: description'"
    if any(skill_utterances(workflow_folderpath).values()):
        listing += ", some followed by example requests for that job (e.g. ...)"

    class MatchSkill(dspy.Signature):
        """Decide which job type a sub-task asks for."""

        subtask: str = dspy.InputField(desc="one sub-task from a user request")
        job_types: str = dspy.InputField(desc=listing)
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
        job_types = _matcher_job_types(workflow_folderpath)
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


def _template_text(steps: tuple[str, ...]) -> str:
    lines = "\n".join(f"- {step}" for step in steps)
    return (f"A job like this is usually done as:\n{lines}\n"
            "Adapt this to the request: drop steps it does not need and add steps it asks for.")


def planner_skill_for(text: str, workflow_folderpath: str) -> tuple[str, SkillCommand | None]:
    """``(template, command)`` for the skill ``text`` asks for: one Jev match.

    ``template`` is ``plan_template_for``'s text ("" when no skill matches); ``command`` is
    the matched skill's parsed form when it has params or phases, else None.
    """
    matched = match_skill(text, workflow_folderpath)
    if matched is None:
        return "", None
    name, steps = matched
    return _template_text(steps), load_skill_commands(workflow_folderpath).get(name)


def plan_template_for(text: str, workflow_folderpath: str) -> str:
    """The matched skill's steps as a planner template for ``text``; "" when no skill matches."""
    return planner_skill_for(text, workflow_folderpath)[0]


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
    request whole. A reply cut off at ``split_max_tokens()`` is not an answer, even
    when it parses (a field header written mid-reasoning, e.g. the output template
    echoed back, parses as "{subtasks}"): it gets the one retry, and a second
    cut-off reply leaves the request whole. The model's split then goes through ``split_merge.merge_subtasks``
    (code only), which joins parts that are not independent jobs and returns the
    request whole when one job is left.
    """
    if planner_lm is None:
        planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
    job_types = _job_types_for(workflow_folderpath)
    # Temperature 0, so the same request splits the same way; a split that is
    # visibly broken or cut off at the cap is retried once at 0.7, which also
    # misses the LM cache.
    # A split whose only flaw is a leftover reference ("each of them") is still
    # better than none, so it is kept when the retry does not fix it.
    usable = None
    for temperature in (0.0, 0.7):
        replies: list = []
        try:
            with dspy.context(lm=_split_lm(planner_lm, temperature),
                              adapter=CommandsSystemPreludeAdapter()), watch_cut_off_replies() as replies:
                signature = _split_signature(workflow_folderpath) if workflow_folderpath else SplitIntoSubtasks
                text = dspy.ChainOfThought(signature)(
                    job_types=job_types, user_query=user_query).subtasks or ""
        except Exception:  # noqa: BLE001 - splitting is an optimisation; never fail the turn
            if any(replies):
                logger.warning(f"the split was cut off at {split_max_tokens()} tokens and did not parse")
                continue
            logger.warning("could not split the request; running it whole", exc_info=True)
            return [user_query]
        if replies and replies[-1]:
            # The parsed reply stopped at the cap: whatever parsed is not the model's answer.
            logger.warning(f"the split was cut off at {split_max_tokens()} tokens; not using it")
            continue
        subtasks = [match.group(1).translate(_HYPHENS) for line in text.splitlines()
                    if (match := _NUMBERED_LINE.match(line))]
        if not subtasks or len(subtasks) > MAX_SUBTASKS:
            return [user_query]
        if not _visibly_broken(subtasks):
            return _merged(user_query, subtasks, workflow_folderpath)
        if usable is None and len(set(subtasks)) == len(subtasks):
            usable = subtasks
    return _merged(user_query, usable, workflow_folderpath) if usable else [user_query]


def _merged(user_query: str, subtasks: list[str], workflow_folderpath: str | None = None) -> list[str]:
    """``split_merge.merge_subtasks`` on the model's split; the split unchanged if the merge fails."""
    try:
        identifiers = identifier_patterns(workflow_folderpath) if workflow_folderpath else ()
        merged = merge_subtasks(user_query, subtasks, identifiers)
    except Exception:  # noqa: BLE001 - the merge is a clean-up; never fail the turn
        logger.warning("could not merge the split; keeping it as the model gave it", exc_info=True)
        return subtasks
    return resolve_referents(user_query, merged, workflow_folderpath) if len(merged) > 1 else merged


#: A sub-task's referent is written in only at or above this Jev probability.
REFERENT_THRESHOLD = 0.5


def resolve_referents(user_query: str, subtasks: list[str], workflow_folderpath: str | None) -> list[str]:
    """Each sub-task with the request's record it acts on but does not name written in (one Jev call each).

    A cut sub-task can lose the record its words point back to: "Tag the customer `x`" after "Ann Lee
    leaves", "Confirm whether Raj Patel is among its holders". For a sub-task that leaves some of
    the request's subjects unnamed, Jev chooses which of them, if any, it is about
    (``split_merge.unnamed_subjects``), told what the workflow counts as a record by its optional
    ``subject_hints`` in ``_skills/_split.py``; ``split_merge.with_referent`` writes the choice in. Without a Jev
    key, or on any failure, the sub-tasks are returned unchanged.

    When a record is chosen and the sub-task holds definite descriptions ("the customer", see
    ``split_merge.definite_descriptions``), a second Jev call chooses which of them, if any, stands
    for it, and the record is written in right after it ("Tag the customer Ann Lee `x`"). Code cannot
    tell "the customer `x`" (a person, `x` a tag put on her) from "the report 'x'" (`x` its title)
    without knowing the workflow's kinds of record; the model, given the hints, can.
    """
    jev_lm = _jev_lm()
    if jev_lm is None:
        return subtasks
    identifiers = identifier_patterns(workflow_folderpath) if workflow_folderpath else ()
    notes = (split_settings(workflow_folderpath).subject_hints if workflow_folderpath else None) or "(none)"
    out = []
    for text in subtasks:
        try:
            candidates = unnamed_subjects(user_query, text, identifiers)
            subject = _choose_referent(user_query, text, candidates, notes, jev_lm) if candidates else None
            descriptions = definite_descriptions(text) if subject else []
            description = _choose_description(user_query, text, subject, descriptions, notes, jev_lm) \
                if descriptions else None
            out.append(with_referent(text, subject, description) if subject else text)
        except Exception:  # noqa: BLE001 - resolving a referent is a clean-up; never fail the turn
            logger.warning("could not resolve the sub-task's referent", exc_info=True)
            out.append(text)
    return out


def _choose_referent(user_query: str, text: str, candidates: list[str], notes: str, jev_lm) -> str | None:
    options = tuple((name, f"The sub-task acts on or asks about {name} without naming it.") for name in candidates) + (
        ("none", "The sub-task names every record it is about, or is about none of these."),)

    class ChooseReferent(dspy.Signature):
        """Decide which record of the request a sub-task is about without naming it."""

        request: str = dspy.InputField(desc="the user's whole request")
        subtask: str = dspy.InputField(desc="one sub-task cut from the request, run on its own")
        notes: str = dspy.InputField(desc="what this workflow counts as a record, and what it does not")
        record: Choice[options] = dspy.OutputField(
            desc="Which record named in the request, but not in the sub-task, is the sub-task about "
                 "(through words like 'it', 'its', 'that team', 'the customer')? 'none' if it names "
                 "every record it is about.")

    with dspy.context(lm=jev_lm):
        choice = dspy.Predict(ChooseReferent)(request=user_query, subtask=text, notes=notes).record
    value, probability = _choice_value_and_probability(choice)
    logger.info(f"referent for {text!r}: {value} (confidence {probability:.2f})")
    return value if value in candidates and probability >= REFERENT_THRESHOLD else None


def _choose_description(user_query: str, text: str, subject: str, descriptions: list[str], notes: str,
                        jev_lm) -> str | None:
    """Which of the sub-task's definite descriptions ("the customer") stands for ``subject``, if any."""
    options = tuple((phrase, f"'{phrase}' in the sub-task means {subject}.") for phrase in descriptions) + (
        ("none", f"None of these phrases means {subject}; each means something else."),)

    class ChooseDescription(dspy.Signature):
        """Decide which phrase of a sub-task stands for a record the sub-task does not name."""

        request: str = dspy.InputField(desc="the user's whole request")
        subtask: str = dspy.InputField(desc="one sub-task cut from the request, run on its own")
        record: str = dspy.InputField(desc="the record of the request the sub-task is about without naming it")
        notes: str = dspy.InputField(desc="what this workflow counts as a record, and what it does not")
        phrase: Choice[options] = dspy.OutputField(
            desc="Which 'the ...' phrase of the sub-task means the record (as 'the customer' can mean "
                 "a customer named earlier)? 'none' if each phrase means something else, such as a tag, a "
                 "request, another record or a record the work is to find.")

    with dspy.context(lm=jev_lm):
        choice = dspy.Predict(ChooseDescription)(request=user_query, subtask=text, record=subject,
                                                 notes=notes).phrase
    value, probability = _choice_value_and_probability(choice)
    logger.info(f"description of {subject!r} in {text!r}: {value} (confidence {probability:.2f})")
    return value if value in descriptions and probability >= REFERENT_THRESHOLD else None


def build_query_with_next_steps(user_query: str,
    chat_session_obj: fastworkflow.ChatSession, with_agent_inputs_and_trajectory: bool = False,
    planner_lm = None,
    trace_trigger: str | None = None,
    planner_user_query: str | None = None,
    plan_template: str | None = None,
    done_only: bool = False,
    skill_command: SkillCommand | None = None) -> str:
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
        skill_command: The matched skill's parsed form when it has params or phases. On the
            turn's initial plan the planner then fills the skill's typed parameters
            and code expands its steps (``skill_commands.plan_skill_call``; span
            ``plan_source`` "skill_call"); Jev, when configured, checks the skill's
            ``verify`` parameters against the request. Any failure falls back to the
            text planner with ``plan_template``. Ignored on a replan.
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

    initial_plan = trace_trigger is None and not with_agent_inputs_and_trajectory
    previous_turn_summaries = _previous_turn_summaries_for_planner(chat_session_obj) if initial_plan else ""
    request_text = (planner_user_query if previous_turn_summaries and planner_user_query is not None
                    else user_query)

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
        if previous_turn_summaries:
            task_planner_func = dspy.ChainOfThought(
                TaskPlannerTextWithPreviousTurnSummariesSignature
            )
            return task_planner_func(
                user_query=with_template(request_text),
                previous_turn_summaries=previous_turn_summaries,
                available_commands=available_commands,
            )
        task_planner_func = dspy.ChainOfThought(TaskPlannerTextSignature)
        return task_planner_func(
            user_query=with_template(user_query),
            available_commands=available_commands) # Note that this is not part of the signature. It is extra metadata that will be picked up by the CommandsSystemPreludeAdapter

    plan_text = ""
    plan_source = None
    skill_args = None
    skill_notes = None
    skill_fallback = None
    if initial_plan:
        # A phased skill call (skill_phases): the runner reads it after this plan.
        chat_session_obj._phased_skill_call = None
    try:
        with dspy.context(lm=planner_lm, adapter=agent_adapter):
            if skill_command is not None and initial_plan:
                try:
                    call = plan_skill_call(skill_command, request_text, available_commands, check_lm=_jev_lm(),
                                           workflow=current_workflow)
                    prediction, plan_text, skill_args = call.prediction, call.plan, call.args
                    skill_notes = list(call.notes) or None
                    plan_source = "skill_call"
                    if call.phases is not None:
                        chat_session_obj._phased_skill_call = (skill_command, call)
                    logger.info(f"skill call {skill_command.name}{call.args}"
                                + (f" ({'; '.join(call.notes)})" if call.notes else ""))
                except Exception as exc:  # noqa: BLE001 - the text planner with the template takes over
                    skill_fallback = f"{type(exc).__name__}: {exc}"[:500]
                    logger.info(f"skill call {skill_command.name} failed, planning from its template: {skill_fallback}")
            if plan_source is None:
                prediction = plan_with()
                plan_text = prediction.next_steps or ""
                plan_source = "text" if plan_text else "none"
    except BaseException:
        tracing.end_span(chat_session_obj, span, status=tracing.STATUS_ERROR)
        raise
    tracing.end_span(
        chat_session_obj,
        span,
        attributes={
            "plan": plan_text,
            "plan_source": plan_source,
            "skill": skill_command.name if skill_command is not None and initial_plan else None,
            "skill_args": skill_args,
            "skill_notes": skill_notes,
            "skill_fallback": skill_fallback,
        },
    )

    if not with_agent_inputs_and_trajectory:
        # The run's initial plan, with its line breaks, for the finish check.
        chat_session_obj._initial_plan_text = plan_text

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
