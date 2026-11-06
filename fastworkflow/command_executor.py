import inspect
import logging
from contextlib import nullcontext

import fastworkflow
from pydantic import BaseModel
from fastworkflow import tracing
from fastworkflow import approval_binding, command_writes
from fastworkflow.command_interfaces import CommandExecutorInterface
from fastworkflow.context_navigation import declared_descends_to

from fastworkflow import Action, CommandOutput, ChatSession
from fastworkflow.observability.execution_recorder import record_execution, recorder_for
from fastworkflow import ModuleType
from fastworkflow.utils.python_utils import legacy_module_names
from fastworkflow.utils.signatures import InputForParamExtraction, parameters_with_unfilled
from pathlib import Path
from fastworkflow.command_routing import RoutingDefinition
from typing import Optional
from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_directory import CommandDirectory


def _declared_output_model(response_generator_class) -> type | None:
    """The command's own ``Signature.Output`` class, or None when its module declares none.

    Read from the globals of the ``__call__`` that the class defines, so it does not depend on the
    module being registered in ``sys.modules``.
    """
    call = getattr(response_generator_class, "__call__", None)
    namespace = getattr(call, "__globals__", None) or {}
    signature = namespace.get("Signature")
    if signature is None:
        signature = getattr(inspect.getmodule(response_generator_class), "Signature", None)
    output = getattr(signature, "Output", None)
    return output if inspect.isclass(output) and issubclass(output, BaseModel) else None


def paging_line(output, command_parameters) -> str:
    """The paging line a framework-written response ends with when rows remain (design 3.14): the
    rows read, their total and the next page's paging value, e.g. ``Rows 1-20 of 57; next page:
    start=20.`` Empty when the Output is not a page, the command has no paging parameter, or no rows remain.
    """
    from fastworkflow.observation_offloading.facts import output_page, paging_parameters

    paging = paging_parameters(type(command_parameters)) if command_parameters is not None else ()
    page = output_page(output)
    if not paging or page is None or page[0] == 0:
        return ""
    rows, total = page
    given = command_parameters.model_dump(exclude_unset=True) if command_parameters is not None else {}
    name = next((n for n in paging if n in given), paging[0])
    start = getattr(command_parameters, name, None) or 0
    if not isinstance(start, int) or isinstance(start, bool):
        start = 0
    after = start + rows
    if after >= total:
        return ""
    return f"Rows {start + 1}-{after} of {total}; next page: {name}={after}."


def _declared_output_message(command_name: str, returned: str) -> str:
    """The call error for a command that declares Signature.Output and returned something else."""
    return (f"{command_name} declares an Output but returned {returned}; return the Output, "
            "or CommandResponse(response=..., output=<Output>)")


def frame_command_result(result, *, command_name: str, output_model: type | None,
                         command_parameters=None) -> CommandOutput:
    """Wrap what a ``ResponseGenerator.__call__`` returned in a CommandOutput (design 3.14).

    The short forms, and only these:

    - the command's own ``Signature.Output``: ``response`` is the Output as JSON and ``output`` the
      instance;
    - a ``str``: the text of the response;
    - a ``CommandResponse``: a CommandOutput around it (its artifacts and next actions), with
      ``success`` left to the framework. A ``CommandResponse`` with ``success=False`` is a call error:
      a refusal is ``raise fastworkflow.Refused(message)``.

    A command that declares ``Signature.Output`` must return it, directly or as
    ``CommandResponse(response=..., output=<Output>)``. A str, or a CommandResponse without that
    output, is a call error (fix-3rsq.declared-output-required).

    A hand-built ``CommandOutput``, or any other value, fails the call, naming the command and the
    type it returned.
    """
    if isinstance(result, fastworkflow.CommandResponse):
        if result.success is False:
            raise TypeError(
                f"{command_name} returned a CommandResponse with success=False: success is the framework's "
                "to set. Refuse with raise fastworkflow.Refused(message)"
            )
        if output_model is not None and not isinstance(result.output, output_model):
            returned = "CommandResponse without output" if result.output is None else (
                f"CommandResponse(output={type(result.output).__qualname__})")
            raise TypeError(_declared_output_message(command_name, returned))
        return CommandOutput(command_response=result)
    if isinstance(result, str):
        if output_model is not None:
            raise TypeError(_declared_output_message(command_name, "str"))
        return CommandOutput(command_response=fastworkflow.CommandResponse(response=result))
    if output_model is not None and type(result) is output_model:
        text = result.model_dump_json()
        line = paging_line(result, command_parameters)
        return CommandOutput(
            command_response=fastworkflow.CommandResponse(
                response=f"{text}\n{line}" if line else text, output=result
            )
        )
    # the module as the workflow's name for it (a namespace digest written back to the bare basename)
    returned = legacy_module_names(f"{type(result).__module__}.{type(result).__qualname__}")
    if output_model is not None and type(result) is not output_model:
        raise TypeError(_declared_output_message(command_name, type(result).__qualname__))
    if output_model is None:
        declared = "its ResponseGenerator declares no Signature.Output"
    else:
        declared = f"its Signature.Output is {output_model.__qualname__}"
    raise TypeError(
        f"{command_name} returned {returned}, which is not a short form its ResponseGenerator may return "
        f"({declared}): return that Output, a str or a CommandResponse (artifacts or next actions), "
        "or raise fastworkflow.Refused(message)"
    )


def _wildcard_command_output(framed: CommandOutput) -> CommandOutput:
    """The turn's CommandOutput for the CME router, built from its typed routing result (fix-3rsq.12.3).

    The router (``_workflows/command_metadata_extraction/_commands/wildcard.py``) returns its
    ``Signature.Output``; the framework frames it here. ``success`` is ``routed``; a handled answer
    carries the artifact ``command_handled``; a routed answer carries the extraction the turn reads
    (``command``, ``command_name``, ``cmd_parameters`` and, when the command ran on an enclosing
    record, ``ran_on_context_name`` / ``ran_on_context_clause``). This is what the router returned
    before it returned its Output: the same response text, success and artifacts.
    """
    route = framed.command_response.output
    if route is None or not all(hasattr(route, name) for name in ("handled", "routed", "response")):
        raise TypeError(
            "wildcard must return its Signature.Output (the routing result), "
            f"not {type(route).__qualname__}"
        )
    artifacts: dict = {}
    if route.handled:
        artifacts["command_handled"] = True
    if route.command_name is not None:
        artifacts["command"] = route.command
        artifacts["command_name"] = route.command_name
        artifacts["cmd_parameters"] = route.cmd_parameters
    if route.ran_on_context_clause is not None:
        artifacts["ran_on_context_name"] = route.ran_on_context_name
        artifacts["ran_on_context_clause"] = route.ran_on_context_clause
    return CommandOutput(
        command_response=fastworkflow.CommandResponse(
            response=route.response,
            success=route.routed,
            artifacts=artifacts,
            output=route.output,
        )
    )


def call_response_generator(response_generator, workflow, command, command_parameters, *, command_name: str = ""):
    """Call ``response_generator(workflow, command, command_parameters)`` and frame its result.

    ``command_parameters`` is the filled ``Input``, or None for a command with no ``Input``. A
    ``fastworkflow.Refused`` the call raises is its answer: a failed CommandOutput with the refusal's
    message. Any other return is framed by ``frame_command_result``.
    """
    try:
        result = response_generator(workflow, command, command_parameters)
    except fastworkflow.Refused as refusal:
        return refusal.as_output()
    return frame_command_result(
        result,
        command_name=command_name or type(response_generator).__qualname__,
        output_model=_declared_output_model(type(response_generator)),
        command_parameters=command_parameters,
    )


logger = logging.getLogger(__name__)

#: (command, target context) pairs already reported, so each undeclared move is logged once per process.
_UNDECLARED_MOVES_WARNED: set[tuple[str, str]] = set()


def _warn_undeclared_move(routing, command_name: str, workflow, context_before) -> None:
    """Warn once when a command moved the workflow into a context its ``descends_to`` does not name.

    Design 3.15: the comparison is the one the `now_in` note makes (the current object before and
    after), made here so assistant mode reports it too. Core commands (go_up, reset_context) are
    exempt, and nothing here may fail the command.
    """
    try:
        if workflow.current_command_context is context_before:
            return
        if command_name in _core_command_names():
            return
        target = workflow.current_command_context_name
        if declared_descends_to(routing, command_name) == target:
            return
        key = (command_name, target)
        if key in _UNDECLARED_MOVES_WARNED:
            return
        _UNDECLARED_MOVES_WARNED.add(key)
        short = command_name.split("/")[-1]
        logger.warning(
            f'{short} moved the workflow into {target} but declares no descends_to; '
            f'declare descends_to = "{target}" so plans can reach {target}\'s commands'
        )
    except Exception:  # noqa: BLE001 - a warning about navigation must not fail the command
        logger.debug("undeclared-move check skipped for %r", command_name, exc_info=True)


def _core_command_names() -> frozenset[str]:
    # Lazy: workflow_agent imports this module.
    from fastworkflow.workflow_agent import core_command_names

    return core_command_names()


# ------------------------------------------------------------------
# Module-level delegation configuration and exceptions
# ------------------------------------------------------------------

MAX_DELEGATION_DEPTH: int = 10  # Safety limit for delegation hops


class CommandNotFoundError(Exception):
    """Raised when a command cannot be resolved in any accessible context."""


class AlreadyInContextError(CommandNotFoundError):
    """The command enters the context the caller is already in."""


def _command_name_token(text: str) -> str:
    parts = (text or "").strip().split(maxsplit=1)
    return parts[0].lstrip("/") if parts else ""


def _resolve_planning_command_name(workflow_folderpath: str, name: str) -> str | None:
    """Return the qualified planning command name *name* refers to, or None."""
    if not name:
        return None
    directory = CommandDirectory.load(workflow_folderpath)
    if name in directory.map_command_2_metadata:
        meta = directory.get_command_metadata(name)
        return name if meta.is_planning else None
    short = name.split("/")[-1].lower()
    matches = [
        qualified_name
        for qualified_name, meta in directory.map_command_2_metadata.items()
        if meta.is_planning and qualified_name.split("/")[-1].lower() == short
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _planning_command_refusal_output(
    qualified_name: str,
    *,
    workflow: fastworkflow.Workflow | None = None,
    message: str | None = None,
) -> CommandOutput:
    """The refusal for a planning command reached outside agent mode: the direct and MCP paths,
    and assistant-mode ``/`` commands that need the agent (``message`` then carries the reasons,
    design 3.4). In agent mode a planning command's steps run it
    (WorkflowExecutionContext.run_planning_command)."""
    if message is None:
        message = f"{qualified_name} is a planning command; it runs in agent mode."
    command_output = CommandOutput(
        command_response=fastworkflow.CommandResponse(response=message, success=False),
        command_name=qualified_name,
    )
    if workflow is not None:
        command_output.workflow_name = workflow.folderpath.split("/")[-1]
        command_output.context = workflow.current_command_context_displayname
    return command_output


def _assistant_planning_refusal(
    folderpath: str, qualified_name: str, workflow: fastworkflow.Workflow
) -> CommandOutput | None:
    """Assistant mode's answer for an exactly named planning command (fix-3rsq.10.8, design 3.4).

    None when the planning command is MECHANICAL: it then runs through the normal run path. A
    refusal (the 3.4 message with its reasons) when it NEEDS THE AGENT, or when it cannot be
    classified at all: a command the framework cannot read is refused rather than run.
    """
    from fastworkflow.planning_needs_agent import planning_command_needs_agent, refusal_message

    try:
        needs_agent, reasons = planning_command_needs_agent(folderpath, qualified_name)
    except ValueError as error:
        return _planning_command_refusal_output(
            qualified_name, workflow=workflow,
            message=f"{qualified_name} is a planning command; it cannot be checked for assistant "
                    f"mode: {error}.")
    if not needs_agent:
        return None
    return _planning_command_refusal_output(
        qualified_name, workflow=workflow, message=refusal_message(qualified_name, reasons))


def _annotate_exception(exc: BaseException, **fields) -> None:
    """Stamp routed identity onto an in-flight exception, first writer wins.

    An exception unwinds outward, so the innermost frame that knew the identity
    is the one that owns it; an enclosing dispatch must not overwrite what a
    nested one already stamped. Attribute setting is best-effort — an exception
    with __slots__ refuses it — because losing the annotation is survivable and
    masking the original exception is not.
    """
    for name, value in fields.items():
        # The READ is inside the guard too, not just the write. An exception
        # whose type defines a __getattr__ that raises something other than
        # AttributeError would otherwise propagate out of here and destroy the
        # command's real exception — the precise failure this function's
        # docstring says must never happen, reintroduced by the probe meant to
        # avoid it.
        try:
            if getattr(exc, name, None) is None:
                setattr(exc, name, value)
        except Exception:
            pass


def _policy_rewrite(chat_session, command_name: str, input_obj) -> Optional[str]:
    """The policy guard's rewrite for this command, or None to run it.

    The guard belongs to the agent serving the session right now, looked up at
    each call: distillation swaps agents mid-session, so a reference taken
    earlier could ask the wrong one. None whenever no policy is enabled.
    """
    from fastworkflow.policy_runtime import policy_of

    policy = policy_of(getattr(chat_session, "workflow_tool_agent", None))
    if policy is None:
        return None
    if input_obj is None:
        params = {}
    elif hasattr(input_obj, "model_dump"):
        try:
            params = input_obj.model_dump()
        except Exception:  # noqa: BLE001 - the guard is still consulted
            params = {}
    else:
        params = dict(input_obj) if isinstance(input_obj, dict) else {}
    return policy.guard(command_name, params)


def _annotation(exc: BaseException, name: str):
    """Read an annotation without letting a hostile __getattr__ escape.

    Same reason the write is guarded: this runs while an exception is already
    unwinding, and anything raised here replaces the failure the caller is
    trying to report with one about the reporting.
    """
    try:
        return getattr(exc, name, None)
    except Exception:
        return None


class CommandExecutor(CommandExecutorInterface):
    @classmethod
    def invoke_command(
        cls,
        chat_session: 'fastworkflow.ChatSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        claim_check = getattr(
            chat_session, "assert_experiment_claim_current", None
        )
        if claim_check is not None:
            claim_check()
        # Whitespace-only text is the same empty command (fix-wfxt): it has no command for intent
        # detection to route, and predict_single_sentence raises on it.
        if not command or not command.strip():
            return CommandOutput(
                command_response=
                    fastworkflow.CommandResponse(
                        response="You just hit the <Enter> key. How about a command or some feedback instead?"
                    )
            )

        # One id per command execution, minted before the span so both sides
        # carry it (arch §12.0 delta 1). This is the join key between the
        # CommandOutput that lands in the turn's record_json and the span that
        # produced it; before it there was none.
        call_id = tracing.new_command_call_id()
        parent_call_id = tracing.current_call_id()

        # fw.command.execute boundary span (observability design §3.1, D3).
        # chat_session is duck-typed (WEC, or ChatSession delegating to its
        # core); with no sink or open turn the helpers no-op.
        span = tracing.start_span(
            chat_session,
            tracing.SPAN_COMMAND_EXECUTE,
            kind=tracing.KIND_TOOL,
            attributes={
                "raw_command": command,
                tracing.ATTR_COMMAND_CALL_ID: call_id,
                tracing.ATTR_PARENT_CALL_ID: parent_call_id,
            },
        )

        # Context type BEFORE execution (arch §12.0 delta 2). Gated on a span
        # having actually opened, for the same reason the attribute prep below
        # is: with tracing off this work must not run at all.
        context_before = (
            tracing.context_type(cls._active_workflow(chat_session))
            if span is not None
            else None
        )

        # Bound before the try so the error path can still file this dispatch's
        # inner hops when call_scope itself is what raised.
        child_calls: list = []
        try:
            # Bind the trace host for the deep NLU emission sites
            # (fw.nlu.intent / fw.nlu.param_extraction) — they run several
            # frames down with no reference to the session ([R28]; D3 as
            # amended). call_scope additionally makes this dispatch the parent
            # of every perform_action hop underneath it.
            with tracing.host_scope(chat_session), tracing.call_scope(
                call_id
            ) as child_calls:
                command_output = cls._invoke_command_impl(chat_session, command)
        except BaseException as exc:
            # CommandCancelledError/AskUserSuspend are BaseException control
            # signals — close the span and always re-raise untouched. Which
            # exceptions those are lives in `tracing`, not here, so this site
            # and the four others cannot drift apart (fix-ajv.19); the local
            # imports that used to name them are gone with the isinstance.
            is_control_signal = tracing.is_control_signal(exc)
            tracing.end_span(
                chat_session,
                span,
                # One mapping, shared with every other dispatch site so they
                # cannot drift apart again. fix-ajv.19.
                status=tracing.status_for_dispatch_exception(exc),
                # Routing usually got far enough to bind these before the
                # generator raised; _invoke_command_impl stamps what it had.
                # Without them the error span was the only one in the taxonomy
                # that could not say which command it covered. fix-ajv.16 FW-3.
                command_name=_annotation(exc, "_fw_command_name"),
                context=_annotation(exc, "_fw_context"),
                attributes={"error_type": type(exc).__name__},
            )
            if not is_control_signal:
                # A failure is still a dispatch that happened. Without the id,
                # the outcome the caller builds from `exc` cannot be joined to
                # this span or to the record below. fix-ajv.16 FW-1.
                _annotate_exception(exc, _fw_call_id=call_id)
                # Paired with the stamp rather than with the span: an outcome
                # carrying a command_call_id must have an ExecutionRecordRef to
                # join to (tests/test_dispatch_path_conformance.py). Control
                # signals yield no outcome, so they stay unrecorded as before.
                try:
                    record_execution(
                        recorder_for(chat_session),
                        command_call_id=call_id,
                        parent_call_id=parent_call_id,
                        span_id=span.span_id if span is not None else None,
                        child_calls=child_calls,
                    )
                except Exception:
                    # Every other observation site on this path is wrapped so
                    # it cannot mask the command's exception; this one was
                    # bare. Recording is best-effort, unwinding is not.
                    pass
            raise

        # Attribute prep stays inside the never-raise boundary and runs only
        # when a span was actually opened: a user-authored parameters model
        # whose model_dump() raises must not fail the turn, and with tracing
        # off this work must not run at all.
        params_dict = None
        if span is not None:
            try:
                params = command_output.command_parameters
                if hasattr(params, "model_dump"):
                    params_dict = params.model_dump()
                elif isinstance(params, dict):
                    params_dict = params
            except Exception:
                params_dict = None

        # Stamped unconditionally, not only when a span opened: the id is what
        # makes the outcome joinable, and a turn recorded with tracing off can
        # still be read back through a public API.
        command_output.command_call_id = call_id

        context_after = None
        if span is not None:
            workflow = cls._active_workflow(chat_session)
            context_after = tracing.context_type(workflow)

        tracing.end_span(
            chat_session,
            span,
            status=(
                tracing.STATUS_OK if command_output.success else tracing.STATUS_ERROR
            ),
            command_name=command_output.command_name or None,
            context=command_output.context or None,
            attributes={
                "parameters": params_dict,
                "response_text": command_output.command_response.response or "",
                "success": bool(command_output.success),
                tracing.ATTR_CONTEXT_BEFORE: context_before,
                tracing.ATTR_CONTEXT_AFTER: context_after,
                # The internal CME/core hops this dispatch made, each naming its
                # parent (arch §12.1 item 5). They have no spans of their own, so
                # this ledger is where their correlation lives. An empty list is
                # recorded rather than omitted: "this dispatch made no inner
                # calls" and "nothing captured them" are different facts, and an
                # absent key cannot tell them apart.
                tracing.ATTR_CHILD_CALLS: list(child_calls),
            },
        )
        record_execution(
            recorder_for(chat_session),
            command_call_id=call_id,
            parent_call_id=parent_call_id,
            span_id=span.span_id if span is not None else None,
            child_calls=child_calls,
        )
        return command_output

    @staticmethod
    def _active_workflow(chat_session: 'fastworkflow.ChatSession'):
        """The workflow whose command context this dispatch acts on, or None.

        Duck-typed and never raising, like the rest of the tracing seam: this is
        called only to build capture attributes, and a host that cannot answer
        must degrade to an absent context type rather than fail the command.
        """
        try:
            return chat_session.get_active_workflow()
        except Exception:
            return None

    @classmethod
    def _invoke_command_impl(
        cls,
        chat_session: 'fastworkflow.ChatSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        app_workflow = chat_session.get_active_workflow()
        # Assistant mode answers an exactly named planning command before any extraction: a
        # command that needs the agent is refused with its reasons. A mechanical one falls
        # through to the extraction hop below, and the post-extraction branch runs it. In agent
        # mode the name goes through the extraction hop and runs as a planning command.
        if app_workflow is not None and (
            "is_assistant_mode_command" in chat_session.cme_workflow._context
        ):
            token = _command_name_token(command)
            if planning_name := _resolve_planning_command_name(
                app_workflow.folderpath, token
            ):
                refusal = _assistant_planning_refusal(
                    app_workflow.folderpath, planning_name, app_workflow
                )
                if refusal is not None:
                    return refusal

        command_output = cls.perform_action(
            chat_session.cme_workflow, 
            Action(
                command_name = "wildcard",
                command = command)
        )

        if command_output.command_handled:       
            # important to clear the current command mode from the workflow context
            if "is_assistant_mode_command" in chat_session.cme_workflow._context:
                del chat_session.cme_workflow._context["is_assistant_mode_command"]
            return command_output
        elif not command_output.success:       
            return command_output

        command_name = command_output.command_response.artifacts["command_name"]
        input_obj = command_output.command_response.artifacts["cmd_parameters"]
        # Set by the CME walk when the command runs on an enclosing record (None otherwise).
        ran_on_name = command_output.command_response.artifacts.get("ran_on_context_name")
        ran_on_clause = command_output.command_response.artifacts.get("ran_on_context_clause")

        workflow = chat_session.get_active_workflow()
        workflow_name = workflow.folderpath.split('/')[-1]
        context = workflow.current_command_context_displayname

        # A command that writes, called inside a planning command's phase whose steps do not name it,
        # is refused unrun (fix-fixv, command_writes): a read-only planning command changes nothing.
        refusal = command_writes.phase_write_refusal(chat_session, workflow.folderpath, command_name)
        if refusal is not None:
            if "is_assistant_mode_command" in chat_session.cme_workflow._context:
                del chat_session.cme_workflow._context["is_assistant_mode_command"]
            # Marked so the agent loop neither logs it as an executed action nor lets the policy
            # remember it as run (workflow_agent._execute_workflow_query skips both, as for policy_blocked).
            refused = CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=refusal, success=False, artifacts={"write_refused": True}),
                command_name=command_name)
            refused.workflow_name = workflow_name
            refused.context = context
            return refused

        if planning_name := _resolve_planning_command_name(
            workflow.folderpath, command_name
        ):
            assistant = "is_assistant_mode_command" in chat_session.cme_workflow._context
            if assistant:
                refusal = _assistant_planning_refusal(
                    workflow.folderpath, planning_name, workflow
                )
                if refusal is not None:
                    del chat_session.cme_workflow._context["is_assistant_mode_command"]
                    return refusal
            # Agent mode: the step's arguments were extracted above; the planning command
            # runs them as a nested frame. Its output is the step's observation. Assistant mode
            # runs a mechanical planning command the same way, with its steps gated as
            # agent-originated (ADR-EXP-025 section 5: a planning command's steps are the agent's
            # calls), and without suspension, since no agent is asking.
            try:
                origin = (approval_binding.origin_scope(approval_binding.Origin.AGENT_DIRECT)
                          if assistant else nullcontext())
                with origin:
                    planning_output = chat_session.run_planning_command(
                        planning_name, input_obj, command
                    )
            finally:
                if assistant and "is_assistant_mode_command" in chat_session.cme_workflow._context:
                    del chat_session.cme_workflow._context["is_assistant_mode_command"]
            planning_output.workflow_name = workflow.folderpath.split("/")[-1]
            planning_output.context = workflow.current_command_context_displayname
            planning_output.command_name = planning_name
            planning_output.command_parameters = input_obj or None
            return planning_output

        command_routing_definition = fastworkflow.RoutingRegistry.get_definition(
            workflow.folderpath
        )

        response_generation_class = command_routing_definition.get_command_class(
            command_name,
            ModuleType.RESPONSE_GENERATION_INFERENCE,
        )
        if not response_generation_class:
            raise ValueError(
                f"Response generation class not found for command name '{command_name}' "
            )
        # Approval gate (ADR-EXP-025): the one choke point the agent's calls and planning-command
        # steps reach. Unannotated and user-originated calls pass straight through.
        held = approval_binding.gate(chat_session, workflow, command_name, input_obj, command)
        if held is not None:
            return held

        response_generation_object = response_generation_class()

        raw_user_message = command
        if "raw_user_message" in workflow.context:
            raw_user_message = workflow.context['raw_user_message']

        # The policy verifier's guard (fastworkflow.policy_runtime), consulted
        # here because only here does the command exist in canonical form: its
        # real name and validated parameters, whatever wording the agent used.
        # A block returns the rewrite as a failed output and nothing executes.
        # All policy logic lives with the guard; this is only the call point.
        if (rewrite := _policy_rewrite(chat_session, command_name, input_obj)) is not None:
            blocked = CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=rewrite,
                    success=False,
                    artifacts={"policy_blocked": True},
                )
            )
            blocked.workflow_name = workflow_name
            blocked.context = context
            blocked.command_name = command_name
            blocked.command_parameters = input_obj or None
            if "is_assistant_mode_command" in chat_session.cme_workflow._context:
                del chat_session.cme_workflow._context["is_assistant_mode_command"]
            return blocked

        context_before = workflow.current_command_context
        try:
            # command_parameters is the filled Input, or None for a command with no Input (3.14).
            command_parameters_class = command_routing_definition.get_command_class(
                command_name, ModuleType.COMMAND_PARAMETERS_CLASS
            )
            command_output = call_response_generator(
                response_generation_object, workflow, raw_user_message,
                input_obj if command_parameters_class else None,
                command_name=command_name,
            )
        except BaseException as exc:
            # Routing resolved this identity above; the assignments that would
            # publish it onto the CommandOutput sit below the raise, so on
            # failure it died with the frame and every caller had to report the
            # command as unnamed. Carry it out on the exception instead, and
            # re-raise unconditionally — this seam observes, it never handles.
            # fix-ajv.16 FW-2.
            _annotate_exception(
                exc,
                _fw_command_name=command_name,
                _fw_workflow_name=workflow_name,
                _fw_context=context,
            )
            raise

        _warn_undeclared_move(command_routing_definition, command_name, workflow, context_before)

        # Set the additional attributes
        command_output.workflow_name = workflow_name
        command_output.context = context
        command_output.command_name = command_name
        command_output.command_parameters = input_obj or None
        if ran_on_clause is not None:
            # A command the walk ran on an enclosing record is filed under that record.
            command_output.context = "global" if not ran_on_name else ran_on_name
            command_output.ran_in_context_name = ran_on_name or ""
            command_output.ran_in_context_clause = ran_on_clause

        # important to clear the current command mode from the workflow context
        if "is_assistant_mode_command" in chat_session.cme_workflow._context:
            del chat_session.cme_workflow._context["is_assistant_mode_command"]

        return command_output

    @classmethod
    def perform_action(
        cls,
        workflow: fastworkflow.Workflow,
        action: fastworkflow.Action,
    ) -> fastworkflow.CommandOutput:  # sourcery skip: extract-method
        workflow.command_context_for_response_generation = \
            workflow.current_command_context

        # One id per dispatch through this method (arch §12.0 delta 1). It is
        # the outermost id on the direct-action, startup-action and MCP paths,
        # and a child id on the internal CME hop that invoke_command makes — in
        # which case call_scope files it under the enclosing command call.
        call_id = tracing.new_command_call_id()

        workflow_name = workflow.folderpath.split('/')[-1]
        context = workflow.current_command_context_displayname

        if action.command_name != "wildcard":
            if planning_name := _resolve_planning_command_name(
                workflow.folderpath, action.command_name
            ):
                return _planning_command_refusal_output(
                    planning_name,
                    workflow=workflow,
                )
        
        command_routing_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)

        response_generation_class = (
            command_routing_definition.get_command_class(
                action.command_name,
                ModuleType.RESPONSE_GENERATION_INFERENCE,
            )
        )
        if not response_generation_class:
            raise ValueError(
                f"Response generation class not found for command name '{action.command_name}'"
            )

        response_generation_object = response_generation_class()

        command_parameters_class = (
            command_routing_definition.get_command_class(
                action.command_name, ModuleType.COMMAND_PARAMETERS_CLASS
            )
        )
        if not command_parameters_class:
            if action.command_name != "wildcard":
                held = approval_binding.gate(tracing.current_host(), workflow, action.command_name,
                                             None, action.command)
                if held is not None:
                    held.command_call_id = call_id
                    return held
            try:
                with tracing.call_scope(call_id, command_name=action.command_name):
                    command_output = call_response_generator(
                        response_generation_object, workflow, action.command, None,
                        command_name=action.command_name,
                    )
            except BaseException as exc:
                # Same seam as `_invoke_command_impl`, for the entry point the
                # other one never reaches: perform_action is the direct-action,
                # startup-action and MCP path, and it is also the CME hop
                # invoke_command makes with command_name="wildcard". Routing is
                # already decided here — `action.command_name` names it, and
                # `workflow_name`/`context` were bound above — but the assignments
                # that publish them onto the CommandOutput sit below the raise, so
                # on failure they died with the frame. Carry them out on the
                # exception and re-raise unconditionally: this seam observes, it
                # never handles. First-writer-wins means a nested dispatch that
                # already stamped a more specific identity keeps it. fix-ajv.16
                # FW-2, extended to this path.
                _annotate_exception(
                    exc,
                    _fw_command_name=action.command_name,
                    _fw_workflow_name=workflow_name,
                    _fw_context=context,
                )
                raise
            
            if action.command_name == "wildcard":
                command_output = _wildcard_command_output(command_output)

            # Validate that response_generation_object returns a CommandOutput, not a string
            if not isinstance(command_output, CommandOutput):
                raise TypeError(f"Response generation object for command '{action.command_name}' did not return a CommandOutput. This indicates an implementation error in the response generator.")
                
            # Set the additional attributes
            command_output.workflow_name = workflow_name
            command_output.context = context
            command_output.command_name = action.command_name
            command_output.command_call_id = call_id
            return command_output

        # Always resolve the command's Signature class via create() so
        # validate_extracted_parameters runs on the direct-action
        # path the same way they do on the NLU path. Validate even when
        # action.parameters is empty/falsy — context preconditions still apply.
        # A required field the caller leaves out is asked for, as on the NLU path (fix-3rsq.plain-required).
        input_obj = parameters_with_unfilled(command_parameters_class, action.parameters or {})

        input_for_param_extraction = InputForParamExtraction.create(
            workflow, action.command_name, action.command
        )
        is_valid, error_msg, _, _ = input_for_param_extraction.validate_parameters(
            workflow, action.command_name, input_obj
        )
        if not is_valid:
            raise ValueError(
                f"Invalid action parameters for command '{action.command_name}'\n{error_msg}"
            )

        if action.command_name != "wildcard":
            held = approval_binding.gate(tracing.current_host(), workflow, action.command_name,
                                         input_obj, action.command)
            if held is not None:
                held.command_call_id = call_id
                return held

        try:
            with tracing.call_scope(call_id, command_name=action.command_name):
                command_output = call_response_generator(
                    response_generation_object, workflow, action.command, input_obj,
                    command_name=action.command_name,
                )
        except BaseException as exc:
            # Same seam as `_invoke_command_impl`, for the entry point the
            # other one never reaches: perform_action is the direct-action,
            # startup-action and MCP path, and it is also the CME hop
            # invoke_command makes with command_name="wildcard". Routing is
            # already decided here — `action.command_name` names it, and
            # `workflow_name`/`context` were bound above — but the assignments
            # that publish them onto the CommandOutput sit below the raise, so
            # on failure they died with the frame. Carry them out on the
            # exception and re-raise unconditionally: this seam observes, it
            # never handles. First-writer-wins means a nested dispatch that
            # already stamped a more specific identity keeps it. fix-ajv.16
            # FW-2, extended to this path.
            _annotate_exception(
                exc,
                _fw_command_name=action.command_name,
                _fw_workflow_name=workflow_name,
                _fw_context=context,
            )
            raise
        
        if action.command_name == "wildcard":
            command_output = _wildcard_command_output(command_output)

        # Validate that response_generation_object returns a CommandOutput, not a string
        if not isinstance(command_output, CommandOutput):
            raise TypeError(f"Response generation object for command '{action.command_name}' did not return a CommandOutput. This indicates an implementation error in the response generator.")
        
        # Set the additional attributes
        command_output.workflow_name = workflow_name
        command_output.context = context
        command_output.command_name = action.command_name
        command_output.command_parameters = input_obj or None
        command_output.command_call_id = call_id

        return command_output

    # MCP-compliant methods
    @classmethod
    def perform_mcp_tool_call(
        cls,
        workflow: fastworkflow.Workflow,
        tool_call: fastworkflow.MCPToolCall,
        command_context: str = '*'
    ) -> fastworkflow.MCPToolResult:
        """
        MCP-compliant tool execution method.
        
        Args:
            workflow: FastWorkflow workflow
            tool_call: MCP tool call request
            workitem_path: The context in which to execute the command. If None, it must be in the arguments.
            
        Returns:
            MCPToolResult: MCP-compliant result format
        """
        try:
            context = tool_call.arguments.get('workitem_path', command_context)
            if not context:
                raise ValueError("Context ('workitem_path') must be provided for an MCP tool call.")

            # Convert MCP tool call to FastWorkflow Action using helper method
            action = fastworkflow.Action(
                command_name=tool_call.name,
                command=tool_call.arguments.get('command', ''),
                parameters=dict(tool_call.arguments.items()),
            )

            # Execute using existing perform_action method. An MCP tool call is agent-originated
            # (owner answer 3); the MCP caller cannot be suspended, so an ask returns its prompt.
            with approval_binding.origin_scope(approval_binding.Origin.AGENT_DIRECT):
                command_output = cls.perform_action(workflow, action)

            # Convert to MCP format
            return command_output.to_mcp_result()

        except Exception as e:
            # Return error in MCP format
            return fastworkflow.MCPToolResult(
                content=[fastworkflow.MCPContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )
