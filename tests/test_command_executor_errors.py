import fastworkflow
import pytest

from fastworkflow import ModuleType
from fastworkflow.command_executor import CommandExecutor, CommandNotFoundError


# ---------------------------------------------------------------------------
# Helpers to monkeypatch registry and extractor
# ---------------------------------------------------------------------------


class FaultyRG:  # noqa: D401
    def __call__(self, *args, **kwargs):  # noqa: D401
        raise RuntimeError("boom")


class DummyCRD:  # minimal stand-in for RoutingDefinition
    def get_command_class(self, name, module_type):  # noqa: D401
        # Only the response generator is faulty. Returning FaultyRG for
        # COMMAND_PARAMETERS_CLASS would make perform_action treat it as a
        # pydantic Input model and run validate_parameters on it.
        if name == "fail" and module_type == ModuleType.RESPONSE_GENERATION_INFERENCE:
            return FaultyRG
        return None


def _monkey_registry(monkeypatch):
    monkeypatch.setattr(
        fastworkflow.RoutingRegistry,
        "get_definition",
        lambda _: DummyCRD(),
    )


def test_perform_action_wraps_error(monkeypatch):
    fastworkflow.init({})

    _monkey_registry(monkeypatch)

    workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_fastworkflow_package_path(),
        workflow_id_str="errs_pa",
    )

    action = fastworkflow.Action(command_name="fail", command="fail")

    with pytest.raises(RuntimeError):
        CommandExecutor.perform_action(workflow, action)


def test_invoke_command_wraps_error(monkeypatch):
    """A failing response generator propagates out of the NLU path too.

    The sibling above covers the direct-action entry point. This one covers
    invoke_command, which reaches the same response generator after intent
    detection has named the command — the two must not diverge on whether a
    command's own exception escapes or gets swallowed into a CommandOutput.

    Driven through a real WorkflowExecutionContext because that is what calls
    invoke_command in production (workflow_execution_context.py:1175 passes
    `self`). The parameter is annotated ChatSession, but only three members are
    touched — cme_workflow, get_active_workflow() and cme_workflow._context —
    and WEC supplies all three, so the annotation is looser than the contract.

    This test previously stubbed CommandExecutor._invoke_command_metadata_extraction_workflow
    and built its session with ChatSession(workflow_folderpath=..., workflow_id_str=...).
    Both are gone: the CME hop is now a perform_action call with the `wildcard`
    command, and ChatSession takes neither argument. It had been skipping on a
    hasattr() guard for the missing method, which meant it silently asserted
    nothing rather than failing when the implementation moved.
    """
    fastworkflow.init({})

    _monkey_registry(monkeypatch)

    from fastworkflow import CommandOutput, CommandResponse
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    # Stand in for intent detection. Everything below the CME hop is real: this
    # only names the command the way a successful extraction would, so the
    # response generator lookup and call are the code under test.
    #
    # perform_action is the seam because that is what the CME hop is now
    # (invoke_command calls it with command_name="wildcard"). It cannot be
    # reached through the stubbed registry instead — DummyCRD has no class for
    # "wildcard", so perform_action would raise ValueError over the RuntimeError
    # this test is about. Only the CME hop goes through perform_action;
    # invoke_command instantiates the resolved command's generator directly, so
    # stubbing it does not hide the call under test.
    def _stub_cme(workflow, action):
        return CommandOutput(
            command_response=CommandResponse(
                response="stub",
                artifacts={
                    "command_name": "fail",
                    "cmd_parameters": None,
                    "command": action.command,
                },
            ),
            success=True,
            command_handled=False,
        )

    monkeypatch.setattr(CommandExecutor, "perform_action", _stub_cme)

    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_fastworkflow_package_path(),
        workflow_id_str="err_ivk",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, session_key="err_ivk")
    ctx.bind_app_workflow(app_workflow)

    # invoke_command resolves the target through get_active_workflow(), which on a
    # WEC reads only the context-local stack — there is no fallback to the bound
    # app workflow, deliberately. Production pushes it for the duration of the
    # turn (workflow_execution_context.py:780), so pushing it here is what makes
    # this the same call and not a contrived one.
    ctx.push_active_workflow(app_workflow)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            CommandExecutor.invoke_command(ctx, "fail")
    finally:
        ctx.pop_active_workflow()
        ctx.close()