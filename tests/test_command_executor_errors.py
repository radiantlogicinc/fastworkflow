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
    fastworkflow.init({})

    _monkey_registry(monkeypatch)

    # Skip this test if _invoke_command_metadata_extraction_workflow doesn't exist
    if not hasattr(CommandExecutor, "_invoke_command_metadata_extraction_workflow"):
        pytest.skip("_invoke_command_metadata_extraction_workflow method not available in current implementation")

    # Monkeypatch _invoke_command_metadata_extraction_workflow to bypass CME workflow.
    from fastworkflow import CommandOutput, CommandResponse

    def _stub_extract(self, ws, cmd):
        co = CommandOutput(
            command_response=
                CommandResponse(response="stub", artifacts={"command_name": "fail", "cmd_parameters": None, "command": cmd}),
            success=True,
            command_handled=False
        )
        return co

    monkeypatch.setattr(CommandExecutor, "_invoke_command_metadata_extraction_workflow", _stub_extract, raising=True)

    # Build minimal ChatSession (no actual _commands needed)
    ws = fastworkflow.ChatSession(
        workflow_folderpath=fastworkflow.get_fastworkflow_package_path(),
        workflow_id_str="err_ivk",
    )

    with pytest.raises(RuntimeError):
        CommandExecutor.invoke_command(ws, "fail") 