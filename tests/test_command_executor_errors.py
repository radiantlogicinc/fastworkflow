import fastworkflow
import pytest

from fastworkflow.command_executor import CommandExecutor, CommandNotFoundError


# ---------------------------------------------------------------------------
# Helpers to monkeypatch registry and extractor
# ---------------------------------------------------------------------------


class FaultyRG:  # noqa: D401
    def __call__(self, *args, **kwargs):  # noqa: D401
        raise RuntimeError("boom")


class DummyCRD:  # minimal stand-in for RoutingDefinition
    def get_command_class(self, name, module_type):  # noqa: D401
        return FaultyRG if name == "fail" else None


def _monkey_registry(monkeypatch):
    monkeypatch.setattr(
        fastworkflow.RoutingRegistry,
        "get_definition",
        lambda _: DummyCRD(),
    )


def test_perform_action_wraps_error(monkeypatch):
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})

    _monkey_registry(monkeypatch)

    executor = CommandExecutor()
    session = fastworkflow.Session.create(
        workflow_folderpath=fastworkflow.get_fastworkflow_package_path(),
        session_id_str="errs_pa",
    )

    action = fastworkflow.Action(command_name="fail", command="fail")

    with pytest.raises(RuntimeError):
        executor.perform_action(session, action)


def test_invoke_command_wraps_error(monkeypatch):
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})

    _monkey_registry(monkeypatch)

    executor = CommandExecutor()

    # Skip this test if _invoke_command_metadata_extraction_workflow doesn't exist
    if not hasattr(executor, "_invoke_command_metadata_extraction_workflow"):
        pytest.skip("_invoke_command_metadata_extraction_workflow method not available in current implementation")

    # Monkeypatch _invoke_command_metadata_extraction_workflow to bypass CME workflow.
    from fastworkflow import CommandOutput, CommandResponse

    def _stub_extract(self, ws, cmd):
        co = CommandOutput(
            command_responses=[
                CommandResponse(response="stub", artifacts={"command_name": "fail", "cmd_parameters": None, "command": cmd})
            ],
            success=True,
            command_handled=False
        )
        return co

    monkeypatch.setattr(CommandExecutor, "_invoke_command_metadata_extraction_workflow", _stub_extract, raising=True)

    # Build minimal WorkflowSession (no actual _commands needed)
    ws = fastworkflow.WorkflowSession(
        executor,
        workflow_folderpath=fastworkflow.get_fastworkflow_package_path(),
        session_id_str="err_ivk",
    )

    with pytest.raises(RuntimeError):
        executor.invoke_command(ws, "fail") 