"""Regression: direct actions must run validate_extracted_parameters (fix-1in).

The NLU path builds InputForParamExtraction via create(), which resolves the
command Signature class. perform_action used to use the bare constructor, so
the hook never ran. These tests lock both paths to the same rejection message
and cover the empty-parameters guard.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.utils.signatures import InputForParamExtraction

from tests.direct_action_validation_workflow._commands.check_ready import (
    HOOK_NOT_READY,
)
from tests.direct_action_validation_workflow._commands.use_payload import (
    HOOK_MISSING_PAYLOAD,
)


@pytest.fixture
def workflow_path() -> str:
    return str(
        Path(__file__).parent.joinpath("direct_action_validation_workflow").resolve()
    )


@pytest.fixture
def initialized(tmp_path, workflow_path: str):
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": str(tmp_path / "speedict")})
    fastworkflow.RoutingRegistry.clear_registry()
    # Force a fresh directory/routing build for the fixture workflow.
    fastworkflow.RoutingRegistry.get_definition(workflow_path, load_cached=False)
    yield
    fastworkflow.RoutingRegistry.clear_registry()


def _make_workflow(workflow_path: str) -> fastworkflow.Workflow:
    return fastworkflow.Workflow.create(
        workflow_path,
        workflow_id_str=f"dav_{uuid.uuid4().hex[:8]}",
    )


def test_create_path_rejects_missing_payload(initialized, workflow_path):
    """Baseline: the factory path already runs the hook."""
    workflow = _make_workflow(workflow_path)
    extractor = InputForParamExtraction.create(
        workflow, "use_payload", "use the payload"
    )
    params_cls = fastworkflow.RoutingRegistry.get_definition(
        workflow_path
    ).get_command_class(
        "use_payload", fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
    )
    assert params_cls is not None
    input_obj = params_cls(note="hello")

    is_valid, error_msg, _, _ = extractor.validate_parameters(
        workflow, "use_payload", input_obj
    )

    assert is_valid is False
    assert HOOK_MISSING_PAYLOAD in error_msg


def test_perform_action_rejects_missing_payload_with_hook_message(
    initialized, workflow_path
):
    workflow = _make_workflow(workflow_path)
    action = fastworkflow.Action(
        command_name="use_payload",
        command="use the payload",
        parameters={"note": "hello"},
    )

    with pytest.raises(ValueError) as exc_info:
        CommandExecutor.perform_action(workflow, action)

    assert HOOK_MISSING_PAYLOAD in str(exc_info.value)


def test_perform_action_succeeds_when_payload_present(initialized, workflow_path):
    workflow = _make_workflow(workflow_path)
    workflow.context["payload"] = "abc"
    action = fastworkflow.Action(
        command_name="use_payload",
        command="use the payload",
        parameters={"note": "hello"},
    )

    result = CommandExecutor.perform_action(workflow, action)

    assert result.success is True
    assert any("payload=abc" in r.response for r in result.command_responses)
    assert any("note=hello" in r.response for r in result.command_responses)


@pytest.mark.parametrize(
    "parameters",
    [
        {},  # default Action.parameters — falsy, used to skip validation
        {"unused": None},
    ],
)
def test_perform_action_rejects_no_params_when_not_ready(
    initialized, workflow_path, parameters
):
    """Empty/absent parameters must still run the context precondition hook."""
    workflow = _make_workflow(workflow_path)
    action = fastworkflow.Action(
        command_name="check_ready",
        command="check ready",
        parameters=parameters,
    )

    with pytest.raises(ValueError) as exc_info:
        CommandExecutor.perform_action(workflow, action)

    assert HOOK_NOT_READY in str(exc_info.value)


def test_perform_action_no_params_succeeds_when_ready(initialized, workflow_path):
    workflow = _make_workflow(workflow_path)
    workflow.context["ready"] = True
    action = fastworkflow.Action(
        command_name="check_ready",
        command="check ready",
        parameters={},
    )

    result = CommandExecutor.perform_action(workflow, action)

    assert result.success is True
    assert any("ready=true" in r.response for r in result.command_responses)
