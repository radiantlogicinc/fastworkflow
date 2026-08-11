import os
from uuid import uuid4

import fastworkflow

from fastworkflow._workflows.command_metadata_extraction._commands.ErrorCorrection.abort import (
    ResponseGenerator as AbortResponseGenerator,
)
from fastworkflow._workflows.command_metadata_extraction._commands.ErrorCorrection.you_misunderstood import (
    ResponseGenerator as MisunderstoodResponseGenerator,
)


def _create_workflows():
    app_workflow_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "todo_list_workflow")
    )
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=app_workflow_path,
        workflow_id_str=f"test-misunderstood-app-{uuid4()}",
    )
    cme_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        ),
        parent_workflow_id=app_workflow.id,
        workflow_context={"app_workflow": app_workflow},
    )
    return app_workflow, cme_workflow


def _assert_neutral_misunderstanding_response(cme_workflow):
    output = MisunderstoodResponseGenerator()(
        cme_workflow, "none of these commands"
    )
    command_response = output.command_response
    response = command_response.response

    assert "couldn't determine which available command matches" in response
    assert "If one of these options applies" in response
    assert "type 'abort' to cancel, then rephrase your request" in response
    assert "rephrase your request or type 'abort'" not in response
    assert "Please select the correct command" not in response

    valid_commands = command_response.artifacts["valid_command_names"]
    assert valid_commands
    assert all(command_name in response for command_name in valid_commands)


def _assert_abort_resets_for_the_next_rephrased_turn(cme_workflow):
    workflow_context = cme_workflow.context
    workflow_context["NLU_Pipeline_Stage"] = (
        fastworkflow.NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
    )
    cme_workflow.context = workflow_context

    abort_output = AbortResponseGenerator()(cme_workflow, "abort")

    assert abort_output.command_response.response == "command aborted\n"
    assert (
        cme_workflow.context["NLU_Pipeline_Stage"]
        == fastworkflow.NLUPipelineStage.INTENT_DETECTION
    )


def test_misunderstanding_response_does_not_presuppose_a_listed_command_is_correct():
    """Out-of-scope requests get this same response, so the list must be optional."""
    fastworkflow.init({})
    app_workflow, cme_workflow = _create_workflows()

    try:
        _assert_neutral_misunderstanding_response(cme_workflow)
        _assert_abort_resets_for_the_next_rephrased_turn(cme_workflow)
    finally:
        app_workflow.close()
