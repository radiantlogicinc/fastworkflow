"""The Input's model validators receive the command text, the workflow and the command in their context (spec 3.2)."""
import fastworkflow
from pydantic import BaseModel, Field, model_validator

from fastworkflow.utils.signatures import InputForParamExtraction


def _validate(input_model, command="list_findings ctrl_ABC"):
    extraction = InputForParamExtraction(command=command, input_for_param_extraction_class=None)
    return extraction.validate_parameters(
        None, "ControlsMonitor/list_findings", input_model.model_construct(control_code="ctrl_ABC"))


def test_an_input_validator_receives_the_command_text_workflow_and_command():
    seen = {}

    class Params(BaseModel):
        control_code: str = Field(description="code")

        @model_validator(mode="after")
        def look(self, info):
            seen.update(info.context)
            return self

    is_valid, *_ = _validate(Params)

    assert is_valid
    assert seen == {"command_text": "list_findings ctrl_ABC", "workflow": None,
                    "command": "ControlsMonitor/list_findings"}


def test_a_validator_refusal_without_a_field_refuses_the_call():
    class Params(BaseModel):
        control_code: str = Field(description="code")

        @model_validator(mode="after")
        def refuse(self, info):
            raise fastworkflow.Refused("rejected")

    is_valid, message, *_ = _validate(Params)

    assert not is_valid and "rejected" in message
