"""validate_extracted_parameters receives the command text only when it asks for it."""
from pydantic import BaseModel, Field

from fastworkflow.utils.signatures import InputForParamExtraction


class Params(BaseModel):
    control_code: str = Field(description="code")


def _validate(hook_class, command="list_findings ctrl_ABC"):
    extraction = InputForParamExtraction(
        command=command, input_for_param_extraction_class=hook_class)
    return extraction.validate_parameters(
        None, "ControlsMonitor/list_findings", Params(control_code="ctrl_ABC"))


def test_a_hook_that_declares_command_text_receives_it():
    seen = {}

    class Hook:
        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters, command_text):
            seen.update(command=command, command_text=command_text)
            return True, ""

    is_valid, *_ = _validate(Hook)

    assert is_valid
    assert seen == {"command": "ControlsMonitor/list_findings",
                    "command_text": "list_findings ctrl_ABC"}


def test_an_existing_three_argument_hook_is_called_as_before():
    seen = []

    class Hook:
        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters):
            seen.append(command)
            return False, "rejected"

    is_valid, message, *_ = _validate(Hook)

    assert not is_valid and "rejected" in message
    assert seen == ["ControlsMonitor/list_findings"]
