from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow

from ..retail_data import load_data
from ..tools.modify_user_address import ModifyUserAddress


class Signature:
    """Modify user address"""
    class Input(BaseModel):
        user_id: str = Field(
            description="The user ID to modify",
            json_schema_extra={"from_command_output": ["find_user_id_by_email.user_id", "find_user_id_by_name_zip.user_id"]},
            pattern=r"^([a-z]+_[a-z]+_\d+)$",
            examples=["sara_doe_496"],
        )
        address1: str = Field(description="First line of address", examples=["123 Main St"])
        address2: str = Field(description="Second line of address", examples=["Apt 1"])
        city: str = Field(description="City name", examples=["San Francisco"])
        state: str = Field(
            description="State code",
            pattern=r"^([A-Z]{2})$",
            examples=["CA"],
        )
        country: str = Field(description="Country name", examples=["USA"])
        zip: str = Field(
            description="ZIP code",
            pattern=r"^(\d{5})$",
            examples=["12345"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="Whether address modification succeeded.")

    plain_utterances: List[str] = [
        "I need to update my shipping address.",
        "Please change my default address to a new one.",
        "Can you modify my address to 123 Main St, Apt 1, San Francisco, CA, USA, 94105?",
        "I want to update my profile address to a new location.",
        "Change my address to 456 Elm Street, Suite 300, New York, NY, USA, 10001.",
        "I recently moved and need to change my delivery address.",
        "Update my account with my new address in Los Angeles.",
        "I̵'d like to edit my saved address details.",
        "Can you update my contact info with a new address?",
        "Switch my default shipping address to 789 Oak Ave, Dallas, TX, USA, 75201.",
    ]
    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> Signature.Output:
        output = self._process_command(workflow, command_parameters)
        return output

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        result = ModifyUserAddress.invoke(
            data=data,
            user_id=input.user_id,
            address1=input.address1,
            address2=input.address2,
            city=input.city,
            state=input.state,
            country=input.country,
            zip=input.zip,
        )
        return Signature.Output(status=result) 