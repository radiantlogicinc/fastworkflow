import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.utils.context_utils import list_context_names
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from ..application.add_two_numbers import add_two_numbers


class Signature:
    class Input(BaseModel):
        first_num: float = Field(description="First number")
        second_num: float = Field(description="Second number")

    class Output(BaseModel):
        sum_of_two_numbers: float = Field(description="The sum of the two provided numbers")

    plain_utterances = [
        "add two numbers",
        "add two numbers {a} {b}",
        "call add_two_numbers with {a} {b}"
    ]

    template_utterances = []

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow, command: str, cmd_parameters: "Signature.Input") -> None:
        return None

class ResponseGenerator:
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """Execute add_two_numbers function"""
        # Call the function
        sum_of_two_numbers = add_two_numbers(a=input.first_num, b=input.second_num)
        return Signature.Output(sum_of_two_numbers=sum_of_two_numbers)

    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> fastworkflow.CommandResponse:
        output = self._process_command(workflow, input)
        response = (
            f'Response: {output.model_dump_json()}'
        )
        return fastworkflow.CommandResponse(response=response, output=output)
