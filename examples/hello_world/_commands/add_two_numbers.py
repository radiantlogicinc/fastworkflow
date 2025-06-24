
import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session, WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.utils.context_utils import list_context_names
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Signature:
    class Input(BaseModel):
        a: float = Field(description="Parameter a")
        b: float = Field(description="Parameter b")

    class Output(BaseModel):
        result: float = Field(description="Result of the function call")

    plain_utterances = [
        "add two numbers",
        "add two numbers {a} {b}",
        "call add_two_numbers with {a} {b}"
    ]

    template_utterances = []

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(session.workflow_snapshot.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        result = generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        utterance_list: list[str] = [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + result
        return utterance_list

    def process_extracted_parameters(self, workflow_snapshot: WorkflowSnapshot, command: str, cmd_parameters: "Signature.Input") -> None:
        pass

class ResponseGenerator:
    def _process_command(self, session: Session, input: Signature.Input) -> Signature.Output:
        """Execute add_two_numbers function"""
        # Call the function
        from ..application.add_two_numbers import add_two_numbers
        result_val = add_two_numbers(a=input.a, b=input.b)
        return Signature.Output(result=result_val)

    def __call__(self, session: Session, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(session, command_parameters)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=output.model_dump_json())
            ]
        )
