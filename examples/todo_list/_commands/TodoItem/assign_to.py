
from pydantic import ConfigDict

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.utils.context_utils import list_context_names
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from ...application.todo_item import TodoItem

class Signature:
    class Input(BaseModel):
        assign_to: str = Field(
            description="name of the person responsible for doing this task",
            examples=['John Doe', 'Jane Smith']
        )

    class Output(BaseModel):
        success: bool = Field(default=True, description="Indicates successful execution.")

    plain_utterances = [
        "assign workitem",
        "assign responsibility"
    ]

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

    def process_extracted_parameters(self, session: fastworkflow.Session, command: str, cmd_parameters: "Signature.Input") -> None:
        pass

class ResponseGenerator:
    def _process_command(self, session: Session, input: Signature.Input) -> Signature.Output:
        """Set the person assigned to the todo item."""
        # Access the application class instance:
        todo_item = session.command_context_for_response_generation  # type: TodoItem
        todo_item.assign_to = input.assign_to
        return Signature.Output(success=True)

    def __call__(self, session: Session, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(session, command_parameters)
        response = (
            f'Context: {session.current_command_context_displayname}\n'
            f'Command: {command}\n'
            f'Command parameters: {command_parameters}\n'
            f'Response: {output.model_dump_json()}'
        )
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )
