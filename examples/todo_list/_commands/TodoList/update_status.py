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
from ...application.todo_list import TodoList
from ...application.todo_item import TodoItem

class Signature:
    """Update the status of this TodoList based on its children"""
    class Output(BaseModel):
        success: bool = Field(
            description="True if the status was updated successfully"
        )

    plain_utterances = [
        "refresh status",
        "sync project status"
    ]

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(session.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        result = generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        utterance_list: list[str] = [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + result
        return utterance_list

    def process_extracted_parameters(self, session: fastworkflow.Session, command: str, cmd_parameters: "Signature.Input") -> None:
        pass

class ResponseGenerator:
    def _process_command(self, session: Session) -> Signature.Output:
        """Update the status of this TodoList based on its children.
If all children are complete, this TodoList is complete. Otherwise, it's incomplete."""
        # Access the application class instance:
        app_instance = session.command_context_for_response_generation  # type: TodoList
        app_instance.update_status()
        return Signature.Output(success=True)

    def __call__(self, session: Session, command: str) -> CommandOutput:
        output = self._process_command(session)
        response = (
            f'Context: {session.current_command_context_name}\n'
            f'Command: {command}\n'
            f'Response: {output.model_dump_json()}'
        )
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )