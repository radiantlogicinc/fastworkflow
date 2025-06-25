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
    """Set multiple properties of this todo list at once"""
    class Input(BaseModel):
        description: Optional[str] = Field(
            default=None, 
            description="Description of the todo list",
            examples=["Weekly shopping", "House chores"]
        )
        assign_to: Optional[str] = Field(
            default=None, 
            description="Person assigned to the todo list",
            examples=["John Smith", "Mary Jones"]
        )
        is_complete: Optional[bool] = Field(
            description="True if complete, False otherwise",
        )

    class Output(BaseModel):
        success: bool = Field(
            description="True if properties were updated successfully"
        )

    plain_utterances = [
        "update project properties",
        "edit project details"
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
    def _process_command(self, session: Session, input: Signature.Input) -> Signature.Output:
        """Sets one or more properties for an instance of TodoList."""
        # Access the application class instance:
        todo_list = session.command_context_for_response_generation  # type: TodoList
        if input.description is not None:
            todo_list.description = input.description
        if input.assign_to is not None:
            todo_list.assign_to = input.assign_to
        if input.is_complete is not None:
            todo_list.assign_to = status = TodoItem.COMPLETE if input.is_complete else TodoItem.INCOMPLETE
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
