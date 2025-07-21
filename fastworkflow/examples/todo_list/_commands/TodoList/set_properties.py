from pydantic import ConfigDict

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.workflow import Workflow
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
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow, command: str, cmd_parameters: "Signature.Input") -> tuple[bool, str]:
        return (True, '')

class ResponseGenerator:
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """Sets one or more properties for an instance of TodoList."""
        # Access the application class instance:
        todo_list = workflow.command_context_for_response_generation  # type: TodoList
        if input.description is not None:
            todo_list.description = input.description
        if input.assign_to is not None:
            todo_list.assign_to = input.assign_to
        if input.is_complete is not None:
            todo_list.assign_to = status = TodoItem.COMPLETE if input.is_complete else TodoItem.INCOMPLETE
        return Signature.Output(success=True)

    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        response = (
            f'Response: {output.model_dump_json()}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )
