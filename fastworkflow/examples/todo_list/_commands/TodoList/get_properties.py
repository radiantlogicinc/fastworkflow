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
    """Get all properties of this todo list"""
    class Output(BaseModel):
        description: str = Field(
            description="Description of this todo item",
            examples=['laundry', 'homework']
        )
        assign_to: str = Field(
            description="name of the person responsible for doing this task",
            examples=['John Doe', 'Jane Smith']
        )
        is_complete: bool = Field(
            description="True if complete, False otherwise",
        )

    plain_utterances = [
        "show project details",
        "get project info"
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
    def _process_command(self, workflow: Workflow) -> Signature.Output:
        """Get all properties of the TodoList class."""
        # Access the application class instance:
        todo_list = workflow.command_context_for_response_generation  # type: TodoList
        return Signature.Output(
            description=todo_list.description, 
            assign_to=todo_list.assign_to, 
            is_complete=todo_list.status == TodoItem.COMPLETE
        )

    def __call__(self, workflow: Workflow, command: str) -> CommandOutput:
        output = self._process_command(workflow)
        response = (
            f'Response: {output.model_dump_json()}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )
