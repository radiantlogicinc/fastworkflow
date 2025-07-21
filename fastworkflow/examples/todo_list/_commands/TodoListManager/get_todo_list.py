
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
from ...application.todo_manager import TodoListManager
from ...application.todo_list import TodoList

class Signature:
    """Set the requested todolist as the command context and return its description"""
    class Input(BaseModel):
        id: int = Field(
            description="id of the todo list",
            examples=['1', '56']
        )

    class Output(BaseModel):
        description: str = Field(
            description="Description of the returned todo list"
        )

    plain_utterances = [
        "show project"
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
        """Get a todo list by ID."""
        # Access the application class instance:
        app_instance = workflow.command_context_for_response_generation  # type: TodoListManager
        if todo_list := app_instance.get_todo_list(id=input.id):
            workflow.current_command_context = todo_list
            return Signature.Output(description=todo_list.description)
        return Signature.Output(description='NOT_FOUND')

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
