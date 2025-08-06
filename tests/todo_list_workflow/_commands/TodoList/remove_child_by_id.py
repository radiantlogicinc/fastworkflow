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
    """Remove a child item or list from this todo list"""
    class Input(BaseModel):
        child_id: int = Field(
            description="ID of the child to remove",
            examples=['2', '5']
        )

    class Output(BaseModel):
        success: bool = Field(
            description="True if the child was removed, False if not found"
        )

    plain_utterances = [
        "remove item",
        "delete workitem"
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
        """Remove a child by its ID."""
        # Access the application class instance:
        app_instance = workflow.command_context_for_response_generation  # type: TodoList
        result = app_instance.remove_child_by_id(child_id=input.child_id)
        return Signature.Output(success=result)

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
