
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
    """Sets the current context to a todoitem and get its properties"""
    class Input(BaseModel):
        todoitem_id: int = Field(
            description="id of todo item",
            examples=['0', '25']
        )

    class Output(BaseModel):
        description: str = Field(
            description="Todo item description")

    plain_utterances = [
        "show workitem",
        "lets work on task"
    ]

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """Get a child by its ID."""
        # Access the application class instance:
        app_instance = workflow.command_context_for_response_generation  # type: TodoList
        if todoitem := app_instance.get_child_by_id(child_id=input.todoitem_id):
            workflow.current_command_context = todoitem
            return Signature.Output(description=todoitem.description)
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
