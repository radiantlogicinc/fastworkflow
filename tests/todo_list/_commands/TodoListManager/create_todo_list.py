
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
    """Create a top-level todolist and set it as the current command context"""
    class Input(BaseModel):
        description: str = Field(
            description="Description of this todo list",
            examples=['groceries', 'Trip preparation']
        )

    class Output(BaseModel):
        current_context: str = Field(
            description="The current context is TodoListManager"
        )
        new_context: str = Field(
            description="Context will be switched to the newly created TodoList"
        )

    plain_utterances = [
        "create project",
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
        """Create a new todo list."""
        # Access the application class instance:
        app_instance = workflow.command_context_for_response_generation  # type: TodoListManager
        todo_list = app_instance.create_todo_list(description=input.description)
        
        current_context = workflow.current_command_context_displayname
        workflow.current_command_context = todo_list
        new_context=workflow.current_command_context_displayname

        return Signature.Output(
            current_context=current_context,
            new_context=new_context
        )

    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        response = (
            f'Context: {output.current_context}\n'
            f'Command: {command}\n'
            f'Command parameters: {command_parameters}\n'
            f'Response: {output.model_dump_json(include={"new_context"})}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )
