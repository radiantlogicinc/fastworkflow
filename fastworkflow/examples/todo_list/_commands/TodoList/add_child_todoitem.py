
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
    """Add a todo item to this todolist and set it as the current command context"""
    class Input(BaseModel):
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

    class Output(BaseModel):
        current_context: str = Field(
            description="The current context is TodoList"
        )
        new_context: str = Field(
            description="Context will be switched to the newly created child TodoItem"
        )

    plain_utterances = [
        "add workitem"
    ]

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """Add a new TodoItem as a child to this TodoList."""
        # Access the application class instance:
        app_instance = workflow.command_context_for_response_generation  # type: TodoList
        todoitem = app_instance.add_child_todoitem(
            description=input.description, 
            assign_to=input.assign_to, 
            status=TodoItem.COMPLETE if input.is_complete else TodoItem.INCOMPLETE
        )
        
        current_context = workflow.current_command_context_displayname
        workflow.current_command_context = todoitem
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
