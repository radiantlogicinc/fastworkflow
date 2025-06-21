
from pydantic import ConfigDict

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session, WorkflowSnapshot
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
        success: bool = Field(
            description="True if item was added, False otherwise"
        )

    plain_utterances = [
        "add workitem"
    ]

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(session.workflow_snapshot.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        result = generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        utterance_list: list[str] = [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + result
        return utterance_list

class ResponseGenerator:
    def _process_command(self, session: Session, input: Signature.Input) -> Signature.Output:
        """Add a new TodoItem as a child to this TodoList."""
        # Access the application class instance:
        app_instance = session.command_context_for_response_generation  # type: TodoList
        todoitem = app_instance.add_child_todoitem(
            description=input.description, 
            assign_to=input.assign_to, 
            status=TodoItem.COMPLETE if input.is_complete else TodoItem.INCOMPLETE
        )
        session.current_command_context = todoitem
        return Signature.Output(success=True)

    def __call__(self, session: Session, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(session, command_parameters)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=output.model_dump_json())
            ]
        )
