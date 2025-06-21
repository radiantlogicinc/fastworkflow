
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
from ...application.todo_manager import TodoListManager
from ...application.todo_list import TodoList

class Signature:
    class Input(BaseModel):
        description: str = Field(description="Parameter description")
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        result: TodoList = Field(description="Result of the method call")

    plain_utterances = [
        "create todo list todolistmanager",
        "Call create_todo_list on todolistmanager",
        "create todo list todolistmanager {description}",
        "Call create_todo_list on todolistmanager with {description}"
    ]

    template_utterances = [
        "TODO: Add template utterances"
    ]

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(session.workflow_snapshot.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        result = generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        utterance_list: list[str] = [command_name] + result
        return utterance_list

    def process_extracted_parameters(self, workflow_snapshot: WorkflowSnapshot, command: str, cmd_parameters: "Signature.Input") -> None:
        pass

class ResponseGenerator:
    def _process_command(self, session: Session, input: Signature.Input) -> Signature.Output:
        """Create a new todo list.

Args:
    description (str): Description of the todo list.

Returns:
    TodoList: The newly created TodoList.

Raises:
    ValueError: If description is empty.

Example:
    >>> manager = TodoListManager()
    >>> list = manager.create_todo_list("Groceries")
    >>> list.id
    1
    >>> list.name
    'Groceries'"""
        # Access the application class instance:
        app_instance = session.workflow_snapshot.context_object  # type: TodoListManager
        result_val = app_instance.create_todo_list(description=input.description)
        return Signature.Output(result=result_val)

    def __call__(self, session: Session, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(session, command_parameters)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=f"result={output.result}")
            ]
        )
