
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
        id: int = Field(description="Parameter id")
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        result: Optional[TodoList] = Field(description="Result of the method call")

    plain_utterances = [
        "get todo list todolistmanager",
        "Call get_todo_list on todolistmanager",
        "get todo list todolistmanager {id}",
        "Call get_todo_list on todolistmanager with {id}"
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
        """Get a todo list by ID.

Args:
    id (int): ID of the todo list to get.

Returns:
    TodoList or None: The TodoList with the given ID, or None if not found."""
        # Access the application class instance:
        app_instance = session.workflow_snapshot.context_object  # type: TodoListManager
        result_val = app_instance.get_todo_list(id=input.id)
        return Signature.Output(result=result_val)

    def __call__(self, session: Session, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(session, command_parameters)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=f"result={output.result}")
            ]
        )
