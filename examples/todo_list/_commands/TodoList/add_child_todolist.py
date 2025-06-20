
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
    class Input(BaseModel):
        description: str = Field(description="Parameter description")
        assign_to: str = Field(description="Parameter assign_to")
        status: str = Field(description="Parameter status")
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        result: 'TodoList' = Field(description="Result of the method call")

    plain_utterances = [
        "add child todolist todolist",
        "Call add_child_todolist on todolist",
        "add child todolist todolist {description} {assign_to} {status}",
        "Call add_child_todolist on todolist with {description} {assign_to} {status}"
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
        """Add a new TodoList as a child to this TodoList.
Args:
    description (str): Description of the todo list.
    assign_to (str): Person assigned to the todo list.
    status (str): Status of the todo list.
Returns:
    TodoList: The newly created child TodoList.
Raises:
    ValueError: If a child with the generated id already exists or if trying to add self as a child."""
        # Access the application class instance:
        app_instance = session.workflow_snapshot.context_object  # type: TodoList
        result_val = app_instance.add_child_todolist(description=input.description, assign_to=input.assign_to, status=input.status)
        return Signature.Output(result=result_val)

    def __call__(self, session: Session, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(session, command_parameters)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=f"result={output.result}")
            ]
        )
