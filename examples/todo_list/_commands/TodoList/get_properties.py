
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
        pass
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        INCOMPLETE: str = Field(description="Value of property INCOMPLETE")
        COMPLETE: str = Field(description="Value of property COMPLETE")
        description: str = Field(description="Get the description of the todo item.\\n\\nReturns:\\n    str: The description of the todo item.")
        assign_to: str = Field(description="Get the person assigned to the todo item.\\n\\nReturns:\\n    str: The person assigned to the todo item.")
        status: str = Field(description="Get the status of the todo item.\\n\\nReturns:\\n    str: The status of the todo item (COMPLETE or INCOMPLETE).")

    plain_utterances = [
        "getproperties todolist",
        "Call getproperties on todolist"
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
        """Get all properties of the TodoList class."""
        # Access the application class instance:
        app_instance = session.workflow_snapshot.context_object  # type: TodoList
        # For get_properties, the primary logic is to gather attribute values,
        # which is handled by constructing the output_return string that references app_instance attributes directly.
        # No additional complex processing steps are typically needed in this block.
        pass # Placeholder if no other pre-return logic is needed
        return Signature.Output(INCOMPLETE=app_instance.INCOMPLETE, COMPLETE=app_instance.COMPLETE, description=app_instance.description, assign_to=app_instance.assign_to, status=app_instance.status)

    def __call__(self, session: Session, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(session, command_parameters)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=f"properties={output.dict()}")
            ]
        )
