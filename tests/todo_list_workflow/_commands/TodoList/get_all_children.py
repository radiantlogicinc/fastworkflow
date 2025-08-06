
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
    class Output(BaseModel):
        todoitem_ids: list[int] = Field(
            description="list of child todoitem or todolist ids",
            examples=['0', '32']
        )

    plain_utterances = [
        "show subprojects",
        "show workitems"
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
    def _process_command(self, workflow: Workflow) -> Signature.Output:
        """Get all children of this TodoList."""
        # Access the application class instance:
        todolist = workflow.command_context_for_response_generation  # type: TodoList
        todoitems = todolist.get_all_children()
        return Signature.Output(todoitem_ids=[todoitem.id for todoitem in todoitems])

    def __call__(self, workflow: Workflow, command: str) -> CommandOutput:
        output = self._process_command(workflow)
        response = (
            f'Response: {output.model_dump_json()}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )
