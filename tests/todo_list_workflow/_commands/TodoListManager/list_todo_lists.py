
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
    class Output(BaseModel):
        todolist_ids: list[int] = Field(
            description="id's of todo lists",
            examples=['0', '43', '79']
        )

    plain_utterances = [
        "show projects",
    ]

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:
    def _process_command(self, workflow: Workflow) -> Signature.Output:
        """List all todo lists."""
        # Access the application class instance:
        app_instance = workflow.command_context_for_response_generation  # type: TodoListManager
        todo_lists = app_instance.list_todo_lists()
        return Signature.Output(todolist_ids=[todolist.id for todolist in todo_lists])

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
