
from pydantic import ConfigDict

import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.utils.context_utils import list_context_names
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from ...application.todo_item import TodoItem

class Signature:
    class Input(BaseModel):
        description: str = Field(
            description="Description of this todo item",
            examples=['laundry', 'homework']
        )

    class Output(BaseModel):
        success: bool = Field(default=True, description="Indicates successful execution.")

    plain_utterances = [
        "update description",
        "change description",
    ]

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """Set the description of the todo item."""
        # Access the application class instance:
        todo_item = workflow.record  # type: TodoItem
        todo_item.description = input.description
        return Signature.Output(success=True)

    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> fastworkflow.CommandResponse:
        output = self._process_command(workflow, input)
        response = (
            f'Response: {output.model_dump_json()}'
        )
        return fastworkflow.CommandResponse(response=response, output=output)
