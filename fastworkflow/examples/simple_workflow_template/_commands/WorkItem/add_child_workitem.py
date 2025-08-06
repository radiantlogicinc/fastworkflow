from typing import Optional
import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.workitem import WorkItem


class Signature:
    """Add a child work item to the current work item."""

    class Input(BaseModel):
        workitem_type: str = Field(
            description="Type of the work item to create",
            examples=["Story", "SubTask", "Task", "Bug"]
        )
        id: Optional[str] = Field(
            default=None,
            description="Optional unique identifier for the work item among its siblings"
        )

    class Output(BaseModel):
        success: bool = Field(description="Whether add child was successful")
        error_msg: str = Field(default='', description="Error message if unsuccessful")
        new_child_path: str = Field(description="Path of the new child workitem")

    plain_utterances = [
        "create new subtask",
        "add bug jh43556"
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow, command: str, cmd_parameters: "Signature.Input") -> tuple[bool, str]:
        if cmd_parameters.id == fastworkflow.get_env_var('NOT_FOUND'):
            cmd_parameters.id = None
        return (True, '')


class ResponseGenerator:
    """Handle command execution and craft the textual response."""

    def _process_command(self, 
                         workflow: fastworkflow.Workflow, 
                         input: Signature.Input) -> Signature.Output:
        parent_workitem: WorkItem = workflow.command_context_for_response_generation
        
        # Create a new child work item with the specified type        
        try:
            child_workitem = parent_workitem._workflow_schema.create_workitem(
                workitem_type=input.workitem_type,
                parent=parent_workitem,
                id=input.id
            )
            return Signature.Output(
                success = True,
                new_child_path = child_workitem.get_absolute_path()
            )
        except ValueError as e:
            return Signature.Output(
                success = False,
                error_msg = str(e)
            )

    def __call__(self, workflow: fastworkflow.Workflow, command: str, 
                 command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        
        response = (
            f'Response: {output.model_dump_json()}'
        )
        
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )