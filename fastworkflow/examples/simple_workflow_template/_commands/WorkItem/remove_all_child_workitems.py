from typing import Optional
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

from ...application.workitem import WorkItem


class Signature:
    """Remove all child work items from the current work item."""

    class Output(BaseModel):
        error_msg: Optional[str] = Field(
            description="Error msg if the removal failed",
            default=None)

    plain_utterances = [
        "delete all child tasks",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Handle command execution and craft the textual response."""

    def _process_command(self, workflow: fastworkflow.Workflow) -> Signature.Output:
        workitem: WorkItem = workflow.command_context_for_response_generation
        
        try:
            # Remove all child work items
            workitem.remove_all_children()
      
            return Signature.Output()
        except ValueError as e:
            return Signature.Output(str(e))

    def __call__(self, workflow: fastworkflow.Workflow, command: str) -> CommandOutput:
        output = self._process_command(workflow)
        
        # Format the status information
        response = (
            f'Response: {output.model_dump_json()}'
        )
        
        if output.error_msg:
            response = output.error_msg
        else:
            workitem: WorkItem = workflow.command_context_for_response_generation
            response = f"Removed all child work items from '{workitem.get_absolute_path()}'."
        
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )