from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

from ...application.workitem import WorkItem


class Signature:
    """Get the status of the current work item and its children."""

    plain_utterances = [
        "show completion status"
    ]

    class Output(BaseModel):
        status_dict: dict = Field(description="Dictionary with status info")

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Handle command execution and craft the textual response."""

    def _process_command(self, workflow: fastworkflow.Workflow) -> Signature.Output:
        workitem: WorkItem = workflow.command_context_for_response_generation
        
        # Get the status of the work item
        return Signature.Output(
            status_dict = workitem.get_status()
        )

    def __call__(self, workflow: fastworkflow.Workflow, command: str) -> CommandOutput:
        output = self._process_command(workflow)
        
        # Format the status information
        response = (
            f'Response: {output.model_dump_json()}'
        )
        
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )