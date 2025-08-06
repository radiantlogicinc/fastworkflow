import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
import json

from ...application.workitem import WorkItem


class Signature:
    """Show the schema information for the current work item type."""

    plain_utterances = [
        "display workflow schema",
        "list all types",
        "what children are possible",
        "how many stories are allowed under an epic"
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Handle command execution and craft the textual response."""

    def _process_command(self, workflow: fastworkflow.Workflow) -> dict:
        workitem: WorkItem = workflow.command_context_for_response_generation
        
        # Get the schema for the current work item type
        return workitem.schema

    def __call__(self, workflow: fastworkflow.Workflow, command: str) -> CommandOutput:
        workitem_schema = self._process_command(workflow)
        
        # Format the status information
        response = (
            f'Response: {json.dumps(workitem_schema)}'
        )
     
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )