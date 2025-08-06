import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.workitem import WorkItem


class Signature:
    """Mark the current work item as complete."""

    class Input(BaseModel):
        is_complete: bool = Field(
            description="Whether to mark the work item as complete or incomplete",
            default=True
        )

    plain_utterances = [
        "finish this task",
        "set as not done"
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Handle command execution and craft the textual response."""

    def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> None:
        workitem: WorkItem = workflow.command_context_for_response_generation
        workitem.is_complete = input.is_complete

    def __call__(self, workflow: fastworkflow.Workflow, command: str, 
                 command_parameters: Signature.Input) -> CommandOutput:
        self._process_command(workflow, command_parameters)
        
        workitem: WorkItem = workflow.command_context_for_response_generation
        status = "complete" if workitem.is_complete else "incomplete"
        
        response = f"Work item '{workitem.get_absolute_path()}' has been marked as {status}."
        
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )