import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field
from typing import Optional

from ...application.workitem import WorkItem


class Signature:
    """Navigate to the previous work item in the hierarchy."""

    class Input(BaseModel):
        workitem_type: Optional[str] = Field(
            description="Filter by work item type",
            examples=["Feature", "Task", "Bug"],
            default=None
        )
        is_complete: Optional[bool] = Field(
            description="Filter by completion status",
            default=None
        )

    class Output(BaseModel):
        current_workitem_has_changed: bool = Field(
            description="Whether the move caused the current workitem to change")
        new_context: Optional[str] = Field(
            default=None,
            description="The new context of the current workitem has changed")

    plain_utterances = [
        "navigate to previous pending feature",
        "previous completed bug",
        "go to previous workitem"
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow, command: str, cmd_parameters: "Signature.Input") -> tuple[bool, str]:
        if cmd_parameters.workitem_type == fastworkflow.get_env_var('NOT_FOUND'):
            cmd_parameters.workitem_type = None
        return (True, '')


class ResponseGenerator:
    """Handle command execution and craft the textual response."""

    def _process_command(self, 
                         workflow: fastworkflow.Workflow, 
                         input: Signature.Input) -> Signature.Output:
        current_workitem: WorkItem = workflow.command_context_for_response_generation
        
        # Get the previous work item based on the filters
        previous_workitem = current_workitem.get_previous_workitem(
            workitem_type=input.workitem_type,
            is_complete=input.is_complete
        )

        new_context = None  
        current_workitem_has_changed = False
        if previous_workitem and previous_workitem is not current_workitem:
            # Change the current context to the first work item
            workflow.current_command_context = previous_workitem
            current_workitem_has_changed = True
            new_context = workflow.current_command_context_displayname
        
        return Signature.Output(
            current_workitem_has_changed = current_workitem_has_changed,
            new_context = new_context)

    def __call__(self, workflow: fastworkflow.Workflow, command: str, 
                 command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        
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