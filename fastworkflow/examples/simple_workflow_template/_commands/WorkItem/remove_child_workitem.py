from typing import Optional
import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.workitem import WorkItem


class Signature:
    """Remove a specific child work item from the current work item."""

    class Input(BaseModel):
        index: int = Field(
            description="Index of the child work item to remove",
            ge=0
        )
        workitem_type: Optional[str] = Field(
            description="Type of the work item to remove",
            examples=["Feature", "Task", "Bug"],
            default=None
        )
        is_complete: Optional[bool] = Field(
            description="Filter by completion status",
            default=None
        )

    class Output(BaseModel):
        error_msg: Optional[str] = Field(
            description="Error msg if the removal failed",
            default=None)

    plain_utterances = [
        "delete the third bug",
        "remove feature number 5",
        "get rid of the last workitem",
        "delete the third pending bug",
        "remove the 9th completed feature"
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
        parent_workitem: WorkItem = workflow.command_context_for_response_generation

        try:
            # Get the child work item at the specified index and type
            child_workitem = parent_workitem.get_child(
                input.index, 
                input.workitem_type,
                input.is_complete)
            
            if child_workitem is None:
                return Signature.Output(
                    f"No child work item of type '{input.workitem_type}' found at index {input.index}"
                )
            
            # Remove the child work item
            parent_workitem.remove_child(child_workitem) 
            return Signature.Output()
        except (ValueError, IndexError) as e:
            return Signature.Output(str(e))

    def __call__(self, workflow: fastworkflow.Workflow, command: str, 
                 command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        
        if output.error_msg:
            response = output.error_msg
        else:
            workitem: WorkItem = workflow.command_context_for_response_generation
            response = f"Removed child work item under '{workitem.get_absolute_path()}'."
        
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )