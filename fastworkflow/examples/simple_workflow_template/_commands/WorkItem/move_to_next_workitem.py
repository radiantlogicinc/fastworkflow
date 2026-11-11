import fastworkflow
from fastworkflow.utils.signatures import NOT_FOUND
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field, model_validator
from typing import Optional

from ...application.workitem import WorkItem
from ._WorkItem import Context as WorkItemContext


class Signature:
    """Navigate to the next work item in the hierarchy."""

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

        @model_validator(mode="after")
        def clear_placeholder_workitem_type(self, info):
            if self.workitem_type == NOT_FOUND:
                self.workitem_type = None
            return self

    class Output(BaseModel):
        current_workitem_has_changed: bool = Field(
            description="Whether the move caused the current workitem to change")
        new_context: Optional[str] = Field(
            default=None,
            description="The new context of the current workitem has changed")

    plain_utterances = [
        "continue to next pending feature",
        "next completed bug",
        "go to next workitem"
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:
    """Handle command execution and craft the textual response."""

    def _process_command(self, 
                         workflow: fastworkflow.Workflow, 
                         input: Signature.Input) -> Signature.Output:
        current_workitem: WorkItem = workflow.record
        
        # Get the next work item based on the filters
        next_workitem = current_workitem.get_next_workitem(
            workitem_type=input.workitem_type,
            is_complete=input.is_complete
        )

        new_context = None  
        current_workitem_has_changed = False
        if next_workitem and next_workitem is not current_workitem:
            # Change the current context to the first work item
            workflow.record = next_workitem
            current_workitem_has_changed = True
            new_context = WorkItemContext.get_displayname(next_workitem)
        
        return Signature.Output(
            current_workitem_has_changed = current_workitem_has_changed,
            new_context = new_context)

    def __call__(self, workflow: fastworkflow.Workflow, command: str, 
                 command_parameters: Signature.Input) -> Signature.Output:
        output = self._process_command(workflow, command_parameters)
        
        # Format the status information
        
        return output