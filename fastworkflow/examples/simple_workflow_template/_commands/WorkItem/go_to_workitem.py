import sys
from typing import Optional
import fastworkflow
from fastworkflow.utils.signatures import NOT_FOUND
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field, model_validator

from ...application.workitem import WorkItem
from ._WorkItem import Context as WorkItemContext


class Signature:
    """Navigate to a specific work item either by its absolute path or 
    relative to the current workitem by providing the child type and optionally the child id or index."""

    class Input(BaseModel):
        path: Optional[str] = Field(
            default=None,
            description="Absolute path to the work item",
            examples=["/Feature[index=0]", "Epic/Task[id=x35]", "story/bug"],
        )
        workitem_type: Optional[str] = Field(
            default=None,
            description="Type of the child work item",
            examples=["Story", "SubTask", "Task", "Bug"]
        )
        id: Optional[str] = Field(
            default=None,
            description="Unique identifier of the child work item",
            examples=["kjhdfg-08435", "1"]
        )
        index: Optional[int] = Field(
            default=None,
            description="index of the child work item",
            examples=["1", "2", "3"]
        )

        @model_validator(mode="after")
        def path_from_workitem_type(self, info):
            workflow = info.context.get("workflow")
            if self.path == NOT_FOUND:
                self.path = None
            if self.workitem_type == NOT_FOUND:
                self.workitem_type = None
            if self.id == NOT_FOUND:
                self.id = None
            if self.index == -sys.maxsize:
                self.index = None

            if not self.path:
                if not self.workitem_type:
                    raise fastworkflow.Refused("Provide the absolute path or the child workitem type and optionally its id or index")

                current_workitem: WorkItem = workflow.record
                absolute_path_of_current_workitem = current_workitem.get_absolute_path()
                if self.id:
                    self.path = f'{absolute_path_of_current_workitem}/{self.workitem_type}[id={self.id}]'
                elif self.index:
                    self.index -= 1   # python indexes are 0 based
                    self.path = f'{absolute_path_of_current_workitem}/{self.workitem_type}[index={self.index}]'
                else:
                    self.path = f'{absolute_path_of_current_workitem}/{self.workitem_type}'
                    self.path = self.path.replace('//', '/')

            self.workitem_type = None
            self.id = None
            self.index = None

            return self

    class Output(BaseModel):
        current_workitem_has_changed: bool = Field(
            description="Whether the move caused the current workitem to change")
        new_context: Optional[str] = Field(
            default=None,
            description="The new context of the current workitem has changed")

    plain_utterances = [
        "navigate to /Feature[index=0]",
        "go to first epic",
        "switch to the x8745 bug"
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
        
        # Get the target work item by absolute path
        target_workitem = current_workitem.get_workitem(input.path)
        
        new_context = None
        current_workitem_has_changed = False
        if target_workitem and target_workitem is not current_workitem:
            # Change the current context to the first work item
            workflow.record = target_workitem
            current_workitem_has_changed = True
            new_context = WorkItemContext.get_displayname(target_workitem)

        return Signature.Output(
            current_workitem_has_changed = current_workitem_has_changed,
            new_context = new_context)

    def __call__(self, workflow: fastworkflow.Workflow, command: str, 
                 command_parameters: Signature.Input) -> Signature.Output:
        output = self._process_command(workflow, command_parameters)
        
        # Format the status information
        
        return output