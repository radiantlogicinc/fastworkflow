import sys
from typing import Optional
import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.workitem import WorkItem


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

    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow, command: str, cmd_parameters: "Signature.Input") -> tuple[bool, str]:
        if cmd_parameters.path == fastworkflow.get_env_var('NOT_FOUND'):
            cmd_parameters.path = None
        if cmd_parameters.workitem_type == fastworkflow.get_env_var('NOT_FOUND'):
            cmd_parameters.workitem_type = None
        if cmd_parameters.id == fastworkflow.get_env_var('NOT_FOUND'):
            cmd_parameters.id = None
        if cmd_parameters.index == -sys.maxsize:
            cmd_parameters.index = None

        if not cmd_parameters.path:
            if not cmd_parameters.workitem_type:
                return (False, "Provide the absolute path or the child workitem type and optionally its id or index")

            current_workitem: WorkItem = workflow.command_context_for_response_generation
            absolute_path_of_current_workitem = current_workitem.get_absolute_path()
            if cmd_parameters.id:
                cmd_parameters.path = f'{absolute_path_of_current_workitem}/{cmd_parameters.workitem_type}[id={cmd_parameters.id}]'
            elif cmd_parameters.index:
                cmd_parameters.index -= 1   # python indexes are 0 based
                cmd_parameters.path = f'{absolute_path_of_current_workitem}/{cmd_parameters.workitem_type}[index={cmd_parameters.index}]'
            else:
                cmd_parameters.path = f'{absolute_path_of_current_workitem}/{cmd_parameters.workitem_type}'
                cmd_parameters.path = cmd_parameters.path.replace('//', '/')

        cmd_parameters.workitem_type = None
        cmd_parameters.id = None
        cmd_parameters.index = None

        return (True, '')


class ResponseGenerator:
    """Handle command execution and craft the textual response."""

    def _process_command(self, 
                         workflow: fastworkflow.Workflow, 
                         input: Signature.Input) -> Signature.Output:
        current_workitem: WorkItem = workflow.command_context_for_response_generation
        
        # Get the target work item by absolute path
        target_workitem = current_workitem.get_workitem(input.path)
        
        new_context = None
        current_workitem_has_changed = False
        if target_workitem and target_workitem is not current_workitem:
            # Change the current context to the first work item
            workflow.current_command_context = target_workitem
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