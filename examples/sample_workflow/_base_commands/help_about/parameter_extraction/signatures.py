from typing import Tuple

from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.session import WorkflowSnapshot


class CommandParameters(BaseModel):
    workitem_type: str = Field(default="NOT_FOUND", description="The workitem type")


class InputForParamExtraction(BaseModel):
    command: str
    current_context: str

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        cls.__doc__ = (
            "Given the following list of workitem types: {workitem_types}\n"
            "Infer the workitem type from the command; and failing that, from the current context\n"
            "Return the default value if the inferred workitem type is not in the list"
        )

        workflow_folderpath = workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)
        workitem_types = ", ".join(workflow_definition.types.keys())
        cls.__doc__ = cls.__doc__.format(workitem_types=workitem_types)

        return cls(command=command, current_context=workflow_snapshot.get_active_workitem().type)

    @classmethod
    def validate_parameters(
        cls, workflow_snapshot: WorkflowSnapshot, cmd_parameters: CommandParameters
    ) -> Tuple[bool, str]:
        """
        Check if the parameters are valid in the current context.
        Parameter is a single field pydantic model.
        Return a tuple with a boolean indicating success or failure.
        And a message with suggested parameter values that are closest matches to the input.
        """
        workflow_folderpath = workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)
        if cmd_parameters.workitem_type in workflow_definition.types:
            return (True, None)

        workitem_types = "\n".join(workflow_definition.types.keys())
        return (
            False,
            (
                f"The workitem type {cmd_parameters.workitem_type} is not in the list of valid workitem types:\n"
                f"{workitem_types}\n"
                "Please choose a valid workitem type from the list"
            ),
        )

    class Config:
        arbitrary_types_allowed = True
