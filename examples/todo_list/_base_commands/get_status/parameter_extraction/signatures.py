from typing import Annotated, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

import fastworkflow
from fastworkflow.session import WorkflowSnapshot


class CommandParameters(BaseModel):
    @field_validator("workitem_path", mode="wrap")
    @staticmethod
    def validate_workitem_path(workitem_path, handler):
        try:
            return handler(workitem_path)
        except ValidationError:
            return "INVALID"

    workitem_path: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The path of the workflow or workitem",
            pattern=r"^(//[^/]+|/[^/]+(?:/[^/]+)*|[^/]+(?:/[^/]+)*)$",
            examples=[
                "/<workflow_name>/<workitem_name>",
                "<another_workitem_name>",
                "//<another_path_name>",
            ],
            invalid_value="INVALID",
        ),
    ]
    workitem_id: Optional[
        Annotated[
            Union[str, int],
            Field(description="The ID of the workitem", examples=["John Doe", "24"]),
        ]
    ]


class InputForParamExtraction(BaseModel):
    """Extract the command parameters from the command"""

    command: str

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        return cls(
            command=command,
        )

    @classmethod
    def validate_parameters(
        cls, workflow_snapshot: WorkflowSnapshot, cmd_parameters: CommandParameters
    ) -> Tuple[bool, str]:
        """
        Check if the parameters are valid in the current context.
        Parameter is a single field pydantic model.
        Return a tuple with a boolean indicating success or failure.
        And a string with helpful information about the error and suggestions for fixing it.
        """
        if cmd_parameters.workitem_path == "INVALID":
            return (False, "Missing or invalid workitem path")

        if cmd_parameters.workitem_path == "NOT_FOUND":
            if active_workitem := workflow_snapshot.active_workitem:
                cmd_parameters.workitem_path = active_workitem.path
                cmd_parameters.workitem_id = active_workitem.id
                return (True, None)

        workflow_folderpath = workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)

        workitem_path = "".join(cmd_parameters.workitem_path.split()).strip(" \"'")
        workitem = workflow_snapshot.workflow.find_workitem(
            workitem_path, cmd_parameters.workitem_id, True
        )
        if workitem is None:
            return (
                False,
                "workitem path was valid but does not exist in this workflow",
            )

        cmd_parameters.workitem_path = workitem.path
        cmd_parameters.workitem_id = workitem.id
        return (True, None)

    class Config:
        arbitrary_types_allowed = True
