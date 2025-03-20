from typing import Annotated, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.workflow_definition import NodeType

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
    """Extract the workitem path and optionally the workitem id from the command. 
Workitem id (if present) can be an integer or a string.
Valid workitem paths are:\n
{workitem_type_list} 
"""

    command: str

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        workflow_folderpath = workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)
        workitem_type_list = "\n".join(workflow_definition.paths_2_typemetadata.keys())

        cls.__doc__ = cls.__doc__.format(workitem_type_list=workitem_type_list)

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
        workflow_folderpath = workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)
        workflow_path_list = [
            path for path, type_metadata in workflow_definition.paths_2_typemetadata.items()
            if type_metadata.node_type == NodeType.Workflow
        ]
        if cmd_parameters.workitem_path in ["NOT_FOUND", "INVALID"]:
            workflow_path_list_str = "\n".join(workflow_path_list)
            return (
                False,
                "Workitem path is missing or invalid.\n"
                f"Valid workitem paths are:\n"
                f"{workflow_path_list_str}\n"
            )

        cmd_parameters.workitem_path = "".join(cmd_parameters.workitem_path.split()).strip(" \"'")
        workitem = workflow_snapshot.workflow.find_workitem(
            cmd_parameters.workitem_path, cmd_parameters.workitem_id
        )
        if workitem is None:
            return (
                False,
                f"workitem path '{cmd_parameters.workitem_path}' does not exist in this workflow\n"
                f"Valid workitem paths are:\n"
                f"{workflow_path_list_str}\n"
            )

        return (True, None)

    class Config:
        arbitrary_types_allowed = True
