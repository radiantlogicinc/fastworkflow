from typing import Annotated, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict

import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.workflow_definition import NodeType
from fastworkflow.utils.signatures import InputForParamExtraction

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
    ] = None

class InputForParamExtraction(BaseModel):
    command: str
    workflow_snapshot: WorkflowSnapshot
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        return cls(command=command, workflow_snapshot=workflow_snapshot)
    
    def db_lookup(self, _:str) -> list[str]: 
        workflow_folderpath = self.workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)

        return workflow_definition.paths_2_typemetadata.keys()
        


    def validate_parameters(
        self, cmd_parameters: CommandParameters
    ) -> Tuple[bool, str]:
        """
        Check if the parameters are valid in the current context.
        Parameter is a single field pydantic model.
        Return a tuple with a boolean indicating success or failure.
        And a message with suggested parameter values that are closest matches to the input.
        """
        if cmd_parameters.workitem_path == "NOT_FOUND":
            cmd_parameters.workitem_path = self.workflow_snapshot.active_workitem.path

        return (True, None)