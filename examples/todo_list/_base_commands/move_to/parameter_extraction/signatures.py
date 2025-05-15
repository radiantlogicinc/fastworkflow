from typing import Annotated, Optional, Tuple, Union

from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator

import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.workflow_definition import NodeType

class CommandParameters(BaseModel):
    workitem_path: str = Field(
        default="NOT_FOUND",
        description="The path of the workflow or workitem",
        pattern=r"^(//[^/]+|/[^/]+(?:/[^/]+)*|[^/]+(?:/[^/]+)*)$",
        examples=[
            "/<workflow_name>/<workitem_name>",
            "<another_workitem_name>",
            "//<another_path_name>",
        ],         
        json_schema_extra={
            "db_lookup": True
        }
    )

    workitem_id: Optional[Union[str, int]] = Field(
        description="The ID of the workitem", 
        examples=["John Doe", "24"]),


class InputForParamExtraction(BaseModel):
    """Extract the workitem path and optionally the workitem id from the command. 
Workitem id (if present) can be an integer or a string.
Valid workitem paths are:\n
{workitem_type_list} 
"""
    command: str
    workflow_snapshot: WorkflowSnapshot
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, command: str):
        return cls(
            command=command,
            workflow_snapshot=workflow_snapshot
        )
    
    def db_lookup(self, _:str) -> list[str]: 
        workflow_folderpath = self.workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)

        return workflow_definition.paths_2_typemetadata.keys()

