from typing import Annotated, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict

import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.workflow_definition import NodeType
from fastworkflow.utils.signatures import InputForParamExtraction

class CommandParameters(BaseModel):
    workitem_path: str = Field(
        default="NOT_FOUND", 
        description="The workitem type",
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
        default=None,
        description="The ID of the workitem", 
        examples=["John Doe", "24"]
    )

class InputForParamExtraction(BaseModel):
    command: str
    workflow_snapshot: WorkflowSnapshot
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, workflow_snapshot: WorkflowSnapshot, _: str, command: str):
        return cls(command=command, workflow_snapshot=workflow_snapshot)
    
    def db_lookup(self, _:str) -> list[str]: 
        workflow_folderpath = self.workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)

        return workflow_definition.paths_2_typemetadata.keys()
