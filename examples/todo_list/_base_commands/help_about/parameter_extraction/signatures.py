from typing import Tuple

from pydantic import BaseModel, Field, ConfigDict

import fastworkflow
from fastworkflow.session import WorkflowSnapshot


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

