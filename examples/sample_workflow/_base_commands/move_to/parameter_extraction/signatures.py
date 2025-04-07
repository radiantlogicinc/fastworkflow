from typing import Annotated, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

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