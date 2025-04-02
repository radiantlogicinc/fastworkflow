from typing import Annotated, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

import dspy
import os
from typing import Annotated, Optional, Tuple, Union, Dict, Any, Type, List, get_args
from enum import Enum
from pydantic import BaseModel, Field, ValidationError, field_validator
import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from datetime import date
import re
import inspect
from difflib import get_close_matches
from fastworkflow.utils.pydantic_model_2_dspy_signature_class import TypedPredictorSignature
from fastworkflow.train.train_airline_workflow import DSPY_LM_MODEL
from fastworkflow.utils.signatures import InputForParamExtraction, DatabaseValidator


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

    workitem_id: Optional[Union[str, int]] = Field(
    default=None, 
    description="The ID of the workitem", 
    examples=["John Doe", "24"]
    )

    workitem_srno: int = Field(default=1, description="the serial number of the workitem", examples=[1,2,3,10,11])




