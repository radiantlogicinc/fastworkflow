from typing import Annotated, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict

from fastworkflow.session import WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction



class CommandParameters(BaseModel):
    order_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The order ID to cancel (must start with #)",
            pattern=r"^(#[\w\d]+|NOT_FOUND)$",
            examples=["#123", "#abc123", "#order456"]
        )
    ]

    reason: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="Reason for cancellation",
            enum=["no longer needed", "ordered by mistake", "NOT_FOUND"],
            examples=["no longer needed", "ordered by mistake"]
        )
    ]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )