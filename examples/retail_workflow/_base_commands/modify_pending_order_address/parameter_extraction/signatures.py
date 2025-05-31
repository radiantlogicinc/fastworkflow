from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction

class CommandParameters(BaseModel):
    """Returns status (whether order delivery address modification succeeded)"""
    order_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The order ID to modify (must start with #)",
            pattern=r"^(#W\d+|NOT_FOUND)$",
            examples=["#W0000000"]
        )
    ]
    
    address1: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="First line of address",
            examples=["123 Main St"]
        )
    ]
    
    address2: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="Second line of address",
            examples=["Apt 1"]
        )
    ]
    
    city: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="City name",
            examples=["San Francisco"]
        )
    ]
    
    state: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="State code",
            pattern=r"^([A-Z]{2}|NOT_FOUND)$",
            examples=["CA"]
        )
    ]
    
    country: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="Country name",
            examples=["USA"]
        )
    ]
    
    zip: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="ZIP code",
            pattern=r"^(\d{5}|NOT_FOUND)$",
            examples=["12345"]
        )
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)