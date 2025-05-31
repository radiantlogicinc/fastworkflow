from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction

class CommandParameters(BaseModel):
    """Returns status (whether user address modification succeeded)"""
    user_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The user ID to modify",
            pattern=r"^([a-z]+_[a-z]+_\d+|NOT_FOUND)$",
            examples=["sara_doe_496"]
        )
    ]
    
    address1: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The first line of the address",
            examples=["123 Main St"]
        )
    ]
    
    address2: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The second line of the address",
            examples=["Apt 1"]
        )
    ]
    
    city: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The city name",
            examples=["San Francisco"]
        )
    ]
    
    state: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The state code",
            pattern=r"^([A-Z]{2}|NOT_FOUND)$",
            examples=["CA"]
        )
    ]
    
    country: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The country name",
            examples=["USA"]
        )
    ]
    
    zip: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The ZIP code",
            pattern=r"^(\d{5}|NOT_FOUND)$",
            examples=["12345"]
        )
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)