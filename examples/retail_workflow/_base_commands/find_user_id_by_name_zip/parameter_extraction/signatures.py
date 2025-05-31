from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction

class CommandParameters(BaseModel):
    """Returns the user_id"""
    first_name: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The first name of the customer",
            pattern=r"^(NOT_FOUND|[A-Za-z]+)$",
            examples=["John"]
        )
    ]
    
    last_name: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The last name of the customer",
            pattern=r"^(NOT_FOUND|[A-Za-z]+)$",
            examples=["Doe"]
        )
    ]
    
    zip: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The zip code of the customer",
            pattern=r"^(NOT_FOUND|\d{5})$",
            examples=["12345"]
        )
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)