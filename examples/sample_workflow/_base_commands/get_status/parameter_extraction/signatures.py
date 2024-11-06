from typing import Annotated, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

from fastworkflow.session import Session


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
    ]


class InputForParamExtraction(BaseModel):
    """Extract the command parameters from the command"""

    command: str

    @classmethod
    def create(cls, session: Session, command: str):
        return cls(
            command=command,
        )

    @classmethod
    def validate_parameters(
        cls, session: Session, cmd_parameters: CommandParameters
    ) -> Tuple[bool, str]:
        """
        Check if the parameters are valid in the current context.
        Parameter is a single field pydantic model.
        Return a tuple with a boolean indicating success or failure.
        And a string with helpful information about the error and suggestions for fixing it.
        """
        if cmd_parameters.workitem_path == "INVALID":
            return (False, "Missing or invalid workitem path")

        if cmd_parameters.workitem_path == "NOT_FOUND":
            active_workitem = session.get_active_workitem()
            if active_workitem:
                return (True, None)

        workitem_path = "".join(cmd_parameters.workitem_path.split()).strip(" \"'")
        relative_to_root = False
        if workitem_path in [
            workitem_type for workitem_type in session.workflow_definition.types
        ]:
            relative_to_root = True
            workitem_path = f"//{workitem_path}"

        workitem = session.workflow.find_workitem(
            workitem_path, cmd_parameters.workitem_id, relative_to_root
        )
        if workitem is None:
            return (
                False,
                "workitem path was valid but does not exist in this workflow",
            )

        return (True, None)

    class Config:
        arbitrary_types_allowed = True
