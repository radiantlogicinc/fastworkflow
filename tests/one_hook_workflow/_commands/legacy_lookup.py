"""A retired db_lookup, declared as a json_schema_extra key: a load error."""

from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    class Input(BaseModel):
        user_name: str = Field(description="a user", json_schema_extra={"db_lookup": True})

    class Output(BaseModel):
        user_name: str = Field(description="the user")

    plain_utterances = ["legacy lookup"]


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(user_name=input.user_name)
