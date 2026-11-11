"""A retired Signature.db_lookup method: a load error."""

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    class Input(BaseModel):
        user_name: str = Field(description="a user")

    class Output(BaseModel):
        user_name: str = Field(description="the user")

    plain_utterances = ["legacy method"]

    @staticmethod
    def db_lookup(workflow: fastworkflow.Workflow, field_name: str, field_value: str):
        return (False, None, [])


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(user_name=input.user_name)
