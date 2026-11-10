"""Execution command whose hook refuses without a field (the whole call is refused)."""

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field, model_validator


class Signature:
    class Input(BaseModel):
        count: int = Field(description="how many at once")

        @model_validator(mode="after")
        def refuse_count_over_two(self, info):
            if self.count > 2:
                raise fastworkflow.Refused("no more than 2 can start at once")
            return self

    class Output(BaseModel):
        count: int = Field(description="how many were started")

    plain_utterances = ["start many"]

class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(count=input.count)
