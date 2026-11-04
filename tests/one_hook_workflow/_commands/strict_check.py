"""Execution command whose hook refuses without a field (the whole call is refused)."""

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    class Input(BaseModel):
        count: int = Field(description="how many at once")

    class Output(BaseModel):
        count: int = Field(description="how many were started")

    plain_utterances = ["start many"]

    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow, command: str,
                                      input: "Signature.Input") -> None:
        if input.count > 2:
            raise fastworkflow.Refused("no more than 2 can start at once")


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(count=input.count)
