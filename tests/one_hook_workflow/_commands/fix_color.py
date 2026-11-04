"""Execution command whose hook corrects a value in place (written back, recorded as validation_hook)."""

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    class Input(BaseModel):
        color: str = Field(description="a color")

    class Output(BaseModel):
        color: str = Field(description="the color used")

    plain_utterances = ["paint it", "use color"]

    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow, command: str,
                                      input: "Signature.Input") -> None:
        if input.color != input.color.lower():
            input.color = input.color.lower()


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(color=input.color)
