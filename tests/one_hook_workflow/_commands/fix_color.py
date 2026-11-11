"""Execution command whose hook corrects a value in place (written back, recorded as validation_hook)."""

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field, model_validator


class Signature:
    class Input(BaseModel):
        color: str = Field(description="a color")

        @model_validator(mode="after")
        def lowercase_color(self, info):
            if self.color != self.color.lower():
                self.color = self.color.lower()
            return self

    class Output(BaseModel):
        color: str = Field(description="the color used")

    plain_utterances = ["paint it", "use color"]

class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(color=input.color)
