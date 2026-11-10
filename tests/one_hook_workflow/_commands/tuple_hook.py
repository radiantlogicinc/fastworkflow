"""An old-form hook: it returns a tuple, which is a call error now."""

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field, model_validator


class Signature:
    class Input(BaseModel):
        name: str = Field(description="a name")

        @model_validator(mode="after")
        def returns_a_tuple(self, info):
            return (True, "")
            return self

    class Output(BaseModel):
        name: str = Field(description="the name")

    plain_utterances = ["old hook"]

class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(name=input.name)
