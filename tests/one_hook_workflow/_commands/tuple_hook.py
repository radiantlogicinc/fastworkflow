"""An old-form hook: it returns a tuple, which is a call error now."""

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    class Input(BaseModel):
        name: str = Field(description="a name")

    class Output(BaseModel):
        name: str = Field(description="the name")

    plain_utterances = ["old hook"]

    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow, command: str,
                                      input: "Signature.Input") -> tuple[bool, str]:
        return (True, "")


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(name=input.name)
