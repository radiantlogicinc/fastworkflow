"""Returns a CommandResponse with success=False: a call error (raise fastworkflow.Refused instead)."""

import fastworkflow
from fastworkflow import CommandResponse
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    class Input(BaseModel):
        word: str = Field(description="a word")

    class Output(BaseModel):
        word: str = Field(description="the word")

    plain_utterances = ["failed response"]


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> CommandResponse:
        return CommandResponse(response="no", success=False)
