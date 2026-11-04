"""Returns a hand-built CommandOutput: a call error now (return the Output, a str or a CommandResponse)."""

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    class Input(BaseModel):
        word: str = Field(description="a word")

    class Output(BaseModel):
        word: str = Field(description="the word")

    plain_utterances = ["hand built envelope"]


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> CommandOutput:
        return CommandOutput(command_response=CommandResponse(response=input.word))
