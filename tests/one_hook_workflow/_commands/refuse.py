"""Refuses a call with raise fastworkflow.Refused(message): the call's answer is the message, success=False."""

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field

REFUSAL = "the vault is closed on Sundays"


class Signature:
    class Input(BaseModel):
        word: str = Field(description="a word")

    class Output(BaseModel):
        word: str = Field(description="the word")

    plain_utterances = ["open the vault"]


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        raise fastworkflow.Refused(REFUSAL)
