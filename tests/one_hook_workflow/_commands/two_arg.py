"""A two-argument __call__: a load error (the call takes workflow, command and input)."""

from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    class Input(BaseModel):
        word: str = Field(description="a word")

    class Output(BaseModel):
        word: str = Field(description="the word")

    plain_utterances = ["two argument call"]


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str) -> Signature.Output:
        return Signature.Output(word="x")
