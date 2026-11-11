"""record_values: records four values, each a plain required field (no default)."""

from typing import List

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field


class Signature:
    """Record four values."""

    class Input(BaseModel):
        name: str = Field(description="a name")
        count: int = Field(description="a count")
        ratio: float = Field(description="a ratio")
        ids: List[str] = Field(description="a list of ids")

    class Output(BaseModel):
        summary: str = Field(description="the recorded values")

    plain_utterances = ["record the values", "record these four values"]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> List[str]:
        return ["record the values"]


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> fastworkflow.CommandResponse:
        p = input
        summary = f"name={p.name}; count={p.count}; ratio={p.ratio}; ids={p.ids}"
        return fastworkflow.CommandResponse(
            response=summary, output=Signature.Output(summary=summary))
