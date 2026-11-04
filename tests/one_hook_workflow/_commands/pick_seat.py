"""Execution command whose hook refuses one value and names its field (Refused(field=...))."""

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field

TAKEN = "seat B9 is taken"


class Signature:
    class Input(BaseModel):
        seat: str = Field(description="the seat to pick")
        note: str = Field(description="a note kept when the seat is refused")

    class Output(BaseModel):
        seat: str = Field(description="the seat picked")
        note: str = Field(description="the note given")

    plain_utterances = ["pick a seat", "choose seat"]

    @staticmethod
    def validate_extracted_parameters(workflow: fastworkflow.Workflow, command: str,
                                      input: "Signature.Input") -> None:
        if input.seat == "B9":
            raise fastworkflow.Refused(f"{TAKEN}; pick another", field="seat")


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(seat=input.seat, note=input.note)
