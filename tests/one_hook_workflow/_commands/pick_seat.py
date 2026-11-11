"""Execution command whose hook refuses one value and names its field (Refused(field=...))."""

import fastworkflow
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field, model_validator

TAKEN = "seat B9 is taken"


class Signature:
    class Input(BaseModel):
        seat: str = Field(description="the seat to pick")
        note: str = Field(description="a note kept when the seat is refused")

        @model_validator(mode="after")
        def refuse_taken_seat(self, info):
            if self.seat == "B9":
                raise fastworkflow.Refused(f"{TAKEN}; pick another", field="seat")
            return self

    class Output(BaseModel):
        seat: str = Field(description="the seat picked")
        note: str = Field(description="the note given")

    plain_utterances = ["pick a seat", "choose seat"]

class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, input: Signature.Input) -> Signature.Output:
        return Signature.Output(seat=input.seat, note=input.note)
