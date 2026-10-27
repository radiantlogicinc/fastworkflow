"""add_note: add a note to the open record, a book's or a member's."""
import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Add a note to the open record."""

    class Input(BaseModel):
        text: str = Field(description="the note to add", examples=["the cover is torn"])

    class Output(BaseModel):
        total: int = Field(description="how many notes the record holds now")

    plain_utterances = ["add a note: card renewed", "note that the cover is torn"]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        record = workflow.command_context_for_response_generation
        record.notes.append(command_parameters.text)
        return Signature.Output(total=len(record.notes))
