"""show_notes: the notes on the open record, a book's or a member's."""
import fastworkflow
from pydantic import BaseModel, Field


class Signature:
    """Show the notes on the open record."""

    class Output(BaseModel):
        notes: list[str] = Field(description="each note, oldest first")
        total: int = Field(description="how many notes the record holds")

    plain_utterances = [
        "show the notes on this record",
        "any notes here",
        "what notes are on this one",
        "read me the notes",
        "list the remarks on this record, oldest first",
        "how many notes does this record hold",
        "is there anything written about this record",
        "let me see the note history",
    ]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: None) -> Signature.Output:
        record = workflow.command_context_for_response_generation
        return Signature.Output(notes=list(record.notes), total=len(record.notes))
