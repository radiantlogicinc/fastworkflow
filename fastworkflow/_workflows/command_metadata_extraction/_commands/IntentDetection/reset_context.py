from fastworkflow import CommandOutput, CommandResponse
import fastworkflow

from fastworkflow.train.generate_synthetic import generate_diverse_utterances


class Signature:  # noqa: D101
    """Reset the current context to the global context (*)."""
    plain_utterances = [
        "reset context",
        "clear context",
    ]

    @staticmethod
    def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
        return generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:  # noqa: D101
    """Handle command execution and craft the textual response."""

    def __call__(self, session: fastworkflow.Session, command: str) -> CommandOutput:
        # Clear the current context so subsequent commands operate at global level
        subject_session = session.workflow_context["subject_session"]
        subject_session.current_command_context = subject_session.root_command_context
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response=f"Context is now '{subject_session.current_command_context_name}'",
                )
            ],
        ) 