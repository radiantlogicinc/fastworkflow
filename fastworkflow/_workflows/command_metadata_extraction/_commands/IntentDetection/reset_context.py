from fastworkflow import CommandOutput, CommandResponse
import fastworkflow
from fastworkflow.session import WorkflowSnapshot


class Signature:  # noqa: D101
    """Reset the current context to the global context (*)."""

    # Parameter-free command – future framework may introspect annotations
    @staticmethod
    def generate_utterances(*_args, **_kwargs):  # noqa: D401
        """Return generic reset utterances list."""
        return [
            "reset context",
            "clear context",
        ]


class ResponseGenerator:  # noqa: D101
    """Handle command execution and craft the textual response."""

    def __call__(self, session: fastworkflow.Session, command: str) -> CommandOutput:
        # Clear the current context so subsequent commands operate at global level
        subject_session = session.workflow_snapshot.workflow_context["subject_session"]
        subject_session.current_command_context = subject_session.root_command_context
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response="Context is now global.",
                )
            ],
        ) 