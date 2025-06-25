from __future__ import annotations

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances


class Signature:  # noqa: D101
    """Change context to the parent of the current context."""

    plain_utterances = [
        "go up",
        "up",
        "parent context",
        "go up a level",
        "expand context",
        "one level up",
        "move up"
    ]

    @staticmethod
    def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:  # noqa: D101
    """Handle command execution and craft the textual response."""
    def __call__(self, session: fastworkflow.Session, command: str) -> CommandOutput:
        # Move the context to its parent.
        subject_session = session.workflow_context["subject_session"]   #type: fastworkflow.Session

        if subject_session.is_current_command_context_root:
            return CommandOutput(
                session_id=session.id,
                command_responses=[
                    CommandResponse(
                        response="Already at the top-level 'global' context.",
                    )
                ],
            )

        parent_context = subject_session.get_parent(subject_session.current_command_context)
        subject_session.current_command_context = parent_context

        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response=f"Context is now '{subject_session.current_command_context_displayname}'",
                )
            ],
        ) 