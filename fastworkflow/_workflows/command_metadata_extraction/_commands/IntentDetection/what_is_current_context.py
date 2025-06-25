from __future__ import annotations

import contextlib

from fastworkflow.train.generate_synthetic import generate_diverse_utterances
"""Core command that reports the current context name and optional properties."""

import fastworkflow



class Signature:  # noqa: D101
    """Show the current context and its properties (if any)."""
    plain_utterances = [
        "what context am I in",
        "current command context",
        "where am I",
    ]

    @staticmethod
    def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
        return generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:  # noqa: D101
    """Generate response describing the current context."""

    def __call__(
        self,
        session: fastworkflow.Session,
        command: str,
    ) -> fastworkflow.CommandOutput:
        subject_session = session.workflow_context["subject_session"]
        current_context = (
            'global' if subject_session.current_command_context_name == '*'
            else subject_session.current_command_context_name
        )
        return fastworkflow.CommandOutput(
            session_id=session.id,
            command_responses=[
                fastworkflow.CommandResponse(
                    response = f"Current context is '{current_context}'"
                )
            ],
        )