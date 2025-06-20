from __future__ import annotations

import contextlib
"""Core command that reports the current context name and optional properties."""

import fastworkflow
from fastworkflow.session import WorkflowSnapshot


class Signature:  # noqa: D101
    """Show the current context and its properties (if any)."""

    @staticmethod
    def generate_utterances(*_args, **_kwargs):  # noqa: D401
        return [
            "what context am I in",
            "current command context",
            "where am I",
        ]


class ResponseGenerator:  # noqa: D101
    """Generate response describing the current context."""

    def __call__(
        self,
        session: fastworkflow.Session,
        command: str,
    ) -> fastworkflow.CommandOutput:
        subject_session = session.workflow_snapshot.workflow_context["subject_session"]
        return fastworkflow.CommandOutput(
            session_id=session.id,
            command_responses=[
                fastworkflow.CommandResponse(
                    response=subject_session.current_command_context_name,
                )
            ],
        )