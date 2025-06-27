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
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:  # noqa: D101
    """Generate response describing the current context."""

    def __call__(
        self,
        workflow: fastworkflow.Workflow,
        command: str,
    ) -> fastworkflow.CommandOutput:
        app_workflow = workflow.context["app_workflow"]
        current_context = (
            'global' if app_workflow.current_command_context_name == '*'
            else app_workflow.current_command_context_name
        )
        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(
                    response = f"Current context is '{current_context}'"
                )
            ],
        )