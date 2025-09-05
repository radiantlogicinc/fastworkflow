from __future__ import annotations

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances


class Signature:  # noqa: D101
    """Change context to the parent of the current context. This could change the commands that are available."""

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
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:  # noqa: D101
    """Handle command execution and craft the textual response."""
    def __call__(self, workflow: fastworkflow.Workflow, command: str) -> CommandOutput:
        # Move the context to its parent.
        app_workflow = workflow.context["app_workflow"]   #type: fastworkflow.Workflow

        if app_workflow.is_current_command_context_root:
            return CommandOutput(
                workflow_id=workflow.id,
                command_responses=[
                    CommandResponse(
                        response="Already at the top-level 'global' context.",
                    )
                ],
            )

        parent_context = app_workflow.get_parent(app_workflow.current_command_context)
        app_workflow.current_command_context = parent_context

        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(
                    response=f"Context is now '{app_workflow.current_command_context_displayname}'",
                )
            ],
        ) 