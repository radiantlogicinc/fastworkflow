from __future__ import annotations

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow_agent import core_command_names


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


#: Most command names the top-level hint lists.
MAX_HINT_COMMANDS = 8


def _top_level_hint(app_workflow: fastworkflow.Workflow) -> str:
    """The " From here, use one of: ..." sentence for the top-level context's own commands; "" if none.

    The core command set (the internal command_metadata_extraction contexts) is
    what is left out: navigation and meta commands are not work to do here.
    """
    routing = fastworkflow.RoutingRegistry.get_definition(app_workflow.folderpath)
    core = core_command_names()
    names = sorted(
        name.split("/")[-1]
        for name in routing.get_command_names(app_workflow.current_command_context_name)
        if name not in core)
    if not names:
        return ""
    return f" From here, use one of: {', '.join(names[:MAX_HINT_COMMANDS])}."


class ResponseGenerator:  # noqa: D101
    """Handle command execution and craft the textual response."""
    def __call__(self, workflow: fastworkflow.Workflow, command: str) -> CommandOutput:
        # Move the context to its parent.
        app_workflow = workflow.context["app_workflow"]   #type: fastworkflow.Workflow

        if app_workflow.is_current_command_context_root:
            return CommandOutput(
                command_response=
                    CommandResponse(
                        response="Already at the top-level 'global' context."
                                 + _top_level_hint(app_workflow),
                    ),
            )

        parent_context = app_workflow.get_parent(app_workflow.current_command_context)
        app_workflow.current_command_context = parent_context

        return CommandOutput(
            command_response=
                CommandResponse(
                    response=f"Context is now '{app_workflow.current_command_context_displayname}'",
                ),
        ) 