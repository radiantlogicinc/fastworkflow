from fastworkflow import CommandOutput, CommandResponse
import fastworkflow

from fastworkflow.train.generate_synthetic import generate_diverse_utterances


class Signature:  # noqa: D101
    """Reset the current context to the global context (*). This could change the commands that are available."""
    plain_utterances = [
        "reset context",
        "clear context",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:  # noqa: D101
    """Handle command execution and craft the textual response."""

    def __call__(self, workflow: fastworkflow.Workflow, command: str) -> CommandOutput:
        # Clear the current context so subsequent commands operate at global level
        app_workflow = workflow.context["app_workflow"]
        app_workflow.current_command_context = app_workflow.root_command_context
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(
                    response=f"Context is now '{app_workflow.current_command_context_name}'",
                )
            ],
        ) 