"""startup: make a fresh Library the root of the workflow."""
import fastworkflow

from ..application.library import Library


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 input: None) -> str:
        workflow.root_command_context = Library()
        return "The library is open."
