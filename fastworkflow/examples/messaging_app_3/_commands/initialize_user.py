import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ..application.user import User, PremiumUser


class Signature:
    """Initialize the workflow with a root User context."""

    class Input(BaseModel):
        name: str = Field(
            description="Name of a person",
            examples=['John', 'Jane Doe'],
            default='DefaultUser'
        )
        is_premium_user: bool = Field(
            description="Whether the person is a regular user or a premium user"
        )

    plain_utterances = [
        "start messaging session as Jack",
        "initialize user context",
        "login as user Billy who is a premium user",
        "startup with Mary, a regular user",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """Generate training utterances for LLM-based intent matching."""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Create a User instance and attach it as the root command context."""

    def __call__(
        self,
        workflow: fastworkflow.Workflow,
        command: str,
        command_parameters: Signature.Input,
    ) -> fastworkflow.CommandOutput:
        # Initialize the current command context
        workflow.current_command_context = (
            PremiumUser(command_parameters.name) if command_parameters.is_premium_user else
            User(command_parameters.name)
        )

        response = (
            f"Root context set to User('{command_parameters.name}')."
            f"Now you can call commands exposed in this context."
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
