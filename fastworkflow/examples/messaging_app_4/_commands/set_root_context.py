import fastworkflow

from ..application.chatroom import ChatRoom


class ResponseGenerator:
    """Create a User instance and attach it as the root command context."""

    def __call__(
        self,
        workflow: fastworkflow.Workflow,
        command: str
    ) -> fastworkflow.CommandOutput:
        # Initialize the root command context
        workflow.root_command_context = ChatRoom()

        response = (
            f"Now you can call commands exposed in this context."
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
