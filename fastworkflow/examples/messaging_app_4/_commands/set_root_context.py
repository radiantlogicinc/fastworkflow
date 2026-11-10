import fastworkflow

from ..application.chatroom import ChatRoom


class ResponseGenerator:
    """Create a User instance and attach it as the root command context."""

    def __call__(
        self,
        workflow: fastworkflow.Workflow,
        command: str, input: None
    ) -> str:
        # Initialize the root command context
        workflow.root = ChatRoom()

        response = (
            f"Now you can call commands exposed in this context."
        )

        return response
