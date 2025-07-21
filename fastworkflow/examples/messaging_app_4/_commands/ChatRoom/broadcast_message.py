import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.chatroom import ChatRoom
from ...application.user import User, PremiumUser

# the signature class defines our intent
class Signature:
    class Input(BaseModel):
        message: str = Field(
            description="The message to be broadcast by the current user",
            examples=['Dont drink and drive', 'Please meet me at noon at the train station'],
            min_length=3,
            max_length=200
        )

    plain_utterances = [
        "Tell everybody dinner is served"
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Call the Chatroom.broadcast instance method."""

    def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> None:
        chatroom: ChatRoom = workflow.command_context_for_response_generation
        chatroom.broadcast(input.message)

    def __call__(self, workflow: 
                 fastworkflow.Workflow, 
                 command: str, 
                 command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """The framework will call this function to process the command"""
        self._process_command(workflow, command_parameters)
        
        response = (
            f'Response: Message has been broadcast'
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
