import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.user import User

# the signature class defines our intent
class Signature:
    class Input(BaseModel):
        to: str = Field(
            description="Who are you sending the message to",
            examples=['jsmith@abc.com', 'jane.doe@xyz.edu'],
            pattern=r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        )
        message: str = Field(
            description="The message you want to send",
            examples=['Hello, how are you?', 'Hi, reaching out to discuss fastWorkflow'],
            min_length=3,
            max_length=500
        )

    plain_utterances = [
        "Tell john@fastworkflow.ai that the build tool needs improvement",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Call the User.send_message instance method."""

    def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> None:
        user: User = workflow.command_context_for_response_generation
        user.send_message(to=input.to, message=input.message)

    def __call__(self, workflow: 
                 fastworkflow.Workflow, 
                 command: str, 
                 command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """The framework will call this function to process the command"""
        self._process_command(workflow, command_parameters)
        
        response = (
            f'Response: The message was printed to the screen'
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
