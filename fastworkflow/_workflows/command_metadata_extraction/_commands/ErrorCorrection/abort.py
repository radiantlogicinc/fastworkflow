from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.workflow import Workflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, ConfigDict


class Signature:
    class Output(BaseModel):
        command: str
        command_name: str

    plain_utterances = [
        "abort",
        "cancel",
        "stop",
        "quit",
        "terminate",
        "end",
        "never mind",
        "exit"
    ]

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    def _process_command(self, workflow: Workflow, command: str) -> Signature.Output:
        workflow.end_command_processing()
        return Signature.Output(command=command, command_name="abort")

    def __call__(self, workflow: Workflow, command: str) -> CommandOutput:
        output = self._process_command(workflow, command)
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(
                    response="command aborted\n",
                    artifacts=output.model_dump(),
                )
            ],
        ) 