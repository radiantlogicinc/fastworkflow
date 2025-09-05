import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.workflow import Workflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, ConfigDict, Field


class Signature:
    plain_utterances = [
        "you_misunderstood",
        "That is not what I meant",
        "Not what I asked",
        "You misunderstood",
        "None of these commands",
        "Incorrect command",
        "Wrong command",
        "Change command",
        "Different command",
    ]

    class Output(BaseModel):
        valid_command_names: list[str]

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    def _process_command(self, workflow: Workflow) -> Signature.Output:
        app_workflow = workflow.context["app_workflow"]  #type: fastworkflow.Workflow
        subject_crd = fastworkflow.RoutingRegistry.get_definition(
            app_workflow.folderpath)
        
        crd = fastworkflow.RoutingRegistry.get_definition(
            workflow.folderpath)
        cme_command_names = crd.get_command_names('IntentDetection')

        fully_qualified_command_names = (
            set(cme_command_names) | 
            set(subject_crd.get_command_names(app_workflow.current_command_context_name))
        )

        valid_command_names = [
            fully_qualified_command_name.split('/')[-1] 
            for fully_qualified_command_name in fully_qualified_command_names
        ]

        return Signature.Output(valid_command_names=sorted(valid_command_names))

    def __call__(self, workflow: Workflow, command: str) -> CommandOutput:
        output = self._process_command(workflow)

        response = (
            "\n".join([
                f"{command_name}"
                for command_name in output.valid_command_names
            ])
        )
        response = (
            "Please enter the correct command from the list below:\n"
            f"{response}\n\nor type 'abort' to cancel"
        )

        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(
                    response=response,
                    artifacts=output.model_dump(),
                )
            ],
        ) 