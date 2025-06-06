import fastworkflow
from fastworkflow import CommandOutput, CommandResponse

from ..response_generation.command_name_prediction import process_command as command_name_prediction
from ..response_generation.command_name_prediction import get_valid_command_names
from ..response_generation.parameter_extraction import process_command as parameter_extraction

class ResponseGenerator:
    def __call__(
        self, 
        session: fastworkflow.Session, 
        command: str,
    ) -> CommandOutput:
        output_of_command_name_prediction = command_name_prediction(session, command)
        if (not output_of_command_name_prediction.command_name):
            return CommandOutput(
                command_responses=[
                    CommandResponse(
                        response=output_of_command_name_prediction.error_msg,
                        success=False
                    )
                ]
            )
        
        # if its a command in command metadata extraction workflow
        valid_command_names = get_valid_command_names(session.workflow_snapshot)
        if output_of_command_name_prediction.command_name in valid_command_names:        
            subject_workflow_snapshot = session.workflow_snapshot
        else:
            subject_workflow_snapshot = session.workflow_snapshot.context["subject_workflow_snapshot"]

        output_of_parameter_extraction = parameter_extraction(
            session,
            subject_workflow_snapshot, 
            output_of_command_name_prediction.command_name,
            command
        )
        if not output_of_parameter_extraction.parameters_are_valid:
            return CommandOutput(
                command_responses=[
                    CommandResponse(
                        response=(
                            f"PARAMETER EXTRACTION ERROR FOR COMMAND '{output_of_command_name_prediction.command_name}'\n"
                            f"{output_of_parameter_extraction.error_msg}"
                        ),
                        success=False
                    )
                ]
            )
        
        session.workflow_snapshot.workflow.is_complete = True

        return CommandOutput(
            command_responses=[
                CommandResponse(
                    response="",
                    artifacts={
                        "command_name": output_of_command_name_prediction.command_name,
                        "cmd_parameters": output_of_parameter_extraction.cmd_parameters,
                    },
                )
            ]
        )

