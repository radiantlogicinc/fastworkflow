from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session

from ..response_generation.command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> CommandOutput:
        output_of_process_command = process_command(session, command)

        if output_of_process_command.parameter_is_valid:
            session.workflow_snapshot.workflow.is_complete = True

            return CommandOutput(
                command_responses=[
                    CommandResponse(
                        response="",
                        artifacts={
                            "cmd_parameters": output_of_process_command.cmd_parameters,
                        },
                    )
                ]
            )

        # Get the command name from session context
        command_name = session.workflow_snapshot.context["subject_command_name"]

        return CommandOutput(
            command_responses=[
                CommandResponse(
                    response=f'CURRENT TOOL: {command_name}, PARAMETER EXTRACTION ERROR MESSAGE: {command}. {output_of_process_command.error_msg}',
                    success=False
                )
            ]
        )
