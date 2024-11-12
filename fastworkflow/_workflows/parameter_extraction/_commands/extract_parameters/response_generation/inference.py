from typing import Optional

from fastworkflow.command_executor import CommandResponse
from fastworkflow.session import Session

from ..response_generation.command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> list[CommandResponse]:
        caller_session = session.caller_session
        if not caller_session:
            raise ValueError("caller_session MUST be set for the parameter extraction workflow")
        if not caller_session.parameter_extraction_info:
            raise ValueError("parameter_extraction_info MUST be set in the caller session for the parameter extraction workflow")       
        if "error_msg" not in caller_session.parameter_extraction_info:
            raise ValueError("error_msg key MUST be set in the caller session's parameter_extraction_info for the parameter extraction workflow")
        
        error_msg = caller_session.parameter_extraction_info["error_msg"]
        if error_msg:   
            caller_session.parameter_extraction_info["error_msg"] = None          
            return [
                CommandResponse(
                    response=error_msg, 
                    artifacts={"abort_command": False}
                )
            ]

        # Note that we are passing the caller workflow's session to process_command
        output_of_process_command = process_command(caller_session, command)

        if output_of_process_command.parameter_is_valid:
            session.workflow.is_complete = True
            return [
                CommandResponse(
                    response="",
                    artifacts={
                        "cmd_parameters": output_of_process_command.cmd_parameters,
                        "abort_command": False,
                    },
                )
            ]

        return [
            CommandResponse(
                response=output_of_process_command.error_msg,
                artifacts={"abort_command": False},
            )
        ]


# if __name__ == "__main__":
