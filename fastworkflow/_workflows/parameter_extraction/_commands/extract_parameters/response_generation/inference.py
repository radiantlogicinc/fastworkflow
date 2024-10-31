from typing import Optional

from fastworkflow.command_executor import CommandOutput
from fastworkflow.session import Session
from ..response_generation.command_implementation import process_command

class ResponseGenerator():
    def __call__(self,
                 session: Session,
                 command: str, 
                 payload: Optional[dict] = None) -> CommandOutput:
        if payload["error_msg"]:
            return CommandOutput(response=payload["error_msg"],
                                 payload={"abort_command": False})

        # Note that we are passing the calling workflow's session (payload["session"]) to process_command
        output_of_process_command = process_command(payload["session"], command, payload)
        if output_of_process_command.parameter_is_valid:
            session.workflow.mark_as_complete()
            return CommandOutput(response="",
                                 payload={
                                     "cmd_parameters": output_of_process_command.cmd_parameters,
                                     "abort_command": False
                                 })

        return CommandOutput(response=output_of_process_command.error_msg,
                             payload={"abort_command": False})


# if __name__ == "__main__":
