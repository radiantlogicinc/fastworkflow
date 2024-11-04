from typing import Optional

from fastworkflow._workflows.parameter_extraction._commands.extract_parameters.response_generation.command_implementation import (
    process_command,
)
from fastworkflow.command_executor import CommandOutput
from fastworkflow.session import Session


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str, payload: Optional[dict] = None
    ) -> CommandOutput:
        return CommandOutput(
            success=False, response="Command aborted", payload={"abort_command": True}
        )


# if __name__ == "__main__":
