from typing import Optional

from fastworkflow.command_executor import CommandResponse
from fastworkflow.session import Session

from ..parameter_extraction.signatures import CommandParameters
from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self,
        session: Session,
        command: str,
        command_parameters: CommandParameters
    ) -> list[CommandResponse]:
        output = process_command(session, command_parameters)
        return [
            CommandResponse(response=f"current status is: {output.status}")
        ]


# if __name__ == "__main__":
