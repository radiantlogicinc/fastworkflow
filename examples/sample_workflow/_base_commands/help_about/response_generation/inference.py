from typing import Optional

import dspy

from fastworkflow.command_executor import CommandOutput
from fastworkflow.session import Session
from fastworkflow.utils.env import get_env_variable

from ..parameter_extraction.signatures import CommandParameters
from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self,
        session: Session,
        command: str,
        command_parameters: CommandParameters,
        payload: Optional[dict] = None,
    ) -> CommandOutput:
        output = process_command(session, command_parameters, payload)

        DSPY_LM_MODEL = get_env_variable("DSPY_LM_MODEL")
        lm = dspy.LM(DSPY_LM_MODEL)
        with dspy.context(lm=lm):
            extract_cmd_params = dspy.Predict("context, question -> answer")
            prediction = extract_cmd_params(
                context=str(output.help_info), question=command
            )

        return CommandOutput(
            response=prediction.answer,
        )
