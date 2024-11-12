from typing import Optional

import dspy

from fastworkflow.command_executor import CommandResponse
from fastworkflow.session import Session
from fastworkflow.utils.env import get_env_variable

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

        DSPY_LM_MODEL = session.get_env_variable("DSPY_LM_MODEL")
        OPENAI_API_KEY = session.get_env_variable("OPENAI_API_KEY")
        lm = dspy.LM(DSPY_LM_MODEL, api_key=OPENAI_API_KEY)
        with dspy.context(lm=lm):
            extract_cmd_params = dspy.Predict("context, question -> answer")
            prediction = extract_cmd_params(
                context=str(output.help_info), question=command
            )

        return [
            CommandResponse(
                response=prediction.answer,
            )
        ]
