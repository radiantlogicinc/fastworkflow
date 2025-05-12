import dspy

import fastworkflow
from fastworkflow.utils.dspy_logger import DSPyRotatingFileLogger, DSPyForward

from ..parameter_extraction.signatures import CommandParameters
from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self,
        session: fastworkflow.Session,
        command: str,
        command_parameters: CommandParameters
    ) -> fastworkflow.CommandOutput:
        output = process_command(session, command_parameters)

        LLM = fastworkflow.get_env_var("LLM")
        LITELLM_API_KEY = fastworkflow.get_env_var("LITELLM_API_KEY")
        lm = dspy.LM(LLM, api_key=LITELLM_API_KEY)
        answer_generator = ResponseGenerator.BasicQA(lm)

        with DSPyRotatingFileLogger("inference_log.jsonl"):
            prediction = answer_generator(
                context=str(output.help_info), question=command
            )

        return fastworkflow.CommandOutput(
            session_id=session.id,
            command_responses=[
                fastworkflow.CommandResponse(
                    response=prediction.answer,
                )
            ]
        )

    class BasicQA(dspy.Module):
        """DSPy Module for answering help questions"""

        def __init__(self, lm: dspy.LM):
            super().__init__()

            self.lm = lm
            self.generate_answer = dspy.Predict("context, question -> answer")

        @DSPyForward.intercept
        def forward(self, context, question):
            """forward pass"""
            with dspy.context(lm=self.lm):
                return self.generate_answer(context=context, question=question)
