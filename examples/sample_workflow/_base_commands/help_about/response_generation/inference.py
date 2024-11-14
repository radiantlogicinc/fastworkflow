import dspy

from fastworkflow.command_executor import CommandResponse
from fastworkflow.session import Session
from fastworkflow.utils.dspy_logger import DSPyRotatingFileLogger, DSPyForward

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

        DSPY_LM_MODEL = session.get_env_var("DSPY_LM_MODEL")
        OPENAI_API_KEY = session.get_env_var("OPENAI_API_KEY")
        lm = dspy.LM(DSPY_LM_MODEL, api_key=OPENAI_API_KEY)
        answer_generator = ResponseGenerator.BasicQA(lm)

        with DSPyRotatingFileLogger("inference_log.jsonl"):
            prediction = answer_generator(
                context=str(output.help_info), question=command
            )

        return [
            CommandResponse(
                response=prediction.answer,
            )
        ]

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
