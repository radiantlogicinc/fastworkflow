"""
DSPy Logging Utilities
Author: Dhar Rawal

Works with DSPy to log forward calls and their results, using a custom handler function.
Works with typed predictors too!
"""
import functools
import json
from typing import Any, Callable, Dict, Optional, Tuple

import logging
from logging.handlers import RotatingFileHandler

import dspy
from pydantic import BaseModel

import threading


class DSPyProgramLog(BaseModel):
    """DSPy Program Log"""

    dspy_program_class: str
    dspy_input_args: Tuple[Any, ...] = ()
    dspy_input_kwargs: Dict[str, Any] = {}
    dspy_completions_dict: Dict[str, Any] = {}
    # dspy_module_logs: list[DSPyModuleLog] = []


class DSPyForward:  # pylint: disable=too-few-public-methods
    """DSPy Forward Interceptor"""

    # class variable for custom handler
    save_dspyprogramlog_func: Optional[Callable[[DSPyProgramLog], None]] = None

    @classmethod
    def intercept(cls, func: Callable) -> Callable:
        """
        Decorator to log forward calls and their results, using a custom handler function.
        Using __call__(...) enables the class itself to be used as a decorator.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            dspy_program_log: DSPyProgramLog = DSPyProgramLog(dspy_program_class=func.__qualname__.split(".")[-2])
            dspy_program_log.dspy_input_args = args[1:] if args else ()
            dspy_program_log.dspy_input_kwargs = kwargs

            result: dspy.Prediction = func(*args, **kwargs)

            if result.completions:
                dspy_program_log.dspy_completions_dict = (
                    result.completions._completions  # pylint: disable=protected-access
                )
            else:
                dspy_program_log.dspy_completions_dict = {}

            if DSPyForward.save_dspyprogramlog_func:
                DSPyForward.save_dspyprogramlog_func(  # pylint: disable=abstract-class-instantiated, not-callable
                    dspy_program_log
                )

            return result

        return wrapper


class DSPyLogger:
    """DSPy Logger"""

    def __init__(self):
        pass

    def __enter__(self) -> "DSPyLogger":
        DSPyForward.save_dspyprogramlog_func = self
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        DSPyForward.save_dspyprogramlog_func = None

    def __call__(dspy_program_log: DSPyProgramLog) -> None:
        """Default handler to save the dspy program log"""
        # args_str = ', '.join([repr(a) for a in dspy_program_log.dspy_input_args] +
        #                      [f"{k}={v!r}" for k, v in dspy_program_log.dspy_input_kwargs.items()])
        # print(args_str)
        print(f"{dspy_program_log.dspy_program_class}")
        print(f"{dspy_program_log.dspy_input_args}")
        print(f"{dspy_program_log.dspy_input_kwargs}")
        print(f"{json.dumps(dspy_program_log.dspy_completions_dict)}")


class DSPyRotatingFileLogger(DSPyLogger):
    """DSPy Rotating File Logger Singleton with Asynchronous Writes"""
    # configurable parameters
    max_file_size = 1024 * 1024  # 1MB
    backup_count = 5

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_file_path: str):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super().__new__(cls)
                    cls._instance.__init__(log_file_path)
        return cls._instance

    def __init__(self, log_file_path: str):
        if hasattr(self, 'logger'):  # Prevent re-initialization
            return

        # Create a logger
        self.logger = logging.getLogger("dspy_log")
        self.logger.setLevel(logging.INFO)

        # Remove any existing handlers to prevent console output
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create a rotating file handler
        self.handler = RotatingFileHandler(log_file_path, 
                                           maxBytes=DSPyRotatingFileLogger.max_file_size, 
                                           backupCount=DSPyRotatingFileLogger.backup_count)
        self.handler.setLevel(logging.INFO)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(self.handler)

        # Prevent propagation to the root logger
        self.logger.propagate = False

    def __call__(self, dspy_program_log: DSPyProgramLog) -> None:
        """Log the dspy program log asynchronously"""
        log_message = dspy_program_log.model_dump_json()
        threading.Thread(target=self._log_to_file, args=(log_message,)).start()

    def _log_to_file(self, log_message: str) -> None:
        """Write log message to file"""
        self.logger.info(log_message)

def _how_to_use():
    """
    how to use:
    Use @DSPyForward.intercept to decorate the forward function of your dspy program
    Call the forward function in the context of the DSPyLogger or DSPyRotatingFileLogger (if you want to log to a rotating file)
    The dspy logger object will get called with an instance of DSPyProgramLog
    """

    class BasicQA(dspy.Module):
        """DSPy Module for testing DSPyLogger"""

        def __init__(self):
            super().__init__()

            self.generate_answer = dspy.Predict("topic, question -> answer")

        gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo", api_key="<YOUR_API_KEY>")

        @DSPyForward.intercept
        def forward(self, topic, question):
            """forward pass"""
            with dspy.context(lm=BasicQA.gpt3_turbo):
                return self.generate_answer(topic=topic, question=question)

    get_answer = BasicQA()
    # If you just want to log to the console, use DSPyLogger
    with DSPyLogger:
        _ = get_answer("geography quiz", question="What is the capital of France?")

    # If you want to log to a file, use DSPyFileLogger
    with DSPyRotatingFileLogger("dspy_logs.jsonl"):
        _ = get_answer("geography quiz", question="What is the capital of France?")

    # This will print:
    # BasicQA
    # ('geography quiz',)
    # {'question': 'What is the capital of France?'}
    # {"answer": ["Topic: geography quiz\nQuestion: What is the capital of France?\nAnswer: Paris"]}


if __name__ == "__main__":
    _how_to_use()
