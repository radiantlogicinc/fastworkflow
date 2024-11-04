""" utility functions """

import logging
from datetime import datetime, timezone
from time import time_ns

from fastworkflow.utils.env import get_env_variable


def format_ns(time_in_ns):
    """convert nanoseconds to text format"""
    formatted_time_upto_seconds = datetime.fromtimestamp(
        time_in_ns / 1e9, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S")

    fractional_sec = (time_in_ns // 10**9) * 10**9
    nanoseconds_string = f"{fractional_sec}"[:9]

    return f"{formatted_time_upto_seconds}.{nanoseconds_string}Z"


# For time in nanoseconds
# https://stackoverflow.com/questions/31328300/python-logging-module-logging-timestamp-to-include-microsecond
class LogRecordNs(logging.LogRecord):  # pylint: disable=too-few-public-methods
    """class that returns nanoseconds"""

    def __init__(self, *args, **kwargs):
        self.created_ns = time_ns()  # Fetch precise timestamp
        super().__init__(*args, **kwargs)


logging.basicConfig(level=logging.NOTSET)


class FormatterNs(logging.Formatter):
    """nanosecond log formatter"""

    default_nsec_format = "%Y-%m-%dT%H:%M:%S.%09dZ"

    def formatTime(self, record, datefmt=None):
        if datefmt is not None:  # Do not handle custom formats here ...
            return super().formatTime(
                record, datefmt
            )  # ... leave to original implementation
        return format_ns(record.created_ns)


logging.setLogRecordFactory(LogRecordNs)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s-%(funcName)s - %(message)s"
log_formatter = FormatterNs(LOG_FORMAT)

if log_level := get_env_variable("LOG_LEVEL", "INFO"):
    if log_level == "DEBUG":
        LOG_LEVEL = logging.DEBUG
    elif log_level == "INFO":
        LOG_LEVEL = logging.INFO
    elif log_level == "WARNING":
        LOG_LEVEL = logging.WARNING
    elif log_level == "ERROR":
        LOG_LEVEL = logging.ERROR
    elif log_level == "CRITICAL":
        LOG_LEVEL = logging.CRITICAL
    else:
        raise ValueError(
            "LOG_LEVEL is not one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
else:
    print("LOG_LEVEL env variable is not specified")
    LOG_LEVEL = logging.DEBUG if __debug__ else logging.INFO

# create logger
logger = logging.getLogger("fastWorkflow")
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # otherwise you will see duplicate log entries

# # clear any existing handlers for our logger
logger.handlers.clear()

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)

# create and add formatter to ch
ch.setFormatter(log_formatter)

# add ch to logger
logger.addHandler(ch)


# create a separate logger for pytest_logger to log assertions
# with a slightly different format
pytest_assertion_logger = logging.getLogger("PyTest_Logger")
pytest_assertion_logger.setLevel(logging.DEBUG)
pytest_assertion_logger.propagate = (
    False  # otherwise you will see duplicate log entries
)
pytest_assertion_logger.handlers.clear()
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(FormatterNs("%(asctime)s - %(levelname)s - %(message)s"))
pytest_assertion_logger.addHandler(ch)

logging.getLogger("dspy").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("speedict").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("semantic_router.utils.logger").setLevel(logging.WARNING)

# some testing code
if __name__ == "__main__":
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warn message")
    logger.error("error message")
    logger.critical("critical message")

# To disable __debug__ and set the log level to INFO, use the -O option as shown below
# python3 -O utils.py
