import os
from typing import Optional


def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
    """
    Helper function to get the environment variable or raise exception.
    Used inside the container applications. DO NOT REMOVE!
    """
    try:
        return os.environ[var_name]
    except KeyError as exc:
        if default is not None:
            return default
        error_msg = f"The environment variable {var_name} was missing, abort..."

        # dynamically import logger
        try:
            from fastworkflow.utils.logging import logger
        except ImportError:
            import logging

            logger = logging.getLogger("fastWorkflow")
        logger.critical("%s", error_msg)

        raise EnvironmentError(error_msg) from exc
