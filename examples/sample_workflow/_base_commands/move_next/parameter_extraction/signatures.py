from typing import Tuple

from pydantic import BaseModel

from fastworkflow.session import WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction


class CommandParameters(BaseModel):
    skip_completed: bool = True