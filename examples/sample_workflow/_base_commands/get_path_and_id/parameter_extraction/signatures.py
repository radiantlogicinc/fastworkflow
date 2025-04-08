from typing import Tuple

from pydantic import BaseModel

from fastworkflow.session import WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction


class CommandParameters(BaseModel):
    for_next_workitem: bool = False
    skip_completed: bool = True
