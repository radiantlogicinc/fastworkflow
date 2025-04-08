from typing import Tuple

from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.session import WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction


class CommandParameters(BaseModel):
    workitem_path: str = Field(default="NOT_FOUND", description="The workitem type")