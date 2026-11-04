"""Live check: an unstated argument comes back NOT_FOUND from a real Bedrock model (fix-1pfu).

Skeleton only. It makes a real provider call, and live calls need owner approval, so it is
marked ``requires_llm_key`` and skipped unconditionally until the owner lifts the skip. Run it
with ``-m requires_llm_key`` and the skip removed, with LLM_PARAM_EXTRACTION set to a Bedrock
Claude model and its credentials in the workflow's passwords env file.

The offline equivalent, which runs by default, is ``tests/test_param_extraction_unknown_marker.py``.
"""
from typing import List

import pytest
from pydantic import BaseModel

from fastworkflow.utils.signatures import NOT_FOUND, InputForParamExtraction

pytestmark = [
    pytest.mark.requires_llm_key,
    pytest.mark.skip(reason="live Bedrock call; needs owner approval (fix-1pfu)"),
]


class _CancelParams(BaseModel):
    order_id: str
    address: str
    item_ids: List[str]


def test_unstated_argument_is_not_found_on_bedrock():
    """The request gives an order id and no address or items. The address must come back as
    NOT_FOUND, not as '<UNKNOWN>', 'UNKNOWN' or an invented value, and the items as an empty list."""
    result = InputForParamExtraction(command="cancel order #W1").extract_parameters(
        CommandParameters=_CancelParams, subject_command_name=None, workflow_folderpath=None)
    assert result.order_id == "#W1"
    assert result.address == NOT_FOUND
    assert result.item_ids == []
