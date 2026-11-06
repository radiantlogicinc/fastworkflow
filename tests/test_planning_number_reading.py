"""The planning side reads the numbers a request writes the way the execution side does.

A planning command's ``int`` input with ``in_request`` must be a number the request writes. The planning
check (``planning_commands._request_numbers``) and the execution check (``argument_checks.written_numbers``)
both read the request's numbers; the thousands separator and the minus sign are read by both.

Integration: real ``check_arguments`` on a planning command loaded from a temporary folder.
"""

from __future__ import annotations

import pytest

from fastworkflow import planning_commands as pc
from fastworkflow.utils import argument_checks
from tests.test_planning_command_files import _write

COPIES = '''"""add_copies: add copies of a book, for the test."""
from pydantic import BaseModel, Field


class Signature:
    """Add copies of a book."""

    class Input(BaseModel):
        copies: int = Field(description="how many copies to add", json_schema_extra={"in_request": True})


class PlanGenerator:
    def steps(self, input):
        return [f"add {input.copies} copies"]
'''


@pytest.fixture
def copies_command(tmp_path):
    folder = _write(tmp_path, {"add_copies": COPIES})
    from fastworkflow import workflow_agent
    return workflow_agent.load_planning_command_map(folder)["add_copies"]


@pytest.mark.parametrize("request_text, value", [
    ("Add 1,000 copies of Dune", 1000),
    ("Remove -5 copies of Dune", -5),
], ids=["thousands-separator", "minus-sign"])
def test_the_planning_check_reads_the_number_the_request_writes(copies_command, request_text, value):
    args, _ = pc.check_arguments(copies_command, {"copies": value}, request_text)

    assert args["copies"] == value


def test_the_execution_side_reads_the_same_two_numbers():
    assert {1000, 1, 0} <= argument_checks.written_numbers("Add 1,000 copies of Dune", [])
    assert -5 in argument_checks.written_numbers("Remove -5 copies of Dune", [])
