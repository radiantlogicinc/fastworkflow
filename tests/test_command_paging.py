"""Paging from the run (fix-3rsq.12.1; design 2.49, 3.14; plan Phase 12 step 3).

A command is paged when its ``Input`` has an int ``start``, ``offset`` or ``skip``. A paged command's
``Output`` declares ``total: int`` (a load error otherwise). ``_next_page`` and ``_more_pages`` read the
run's Input and Output, and the framework-written response ends with a paging line when rows remain.

Integration tests: the loader runs on a copy of the todo list workflow, the framework's response
builder runs on real Output models, and the code-steps runner runs on the todo workflow's test
harness (``tests/test_planning_command_code_steps.py``). The only seam is the scripted command table
that stands in for the NLU, as in that file.
"""
from __future__ import annotations

import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

import fastworkflow
from fastworkflow import ModuleType, planning_phases
from fastworkflow.command_executor import frame_command_result
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry
from fastworkflow.observation_offloading.facts import output_page, paging_parameters
from fastworkflow.planning_commands import PlanningCommandFormatError
from tests.test_planning_command_code_steps import (  # noqa: F401 - the todo fixtures of that file
    ItemsParams,
    Phase,
    _plan,
    paged,
    todo,
)
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    initialized_fastworkflow,
    no_review_gate,
    todo_workflow_path,
)

TODO = Path(__file__).resolve().parent / "todo_list_workflow"
GET_TODO_LIST = "TodoListManager/get_todo_list"

_COPY_IGNORE = shutil.ignore_patterns("___workflow_contexts", "___convo_info", "__pycache__", "versions")

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


@pytest.fixture(autouse=True)
def _loader_caches_per_test():
    # The routing cache is keyed by command name alone, and these tests rewrite the same command.
    RoutingDefinition._GLOBAL_COMMAND_CLASS_CACHE.clear()
    yield
    RoutingDefinition._GLOBAL_COMMAND_CLASS_CACHE.clear()


# ----------------------------------------------------------------------
# A copy of the todo workflow whose get_todo_list is rewritten
# ----------------------------------------------------------------------

_GET_TODO_LIST_HEAD = '''"""get_todo_list, rewritten for a paging test."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

import fastworkflow


class Signature:
    """Show one todo list, for the test."""

    class Input(BaseModel):
        id: int = Field(default=1, description="id of the todo list", examples=["1"])
%(extra_input)s

    class Output(BaseModel):
%(output)s


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, command_parameters) -> Signature.Output:
        return Signature.Output(description="one list")
'''

_TOTAL = "        description: str = Field(description='one list')\n        total: int = Field(description='how many rows')"
_NO_TOTAL = "        description: str = Field(description='one list')"


def _rewrite_get_todo_list(folder: Path, extra_input: str, output: str) -> None:
    text = _GET_TODO_LIST_HEAD % {"extra_input": textwrap.indent(extra_input, "        "), "output": output}
    (folder / "_commands" / "TodoListManager" / "get_todo_list.py").write_text(text)


@pytest.fixture
def copy_of_todo(tmp_path) -> Path:
    folder = tmp_path / "todo_list_workflow"
    shutil.copytree(TODO, folder, ignore=_COPY_IGNORE, symlinks=True)
    versions = TODO / "___command_info" / "versions"
    if versions.exists():
        (folder / "___command_info").mkdir(exist_ok=True)
        (folder / "___command_info" / "versions").symlink_to(versions)
    return folder


def _load_input_class(folder: Path, command: str = GET_TODO_LIST):
    RoutingRegistry.clear_registry()
    routing = RoutingRegistry.get_definition(str(folder))
    return routing.get_command_class(command, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)


# ----------------------------------------------------------------------
# The paging parameter and the page a run's Output is
# ----------------------------------------------------------------------


class TestPagingParameter:
    def test_an_int_or_optional_int_start_is_the_paging_parameter(self):
        class Input(BaseModel):
            start: Optional[int] = None
            query: str = ""

        class Other(BaseModel):
            offset: int = 0      # not a paging name: only start pages

        assert paging_parameters(Input) == ("start",)
        assert paging_parameters(Other) == ()

    def test_a_start_that_is_a_datetime_or_a_field_named_skip_is_not_paged(self):
        class Dated(BaseModel):
            start: Optional[datetime] = None

        class Flagged(BaseModel):
            skip: bool = False

        class Counted(BaseModel):
            skip: int = 0        # skip is no paging name, whatever its type

        assert paging_parameters(Dated) == ()
        assert paging_parameters(Flagged) == ()
        assert paging_parameters(Counted) == ()
        assert paging_parameters(None) == ()

    def test_output_page_reads_the_rows_of_one_length_and_the_total(self):
        class Output(BaseModel):
            uids: list[str]
            titles: list[str]
            total: int

        assert output_page(Output(uids=["a", "b"], titles=["x", "y"], total=5)) == (2, 5)
        assert output_page({"uids": [], "titles": [], "total": 0}) == (0, 0)
        assert output_page({"uids": ["a", "b"], "titles": ["x"], "total": 3}) is None
        assert output_page({"uids": ["a"]}) is None


# ----------------------------------------------------------------------
# The load check
# ----------------------------------------------------------------------


class TestLoadCheck:
    def test_a_paged_command_whose_output_declares_no_total_is_a_load_error(self, copy_of_todo):
        _rewrite_get_todo_list(copy_of_todo, "start: Optional[int] = Field(default=None, description='first row')",
                               _NO_TOTAL)
        with pytest.raises(PlanningCommandFormatError) as caught:
            _load_input_class(copy_of_todo)
        message = str(caught.value)
        assert "Input.start makes TodoListManager/get_todo_list a paged command" in message
        assert "its Output declares no total" in message
        assert "declare total: int" in message
        assert "or rename start if it is not a page" in message

    def test_a_paged_command_whose_output_declares_total_loads(self, copy_of_todo):
        _rewrite_get_todo_list(copy_of_todo, "start: Optional[int] = Field(default=None, description='first row')",
                               _TOTAL)
        cls = _load_input_class(copy_of_todo)
        assert cls is not None and "start" in cls.Input.model_fields

    def test_a_start_that_is_a_datetime_is_not_paged_and_loads_without_a_total(self, copy_of_todo):
        _rewrite_get_todo_list(copy_of_todo,
                               "start: Optional[datetime] = Field(default=None, description='a moment')",
                               _NO_TOTAL)
        cls = _load_input_class(copy_of_todo)
        assert cls is not None and "start" in cls.Input.model_fields

    def test_a_skip_that_is_a_bool_is_not_paged_and_loads_without_a_total(self, copy_of_todo):
        _rewrite_get_todo_list(copy_of_todo,
                               "skip: bool = Field(default=False, description='leave the done rows out')",
                               _NO_TOTAL)
        cls = _load_input_class(copy_of_todo)
        assert cls is not None and "skip" in cls.Input.model_fields


# ----------------------------------------------------------------------
# The paging line the framework writes from an Output
# ----------------------------------------------------------------------


class ItemsPage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    labels: list[str]
    total: int


class PagedInput(BaseModel):
    start: Optional[int] = Field(default=None)


class TestPagingLine:
    def test_the_paging_line_appears_when_rows_remain(self):
        built = frame_command_result(ItemsPage(labels=["a"] * 10, total=57), command_name="list_items",
                                     output_model=ItemsPage, command_parameters=PagedInput())
        assert built.command_response.response.endswith(
            "\nRows 1-10 of 57; next page: start=10.")
        assert built.command_response.response.startswith('{"labels"')

    def test_the_paging_line_follows_the_page_the_run_read(self):
        built = frame_command_result(ItemsPage(labels=["a"] * 10, total=57), command_name="list_items",
                                     output_model=ItemsPage, command_parameters=PagedInput(start=20))
        assert built.command_response.response.endswith("\nRows 21-30 of 57; next page: start=30.")

    def test_the_paging_line_is_absent_when_no_rows_remain(self):
        built = frame_command_result(ItemsPage(labels=["a"] * 7, total=57), command_name="list_items",
                                     output_model=ItemsPage, command_parameters=PagedInput(start=50))
        assert "next page" not in built.command_response.response
        assert built.command_response.response == ItemsPage(labels=["a"] * 7, total=57).model_dump_json()

    def test_an_unpaged_command_gets_no_paging_line(self):
        class Plain(BaseModel):
            query: str = ""

        class Output(BaseModel):
            labels: list[str]
            total: int

        built = frame_command_result(Output(labels=["a"], total=9), command_name="find",
                                     output_model=Output, command_parameters=Plain())
        assert "next page" not in built.command_response.response


# ----------------------------------------------------------------------
# A full-list step reads every page from the run's Output
# ----------------------------------------------------------------------

TOTAL_ITEMS = 30


def _json_page(command: str):
    """The scripted reply of ``list_items``: the Output as the framework frames it, JSON only, so the
    code-steps runner has to read the next page from the run's Output rather than from the text."""
    start = int((command.split("start=") + ["0"])[1].split()[0]) if "start=" in command else 0
    page = ItemsPage(labels=[f"i{start + n}" for n in range(10)], total=TOTAL_ITEMS)
    built = frame_command_result(page, command_name="list_items", output_model=ItemsPage,
                                 command_parameters=PagedInput(start=start or None))
    response = built.command_response.response
    return response.split("\nRows ")[0], built.command_response.output


def test_a_full_list_step_on_a_paged_json_command_reads_every_page(paged, no_review_gate):
    # no_review_gate: this turn is one sub-task with no head to match it, which the round-1 gate now reviews; the test
    # is about the paged full-list step, so the review is off here (fix-3rsq.plan-review.2.6).
    paged.toys["list_items"] = ItemsParams
    paged.workflow.scripted["list_items"] = _json_page
    paged.plan = _plan(Phase("report", lambda: [
        "list_items (full list: next page with start=<n> until complete)", "list_todo_lists"]))
    paged.turn()

    # Code read page 1, then each page the run's Output says remains: no model step.
    assert paged.model.asked == []
    assert paged.workflow.sent == ["list_items", "list_items start=10", "list_items start=20",
                                   "list_todo_lists"]


def test_next_page_reads_the_run_not_the_text():
    run = {"input": {"start": 0}, "output": {"labels": ["a"] * 10, "total": 25}}
    # The text names no page at all: the run's Output says rows remain after the page it read.
    assert planning_phases._next_page("ten labels", ["start"], 0, run=run) == ("start", 10)
    assert planning_phases._more_pages("ten labels", ["start"], run=run) is True
    last = {"input": {"start": 20}, "output": {"labels": ["a"] * 5, "total": 25}}
    assert planning_phases._next_page("Next page: start=99", ["start"], 20, run=last) is None
    assert planning_phases._more_pages("Next page: start=99", ["start"], run=last) is False


def test_next_page_falls_back_to_the_text_for_a_run_with_no_output():
    run = {"input": {"start": 0}, "output": None}
    assert planning_phases._next_page("Next page: start=10.", ["start"], 0, run=run) == ("start", 10)
    assert planning_phases._more_pages("Next page: start=10.", ["start"], run=run) is True
