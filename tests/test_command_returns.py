"""What a command returns, and the one ``__call__`` signature (fix-3rsq.12.early; design 3.14, 2.29; plan
Phase 12 steps 1-2).

Integration tests through ``CommandExecutor.perform_action`` on copies of the example workflow, the todo
list workflow and the library. A test that needs a command to return another form rewrites that command's
``__call__`` in its own copy; nothing is mocked.
"""
from __future__ import annotations

import json
import shutil
import textwrap
import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import Action, CommandOutput, CommandResponse
from fastworkflow.command_executor import CommandExecutor, frame_command_result
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.examples.library.application.library import Library
from tests.example_workflow.application.todo_manager import TodoListManager as ExampleTodoListManager
from tests.todo_list_workflow.application.todo_manager import TodoListManager
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture

TESTS = Path(__file__).resolve().parent
EXAMPLE = TESTS / "example_workflow"
TODO = TESTS / "todo_list_workflow"
LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"
LIST_TODO_LISTS = "TodoListManager/list_todo_lists"
GET_TODO_LIST = "TodoListManager/get_todo_list"

_COPY_IGNORE = shutil.ignore_patterns(
    "___workflow_contexts", "___convo_info", "__pycache__", "versions",
)

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


# ----------------------------------------------------------------------
# Copies and helpers
# ----------------------------------------------------------------------


def _copy(source: Path, into: Path) -> Path:
    """A copy of a workflow folder, its trained artifacts included (the model versions are linked, not copied)."""
    shutil.copytree(source, into, ignore=_COPY_IGNORE, symlinks=True)
    versions = source / "___command_info" / "versions"
    if versions.exists():
        (into / "___command_info").mkdir(exist_ok=True)
        (into / "___command_info" / "versions").symlink_to(versions)
    return into


def _rewrite_call(folder: Path, command_file: str, body: str,
                  signature: str = "(self, workflow: Workflow, command: str)") -> None:
    """Replace the ``__call__`` of a command in this copy. Every shipped command file defines ``__call__`` last."""
    path = folder / "_commands" / command_file
    head, marker, _ = path.read_text().partition("    def __call__(")
    assert marker, f"{path} has no __call__ to rewrite"
    new_call = f"    def __call__{signature}:\n" + textwrap.indent(textwrap.dedent(body).strip("\n") + "\n", "        ")
    path.write_text(head + new_call)


def _workflow(folder: Path, root, workflow_tag: str):
    """The workflow at *folder*, its root context set as ``startup`` would set it."""
    RoutingRegistry.clear_registry()
    wf = fastworkflow.Workflow.create(str(folder), workflow_id_str=f"returns-{workflow_tag}-{uuid.uuid4().hex}")
    wf.root_command_context = root
    return wf


def _run(folder: Path, root, command_name: str, command: str = "", parameters: dict | None = None) -> CommandOutput:
    wf = _workflow(folder, root, uuid.uuid4().hex[:8])
    action = Action(command_name=command_name, command=command or command_name, parameters=parameters or {})
    return CommandExecutor.perform_action(wf, action)


def _example_root(tmp_path: Path):
    return ExampleTodoListManager(str(tmp_path / "example_todo.json"))


def _todo_root(tmp_path: Path):
    return TodoListManager(str(tmp_path / "todo_list.json"))


@pytest.fixture
def example_copy(tmp_path) -> Path:
    return _copy(EXAMPLE, tmp_path / "example_workflow")


@pytest.fixture
def todo_copy(tmp_path) -> Path:
    return _copy(TODO, tmp_path / "todo_list_workflow")


# ----------------------------------------------------------------------
# A command that returns its Signature.Output: the same envelope as the hand-built form
# ----------------------------------------------------------------------


class TestOutputReturn:
    def test_an_output_return_has_the_envelope_of_the_hand_built_command_output(self, tmp_path):
        # Two copies of the same command: the shipped one builds a CommandOutput by hand; the other returns
        # its Signature.Output and leaves the framework to build the same envelope.
        # Both copies share a folder name, so workflow_name is the same for both.
        shipped = _copy(EXAMPLE, tmp_path / "shipped" / "example_workflow")
        returned = _copy(EXAMPLE, tmp_path / "returned" / "example_workflow")
        _rewrite_call(returned, "TodoListManager/list_todo_lists.py", """
            app = workflow.command_context_for_response_generation
            return Signature.Output(result=app.list_todo_lists())
            """)

        built = _run(shipped, _example_root(tmp_path / "a"), LIST_TODO_LISTS)
        framed = _run(returned, _example_root(tmp_path / "b"), LIST_TODO_LISTS)

        output = framed.command_response.output
        assert output is not None and type(output).__name__ == "Output"
        assert framed.command_response.response == output.model_dump_json()
        assert json.loads(framed.command_response.response) == {"result": []}
        # The envelope a hand-built CommandOutput carries, with the Output set as the framework sets it.
        hand_built = CommandOutput(command_response=CommandResponse(
            response=output.model_dump_json(), output=output))
        assert framed.command_response.model_dump() == hand_built.command_response.model_dump()
        # The shipped command's text is the same JSON, so the client sees the same response.
        assert framed.command_response.response == built.command_response.response
        assert framed.success is True and built.success is True
        for field in ("workflow_name", "context", "command_name"):
            assert getattr(framed, field) == getattr(built, field)
        assert framed.command_name == LIST_TODO_LISTS
        assert framed.command_call_id is not None

    def test_a_command_that_declares_an_output_and_returns_it_is_not_warned_about(self, tmp_path, caplog):
        returned = _copy(EXAMPLE, tmp_path / "returned")
        _rewrite_call(returned, "TodoListManager/list_todo_lists.py", """
            app = workflow.command_context_for_response_generation
            return Signature.Output(result=app.list_todo_lists())
            """)
        with caplog.at_level("WARNING"):
            _run(returned, _example_root(tmp_path), LIST_TODO_LISTS)
        assert not [r for r in caplog.records if "declares an Output" in r.getMessage()]


# ----------------------------------------------------------------------
# The other shipped and taught forms: str, CommandResponse, CommandOutput, Refused
# ----------------------------------------------------------------------


class TestOtherForms:
    def test_a_str_return_is_the_response_text(self, todo_copy, tmp_path):
        _rewrite_call(todo_copy, "TodoListManager/list_todo_lists.py", """
            return "two todo lists"
            """)
        out = _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)
        assert out.command_response.response == "two todo lists"
        assert out.command_response.output is None
        assert out.success is True
        assert out.command_name == LIST_TODO_LISTS

    def test_a_command_response_return_keeps_its_artifacts(self, todo_copy, tmp_path):
        _rewrite_call(todo_copy, "TodoListManager/list_todo_lists.py", """
            return fastworkflow.CommandResponse(response="listed", artifacts={"count": 0})
            """)
        out = _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)
        assert out.command_response.response == "listed"
        assert out.command_response.artifacts == {"count": 0}
        assert out.success is True
        assert out.command_name == LIST_TODO_LISTS

    def test_a_shipped_command_output_return_is_kept_as_it_is(self, todo_copy, tmp_path):
        # The unchanged todo command builds its CommandOutput by hand; the framework leaves it as returned.
        out = _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)
        assert isinstance(out, CommandOutput)
        assert out.command_response.response.startswith("Response: ")
        assert out.success is True
        assert out.command_name == LIST_TODO_LISTS

    def test_frame_command_result_returns_a_command_output_unchanged(self):
        built = CommandOutput(command_response=CommandResponse(response="as built"))
        assert frame_command_result(built, command_name="x", output_model=None) is built

    def test_refused_gives_a_failed_output_with_its_message(self, todo_copy, tmp_path):
        _rewrite_call(todo_copy, "TodoListManager/list_todo_lists.py", """
            raise fastworkflow.Refused("there are no todo lists to list")
            """)
        out = _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)
        assert out.success is False
        assert out.command_response.response == "there are no todo lists to list"
        assert out.command_name == LIST_TODO_LISTS
        assert out.command_response.output is None


# ----------------------------------------------------------------------
# One __call__ signature: command_parameters is None for a command with no Input
# ----------------------------------------------------------------------


class TestCallSignature:
    def test_a_three_argument_call_gets_none_for_a_command_with_no_input(self, todo_copy, tmp_path):
        _rewrite_call(todo_copy, "TodoListManager/list_todo_lists.py", """
            return fastworkflow.CommandResponse(response=f"got {command_parameters!r}")
            """, signature="(self, workflow: Workflow, command: str, command_parameters)")
        out = _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)
        assert out.command_response.response == "got None"
        assert out.success is True

    def test_a_three_argument_call_gets_the_filled_input_for_a_command_with_one(self, example_copy, tmp_path):
        _rewrite_call(example_copy, "TodoListManager/get_todo_list.py", """
            return fastworkflow.CommandResponse(response=f"got {type(command_parameters).__name__} id={command_parameters.id}")
            """, signature="(self, workflow: Workflow, command: str, command_parameters)")
        out = _run(example_copy, _example_root(tmp_path), GET_TODO_LIST, parameters={"id": 7})
        assert out.command_response.response == "got Input id=7"
        assert out.command_parameters is not None and out.command_parameters.id == 7

    def test_a_two_argument_call_still_works_for_a_command_with_no_input(self, example_copy, tmp_path):
        out = _run(example_copy, _example_root(tmp_path), LIST_TODO_LISTS)
        assert out.success is True
        assert out.command_response.response == '{"result":[]}'

    def test_a_two_argument_call_still_works_for_a_command_with_an_input(self, todo_copy, tmp_path):
        # The unchanged todo command takes the Input in its three-argument __call__; a two-argument
        # rewrite of the same command gets no parameters and still runs.
        _rewrite_call(todo_copy, "TodoListManager/get_todo_list.py", """
            return fastworkflow.CommandResponse(response="two arguments")
            """)
        out = _run(todo_copy, _todo_root(tmp_path), GET_TODO_LIST, parameters={"id": 7})
        assert out.command_response.response == "two arguments"
        assert out.success is True


# ----------------------------------------------------------------------
# A wrong model or type fails the call, naming the command and the type
# ----------------------------------------------------------------------


class TestWrongReturn:
    def test_another_models_instance_fails_naming_the_command_and_the_type(self, todo_copy, tmp_path):
        _rewrite_call(todo_copy, "TodoListManager/list_todo_lists.py", """
            from pydantic import BaseModel

            class Stray(BaseModel):
                n: int = 1

            return Stray()
            """)
        with pytest.raises(TypeError) as raised:
            _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)
        message = str(raised.value)
        assert LIST_TODO_LISTS in message
        assert "Stray" in message
        assert "Signature.Output" in message

    def test_none_fails_naming_the_command_and_the_type(self, todo_copy, tmp_path):
        _rewrite_call(todo_copy, "TodoListManager/list_todo_lists.py", """
            return None
            """)
        with pytest.raises(TypeError) as raised:
            _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)
        assert LIST_TODO_LISTS in str(raised.value)
        assert "NoneType" in str(raised.value)

    def test_an_int_fails_naming_the_command_and_the_type(self, todo_copy, tmp_path):
        _rewrite_call(todo_copy, "TodoListManager/list_todo_lists.py", """
            return 3
            """)
        with pytest.raises(TypeError) as raised:
            _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)
        assert LIST_TODO_LISTS in str(raised.value)
        assert "int" in str(raised.value)


# ----------------------------------------------------------------------
# The library's own commands run through the executor (fix-3rsq.library-bare-output)
# ----------------------------------------------------------------------


class TestLibraryCommandsRunThroughExecutor:
    @pytest.fixture
    def library_copy(self, tmp_path) -> Path:
        return _copy(LIBRARY, tmp_path / "library")

    def test_find_member_returns_its_output_through_the_executor(self, library_copy, tmp_path, caplog):
        library = Library()
        wf = _workflow(library_copy, library, "library-find")
        wf.current_command_context = library
        with caplog.at_level("WARNING"):
            out = CommandExecutor.perform_action(
                wf, Action(command_name="Library/find_member", command="find the member Ann Lee",
                           parameters={"query": "Ann Lee"}))
        assert out.success is True
        assert out.command_name == "Library/find_member"
        assert type(out.command_response.output).__name__ == "Output"
        assert out.command_response.output.member_uids == ["m1"]
        assert out.command_response.output.total == 1
        assert json.loads(out.command_response.response) == {
            "member_uids": ["m1"], "names": ["Ann Lee"], "total": 1}
        assert not [r for r in caplog.records if "declares an Output" in r.getMessage()]

    def test_show_notes_takes_no_input_and_returns_its_output_through_the_executor(self, library_copy, tmp_path,
                                                                                  caplog):
        library = Library()
        book = library.books["b1"]
        wf = _workflow(library_copy, library, "library-notes")
        wf.current_command_context = book
        with caplog.at_level("WARNING"):
            out = CommandExecutor.perform_action(
                wf, Action(command_name="Record/show_notes", command="show the notes on this record"))
        assert out.success is True
        assert out.command_name == "Record/show_notes"
        assert out.command_response.output.notes == list(book.notes)
        assert out.command_response.output.total == len(book.notes)
        assert out.command_parameters is None
        assert not [r for r in caplog.records if "declares an Output" in r.getMessage()]
