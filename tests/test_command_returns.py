"""What a command returns, and the one ``__call__`` signature (fix-3rsq.12.early, fix-3rsq.12.4; design 3.14,
2.29; plan Phase 12 steps 1-2 and 6).

Integration tests through ``CommandExecutor.perform_action`` on copies of the example workflow, the todo
list workflow and the library. A test that needs a command to return another form rewrites that command's
``__call__`` in its own copy; nothing is mocked. The short forms only: the command's own Output, a str, or a
CommandResponse; a refusal is ``raise fastworkflow.Refused(message)``. A hand-built CommandOutput return is a
call error, and so is a two-argument ``__call__`` a load error.

Plan Phase 12 step 6 bullets, and where each one is covered:
- an Output return has the envelope of the same response built by hand: ``TestOutputReturn``;
- str, CommandResponse with artifacts, Refused: ``TestOtherForms``; a shipped or hand-built CommandOutput
  return is a call error (``TestOtherForms``, ``TestWrongReturn``);
- a two-argument ``__call__`` is a load error and a three-argument one gets ``command_parameters=None``
  for a command with no Input: ``TestCallSignature``;
- paged commands (paged without total is a load error; a datetime start or bool skip is not paged; a
  full_list step reads every page; the paging line appears when rows remain): covered in
  ``tests/test_command_paging.py`` (``TestLoadCheck``, ``TestPagingLine``, the full_list and next-page tests),
  not repeated here;
- ``GET /command_schemas`` and the per-command MCP tool's ``outputSchema``/``structuredContent``: covered in
  ``tests/test_command_schemas.py`` (``test_command_schemas_*``, ``test_mcp_tool_declares_output_schema_*``);
- the REST turn body keeps ``next_actions`` and ``recommendations``: covered in
  ``tests/test_command_schemas.py::test_turn_body_command_response_keeps_next_actions_and_recommendations``.
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
from fastworkflow import ModuleType
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry
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

THREE_ARGUMENT_CALL = "(self, workflow: Workflow, command: str, command_parameters)"


@pytest.fixture(autouse=True)
def _loader_caches_per_test():
    # The routing cache is keyed by command name alone, and these tests rewrite the same commands.
    RoutingDefinition._GLOBAL_COMMAND_CLASS_CACHE.clear()
    yield
    RoutingDefinition._GLOBAL_COMMAND_CLASS_CACHE.clear()


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


def _rewrite_call(folder: Path, command_file: str, body: str, signature: str = THREE_ARGUMENT_CALL) -> None:
    """Replace the ``__call__`` of a command in this copy. Every shipped command file defines ``__call__`` last."""
    path = folder / "_commands" / command_file
    head, marker, _ = path.read_text().partition("    def __call__(")
    assert marker, f"{path} has no __call__ to rewrite"
    if "import fastworkflow" not in head:
        head = "import fastworkflow\n" + head  # the rewritten body builds a fastworkflow.CommandResponse
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


def _load_response_generator(folder: Path, command: str):
    RoutingRegistry.clear_registry()
    routing = RoutingRegistry.get_definition(str(folder))
    return routing.get_command_class(command, ModuleType.RESPONSE_GENERATION_INFERENCE)


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
# A command that returns its Signature.Output: the same envelope as a response built by hand
# ----------------------------------------------------------------------


class TestOutputReturn:
    def test_an_output_return_has_the_envelope_of_the_same_response_built_by_hand(self, tmp_path):
        # Two copies of the same command: one returns its Signature.Output and leaves the framework to build the
        # envelope; the other builds the same CommandResponse by hand (response, and the output it carries) and
        # returns that. A hand-built CommandOutput is a call error (TestOtherForms), so this is the hand-built form.
        returned = _copy(EXAMPLE, tmp_path / "returned" / "example_workflow")
        built_copy = _copy(EXAMPLE, tmp_path / "built" / "example_workflow")
        _rewrite_call(returned, "TodoListManager/list_todo_lists.py", """
            app = workflow.command_context_for_response_generation
            return Signature.Output(result=app.list_todo_lists())
            """)
        _rewrite_call(built_copy, "TodoListManager/list_todo_lists.py", """
            app = workflow.command_context_for_response_generation
            output = Signature.Output(result=app.list_todo_lists())
            return fastworkflow.CommandResponse(response=output.model_dump_json(), output=output)
            """)

        framed = _run(returned, _example_root(tmp_path / "a"), LIST_TODO_LISTS)
        built = _run(built_copy, _example_root(tmp_path / "b"), LIST_TODO_LISTS)

        output = framed.command_response.output
        assert output is not None and type(output).__name__ == "Output"
        assert framed.command_response.response == output.model_dump_json()
        assert json.loads(framed.command_response.response) == {"result": []}
        assert framed.command_response.model_dump() == built.command_response.model_dump()
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
# The other short forms: str, CommandResponse with artifacts, Refused; the hand-built CommandOutput is refused
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

    def test_a_hand_built_command_output_return_is_a_call_error(self, todo_copy, tmp_path):
        # The old form: a command that builds a CommandOutput by hand. It is refused, naming the command.
        _rewrite_call(todo_copy, "TodoListManager/list_todo_lists.py", """
            return fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(response="built"))
            """)
        with pytest.raises(TypeError) as raised:
            _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)
        assert LIST_TODO_LISTS in str(raised.value)
        assert "CommandOutput" in str(raised.value)
        assert "not a short form" in str(raised.value)

    def test_frame_command_result_refuses_a_command_output_it_is_handed(self):
        built = CommandOutput(command_response=CommandResponse(response="as built"))
        with pytest.raises(TypeError, match="not a short form"):
            frame_command_result(built, command_name="x", output_model=None)

    def test_a_command_response_with_success_false_is_a_call_error(self, todo_copy, tmp_path):
        _rewrite_call(todo_copy, "TodoListManager/list_todo_lists.py", """
            return fastworkflow.CommandResponse(response="nope", success=False)
            """)
        with pytest.raises(TypeError, match="success=False"):
            _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)

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
            """)
        out = _run(todo_copy, _todo_root(tmp_path), LIST_TODO_LISTS)
        assert out.command_response.response == "got None"
        assert out.success is True

    def test_a_three_argument_call_gets_the_filled_input_for_a_command_with_one(self, example_copy, tmp_path):
        _rewrite_call(example_copy, "TodoListManager/get_todo_list.py", """
            return fastworkflow.CommandResponse(response=f"got {type(command_parameters).__name__} id={command_parameters.id}")
            """)
        out = _run(example_copy, _example_root(tmp_path), GET_TODO_LIST, parameters={"id": 7})
        assert out.command_response.response == "got Input id=7"
        assert out.command_parameters is not None and out.command_parameters.id == 7

    def test_a_two_argument_call_is_a_load_error(self, todo_copy):
        _rewrite_call(todo_copy, "TodoListManager/list_todo_lists.py", """
            return fastworkflow.CommandResponse(response="two arguments")
            """, signature="(self, workflow: Workflow, command: str)")
        with pytest.raises(ValueError) as raised:
            _load_response_generator(todo_copy, LIST_TODO_LISTS)
        assert "ResponseGenerator.__call__ must take three arguments" in str(raised.value)
        assert "(self, workflow, command, input)" in str(raised.value)

    def test_a_two_argument_call_with_an_input_is_a_load_error_too(self, todo_copy):
        _rewrite_call(todo_copy, "TodoListManager/get_todo_list.py", """
            return fastworkflow.CommandResponse(response="two arguments")
            """, signature="(self, workflow: Workflow, command: str)")
        with pytest.raises(ValueError, match="must take three arguments"):
            _load_response_generator(todo_copy, GET_TODO_LIST)


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
