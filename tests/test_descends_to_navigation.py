"""Navigation is declared once, on the command: ``descends_to`` (design 3.15, Phase 10 step 2).

Each case runs on a COPY of a real workflow (the library example, or ``tests/todo_list_workflow``).
Nothing is mocked: the routing definition, the context classes and the command classes are the
copy's own. The one stand-in is the CME hop (``script_command_extraction``, from
``tests/test_step_helpers``): the copy has no trained intent heads, so the command name and its
``name=value`` arguments are handed over as a successful extraction would, and
``CommandExecutor._invoke_command_impl`` runs the rest for real.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow import command_executor
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.context_navigation import DescendEdge, collect_descend_edges
from tests.test_step_helpers import script_command_extraction
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    initialized_fastworkflow,
)  # the registry is cleared per test: command classes are cached by name alone

LIBRARY = os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "library")
TODO = os.path.join(os.path.dirname(__file__), "todo_list_workflow")

_COPY_IGNORE = shutil.ignore_patterns(
    "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
)

#: The library's navigation edges, recorded before Phase 10 step 2 (the library declared
#: ``descends_to`` on open_book and open_member and no ``enter_command`` anywhere).
LIBRARY_EDGES_BEFORE = (
    DescendEdge("Library", "open_book", "Book", ("book_uid",), ()),
    DescendEdge("Library", "open_member", "Member", ("member_uid",), ()),
)

UNDECLARED_WARNING = (
    'open_book moved the workflow into Book but declares no descends_to; '
    'declare descends_to = "Book" so plans can reach Book\'s commands'
)


def _copy(tmp_path, source: str, name: str):
    folder = tmp_path / name
    shutil.copytree(source, folder, ignore=_COPY_IGNORE, symlinks=True)
    return folder


def _edit(path, marker: str, insertion: str) -> None:
    text = path.read_text()
    assert marker in text, path
    path.write_text(text.replace(marker, insertion, 1))


def _declare_descends_to(folder, command_file: str, context: str) -> None:
    _edit(folder / "_commands" / command_file, "class ResponseGenerator:\n",
          f'class ResponseGenerator:\n    descends_to = "{context}"\n')


def _undeclare_open_book(folder) -> None:
    path = folder / "_commands" / "Library" / "open_book.py"
    text = path.read_text()
    assert 'descends_to = "Book"\n' in text, path
    path.write_text(text.replace('    descends_to = "Book"\n', "", 1))


def _application(folder):
    return fastworkflow.Workflow.create(workflow_folderpath=str(folder),
                                        workflow_id_str=f"descends-{folder.name}")


def _cme_for(app):
    return fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id,
        workflow_context={"app_workflow": app},
    )


def _session(app):
    """The three members _invoke_command_impl reads off a session: the active workflow, the CME
    workflow and its context. The workflows are real."""
    return SimpleNamespace(get_active_workflow=lambda: app, cme_workflow=_cme_for(app))


@pytest.fixture
def warnings_reset(monkeypatch):
    """The once-per-process record of undeclared moves starts empty in every test."""
    monkeypatch.setattr(command_executor, "_UNDECLARED_MOVES_WARNED", set())


# ---------------------------------------------------------------------------
# A retired declaration is a load error
# ---------------------------------------------------------------------------

def test_enter_command_on_a_context_class_fails_to_load_naming_descends_to(
        tmp_path, initialized_fastworkflow):
    folder = _copy(tmp_path, TODO, "todo_enter_command")
    _edit(folder / "_commands" / "TodoList" / "_TodoList.py", "class Context:\n",
          "class Context:\n    enter_command = 'get_todo_list <todo_list_id>'\n")

    with pytest.raises(ValueError) as raised:
        fastworkflow.RoutingRegistry.get_definition(str(folder))

    assert str(raised.value) == (
        '_commands/TodoList/_TodoList.py: Context.enter_command is no longer read; '
        'declare descends_to = "TodoList" on get_todo_list\'s ResponseGenerator'
    )


def test_enter_command_fails_again_on_every_read_because_nothing_is_cached(
        tmp_path, initialized_fastworkflow):
    """Workflow.create is lazy; the routing read is where the workflow loads, and a failed
    load is not cached, so a second read fails the same way."""
    folder = _copy(tmp_path, TODO, "todo_enter_command_twice")
    _edit(folder / "_commands" / "TodoList" / "_TodoList.py", "class Context:\n",
          "class Context:\n    enter_command = 'get_todo_list <todo_list_id>'\n")
    _application(folder)  # creating the workflow does not read its routing

    for _ in range(2):
        with pytest.raises(ValueError, match=re.escape('declare descends_to = "TodoList" on get_todo_list')):
            fastworkflow.RoutingRegistry.get_definition(str(folder))


def test_enter_commands_list_fails_to_load_naming_each_command(tmp_path, initialized_fastworkflow):
    folder = _copy(tmp_path, TODO, "todo_enter_commands")
    _edit(folder / "_commands" / "TodoList" / "_TodoList.py", "class Context:\n",
          "class Context:\n    enter_commands = ['get_todo_list <todo_list_id>', 'add_child_todolist x']\n")

    with pytest.raises(ValueError) as raised:
        fastworkflow.RoutingRegistry.get_definition(str(folder))

    message = str(raised.value)
    assert "Context.enter_commands is no longer read" in message
    assert 'declare descends_to = "TodoList" on get_todo_list\'s ResponseGenerator' in message
    assert 'declare descends_to = "TodoList" on add_child_todolist\'s ResponseGenerator' in message


def test_enter_command_stays_a_load_error_when_the_definition_was_persisted(
        tmp_path, initialized_fastworkflow):
    """A routing definition saved before the retirement is not trusted to skip the check."""
    folder = _copy(tmp_path, TODO, "todo_enter_command_persisted")
    fastworkflow.RoutingRegistry.get_definition(str(folder))  # saves routing_definition.json
    _edit(folder / "_commands" / "TodoList" / "_TodoList.py", "class Context:\n",
          "class Context:\n    enter_command = 'get_todo_list <todo_list_id>'\n")
    fastworkflow.RoutingRegistry.clear_registry()

    with pytest.raises(ValueError, match="Context.enter_command is no longer read"):
        fastworkflow.RoutingRegistry.get_definition(str(folder))


# ---------------------------------------------------------------------------
# descends_to is the edge
# ---------------------------------------------------------------------------

def test_descends_to_on_a_command_is_the_collected_edge(tmp_path, initialized_fastworkflow):
    folder = _copy(tmp_path, TODO, "todo_descends")
    _declare_descends_to(folder, "TodoListManager/get_todo_list.py", "TodoList")

    edges = collect_descend_edges(str(folder))

    assert edges == (
        DescendEdge("TodoListManager", "get_todo_list", "TodoList", ("id",), ()),
    )


def test_a_command_without_descends_to_collects_no_edge(tmp_path, initialized_fastworkflow):
    folder = _copy(tmp_path, TODO, "todo_no_descends")

    assert collect_descend_edges(str(folder)) == ()


def test_library_navigation_edges_are_unchanged_by_the_migration(tmp_path, initialized_fastworkflow):
    """Before and after: the library never declared enter_command, so its descends_to edges
    (recorded before this change) are the same set."""
    folder = _copy(tmp_path, LIBRARY, "library_edges")

    assert set(collect_descend_edges(str(folder))) == set(LIBRARY_EDGES_BEFORE)


# ---------------------------------------------------------------------------
# A move the command does not declare is reported, once
# ---------------------------------------------------------------------------

def test_an_undeclared_move_warns_once_per_command_and_context(
        tmp_path, initialized_fastworkflow, monkeypatch, warnings_reset, caplog):
    folder = _copy(tmp_path, LIBRARY, "library_undeclared")
    _undeclare_open_book(folder)
    app = _application(folder)
    app.root_command_context = _library_root()
    app.current_command_context = app.root_command_context
    session = _session(app)
    script_command_extraction(monkeypatch, app)

    with caplog.at_level(logging.WARNING, logger=command_executor.__name__):
        CommandExecutor._invoke_command_impl(session, "open_book book_uid=b1")
        assert app.current_command_context_name == "Book"
        CommandExecutor._invoke_command_impl(session, "open_book book_uid=b2")

    warnings = [r for r in caplog.records if "declares no descends_to" in r.getMessage()]
    assert [r.getMessage() for r in warnings] == [UNDECLARED_WARNING]
    assert warnings[0].levelno == logging.WARNING


def test_a_declared_move_does_not_warn(
        tmp_path, initialized_fastworkflow, monkeypatch, warnings_reset, caplog):
    folder = _copy(tmp_path, LIBRARY, "library_declared")
    app = _application(folder)
    app.root_command_context = _library_root()
    app.current_command_context = app.root_command_context
    session = _session(app)
    script_command_extraction(monkeypatch, app)

    with caplog.at_level(logging.WARNING, logger=command_executor.__name__):
        CommandExecutor._invoke_command_impl(session, "open_book book_uid=b1")

    assert app.current_command_context_name == "Book"
    assert not [r for r in caplog.records if "declares no descends_to" in r.getMessage()]


def _library_root():
    from fastworkflow.examples.library.application.library import Library

    return Library()
