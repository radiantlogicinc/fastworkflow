"""Two workflows in one process do not share what is keyed by a name (fix-3rsq.phase10-routing-class-cache).

The command-class cache is keyed by (workflow folder, command), and a workflow's command modules are loaded
under a namespace derived from the folder's resolved path, so two folders with the same basename never share
modules or relative imports. Each case loads real copies of the library and todo-list workflows; no model is
called.
"""
from __future__ import annotations

import inspect
import os
import re
import shutil
import sys
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingDefinition
from fastworkflow.planning import run_code
from fastworkflow.utils import python_utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")
TODO = os.path.join(ROOT, "tests", "todo_list_workflow")
_IGNORE = shutil.ignore_patterns("__pycache__", "___*")


def _copy(source: str, destination) -> str:
    shutil.copytree(source, destination, ignore=_IGNORE)
    return str(destination)


@pytest.fixture(autouse=True)
def _registries():
    if fastworkflow.RoutingRegistry is None:
        fastworkflow.init({})


@pytest.fixture
def no_model(monkeypatch):
    """A seam: any model call in a run fails the test (run-code makes no model calls by design)."""
    import litellm

    def refuse(*args, **kwargs):
        raise AssertionError("the workflow called a model")

    monkeypatch.setattr(litellm, "completion", refuse)
    monkeypatch.setattr(litellm, "acompletion", refuse)
    import dspy

    monkeypatch.setattr(dspy.LM, "__call__", refuse)


def _startup_class(folder: str):
    definition = RoutingDefinition.build(folder)
    return definition.get_command_class("startup", fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)


def test_two_copies_with_a_same_named_command_each_get_their_own_class(tmp_path):
    first = _copy(LIBRARY, tmp_path / "first" / "library")
    second = _copy(LIBRARY, tmp_path / "second" / "library")
    startup = os.path.join(second, "_commands", "startup.py")
    with open(startup, encoding="utf-8") as fh:
        source = fh.read()
    with open(startup, "w", encoding="utf-8") as fh:
        fh.write(source.replace("The library is open.", "The archive is open."))

    # The second copy is loaded first, then the first, then the second again: an order-dependent
    # cache would hand the first copy's class to whichever copy is asked for second.
    second_class = _startup_class(second)
    first_class = _startup_class(first)
    again_second = _startup_class(second)

    assert first_class is not second_class
    assert again_second is second_class
    assert os.path.realpath(inspect.getfile(first_class)) == os.path.realpath(
        os.path.join(first, "_commands", "startup.py"))
    assert os.path.realpath(inspect.getfile(second_class)) == os.path.realpath(startup)
    assert first_class()(SimpleNamespace(), "startup", None) == "The library is open."
    assert second_class()(SimpleNamespace(), "startup", None) == "The archive is open."


def test_two_folders_with_one_basename_import_their_own_modules_and_relative_imports(tmp_path):
    first = _copy(LIBRARY, tmp_path / "first" / "library")
    second = _copy(LIBRARY, tmp_path / "second" / "library")
    for folder, origin in ((first, "first copy"), (second, "second copy")):
        with open(os.path.join(folder, "application", "library.py"), "a", encoding="utf-8") as fh:
            fh.write(f'\nORIGIN = "{origin}"\n')

    first_startup = python_utils.get_module(os.path.join(first, "_commands", "startup.py"), first)
    second_startup = python_utils.get_module(os.path.join(second, "_commands", "startup.py"), second)

    assert first_startup.__name__ != second_startup.__name__
    # startup.py does `from ..application.library import Library`: each copy's relative import must
    # resolve inside its own folder, not to the module the other copy loaded under the same basename.
    first_library = sys.modules[first_startup.Library.__module__]
    second_library = sys.modules[second_startup.Library.__module__]
    assert first_library.ORIGIN == "first copy"
    assert second_library.ORIGIN == "second copy"
    assert os.path.realpath(first_library.__file__) == os.path.realpath(
        os.path.join(first, "application", "library.py"))
    assert python_utils.get_module(os.path.join(first, "_commands", "startup.py"), first) is first_startup


def test_run_code_on_two_workflows_in_one_process_gives_each_its_own_startup(tmp_path, no_model):
    library = _copy(LIBRARY, tmp_path / "library")
    todo = _copy(TODO, tmp_path / "todo_list_workflow")

    library_run = run_code(library, "book_by_title", {"title": "Dun"})
    todo_run = run_code(todo, "TodoListManager/list_todo_lists")
    # The todo run's startup opens a TodoListManager; a class loaded for the library's startup would
    # open a Library instead and the context would read Library.
    assert "  context: TodoListManager" in todo_run.text
    assert "output: Output(todolist_ids=[1])" in todo_run.text

    # The library run after the todo run gets the library's startup back (the planning command starts).
    again = run_code(library, "book_by_title", {"title": "Dun"})
    assert again.text == library_run.text
    assert again.open_step is not None and again.open_step.startswith("step 1 of book")


def test_a_namespaced_module_name_digests_as_the_bare_name_it_had_before(tmp_path):
    """A parameter-example cache key written before the namespace existed still matches: the namespace
    is written back to the bare basename wherever a module name is digested."""
    from fastworkflow.train import param_example_cache as cache

    folder = _copy(LIBRARY, tmp_path / "library")
    namespace = python_utils.workflow_namespace(folder)
    assert namespace.startswith("library__") and namespace != "library"
    assert python_utils.workflow_namespace(folder) == namespace   # stable for one folder
    assert python_utils.legacy_module_names(f"{namespace}._commands.startup") == "library._commands.startup"
    assert cache._canonical_digest({"type": f"{namespace}.application.library.Book"}) == \
        cache._canonical_digest({"type": "library.application.library.Book"})


def test_two_copies_of_the_tutorial_workflow_at_two_paths_preview_identically(tmp_path):
    """The previews name a planning command's types by their plain names: a workflow folder's namespace
    digest (set by its path) never reaches a preview, so two copies at two paths preview the same text."""
    from fastworkflow import command_preview

    tutorial = os.path.join(ROOT, "tests", "planning_tutorial_workflow")
    first = command_preview.render_previews(_copy(tutorial, tmp_path / "first" / "planning_tutorial_workflow"))
    second = command_preview.render_previews(_copy(tutorial, tmp_path / "second" / "planning_tutorial_workflow"))
    assert first and first == second
    for name, text in first.items():
        assert re.search(r"__[0-9a-f]{8}", text) is None, name
        assert str(tmp_path) not in text, name
