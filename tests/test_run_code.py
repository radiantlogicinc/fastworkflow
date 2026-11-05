"""``fastworkflow command run-code`` and ``fastworkflow.planning.run_code`` (design 2.39, 3.18), on real workflows.

The sample workflow (``fastworkflow/examples/library``) and ``tests/todo_list_workflow`` are copied to ``tmp_path``
for every test, since a run changes the session's state (and a copy keeps the repository clean). No model is
called: the run is checked for that with a seam that fails if one is reached, and by a CLI run with no LLM
environment at all. The load-problem cases write a broken command into a copy and check ``fastworkflow run``
(its output before the first prompt), ``fastworkflow train`` (it stops before training) and the FastAPI
server's startup log.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

import fastworkflow
from fastworkflow.planning import run_code
from fastworkflow.planning_code_run import RunCodeError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")
TODO = os.path.join(ROOT, "tests", "todo_list_workflow")

#: a command file that does not load: its Signature has no docstring (the planning command's description)
BROKEN_PLANNING_COMMAND = '''class Signature:
    plain_utterances = ["a broken command"]


class PlanGenerator:
    def find(self, input):
        return ["find_book query=x"]
'''


def _copy(source: str, destination) -> str:
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "___*"))
    return str(destination)


@pytest.fixture
def library(tmp_path) -> str:
    return _copy(LIBRARY, tmp_path / "library")


@pytest.fixture
def todo(tmp_path) -> str:
    return _copy(TODO, tmp_path / "todo_list_workflow")


def _no_llm_env() -> dict:
    """The process environment without any model or key setting."""
    prefixes = ("LITELLM", "LLM_", "JEV", "OPENAI", "MISTRAL", "ANTHROPIC", "FW_")
    return {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}


def _cli(*args, env=None, stdin=subprocess.DEVNULL, timeout=600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "fastworkflow.cli", *args], cwd=ROOT, env=env,
                          stdin=stdin, capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def no_model(monkeypatch):
    """A seam: any model call in the run fails the test (run-code makes no model calls by design)."""
    import litellm

    def refuse(*args, **kwargs):
        raise AssertionError("run-code called a model")

    monkeypatch.setattr(litellm, "completion", refuse)
    monkeypatch.setattr(litellm, "acompletion", refuse)
    try:
        import dspy

        monkeypatch.setattr(dspy.LM, "__call__", refuse)
    except (ImportError, AttributeError):
        pass
    yield


# ------------------------------------------------------------------ a planning command, phase by phase

def test_a_planning_command_prints_its_phases_and_stops_at_the_first_open_step(library, no_model):
    """book_by_title with a title two books share: find lists them, the pick is open, the run stops (exit 0)."""
    result = run_code(library, "book_by_title", {"title": "Dun"})
    assert result.open_step is not None and result.open_step.startswith("step 1 of book")
    assert result.text.splitlines()[:4] == [
        "Phase find",
        "  1. find_book query=Dun",
        "  code runs: find_book query=Dun",
        "  facts: Output(book_uids=['b1', 'b2'], titles=['Dune', 'Dune Messiah'], total=2)",
    ]
    assert "  step 1 is open: the agent would run it and the rest" in result.text
    assert "answer written" not in result.text and result.answer == ""

    cli = _cli("command", "run-code", library, "book_by_title", "--input", '{"title": "Dun"}', env=_no_llm_env())
    assert cli.returncode == 0, cli.stderr
    assert "step 1 is open: the agent would run it and the rest" in cli.stdout


def test_a_planning_command_that_completes_prints_its_answer_and_output(library, no_model):
    result = run_code(library, "book_by_title", {"title": "Emma"})
    assert result.open_step is None
    assert result.answer == "Emma is by Jane Austen; the library owns 2 copies."
    assert "output: Output(found=True, book_uid='b3', title='Emma', author='Jane Austen', copies=2)" in result.text
    assert "  now in: Book" in result.text   # the pick opened the book, so the workflow moved onto it


# ------------------------------------------------------------------ --start-with

def test_start_with_puts_the_workflow_on_a_record_before_a_context_planning_command(library, no_model):
    """shelf_report is a Book command: it needs the workflow on a book first."""
    with pytest.raises(RunCodeError, match="must be on a Book record first"):
        run_code(library, "shelf_report")

    result = run_code(library, "shelf_report", start_with=["open_book book_uid=b1"])
    assert result.text.startswith("started with: open_book book_uid=b1")
    assert result.answer == "Dune is held at Central, 3 copies; on loan to 2 borrowers (Ann Lee, Raj Patel)."


def test_a_start_step_code_cannot_run_is_an_error(library, no_model):
    """open_book needs its book_uid: a step without it is not something code can run."""
    with pytest.raises(RunCodeError, match="open_book") as caught:
        run_code(library, "shelf_report", start_with=["open_book"])
    assert "started with" not in caught.value.text


# ------------------------------------------------------------------ a step that names a planning command

def test_a_step_that_names_a_planning_command_runs_it_indented(library, no_model):
    """book_loans' book phase is one step: book_by_title, a planning command. It runs as its own call, indented."""
    result = run_code(library, "book_loans", {"title": "Dune"})
    lines = result.text.splitlines()
    assert lines[0] == "Phase book"
    assert "  step 1 runs planning command book_by_title:" in lines
    nested = lines.index("  step 1 runs planning command book_by_title:")
    assert lines[nested + 1] == "    Phase find"
    assert "    Phase book" in lines[nested:]
    assert "  facts: Output(found=True, book_uid='b1', title='Dune', author='Frank Herbert', copies=3)" in lines
    assert result.answer == "Dune is on loan to 2 borrowers (Ann Lee, Raj Patel)."


# ------------------------------------------------------------------ an execution command

def test_an_execution_command_runs_and_prints_its_command_output(library, todo, no_model):
    result = run_code(library, "Library/find_book", {"query": "Dune"})
    assert result.output is not None
    assert result.output.book_uids == ["b1", "b2"]
    assert "output: Output(book_uids=['b1', 'b2'], titles=['Dune', 'Dune Messiah'], total=2)" in result.text

    result = run_code(todo, "TodoListManager/list_todo_lists")
    assert result.text.startswith("TodoListManager/list_todo_lists\n")
    assert "  context: TodoListManager" in result.text
    assert "output: Output(todolist_ids=[1])" in result.text


def test_a_command_that_is_not_in_the_workflow_is_an_error(library, no_model):
    with pytest.raises(RunCodeError, match="no command named 'no_such_command'"):
        run_code(library, "no_such_command")


# ------------------------------------------------------------------ a hook that raises

def test_a_hook_that_raises_exits_1_naming_it(library, no_model):
    path = os.path.join(library, "_commands", "Library", "book_loans.py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    marker = "    def __result__(self, input, book, loans):\n"
    assert source.count(marker) == 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source.replace(marker, marker + '        if input.title == "Dune":\n'
                                                  '            raise RuntimeError("the answer hook failed")\n'))
    with pytest.raises(RunCodeError) as caught:
        run_code(library, "book_loans", {"title": "Dune"})
    assert "the hook __result__() of Library/book_loans raised RuntimeError: the answer hook failed" in str(caught.value)
    assert "Phase book" in caught.value.text   # what it printed before it stopped

    cli = _cli("command", "run-code", library, "book_loans", "--input", '{"title": "Dune"}', env=_no_llm_env())
    assert cli.returncode == 1
    assert "__result__" in cli.stderr and "Library/book_loans" in cli.stderr


# ------------------------------------------------------------------ a command file that does not load

def _break_a_command_file(folder: str) -> None:
    with open(os.path.join(folder, "_commands", "Library", "broken_thing.py"), "w", encoding="utf-8") as fh:
        fh.write(BROKEN_PLANNING_COMMAND)


def test_fastworkflow_run_prints_the_load_problem_before_its_first_prompt(library):
    _break_a_command_file(library)
    env_files = [os.path.join(library, "fastworkflow.env"), os.path.join(library, "fastworkflow.passwords.env")]
    cli = _cli("run", library, *env_files, env=_no_llm_env())
    out = cli.stdout + cli.stderr
    problem = "Load problem: Library/broken_thing: PlanningCommandFormatError"
    assert problem in out
    assert "User >" not in out   # no prompt was shown: the run stopped before one (the copy is not trained)
    assert out.index(problem) < out.index("is not trained")


def test_fastworkflow_train_exits_with_an_error_listing_the_problem_before_training(library):
    _break_a_command_file(library)
    env_files = [os.path.join(library, "fastworkflow.env"), os.path.join(library, "fastworkflow.passwords.env")]
    cli = _cli("train", library, *env_files, env=_no_llm_env())
    assert cli.returncode == 1
    assert "training stopped before it started" in cli.stderr
    assert "Library/broken_thing: PlanningCommandFormatError" in cli.stderr
    # Training writes its artifacts into ___command_info; the load check writes only its two caches there.
    info = os.path.join(library, "___command_info")
    written = set(os.listdir(info)) if os.path.isdir(info) else set()
    assert written <= {"command_directory.json", "routing_definition.json"}, f"training began: {sorted(written)}"


def test_the_server_logs_the_load_problem_at_startup(library, tmp_path):
    _break_a_command_file(library)
    log = tmp_path / "server.log"
    env_files = [os.path.join(library, "fastworkflow.env"), os.path.join(library, "fastworkflow.passwords.env")]
    with open(log, "w", encoding="utf-8") as handle:
        try:
            subprocess.run([sys.executable, "-m", "fastworkflow.cli", "run_fastapi_mcp", library, *env_files],
                           cwd=ROOT, env=_no_llm_env(), stdin=subprocess.DEVNULL, stdout=handle,
                           stderr=subprocess.STDOUT, timeout=45)
        except subprocess.TimeoutExpired:
            pass   # the server runs until stopped: the startup lines are what is read
    text = log.read_text(encoding="utf-8")
    assert "command file does not load, so the server does not serve it: Library/broken_thing" in text
    assert "Application ready to accept traffic" in text


# ------------------------------------------------------------------ no model, by the environment

def test_run_code_on_the_cli_needs_no_llm_environment(library):
    cli = _cli("command", "run-code", library, "book_loans", "--input", '{"title": "Dune"}', env=_no_llm_env())
    assert cli.returncode == 0, cli.stderr
    assert "Dune is on loan to 2 borrowers" in cli.stdout


def test_the_cli_refuses_input_that_is_not_a_json_object(library):
    cli = _cli("command", "run-code", library, "book_loans", "--input", "[1, 2]", env=_no_llm_env())
    assert cli.returncode == 1
    assert "must be a JSON object" in cli.stderr
