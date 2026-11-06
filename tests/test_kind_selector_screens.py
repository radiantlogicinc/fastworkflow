"""Kind selectors, and the ``None`` marker.

A ``str`` field whose description lists the kinds it takes ("One of: application, group, ...") or
whose labeled examples give kind words as its value is a kind selector: a value it lists is a kind
it takes, so the kind-word screen does not refuse it. The same word in a free-text field with no
such description is still refused. The pronoun refusal applies to a kind selector too. The text
``None`` is the missing-value marker, not a kind word: a required field holding it is missing, and
an optional field takes its default.

Integration only: a temporary copy of ``todo_list_workflow`` (its contexts are the kind words) and of
``planning_tutorial_workflow`` (with one skill added), the real vocabulary builder, loader, screens and
``check_arguments``. The repo's own ``___command_info`` is never read for writing or touched.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import planning_commands as sc
from fastworkflow import workflow_agent
from fastworkflow.utils import signatures
from fastworkflow.utils import workflow_vocabulary as wv
from fastworkflow.utils.signatures import InputForParamExtraction

TESTS = Path(__file__).resolve().parent
TODO = TESTS / "todo_list_workflow"
TUTORIAL = TESTS / "planning_tutorial_workflow"
ASSIGN_TO = "TodoItem/assign_to"
GENERIC_DESCRIPTION = "name of the person responsible for doing this task"
LISTED_DESCRIPTION = "the kind of record responsible for this task. One of: todoitem, todolist"

#: A skill with a declared kind selector (``resource_type``: its description lists the kinds it
#: takes), a free-text input (``query``) and an optional one (``note``). Written into the copy only.
FILE_REQUEST = '''
from pydantic import BaseModel, Field


class Signature:
    """File a request about one thing."""

    class Input(BaseModel):
        resource_type: str = Field(description="which kind of thing the request is about. One of: book, member, branch")
        query: str = Field(description="free text to search for")
        note: str = Field(default="", description="a note on the request")


class PlanGenerator:
    def steps(self, input):
        return [f"look up {input.query}"]
'''


def _copy(source: Path, destination: Path) -> str:
    shutil.copytree(source, destination,
                    ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    return str(destination)


@pytest.fixture
def todo_copy(tmp_path):
    """A copy of the todo workflow with ``TodoItem/assign_to`` described as a kind selector."""
    folder = _copy(TODO, tmp_path / "todo")
    command = Path(folder) / "_commands" / "TodoItem" / "assign_to.py"
    text = command.read_text(encoding="utf-8")
    assert GENERIC_DESCRIPTION in text
    command.write_text(text.replace(GENERIC_DESCRIPTION, LISTED_DESCRIPTION), encoding="utf-8")
    return folder


@pytest.fixture
def tutorial_copy(tmp_path):
    """A copy of the tutorial workflow with the ``file_request`` skill added."""
    folder = _copy(TUTORIAL, tmp_path / "tutorial")
    (Path(folder) / "_commands" / "file_request.py").write_text(FILE_REQUEST, encoding="utf-8")
    return folder


@pytest.fixture
def workflow(tmp_path):
    """The tutorial workflow loaded for ``validate_parameters``; the state root is a temp folder."""
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    fastworkflow.RoutingRegistry.get_definition(str(TUTORIAL), load_cached=False)
    app_workflow = fastworkflow.Workflow.create(str(TUTORIAL), workflow_id_str=f"kind_{uuid.uuid4().hex[:8]}")
    yield app_workflow
    fastworkflow.RoutingRegistry.clear_registry()


def _labeled(folder: str, command: str, examples: list[dict]) -> Path:
    """Write ``<bare command>_param_labeled.json`` (the file the trainset reader and train write)."""
    path = Path(folder) / "___command_info" / f"{command.rpartition('/')[2]}_param_labeled.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"command_name": command.rpartition("/")[2],
               "valid_examples": [{"fields": fields, "inputs": ["command"]} for fields in examples],
               "rejected_examples": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load(folder: str) -> wv.WorkflowVocabulary:
    return wv.load_workflow_vocabulary(folder)


# ---------------------------------------------------------------- what a description lists

@pytest.mark.parametrize("description, kinds, listed", [
    ("Which kind of thing access is being asked about. One of: application, group, permission, "
     "repository, resource", {"application", "group", "permission", "repository", "resource", "account"},
     {"application", "group", "permission", "repository", "resource"}),
    ("The kind. one of a, b or c", {"a"}, {"a", "b", "c"}),
    ("e.g. x, y", set(), {"x", "y"}),
    ("the group and the account to list", {"group", "account"}, {"group", "account"}),
], ids=["one-of-list", "one-of-or", "e-g", "two-kind-words-named"])
def test_a_description_that_lists_kinds_gives_the_listed_options(description, kinds, listed):
    assert wv.description_kind_options(description, frozenset(kinds)) == frozenset(listed)


@pytest.mark.parametrize("description, kinds", [
    ("the group's name", {"group"}),          # one kind word is not a list
    ("free text to search for", {"group"}),   # no kind word, no list
], ids=["one-kind-word", "none"])
def test_a_description_that_names_no_list_gives_no_options(description, kinds):
    assert wv.description_kind_options(description, frozenset(kinds)) == frozenset()


# ---------------------------------------------------------------- the generic screen

def _screened(command: str, vocabulary: wv.WorkflowVocabulary, **values) -> dict:
    """The fields of a model with the given values that the kind-word screen refuses."""
    class Params(BaseModel):
        assign_to: str = Field(description=LISTED_DESCRIPTION)
        query: str = Field(description="free text to search for")

    params = Params.model_construct(**values)
    return signatures.word_screen_refusals(params, vocabulary, frozenset(), command)


def test_a_kind_listed_by_the_description_passes_and_the_same_word_elsewhere_is_refused(todo_copy):
    wv.write_workflow_vocabulary(todo_copy)
    vocabulary = _load(todo_copy)
    assert vocabulary.kind_options(ASSIGN_TO, "assign_to") == frozenset({"todoitem", "todolist"})

    # listed by the description of this command's field: a kind it takes
    assert _screened(ASSIGN_TO, vocabulary, assign_to="todolist", query="engineering") == {}
    # the free-text field of the same command, which lists nothing: refused
    refused = _screened(ASSIGN_TO, vocabulary, assign_to="John Smith", query="todolist")
    assert set(refused) == {"query"}
    assert refused["query"][1].startswith("query 'todolist' is a kind word of this workflow")


def test_the_same_word_in_a_field_with_no_listing_is_refused(todo_copy):
    """The kind-selector exemption is per command and field: another command's ``assign_to`` lists nothing."""
    wv.write_workflow_vocabulary(todo_copy)
    vocabulary = _load(todo_copy)

    refused = _screened("TodoItem/get_properties", vocabulary, assign_to="todolist", query="engineering")

    assert set(refused) == {"assign_to"}


def test_the_pronoun_refusal_still_applies_to_a_kind_selector(todo_copy):
    wv.write_workflow_vocabulary(todo_copy)
    vocabulary = _load(todo_copy)

    refused = _screened(ASSIGN_TO, vocabulary, assign_to="her", query="engineering")

    assert set(refused) == {"assign_to"}
    assert refused["assign_to"][1].startswith("assign_to 'her' is a pronoun")


# ---------------------------------------------------------------- labeled-example inference

def test_a_kind_value_in_the_labeled_examples_makes_the_field_a_kind_selector(todo_copy):
    _labeled(todo_copy, ASSIGN_TO, [
        {"command": "give the report to the todolist team", "assign_to": "todolist"},
        {"command": "ask Jane Smith", "assign_to": "Jane Smith"},
    ])
    # The description of the copy does not list kinds for this field: the labeled example is the only source.
    generic = Path(todo_copy) / "_commands" / "TodoItem" / "assign_to.py"
    generic.write_text(generic.read_text(encoding="utf-8").replace(LISTED_DESCRIPTION, GENERIC_DESCRIPTION),
                       encoding="utf-8")

    wv.write_workflow_vocabulary(todo_copy)
    vocabulary = _load(todo_copy)

    assert vocabulary.kind_options(ASSIGN_TO, "assign_to") == frozenset({"todolist"})
    assert _screened(ASSIGN_TO, vocabulary, assign_to="todolist", query="x") == {}
    # a kind word the labeled examples do not give for this field is still refused
    assert set(_screened(ASSIGN_TO, vocabulary, assign_to="todoitem", query="x")) == {"assign_to"}


def test_the_labeled_files_of_the_real_workflow_are_not_read_for_writing(todo_copy):
    """The labeled-example source reads the copy's ``___command_info`` only; the repo's is untouched."""
    real = TODO / "___command_info"
    before = sorted(real.glob("*")) if real.exists() else []
    _labeled(todo_copy, ASSIGN_TO, [{"command": "x", "assign_to": "todolist"}])
    wv.write_workflow_vocabulary(todo_copy)

    after = sorted(real.glob("*")) if real.exists() else []
    assert after == before


# ---------------------------------------------------------------- the stored vocabulary

def test_the_stored_vocabulary_is_format_two_with_kind_words_and_selectors(todo_copy):
    path = wv.write_workflow_vocabulary(todo_copy)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == wv.FORMAT_VERSION == 2
    assert payload["labeled_fingerprint"] == wv.labeled_fingerprint(todo_copy)
    assert set(payload["kind_words"]) == {"todoitem", "todolist", "todolistmanager"}
    assert payload["kind_selectors"][ASSIGN_TO]["assign_to"] == ["todoitem", "todolist"]
    assert _load(todo_copy).origin == "file"


def test_a_stored_vocabulary_read_back_equals_the_derived_one(todo_copy):
    wv.write_workflow_vocabulary(todo_copy)
    stored = _load(todo_copy)
    wv.vocabulary_path(todo_copy).unlink()
    wv._IN_MEMORY.pop(str(Path(todo_copy).resolve()), None)  # the process cache would answer otherwise
    derived = _load(todo_copy)

    assert stored.origin == "file" and derived.origin == "derived"
    assert stored.kind_selectors == derived.kind_selectors
    assert stored.kind_words == derived.kind_words
    assert stored.words == derived.words


def test_a_version_one_file_is_stale_and_the_loader_derives_in_memory(todo_copy):
    wv.write_workflow_vocabulary(todo_copy)
    payload = json.loads(wv.vocabulary_path(todo_copy).read_text(encoding="utf-8"))
    payload["format"] = 1
    del payload["kind_selectors"], payload["kind_words"], payload["labeled_fingerprint"]
    payload["words"] = ["sentinel_word"]
    wv.vocabulary_path(todo_copy).write_text(json.dumps(payload), encoding="utf-8")

    vocabulary = _load(todo_copy)

    assert vocabulary.origin == "derived"
    assert not vocabulary.has_word("sentinel_word")
    assert vocabulary.kind_options(ASSIGN_TO, "assign_to") == frozenset({"todoitem", "todolist"})


def test_a_changed_labeled_example_makes_the_stored_vocabulary_stale(todo_copy):
    _labeled(todo_copy, ASSIGN_TO, [{"command": "x", "assign_to": "todolist"}])
    wv.write_workflow_vocabulary(todo_copy)
    assert _load(todo_copy).origin == "file"

    _labeled(todo_copy, ASSIGN_TO, [{"command": "x", "assign_to": "todoitem"}])
    vocabulary = _load(todo_copy)

    assert vocabulary.origin == "derived"
    assert vocabulary.kind_options(ASSIGN_TO, "assign_to") == frozenset({"todoitem", "todolist"})


# ---------------------------------------------------------------- the typed skill screen

def _planning_command(folder: str, name: str) -> sc.PlanningCommandCommand:
    return workflow_agent.load_planning_command_map(folder)[name]


def _raw(command: sc.PlanningCommandCommand, **values) -> dict:
    return {p.name: values.get(p.name) for p in command.params}


def test_a_planning_command_argument_whose_description_lists_a_kind_takes_it(tutorial_copy):
    command = _planning_command(tutorial_copy, "file_request")

    args, _ = sc.check_arguments(command, _raw(command, resource_type="book", query="Dune"),
                                 "show the book Dune")

    assert args["resource_type"] == "book"


def test_the_same_kind_word_in_a_free_text_planning_command_argument_is_refused(tutorial_copy):
    command = _planning_command(tutorial_copy, "file_request")
    with pytest.raises(sc.PlanningCommandInputError) as refused:
        sc.check_arguments(command, _raw(command, resource_type="book", query="book"), "show the book Dune")

    assert "query='book' is a kind word of this workflow" in str(refused.value)


def test_the_none_marker_makes_a_required_planning_command_argument_missing(tutorial_copy):
    command = _planning_command(tutorial_copy, "file_request")
    with pytest.raises(sc.PlanningCommandInputError) as refused:
        sc.check_arguments(command, _raw(command, resource_type="book", query="None"), "show the book Dune")

    assert str(refused.value) == "missing required argument query"
    assert sc.refusal_question(command, refused.value) == (
        "I can't run file request yet: your request does not say which query to use. Which query should I use?")


def test_the_none_marker_gives_an_optional_planning_command_argument_its_default(tutorial_copy):
    command = _planning_command(tutorial_copy, "file_request")

    args, _ = sc.check_arguments(command, _raw(command, resource_type="book", query="Dune", note="None"),
                                 "show the book Dune")

    assert args["note"] == ""


# ---------------------------------------------------------------- the generic None marker

class _Request(BaseModel):
    title: str = Field(description="the title of the book")
    subtitle: Optional[str] = Field(default=None, description="a subtitle")


def test_the_none_marker_in_a_required_field_becomes_absent_and_the_screen_says_nothing_of_kinds():
    params = _Request.model_construct(title="None", subtitle="None")

    changed = signatures.default_unfilled_optional_strs(params, "NOT_FOUND")

    assert params.title is None          # absent: the missing check reports it
    assert params.subtitle is None       # an optional field takes its default (None here)
    assert set(changed) == {"title", "subtitle"}
    assert signatures.word_screen_refusals(params, None, frozenset(), "Any/command") == {}


class _Reasoned(BaseModel):
    title: str = Field(description="the title of the book")
    reason: str = Field(default="ordered by mistake", description="why")


@pytest.mark.parametrize("marker", ["NOT_FOUND", "None"])
def test_a_str_field_with_a_declared_default_takes_it_when_not_given(marker):
    """A plain ``str`` field with a default is optional: not given, it takes the default rather than
    being reported missing (which would ask the user); a required field is left to the missing check."""
    params = _Reasoned.model_construct(title="NOT_FOUND", reason=marker)

    changed = signatures.default_unfilled_optional_strs(params, "NOT_FOUND")

    assert params.reason == "ordered by mistake"
    assert params.title == "NOT_FOUND"
    assert changed == ["reason"]


def test_a_defaulted_str_field_not_given_is_not_missing_in_validate_parameters(workflow):
    extraction = InputForParamExtraction(command="cancel it", input_for_param_extraction_class=None)
    params = _Reasoned.model_construct(title="Dune", reason="NOT_FOUND")

    _, _, _, fields = extraction.validate_parameters(workflow, "show_book", params)

    assert "reason" not in fields
    assert params.reason == "ordered by mistake"


def test_the_none_marker_is_missing_in_validate_parameters(workflow):
    extraction = InputForParamExtraction(command="show the book", input_for_param_extraction_class=None)
    params = _Request.model_construct(title="None", subtitle="None")

    is_valid, _, _, fields = extraction.validate_parameters(workflow, "show_book", params)

    assert not is_valid
    assert "title" in fields
    assert params.subtitle is None
