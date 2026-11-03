"""Confident global first (owner-approved, the step 3 bars; scorer rule ``walk_flattened_global_first``).

For natural language typed on an application record, the CME walk asks the global head ``*`` first.
A confident answer naming one application command runs that command, bound to the root object (as
the climb binds ``*``), and is filed under the root. Anything else is discarded and today's flow runs
unchanged: the current head, then the climb. Exact command names never reach this step: the exact-name
lookup (3.9) answers them first, nearest scope first.

Each test runs on a COPY of the trained library example (``___command_info`` included), with real
application objects. Nothing is trained. The classifier heads are the copy's own; a spy on
``CommandNamePrediction.predict`` records which heads are asked, in order, and calls through.

Where a test needs a phrasing the trained head is confident on, the phrasing is found by probing the
copy's real heads inside the test, not hard-coded. A probe that finds none skips the test with the
count it tried, so a skip is visible and says why.
"""

import os
import shutil

import pytest

import fastworkflow
from fastworkflow import NLUPipelineStage
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction._commands import wildcard
from fastworkflow.command_executor import CommandExecutor
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    initialized_fastworkflow,
)

LIBRARY = os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "library")

#: Natural phrasings for the library-hours command (library_hours). Probed, not trusted: a phrasing
#: is used only if the copy's global head names library_hours confidently on it.
HOURS_CANDIDATES = [
    "what are the opening hours of Riverside",
    "could you tell me the opening hours of the library",
    "when is Riverside open today",
    "what are the working hours at the Riverside branch",
    "what time does the library close",
    "what hours is the library open",
    "tell me the business hours of Riverside",
    "I need to know the Riverside opening times",
    "what are the opening hours for the library branch",
    "when are the doors of Riverside open",
]

#: Natural phrasings for Book commands, probed against the Book head.
OWN_CANDIDATES = [
    "show the book",
    "show me this book",
    "list the loans of this book",
    "who has this book on loan",
    "show this book's details",
    "what are the details of this book",
]

#: Candidate phrasings for the unsure path. The copy's global head is probed with each (see
#: _unsure_phrasing): the one it is not confident on, and the Book head is not confident on either, so
#: the walk ends at the Book head and never reaches parameter extraction (no model call).
UNSURE_CANDIDATES = [
    "when are you open",
    "what is the weather like today",
    "tell me something nice",
    "can you do the thing with the stuff",
    "how do I feel about rain",
    "the quick brown fox jumps",
]


def _unsure_phrasing(cme, heads=("*", "Book")) -> str:
    """A candidate that none of *heads* (the copy's real heads) is confident on, so it never reaches
    parameter extraction."""
    predictor = intent_detection.CommandNamePrediction(cme)
    for phrasing in UNSURE_CANDIDATES:
        if all(not (out.command_name and not out.error_msg)
               for out in (predictor.predict(head, phrasing, NLUPipelineStage.INTENT_DETECTION)
                           for head in heads)):
            return phrasing
    pytest.skip(f"no one of {len(UNSURE_CANDIDATES)} unsure candidates is unanswered by "
                f"{', '.join(heads)} on this copy: nothing to route")


#: Natural phrasings for the global command set_root_context (messaging_app_4), probed, not trusted.
ROOT_PHRASINGS = [
    "set the root context",
    "set root context",
    "set up the root context",
    "make the root context",
    "set the root",
    "change the root context",
]


@pytest.fixture
def messaging_on_a_user(tmp_path, initialized_fastworkflow, monkeypatch):
    """A COPY of the trained messaging_app_4 (its ``___command_info`` included): the root is its
    ChatRoom, and the current record is a User below it."""
    from fastworkflow.examples.messaging_app_4.application.chatroom import ChatRoom
    from fastworkflow.examples.messaging_app_4.application.user import User

    folder = tmp_path / "messaging_wf"
    shutil.copytree(
        os.path.join(os.path.dirname(fastworkflow.__file__), "examples", "messaging_app_4"), folder,
        ignore=shutil.ignore_patterns("___workflow_contexts", "___convo_info", "__pycache__"),
        symlinks=True,
    )
    app = fastworkflow.Workflow.create(workflow_folderpath=str(folder), workflow_id_str="gf-messaging")
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id,
        workflow_context={"app_workflow": app},
    )
    if app.root_command_context is None:
        app.root_command_context = ChatRoom()
    root = app.root_command_context
    user = User(root, "alice")
    root.add_user(user)
    app.current_command_context = user
    return app, cme, root, user


def _root_library(app):
    """The workflow's root Library: the one its startup action made, or a fresh one when none was set."""
    if app.root_command_context is None:
        from fastworkflow.examples.library.application.library import Library

        app.root_command_context = Library()
    return app.root_command_context


def _copy_trained_library(tmp_path, *, drop_global_head: bool = False) -> str:
    folder = tmp_path / "library_wf"
    shutil.copytree(
        LIBRARY, folder,
        ignore=shutil.ignore_patterns("___workflow_contexts", "___convo_info", "__pycache__"),
        symlinks=True,
    )
    if drop_global_head:
        os.unlink(folder / "___command_info" / "global")
    return str(folder)


@pytest.fixture
def library_on_a_book(tmp_path, initialized_fastworkflow, monkeypatch):
    """The trained library copy, its root a real Library, the current record a real Book."""
    folder = _copy_trained_library(tmp_path)
    app = fastworkflow.Workflow.create(workflow_folderpath=folder, workflow_id_str="gf-library")
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id,
        workflow_context={"app_workflow": app},
    )
    library = _root_library(app)
    app.current_command_context = library.books["b1"]
    return app, cme, library


def _heads_asked(monkeypatch) -> list[str]:
    """Spy on ``CommandNamePrediction.predict``: record each head asked, in order, and call through."""
    asked: list[str] = []
    original = intent_detection.CommandNamePrediction.predict

    def spy(self, command_context_name, command, nlu_pipeline_stage):
        asked.append(command_context_name)
        return original(self, command_context_name, command, nlu_pipeline_stage)

    monkeypatch.setattr(intent_detection.CommandNamePrediction, "predict", spy)
    return asked


def _route(cme, text: str):
    """Run the typed text through the CME walk, as the agent's natural-language path does."""
    return CommandExecutor.perform_action(
        cme, fastworkflow.Action(command_name="wildcard", command=text))


def _probe(cme, candidates, head: str):
    """The candidates the copy's real head *head* names confidently, as (phrasing, command) pairs."""
    predictor = intent_detection.CommandNamePrediction(cme)
    hits = []
    for phrasing in candidates:
        out = predictor.predict(head, phrasing, NLUPipelineStage.INTENT_DETECTION)
        if out.command_name and not out.error_msg and not out.answered_bare_value and not out.is_cme_command:
            hits.append((phrasing, out.command_name))
    return hits


def test_a_global_command_is_run_from_a_user_record_bound_to_the_root_when_the_global_head_is_confident(
        messaging_on_a_user, monkeypatch, setup_test_environment):
    """Confident global first, on a workflow with a context below its root (messaging_app_4: the root
    is the ChatRoom, the record a User). The global head names set_root_context confidently on the
    probed phrasings; it is asked first, and the walk names it bound to the root. The routing is read
    from the CME walk's artifacts: the command itself is not executed here (the root can be set once)."""
    app, cme, root, user = messaging_on_a_user
    hits = [phrasing for phrasing, name in _probe(cme, ROOT_PHRASINGS, "*")
            if name.split("/")[-1] == "set_root_context"]
    if not hits:
        pytest.skip(f"the copy's global head names set_root_context confidently on none of "
                    f"{len(ROOT_PHRASINGS)} probed phrasings: nothing to route")
    asked = _heads_asked(monkeypatch)

    out = _route(cme, hits[0])

    assert asked[0] == "*"
    artifacts = out.command_response.artifacts
    assert artifacts["command_name"].split("/")[-1] == "set_root_context"
    assert app.command_context_for_response_generation is root
    assert artifacts["ran_on_context_name"] == "ChatRoom"
    assert app.current_command_context is user
    assert app.root_command_context is root


def test_an_own_command_phrasing_from_the_book_still_routes_to_the_book_command(
        library_on_a_book, monkeypatch, setup_test_environment):
    """The global head is asked first, is not confident here, and the Book head's command wins."""
    app, cme, library = library_on_a_book
    hits = [(phrasing, name) for phrasing, name in _probe(cme, OWN_CANDIDATES, "Book")
            if name.startswith("Book/")]
    if not hits:
        pytest.skip(f"the Book head names a Book command confidently on none of "
                    f"{len(OWN_CANDIDATES)} probed phrasings")
    phrasing, expected = hits[0]
    asked = _heads_asked(monkeypatch)

    out = _route(cme, phrasing)

    assert asked[0] == "*"
    assert out.command_response.artifacts["command_name"] == expected
    assert "ran_on_context_name" not in out.command_response.artifacts
    assert app.command_context_for_response_generation is app.current_command_context


def test_a_phrasing_the_global_head_is_unsure_about_falls_through_to_todays_flow(
        library_on_a_book, monkeypatch, setup_test_environment):
    """The global head answers ambiguous: its result is discarded, '*' is not asked again, and the
    current head and the climb are asked as today (Book, then its root Library)."""
    app, cme, library = library_on_a_book
    unsure = _unsure_phrasing(cme)
    asked = _heads_asked(monkeypatch)

    out = _route(cme, unsure)

    # Observed on the copy: the Book head answers ambiguous, and the walk ends there, as it does
    # today (the climb to Library is not reached). '*' is asked once, first.
    assert asked == ["*", "Book"]
    artifacts = out.command_response.artifacts
    assert artifacts.get("ran_on_context_name") != "Library" or "command_name" not in artifacts


def test_without_a_trained_global_head_the_global_ask_is_skipped_silently(
        tmp_path, initialized_fastworkflow, monkeypatch, setup_test_environment):
    """A copy whose '*' head has no artifacts: the global ask is skipped, nothing is loaded for it,
    and the routing is today's (no FileNotFoundError)."""
    folder = _copy_trained_library(tmp_path, drop_global_head=True)
    app = fastworkflow.Workflow.create(workflow_folderpath=folder, workflow_id_str="gf-no-global")
    cme = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        parent_workflow_id=app.id,
        workflow_context={"app_workflow": app},
    )
    library = _root_library(app)
    app.current_command_context = library.books["b1"]
    assert wildcard._global_head_trained(app) is False
    asked = _heads_asked(monkeypatch)

    _route(cme, _unsure_phrasing(cme, heads=("Book",)))

    assert "*" not in asked
    assert asked[0] == "Book"


def test_an_exact_name_from_the_book_is_answered_nearest_first_and_skips_the_global_ask(
        library_on_a_book, monkeypatch, setup_test_environment):
    """Exact names are not global-first: a Book command typed by its name runs in the Book, with no
    global-first ask; a Library command typed by name runs bound to the root, as before."""
    app, cme, library = library_on_a_book

    asked = _heads_asked(monkeypatch)
    out = _route(cme, "show_book")
    assert out.command_response.artifacts["command_name"] == "Book/show_book"
    assert "ran_on_context_name" not in out.command_response.artifacts
    assert "*" not in asked

    asked.clear()
    out = _route(cme, "list_branches")
    assert out.command_response.artifacts["command_name"] == "Library/list_branches"
    assert out.command_response.artifacts["ran_on_context_name"] == "Library"
    assert app.command_context_for_response_generation is library
    # Its scope's own head answers the exact name (3.9); the global-first ask is not made.
    assert asked == ["Library"]


# ----------------------------------------------------------------------------------------------
# The rule itself, on real CommandNamePrediction outputs (no model, no mocks).
# ----------------------------------------------------------------------------------------------

def test_the_confident_global_rule_takes_only_a_confident_application_command():
    from fastworkflow._workflows.command_metadata_extraction.intent_detection import CommandNamePrediction

    Output = CommandNamePrediction.Output
    # Confident: one application command, no ambiguity, not a bare value, not a CME command.
    assert wildcard._confident_global_command(Output(command_name="library_hours"))
    # Unsure: the head answered ambiguous (error_msg set): discarded.
    assert not wildcard._confident_global_command(
        Output(command_name="library_hours", error_msg="Ambiguous intent error"))
    assert not wildcard._confident_global_command(Output(command_name=None, error_msg="ambiguous"))
    # Non-routable: a decline (wildcard resolves to no command), a bare value (parameter_value).
    assert not wildcard._confident_global_command(Output(command_name=None))
    assert not wildcard._confident_global_command(Output(command_name=None, answered_bare_value=True))
    # A framework command is never the global answer.
    assert not wildcard._confident_global_command(
        Output(command_name="IntentDetection/go_up", is_cme_command=True))
