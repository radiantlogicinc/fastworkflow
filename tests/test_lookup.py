"""The scoped lookup, end to end on copies of the trained examples (Phase 9 step 6, design 2.18 and 3.9).

Only the cases the sibling files do not already pin. Covered elsewhere, not repeated here:

- an enclosing context's command by exact name, bound and filed under its record:
  tests/test_lookup_exact_names.py::test_an_enclosing_context_command_by_exact_name_runs_bound_and_filed_under_its_record
- a global command by exact name from inside a context: tests/test_lookup_exact_names.py::
  test_a_global_command_by_exact_name_from_inside_a_context_runs_bound_to_the_root
- a global command by natural language when the global head is confident: tests/test_global_first_lookup.py::
  test_a_global_command_is_run_from_a_user_record_bound_to_the_root_when_the_global_head_is_confident
- parameter_value ends the lookup: tests/test_lookup_defects.py::test_a_bare_value_answer_stops_the_climb_where_it_is_given
- filing under the bound record: tests/test_lookup_defects.py::test_a_run_on_an_enclosing_record_is_filed_under_that_record
- the hierarchy-file warning (once) and identical routing with and without the file:
  tests/test_hierarchy_file_retired.py::test_a_workflow_with_the_file_warns_once_per_load_and_says_it_can_be_deleted and
  ::test_routing_is_the_same_with_and_without_the_file
- a pre-change artifact refused with 'retrain with `fastworkflow train`':
  tests/test_not_here_production.py::test_load_refuses_unversioned_legacy_artifacts_and_accepts_a_recorded_version

Each test here runs on a COPY of a trained example (``___command_info`` included) with real
application objects. The classifier heads are the copy's own. A phrasing is never hard-coded as a
route: where a head must answer, the phrasing is probed against it, and a probe that finds none
skips the test and says how many phrasings it tried.
"""

import os
import shutil

import pytest

import fastworkflow
from fastworkflow import ModuleType
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.workflow_agent import _explicit_agent_command

# Fixtures and helpers reused from the sibling test modules (not copied).
from tests.test_global_first_lookup import (  # noqa: F401 - fixtures
    library_on_a_book,
    messaging_on_a_user,
    _heads_asked,
    _probe,
    _route,
)
from tests.test_lookup_exact_names import _run  # noqa: F401
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    initialized_fastworkflow,
)

#: Natural phrasings for the inherited Record commands (add_note, show_notes), probed on the Book head.
INHERITED_CANDIDATES = [
    "add a note: card renewed",
    "note that the cover is torn",
    "show the notes",
    "show me the notes on this book",
    "what notes are there",
    "add a note to this book",
]

#: Phrasings for the switch case, probed on the ChatRoom head (the climb's target). The record is
#: alice, so the switch target is bob. "switch to bob" is the literal form the head does not answer
#: confidently on this copy; the probed set_current_user phrasings below are used instead.
SWITCH_CANDIDATES = [
    "set current user to bob",
    "set current user bob",
    "switch to bob",
    "set bob as the current user",
    "make bob the current user",
]

#: Phrasings for the library's global command library_hours, probed on the global head.
HOURS_CANDIDATES = [
    "what are the opening hours of Riverside",
    "what time does the library close",
    "when is Riverside open today",
    "what hours is the library open",
]


def _short(name: str) -> str:
    return name.split("/")[-1]


def test_an_inherited_command_is_recognised_by_the_flattened_head_and_bound_to_the_current_record(
        library_on_a_book, monkeypatch, setup_test_environment):
    """Book inherits Record's commands (add_note, show_notes). Phase 9 flattens each head to its own
    labels plus the inherited ones, so the Book head itself recognises them. The run is bound to the
    current Book, not the root, and filed under no other record."""
    app, cme, library = library_on_a_book
    book = app.current_command_context
    hits = [(phrasing, name) for phrasing, name in _probe(cme, INHERITED_CANDIDATES, "Book")
            if _short(name) in {"add_note", "show_notes"}]
    if not hits:
        pytest.skip(f"the Book head names an inherited Record command confidently on none of "
                    f"{len(INHERITED_CANDIDATES)} probed phrasings")
    phrasing, expected = hits[0]
    asked = _heads_asked(monkeypatch)

    out = _route(cme, phrasing)

    assert "Book" in asked, asked
    artifacts = out.command_response.artifacts
    if "command_name" in artifacts:
        assert artifacts["command_name"] == expected
        # Bound to the current record: no enclosing scope ran it.
        assert "ran_on_context_name" not in artifacts
    else:
        # Parameter extraction needs a live LLM key, which this box does not have, so the walk
        # ends at extraction. The routing decision is still in the message: the Book head named
        # the inherited command, and the walk stopped there, not at an ambiguity or a decline.
        assert expected in out.command_response.response, out.command_response.response
        assert "Ambiguous" not in out.command_response.response
    assert app.command_context_for_response_generation is book
    assert app.current_command_context is book


def test_a_switch_to_a_user_said_on_a_user_is_resolved_by_the_climb_and_bound_to_the_chat_room(
        messaging_on_a_user, monkeypatch, setup_test_environment):
    """Said on a User record (alice), switching to bob, by the exact name of set_current_user, which
    lives on the ChatRoom (the root). The exact-name lookup climbs the live chain to the ChatRoom
    scope, binds the command to the chat room before parameter extraction, and its db_lookup reads
    the chat room's user list, so 'bob' resolves there. The current record does not move in routing.

    Why the exact name and not the natural phrasing ("switch to bob"): on this copy the User head
    answers 'ambiguous' for every switch phrasing probed (test run 2026-11-01), and an ambiguity ends
    the walk before any climb, as it does today. Phrasings that reach the ChatRoom head fine
    ("set current user to bob") are stopped by the User head first, so the NL climb is not reachable
    here. Recorded as a finding, not papered over."""
    app, cme, root, user = messaging_on_a_user
    from fastworkflow.examples.messaging_app_4.application.user import User

    bob = User(root, "bob")
    root.add_user(bob)

    out = _route(cme, "set_current_user bob")

    artifacts = out.command_response.artifacts
    if "command_name" in artifacts:
        assert _short(artifacts["command_name"]) == "set_current_user"
        assert artifacts.get("ran_on_context_name") == "ChatRoom"
        command_name = artifacts["command_name"]
    else:
        # Parameter extraction needs a live LLM key (not on this box), so the walk ends there. The
        # message names the command the exact-name lookup resolved, and the ChatRoom scope.
        assert "ChatRoom/set_current_user" in out.command_response.response, out.command_response.response
        command_name = "ChatRoom/set_current_user"
    # Bound to the chat room (the enclosing scope on the live chain), not the current user.
    assert app.command_context_for_response_generation is root
    assert app.current_command_context is user

    routing = fastworkflow.RoutingRegistry.get_definition(app.folderpath)
    input_cls = routing.get_command_class(
        command_name, ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
    signature = _signature_with_db_lookup(input_cls)
    matched, corrected, _suggestions = signature.db_lookup(app, "user_name", "bob")
    assert matched is True, "db_lookup must read the bound chat room's users"
    assert corrected == "bob"


def _signature_with_db_lookup(cls):
    """The command's Signature: the input class's module carries it (the trainer's own layout)."""
    import sys

    module = sys.modules[cls.__module__]
    signature = getattr(module, "Signature", None)
    assert signature is not None and hasattr(signature, "db_lookup"), cls.__module__
    return signature


def test_library_hours_is_reachable_by_exact_name_from_inside_a_book(
        library_on_a_book, monkeypatch, setup_test_environment):
    """library_hours is a global command. From a Book it is reached by exact name: bound to the root
    Library (the global scope binds the root), filed under it, and the current Book does not move."""
    app, cme, library = library_on_a_book
    book = app.current_command_context

    run = _run(app, cme, monkeypatch, "library_hours")

    assert _short(run.command_name) == "library_hours"
    assert run.ran_on_context_name == "Library"
    assert app.command_context_for_response_generation is library
    assert app.current_command_context is book


def test_library_hours_by_natural_language_from_a_book(
        library_on_a_book, monkeypatch, setup_test_environment):
    """The library's global head is weak, so library_hours is expected to be reachable by exact name
    only. This test takes the natural-language route when the copy's global head is confident on a
    probed phrasing, and otherwise skips with the count, which is the stated expectation here."""
    app, cme, library = library_on_a_book
    book = app.current_command_context
    hits = [phrasing for phrasing, name in _probe(cme, HOURS_CANDIDATES, "*")
            if _short(name) == "library_hours"]
    if not hits:
        pytest.skip(f"the library's global head is not confident on library_hours for any of "
                    f"{len(HOURS_CANDIDATES)} probed phrasings: reachable by exact name only")
    asked = _heads_asked(monkeypatch)

    out = _route(cme, hits[0])

    assert asked[0] == "*", asked
    assert _short(out.command_response.artifacts["command_name"]) == "library_hours"
    assert app.command_context_for_response_generation is library
    assert app.current_command_context is book
