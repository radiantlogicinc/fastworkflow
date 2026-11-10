"""Assistant mode and planning commands that need the agent (fix-3rsq.10.8, design 3.4).

Assistant mode has no intent detection: an EXACTLY named planning command runs there when it is
mechanical, and is refused with the 3.4 message, extended by the reasons, when it needs the agent.
The classifier (``fastworkflow.planning_needs_agent``) is tested here on the library's real planning
commands and on copies with a mechanical and a needs-agent command added.

The end-to-end dispatch tests run the real executor: a mechanical planning command runs by its exact
name, and a needs-agent one is refused with the reasons (``command_executor._assistant_planning_refusal``).
"""
from __future__ import annotations

import os
import uuid

import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.planning_needs_agent import planning_command_needs_agent, refusal_message
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_planning_command_code_steps import Model
from tests.test_step_helpers import (
    LibraryRunner,
    _library_workflow,
    script_command_extraction,
    script_every_agent,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


LIBRARY = os.path.join(ROOT, "fastworkflow", "examples", "library")

MECHANICAL_LIBRARY = [
    "Book/loan_report",
    "Book/shelf_report",
    "Library/author_books",
    "Library/book_by_title",
    "Library/book_loans",
    "Library/count_books",
    "Library/find_record",
]
NEEDS_AGENT_LIBRARY = {
    "Library/member_report": ("phase 'loans' is decided at run time",
                              "phase 'notes' is decided at run time"),
    "Library/note_member": ("phase 'note' has a prose step the agent carries out",
                            "it has no __result__"),
    "Library/overdue_summary": ("phase 'overdue' has a prose step the agent carries out",
                                "it has no __result__"),
}

# A mechanical planning command: code steps only, a __result__ that writes the answer, and no
# condition on an earlier phase's facts. Added to a copy of the library.
MECHANICAL_COUNT_MEMBERS = '''"""count_members: how many members match the words the request gives."""
from pydantic import BaseModel, Field

from fastworkflow.planning import run

from .find_member import Signature as FindMember


class Signature:
    """Count the members whose name holds some words."""

    plain_utterances = ["how many members are called Lee"]

    class Input(BaseModel):
        query: str = Field(description="words of the member's name",
                           json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input) -> FindMember.Output:
        return [run("find_member", query=input.query)]

    def __result__(self, input, find):
        return f"{input.query} matches {find.total} {'member' if find.total == 1 else 'members'}."
'''

# A needs-agent planning command: a prose step the agent carries out, with a __result__, so the one
# reason is the prose step.
NEEDS_AGENT_TELL_MEMBERS = '''"""tell_members: names the members the request names, in the agent's words."""
from pydantic import BaseModel, Field

from fastworkflow.planning import run

from .find_member import Signature as FindMember


class Signature:
    """Tell the user which members the request names."""

    plain_utterances = ["who is called Lee"]

    class Input(BaseModel):
        query: str = Field(description="words of the member's name",
                           json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def find(self, input) -> FindMember.Output:
        return [run("find_member", query=input.query),
                "say which members the request names"]

    def __result__(self, input, find):
        return f"{input.query} matches {find.total} members."
'''

# A needs-agent planning command: a verify parameter, decided by a classifier from the request.
VERIFY_COUNT_MEMBERS = '''"""count_members_checked: the count of members, with a check on the request's wording."""
from pydantic import BaseModel, Field

from fastworkflow.planning import run

from .find_member import Signature as FindMember


class Signature:
    """Count the members whose name holds some words, when the request asks."""

    plain_utterances = ["how many members are called Lee, please"]

    class Input(BaseModel):
        query: str = Field(description="words of the member's name",
                           json_schema_extra={"copied_from_command_text": True})
        recent: bool = Field(default=False, description="the request asks for recent members",
                             json_schema_extra={"classifier_confirms": True})


class PlanGenerator:
    def find(self, input) -> FindMember.Output:
        return [run("find_member", query=input.query)]

    def __result__(self, input, find):
        return f"{input.query} matches {find.total} {'member' if find.total == 1 else 'members'}."
'''


def _copy_with(tmp_path, name: str, text: str) -> str:
    """A copy of the library with the planning command *name* (``Library/x``) added."""
    return _library_workflow(tmp_path, extra_planning_commands={name: text})


# ---------------------------------------------------------------- the classifier on the library

@pytest.mark.parametrize("name", MECHANICAL_LIBRARY)
def test_library_mechanical_planning_commands_are_mechanical(name):
    needs, reasons = planning_command_needs_agent(LIBRARY, name)
    assert needs is False, reasons
    assert reasons == []


@pytest.mark.parametrize("name", sorted(NEEDS_AGENT_LIBRARY))
def test_library_needs_agent_planning_commands_give_their_reasons(name):
    needs, reasons = planning_command_needs_agent(LIBRARY, name)
    assert needs is True
    for expected in NEEDS_AGENT_LIBRARY[name]:
        assert any(expected in reason for reason in reasons), (expected, reasons)


def test_library_member_report_reason_names_the_fact_that_decides_it():
    _, reasons = planning_command_needs_agent(LIBRARY, "Library/member_report")
    assert any("when the earlier facts member are absent" in r for r in reasons), reasons


def test_unknown_name_is_refused():
    with pytest.raises(ValueError, match="is not a planning command"):
        planning_command_needs_agent(LIBRARY, "Library/no_such_command")


# ---------------------------------------------------------------- copies with an added command

def test_copy_code_only_planning_command_is_mechanical(tmp_path):
    folder = _copy_with(tmp_path, "Library/count_members", MECHANICAL_COUNT_MEMBERS)
    needs, reasons = planning_command_needs_agent(folder, "Library/count_members")
    assert (needs, reasons) == (False, [])


def test_copy_prose_step_planning_command_needs_agent_with_reason(tmp_path):
    folder = _copy_with(tmp_path, "Library/tell_members", NEEDS_AGENT_TELL_MEMBERS)
    needs, reasons = planning_command_needs_agent(folder, "Library/tell_members")
    assert needs is True
    assert reasons == [
        "phase 'find' has a prose step the agent carries out: 'say which members the request "
        "names' (a code step (run, run with named=, or run with all_pages=True) would make it mechanical)",
    ]


def test_copy_verify_parameter_is_a_reason(tmp_path):
    folder = _copy_with(tmp_path, "Library/count_members_checked", VERIFY_COUNT_MEMBERS)
    needs, reasons = planning_command_needs_agent(folder, "Library/count_members_checked")
    assert needs is True
    assert len(reasons) == 1
    assert reasons[0].startswith("field 'recent' carries verify: a classifier model decides it")


# ---------------------------------------------------------------- the refusal text

def test_refusal_keeps_the_3_4_sentence_and_adds_the_reasons():
    message = refusal_message("Library/note_member", [
        "phase 'note' has a prose step the agent carries out: 'say x'",
        "it has no __result__, so the agent writes the answer",
    ])
    assert message.startswith(
        "Library/note_member is a planning command; it runs in agent mode. It needs agent mode "
        "because phase 'note' has a prose step the agent carries out: 'say x'; it has no __result__")
    assert message.endswith(".")


# ---------------------------------------------------------------- assistant-mode dispatch

@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()


def _assistant_ctx(tmp_path, monkeypatch, name: str, text: str):
    """(ctx, folder): an assistant-mode context on a library copy holding the planning command *name*."""
    from fastworkflow.command_routing import RoutingRegistry

    folder = _copy_with(tmp_path, name, text)
    RoutingRegistry.clear_registry()
    RoutingRegistry.get_definition(folder, load_cached=False)
    wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"am_{uuid.uuid4().hex[:8]}")
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(wf)
    ctx.push_active_workflow(wf)
    LibraryRunner(wf)(ctx, "startup")
    ctx._cme_workflow.context["is_assistant_mode_command"] = True
    script_command_extraction(monkeypatch, wf)
    return ctx, folder


def test_assistant_mode_runs_a_mechanical_planning_command_by_exact_name(tmp_path, monkeypatch):
    ctx, _ = _assistant_ctx(tmp_path, monkeypatch, "Library/count_members",
                            MECHANICAL_COUNT_MEMBERS)
    # Its phases are code steps: the scripted model only closes each phase (no step is asked for).
    script_every_agent(monkeypatch, Model(None))
    out = CommandExecutor.invoke_command(ctx, "count_members query=Lee")
    assert out.success is True
    assert out.command_response.response == "Lee matches 1 member."  # only Ann Lee is seeded


def test_assistant_mode_refuses_a_needs_agent_planning_command_with_reasons(tmp_path, monkeypatch):
    ctx, folder = _assistant_ctx(tmp_path, monkeypatch, "Library/tell_members",
                                 NEEDS_AGENT_TELL_MEMBERS)
    needs, reasons = planning_command_needs_agent(folder, "Library/tell_members")
    assert needs is True
    out = CommandExecutor.invoke_command(ctx, "tell_members query=Lee")
    assert out.success is False
    assert out.command_response.response == refusal_message("Library/tell_members", reasons)
