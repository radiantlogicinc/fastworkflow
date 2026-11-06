"""Planning commands called from a step run as nested frames (design 3.5, Phase 5 step 1).

Each case runs the library copy through the executor's real dispatch: the extraction
hop is scripted (``script_command_extraction``), and every command, planning or not,
goes through ``CommandExecutor._invoke_command_impl``. The nested frames' agents answer
from ``state.nested``; a step that code runs itself is never asked of them.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import types

import pytest
from pydantic import BaseModel, Field

import fastworkflow
import fastworkflow.workflow_agent as workflow_agent
from fastworkflow import planning_commands, planning_phases
from fastworkflow import plan_frame as plan_frame_module
from fastworkflow import tracing
from fastworkflow.command_executor import CommandExecutor, CommandNotFoundError
from fastworkflow.context_identity import context_class_for
from fastworkflow.planning import run
from fastworkflow.planning import phase
from fastworkflow.observation_offloading.facts import MissingFactError, workflow_signatures
from fastworkflow.observation_offloading.labels import printed_alias
from fastworkflow.observation_offloading.state import (
    archive_for_path,
    observability_db_path,
    scope_for_host,
)
from fastworkflow.plan_frame import TURN_MAX_STEPS_ENV, PlanFrame, runs_outside_ranges, turn_max_steps
from fastworkflow.typed_phase_facts import fill_typed_phase_output
from fastworkflow.examples.library._commands.Library.book_by_title import Signature as BookByTitle
from fastworkflow.examples.library._commands.Library.find_book import Signature as FindBook
from fastworkflow.planning_commands import PlanningCommandInputError
from fastworkflow.state_serialization import validate_state
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_execution_context import (
    AWAITING_ANSWER_REFUSAL_MESSAGE,
    NESTED_BUDGET_SPENT_MESSAGE,
)
from tests.test_turn_result_capture import no_review_gate  # noqa: F401 - the review gate is off for these turns
from tests.test_plan_frames import no_turn_budget_env, warnings_logged  # noqa: F401 - fixtures
from tests.test_step_helpers import (
    LIBRARY,
    FoundBooks,
    _library_workflow,
    _make_library_runner_env,
    _real_planning_command,
)
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    initialized_fastworkflow,
)

BOOK_LOANS = "Library/book_loans"
BOOK_BY_TITLE = "Library/book_by_title"
CHILD_ALIAS = "/book_by_title#1"
NESTED_COMMANDS = (
    "_commands/Library/book_by_title.py",
    "_commands/Library/book_loans.py",
    "_commands/Library/_wording.py",
)
BOOK_LOANS_ANSWER = "Dune is on loan to 2 borrowers (Ann Lee, Raj Patel)."
BOOK_BY_TITLE_RESPONSE = "Dune is by Frank Herbert; the library owns 3 copies."


@pytest.fixture
def nested_env(tmp_path, initialized_fastworkflow, monkeypatch):
    monkeypatch.delenv(TURN_MAX_STEPS_ENV, raising=False)
    state = _make_library_runner_env(
        tmp_path,
        initialized_fastworkflow,
        monkeypatch,
        extra_commands=NESTED_COMMANDS,
        real_dispatch=True,
    )
    yield state
    state.ctx.pop_active_workflow()


@pytest.fixture
def nested_book_env(tmp_path, initialized_fastworkflow, monkeypatch):
    """nested_env plus Book/loan_report, so a Library step can name a Book planning command."""
    monkeypatch.delenv(TURN_MAX_STEPS_ENV, raising=False)
    state = _make_library_runner_env(
        tmp_path,
        initialized_fastworkflow,
        monkeypatch,
        extra_commands=NESTED_COMMANDS + (LOAN_REPORT_FILE,),
        real_dispatch=True,
    )
    yield state
    state.ctx.pop_active_workflow()


def _run_book_loans(state, title: str = "Dune"):
    """A turn whose planned call is ``book_loans`` for *title*. Code runs every step of its
    phases, including the ``book`` phase's ``book_by_title`` step, which is a nested planning
    command, so no model step is scripted for the turn."""
    state.planning_command = _real_planning_command(BOOK_LOANS, "_commands/Library/book_loans.py").command
    state.args = {"title": title}
    return state.ctx.process_turn(f"who has {title} on loan")


def _phase_spans(state, planning_command: str):
    return [s for s in state.sink.named(tracing.SPAN_PLANNING_COMMAND_PHASE)
            if s.attributes.get("skill") == planning_command]


def test_a_agent_step_runs_book_loans_and_its_nested_book_by_title(nested_env):
    result = _run_book_loans(nested_env)

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == BOOK_LOANS_ANSWER
    # The nested run is held under the step that called it, with the child's own output.
    child_output = next(o for o in nested_env.ctx._turn_outputs if o.command_name == BOOK_BY_TITLE)
    assert child_output.success is True
    assert child_output.command_response.response == BOOK_BY_TITLE_RESPONSE
    assert child_output.command_response.output.found is True
    assert child_output.command_response.output.book_uid == "b1"
    # The child's two phases ran in its own frame; the parent's phases ran in the turn frame.
    child_phases = [s.attributes["phase"] for s in nested_env.sink.named(tracing.SPAN_PLANNING_COMMAND_PHASE)
                    if s.attributes.get(tracing.ATTR_FRAME_ALIAS) == CHILD_ALIAS]
    assert child_phases == ["find", "book"]
    book_span, loans_span = _phase_spans(nested_env, BOOK_LOANS)
    assert [book_span.attributes["phase"], loans_span.attributes["phase"]] == ["book", "loans"]
    # Code ran the book phase's book_by_title step and the loans phase's list_loans step:
    # no phase agent (turn or nested) was asked for a step.
    assert book_span.attributes["steps"] == "1. book_by_title title=Dune"
    assert book_span.attributes["steps_by_code"] == 1
    assert book_span.attributes["code_stop"] == "done"
    assert loans_span.attributes["steps_by_code"] == 1
    assert loans_span.attributes["code_stop"] == "done"
    assert nested_env.model.asked == []
    assert nested_env.nested.asked == []
    # The book phase received book_by_title's Output, and loans read the book it opened.
    assert book_span.attributes["facts"] == {
        "found": True, "book_uid": "b1", "title": "Dune", "author": "Frank Herbert", "copies": 3,
    }
    assert loans_span.attributes["facts"] == {
        "borrowers": ["Ann Lee", "Raj Patel"], "due_dates": ["2026-11-02", "2026-10-20"], "total": 2,
    }
    # The parent's run of the nested call sits between the child's own runs and the
    # parent's later list_loans run.
    assert [o.command_name for o in nested_env.ctx._turn_outputs] == [
        "Library/find_book", "Library/open_book", BOOK_BY_TITLE, "Book/list_loans", BOOK_LOANS,
    ]
    assert len(nested_env.ctx._frame_stack()) == 1


def test_b_a_planning_command_already_on_the_stack_is_refused_without_a_frame(nested_env):
    ctx = nested_env.ctx
    running = PlanFrame(kind="planning", command_name=BOOK_BY_TITLE, depth=1, alias=CHILD_ALIAS)
    ctx._frame_stack().append(running)
    try:
        out = ctx.run_planning_command(BOOK_BY_TITLE, {"title": "Dune"},
                                       "book_by_title title=Dune")
        frames_after = list(ctx._frame_stack())
    finally:
        ctx._frame_stack().remove(running)

    assert out.success is False
    assert out.command_response.response == (
        f"{BOOK_BY_TITLE} is already running; call its steps directly.")
    assert frames_after == [ctx._turn_frame(), running]


def test_c_a_spent_turn_budget_refuses_a_further_nested_call(nested_env, no_turn_budget_env,
                                                             monkeypatch):
    ctx = nested_env.ctx
    monkeypatch.setitem(fastworkflow._env_vars, TURN_MAX_STEPS_ENV, "3")
    assert turn_max_steps() == 3
    ctx._turn_steps_used = 3

    out = ctx.run_planning_command(BOOK_BY_TITLE, {"title": "Dune"},
                                   "book_by_title title=Dune")

    assert out.success is False
    assert out.command_response.response == NESTED_BUDGET_SPENT_MESSAGE
    assert len(ctx._frame_stack()) == 1
    assert ctx._turn_steps_used == 3


def test_d_nested_steps_are_numbered_once_and_their_ranges_recorded(nested_env, monkeypatch):
    ctx = nested_env.ctx
    recorded_ranges: list[list[list[int]]] = []
    original = type(ctx).run_planning_command

    def recording(self, qualified_name, params, step_text):
        output = original(self, qualified_name, params, step_text)
        recorded_ranges.append([list(r) for r in self._turn_frame().child_step_ranges])
        return output

    monkeypatch.setattr(type(ctx), "run_planning_command", recording)
    result = _run_book_loans(nested_env)
    assert result.status == TurnStatus.COMPLETED

    # One nested call, recorded once on the turn frame.
    assert len(recorded_ranges) == 1
    [[first, last]] = recorded_ranges[0]
    assert first == 1 and last >= first
    # The parent's own later step starts after the child's range.
    loans = next(s for s in _phase_spans(nested_env, BOOK_LOANS)
                 if s.attributes["phase"] == "loans")
    assert loans.attributes["first_step"] > last
    # Every archived run has its own step index: none of the frames reused another's.
    runs = archive_for_path(observability_db_path(ctx)).command_runs(scope_for_host(ctx))
    step_of = [(r["step_index"], r["command"]) for r in runs]
    assert len({step for step, _ in step_of}) == len(step_of)
    assert [step for step, _ in step_of if first <= step <= last] == [1, 2]
    assert (loans.attributes["first_step"], "Book/list_loans") in step_of


def test_e_live_frames_are_one_after_completion(nested_env):
    result = _run_book_loans(nested_env)

    assert result.status == TurnStatus.COMPLETED
    assert len(nested_env.ctx._frame_stack()) == 1


def test_f_nested_spans_name_their_frame_and_the_root_names_none(nested_env):
    result = _run_book_loans(nested_env)
    assert result.status == TurnStatus.COMPLETED

    # The root is emitted at open and again at close under one span id (tracing_phase1);
    # the closed emission carries the turn's attributes.
    roots = nested_env.sink.named(tracing.SPAN_TURN)
    assert len(roots) == 2
    closed_root = roots[-1]
    assert closed_root.end_ns is not None
    assert {r.attributes[tracing.ATTR_FRAME_ALIAS] for r in roots} == {""}
    child = [s for s in nested_env.sink.named(tracing.SPAN_PLANNING_COMMAND_PHASE)
             if s.attributes.get(tracing.ATTR_FRAME_ALIAS) == CHILD_ALIAS]
    assert len(child) == 2
    assert all(s.attributes["skill"] == BOOK_BY_TITLE for s in child)


def test_g_a_bad_turn_budget_warns_once_however_often_it_is_read(
        monkeypatch, no_turn_budget_env, warnings_logged):
    monkeypatch.setitem(fastworkflow._env_vars, TURN_MAX_STEPS_ENV, "abc")

    for _ in range(3):
        assert turn_max_steps() == plan_frame_module.DEFAULT_TURN_MAX_STEPS

    warned = [r for r in warnings_logged if TURN_MAX_STEPS_ENV in r.getMessage()]
    assert len(warned) == 1
    assert warned[0].levelno == logging.WARNING


#: A test-copy planning command whose one step names a command but is not one code runs
#: (the command is not first in the step), so the phase agent takes a model step for it, and
#: each model step counts against the turn's budget. (A step naming no command is answer-only.)
TITLE_LOOKUP = '''from pydantic import BaseModel, Field


class Signature:
    """Look a title up, one step at a time."""

    class Input(BaseModel):
        title: str = Field(description="the book's title",
                           json_schema_extra={"in_request": True})


class PlanGenerator:
    def look(self, input):
        return ["look the title up with find_book"]
'''


@pytest.fixture
def budget_env(tmp_path, initialized_fastworkflow, monkeypatch):
    monkeypatch.delenv(TURN_MAX_STEPS_ENV, raising=False)
    state = _make_library_runner_env(
        tmp_path,
        initialized_fastworkflow,
        monkeypatch,
        extra_planning_commands={"Library/title_lookup": TITLE_LOOKUP},
        extra_commands=NESTED_COMMANDS,
        real_dispatch=True,
    )
    yield state
    state.ctx.pop_active_workflow()


def test_h_the_budget_stops_a_call_partway_and_the_turn_goes_on(budget_env, no_turn_budget_env,
                                                                monkeypatch):
    state = budget_env
    ctx = state.ctx
    monkeypatch.setitem(fastworkflow._env_vars, TURN_MAX_STEPS_ENV, "3")
    # Four model steps for the nested call; the budget of three stops it after the third.
    for _ in range(4):
        state.nested.run("find_book query=Dune")
    # The turn's own agent: the nested call, then another one that the spent budget refuses,
    # then the finish.
    state.model.run("title_lookup title=Dune")
    state.model.run("book_by_title title=Dune")
    state.model.finish()

    result = ctx.process_turn("look up Dune")

    assert result.status == TurnStatus.COMPLETED
    # The budget stopped the running call after three of its model steps.
    assert len(state.nested.asked) == 3
    assert len(state.nested.steps) == 1
    assert ctx._turn_steps_used == 3
    lookup = next(o for o in ctx._turn_outputs if o.command_name == "Library/title_lookup")
    assert lookup.success is False
    assert lookup.command_response.response == NESTED_BUDGET_SPENT_MESSAGE
    # The turn went on: its next nested call was refused the same way, and pushed no frame.
    refused = [o for o in ctx._turn_outputs if o.command_name == BOOK_BY_TITLE]
    assert [(o.success, o.command_response.response) for o in refused] == [
        (False, NESTED_BUDGET_SPENT_MESSAGE)]
    assert not [s for s in state.sink.named(tracing.SPAN_PLANNING_COMMAND_PHASE)
                if s.attributes.get(tracing.ATTR_FRAME_ALIAS) == "/book_by_title#1"]
    assert len(ctx._frame_stack()) == 1


BOOK_BY_TITLE_FILE = "_commands/Library/book_by_title.py"
_BOOK_BY_TITLE_SOURCE = "import fastworkflow\n" + open(os.path.join(LIBRARY, BOOK_BY_TITLE_FILE), encoding="utf-8").read()
HOOK_CALLS = "title_hook_calls"

#: A test copy of book_by_title whose hook counts its calls (in the workflow's context), corrects
#: a lower-case title and refuses one title.
_HOOK = '''
    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
        if workflow is not None:
            workflow.context.setdefault("title_hook_calls", []).append(command_text)
        if cmd_parameters.title.lower() == "dune":
            cmd_parameters.title = "Dune"
        if cmd_parameters.title == "Emma":
            raise fastworkflow.Refused("the library holds no Emma")
        return None
'''
_HOOK_ANCHOR = '''    plain_utterances = [
        "who wrote Dune",
        "how many copies of Dune Messiah are there",
        "who is the author of Persuasion, and how many copies do we have",
        "tell me about the book Emma",
        "what is Dune Messiah by, and how many copies",
        "give me the author and copy count for the title Emma",
        "does the library own Persuasion, and who wrote it",
        "how many copies of Emma do we own and who wrote it",
    ]
'''

#: A test copy of book_by_title whose Input rule reads the request from the validation context:
#: a request that asks about a withdrawn book is refused.
_INPUT_RULE = '''        @field_validator("title")
        @classmethod
        def title_is_not_withdrawn(cls, value, info):
            if "withdrawn" in (info.context or {}).get("request", "").lower():
                raise ValueError("the library does not look up withdrawn books")
            return value

'''
_TITLE_FIELD_ANCHOR = '                           json_schema_extra={"in_request": True})\n'
_PYDANTIC_ANCHOR = "from pydantic import BaseModel, Field\n"

#: A test copy of book_by_title with a verify bool the classifier decides from the request.
_VERIFY_FIELD = '''        flagged: bool = Field(default=False, description="whether the book is flagged",
                              json_schema_extra={"verify": True})
'''


def _insert_after(source: str, anchor: str, addition: str) -> str:
    assert source.count(anchor) == 1, f"the library's book_by_title changed: {anchor!r}"
    return source.replace(anchor, anchor + addition)


def _hooked_book_by_title() -> str:
    return _insert_after(_BOOK_BY_TITLE_SOURCE, _HOOK_ANCHOR, _HOOK)


def _input_rule_book_by_title() -> str:
    text = _insert_after(_BOOK_BY_TITLE_SOURCE, _TITLE_FIELD_ANCHOR, _INPUT_RULE)
    return text.replace(_PYDANTIC_ANCHOR, "from pydantic import BaseModel, Field, field_validator\n")


def _verify_book_by_title() -> str:
    return _insert_after(_BOOK_BY_TITLE_SOURCE, _TITLE_FIELD_ANCHOR, _VERIFY_FIELD)


def _nested_state(tmp_path, initialized_fastworkflow, monkeypatch, *, command_texts=None,
                  extracted=None):
    monkeypatch.delenv(TURN_MAX_STEPS_ENV, raising=False)
    return _make_library_runner_env(
        tmp_path,
        initialized_fastworkflow,
        monkeypatch,
        extra_commands=NESTED_COMMANDS,
        real_dispatch=True,
        command_texts=command_texts,
        extracted=extracted,
    )


def _book_by_title_calls(state, name: str):
    return [o for o in state.ctx._turn_outputs if o.command_name == name]


def _turn_calling(state, *steps: str, query: str = "look up the book"):
    for step in steps:
        state.model.run(step)
    state.model.finish()
    return state.ctx.process_turn(query)


def _record_planning_entries(monkeypatch, ctx) -> list:
    """The names of the planning calls that reach ``run_planning_command``."""
    entered: list = []
    original = type(ctx).run_planning_command

    def recording(self, qualified_name, params, step_text):
        entered.append((qualified_name, step_text))
        return original(self, qualified_name, params, step_text)

    monkeypatch.setattr(type(ctx), "run_planning_command", recording)
    return entered


def test_i_a_guard_refusal_is_the_invalid_value_response_and_pushes_no_frame(
        tmp_path, initialized_fastworkflow, monkeypatch):
    state = _nested_state(
        tmp_path, initialized_fastworkflow, monkeypatch,
        command_texts={BOOK_BY_TITLE_FILE: _hooked_book_by_title()},
        # The extraction returns a title the step does not write.
        extracted={BOOK_BY_TITLE: {"title": "Foundation"}},
    )
    try:
        entered = _record_planning_entries(monkeypatch, state.ctx)
        result = _turn_calling(state, "book_by_title title=Dune")

        assert result.status == TurnStatus.COMPLETED
        refused = [o for o in state.ctx._turn_outputs if o.command_name == BOOK_BY_TITLE]
        assert [(o.success, o.command_response.response) for o in refused] == [(
            False,
            f"PARAMETER EXTRACTION ERROR FOR COMMAND '{BOOK_BY_TITLE}'\n"
            "title='Foundation' is not written in the request",
        )]
        # The guard came before the hook and the Input: neither ran, and no frame was pushed.
        assert state.wf.context.get(HOOK_CALLS) is None
        assert entered == []
        assert not [s for s in state.sink.named(tracing.SPAN_PLANNING_COMMAND_PHASE)
                    if s.attributes.get(tracing.ATTR_FRAME_ALIAS) == CHILD_ALIAS]
        assert len(state.ctx._frame_stack()) == 1
    finally:
        state.ctx.pop_active_workflow()


def test_j_the_hook_runs_once_after_the_guards_and_its_correction_reaches_the_child(
        tmp_path, initialized_fastworkflow, monkeypatch):
    state = _nested_state(
        tmp_path, initialized_fastworkflow, monkeypatch,
        command_texts={BOOK_BY_TITLE_FILE: _hooked_book_by_title()},
    )
    try:
        entered = _record_planning_entries(monkeypatch, state.ctx)
        result = _turn_calling(state, "book_by_title title=dune")

        assert result.status == TurnStatus.COMPLETED
        # One hook call, with the step's text as command_text, after the guards passed.
        assert state.wf.context[HOOK_CALLS] == ["book_by_title title=dune"]
        assert entered == [(BOOK_BY_TITLE, "book_by_title title=dune")]
        # The child's find phase searched for the corrected title.
        find_span, _book_span = _phase_spans(state, BOOK_BY_TITLE)
        assert find_span.attributes["steps"] == "1. find_book query=Dune"
        child_output = next(o for o in state.ctx._turn_outputs if o.command_name == BOOK_BY_TITLE)
        assert child_output.success is True
        assert child_output.command_response.response == BOOK_BY_TITLE_RESPONSE
    finally:
        state.ctx.pop_active_workflow()


def test_k_a_refusal_by_the_hook_is_returned_as_the_invalid_value_response(
        tmp_path, initialized_fastworkflow, monkeypatch):
    state = _nested_state(
        tmp_path, initialized_fastworkflow, monkeypatch,
        command_texts={BOOK_BY_TITLE_FILE: _hooked_book_by_title()},
    )
    try:
        entered = _record_planning_entries(monkeypatch, state.ctx)
        result = _turn_calling(state, "book_by_title title=Emma")

        assert result.status == TurnStatus.COMPLETED
        assert state.wf.context[HOOK_CALLS] == ["book_by_title title=Emma"]
        refused = [o for o in state.ctx._turn_outputs if o.command_name == BOOK_BY_TITLE]
        assert [(o.success, o.command_response.response) for o in refused] == [(
            False,
            f"PARAMETER EXTRACTION ERROR FOR COMMAND '{BOOK_BY_TITLE}'\n"
            "the library holds no Emma",
        )]
        assert entered == []
        assert len(state.ctx._frame_stack()) == 1
    finally:
        state.ctx.pop_active_workflow()


def test_l_an_input_validator_reading_the_request_refuses_on_the_nested_path(
        tmp_path, initialized_fastworkflow, monkeypatch):
    state = _nested_state(
        tmp_path, initialized_fastworkflow, monkeypatch,
        command_texts={BOOK_BY_TITLE_FILE: _input_rule_book_by_title()},
    )
    try:
        entered = _record_planning_entries(monkeypatch, state.ctx)
        # The first step asks about a withdrawn book; the second asks about the book itself.
        result = _turn_calling(state, "book_by_title title=Dune withdrawn",
                               "book_by_title title=Dune")

        assert result.status == TurnStatus.COMPLETED
        outputs = [o for o in state.ctx._turn_outputs if o.command_name == BOOK_BY_TITLE]
        assert outputs[0].success is False
        assert outputs[0].command_response.response == (
            f"PARAMETER EXTRACTION ERROR FOR COMMAND '{BOOK_BY_TITLE}'\n"
            f"the arguments break a rule of {BOOK_BY_TITLE}: "
            "title: the library does not look up withdrawn books")
        assert outputs[1].success is True
        # Only the accepted step entered the planning path.
        assert entered == [(BOOK_BY_TITLE, "book_by_title title=Dune")]
    finally:
        state.ctx.pop_active_workflow()


def test_m_the_input_rule_refuses_the_same_call_at_the_turn_frame(
        tmp_path, initialized_fastworkflow, monkeypatch):
    state = _nested_state(
        tmp_path, initialized_fastworkflow, monkeypatch,
        command_texts={BOOK_BY_TITLE_FILE: _input_rule_book_by_title()},
    )
    try:
        command = planning_commands.planning_command(state.wf.folderpath, BOOK_BY_TITLE)
        assert command is not None
        with pytest.raises(PlanningCommandInputError, match="the library does not look up withdrawn books"):
            planning_commands._validate_arguments(command, {"title": "Dune"},
                                         "book_by_title title=Dune withdrawn", state.wf)
        assert planning_commands._validate_arguments(
            command, {"title": "Dune"}, "book_by_title title=Dune", state.wf) == {"title": "Dune"}
    finally:
        state.ctx.pop_active_workflow()


def test_n_a_verify_field_is_checked_once_on_the_request_before_the_guards(
        tmp_path, initialized_fastworkflow, monkeypatch):
    state = _nested_state(
        tmp_path, initialized_fastworkflow, monkeypatch,
        command_texts={BOOK_BY_TITLE_FILE: _verify_book_by_title()},
    )
    checks: list[tuple[str, str]] = []

    def fake_check(lm, command, p, request):
        checks.append((p.name, request))
        return True, 1.0

    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: "classifier")
    monkeypatch.setattr(planning_commands, "_check_one", fake_check)
    try:
        result = _turn_calling(state, "book_by_title title=Dune")

        assert result.status == TurnStatus.COMPLETED
        assert checks == [("flagged", "book_by_title title=Dune")]
        child_output = next(o for o in state.ctx._turn_outputs if o.command_name == BOOK_BY_TITLE)
        assert child_output.success is True
    finally:
        state.ctx.pop_active_workflow()


class TitleInput(BaseModel):
    title: str = Field(description="the book's title", json_schema_extra={"in_request": True})


def _planning_command(name: str, plan_cls, input_cls=None):
    """A planning command built in memory: its signature (with *input_cls* as its Input, or
    none) and its ``plan()`` class, as a skill file of the workflow would give them."""
    attributes = {"__doc__": f"{name}, for the test."}
    if input_cls is not None:
        attributes["Input"] = input_cls
    module = types.ModuleType("test_planning_command")
    module.Signature = type("Signature", (), attributes)
    module.PlanGenerator = plan_cls
    return planning_commands.command_from_module(name, module, f"_commands/{name}.py").command


def test_o_a_planning_command_with_no_input_and_one_phase_runs_through_the_phase_runner(
        nested_env):
    class Plan:
        def summary(self):
            return ["find_book query=Dune"]

    nested_env.planning_command = _planning_command("Library/dune_count", Plan)
    result = nested_env.ctx.process_turn("how many books are called Dune")

    assert result.status == TurnStatus.COMPLETED
    [span] = _phase_spans(nested_env, "Library/dune_count")
    assert span.attributes["phase"] == "Library/dune_count"
    assert span.attributes["steps_by_code"] == 1
    assert span.attributes["code_stop"] == "done"
    assert nested_env.model.asked == []
    assert nested_env.nested.asked == []


def test_p_a_one_phase_call_with_an_annotated_phase_has_the_agents_step_budget(
        nested_env, monkeypatch):
    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title), "say how many books match"]

    monkeypatch.setattr(planning_phases, "PHASE_MAX_STEPS", 7)
    nested_env.planning_command = _planning_command("Library/title_find", Plan, TitleInput)
    nested_env.args = {"title": "Dune"}
    result = nested_env.ctx.process_turn("how many books are called Dune")

    assert result.status == TurnStatus.COMPLETED
    [span] = _phase_spans(nested_env, "Library/title_find")
    assert span.attributes["phase_count"] == 1
    assert span.attributes["step_budget"] == nested_env.ctx._workflow_tool_agent.max_iters
    assert span.attributes["step_budget"] != planning_phases.phase_step_budget()


def test_q_each_phase_of_a_two_phase_call_has_the_phase_step_budget(nested_env, monkeypatch):
    monkeypatch.setattr(planning_phases, "PHASE_MAX_STEPS", 7)
    assert planning_phases.phase_step_budget() == 7
    result = _run_book_loans(nested_env)

    assert result.status == TurnStatus.COMPLETED
    book_span, loans_span = _phase_spans(nested_env, BOOK_LOANS)
    assert book_span.attributes["phase_count"] == loans_span.attributes["phase_count"] == 2
    assert book_span.attributes["step_budget"] == loans_span.attributes["step_budget"] == 7


#: The find step of book_by_title, which code runs as a find_book run, and the same step left
#: to the phase's agent: a step naming no command, so the agent takes the model step and can ask.
_FIND_STEP = '        return [run("find_book", query=input.title)]\n'
_OPEN_FIND_STEP = '        return ["look the title up with find_book"]\n'


def _open_find_book_by_title() -> str:
    assert _BOOK_BY_TITLE_SOURCE.count(_FIND_STEP) == 1, "the library's book_by_title changed"
    return _BOOK_BY_TITLE_SOURCE.replace(_FIND_STEP, _OPEN_FIND_STEP)


def _open_find_state(tmp_path, initialized_fastworkflow, monkeypatch):
    return _nested_state(tmp_path, initialized_fastworkflow, monkeypatch,
                         command_texts={BOOK_BY_TITLE_FILE: _open_find_book_by_title()})


@pytest.fixture
def open_find_env(tmp_path, initialized_fastworkflow, monkeypatch):
    state = _open_find_state(tmp_path, initialized_fastworkflow, monkeypatch)
    yield state
    state.ctx.pop_active_workflow()


def _ask_in_find(state, clarification: str = "Which Dune?") -> None:
    """The nested find asks *clarification*; once answered it runs find_book and finishes."""
    state.nested.steps.append(("ask", "ask_user", {"clarification_request": clarification}))
    state.nested.run("find_book query=Dune")
    state.nested.finish()


def _book_loans_asks(state, clarification: str = "Which Dune?"):
    """The turn calls book_loans, whose nested book_by_title's find asks *clarification*."""
    _ask_in_find(state, clarification)
    return _turn_calling(state, "book_loans title=Dune", query="who has Dune on loan")


def _output_names(ctx) -> list[str]:
    return [o.command_name for o in ctx._turn_outputs if not o.is_ask_user]


def test_r_an_ask_user_in_a_nested_call_suspends_the_whole_turn(open_find_env):
    state = open_find_env
    ctx = state.ctx

    suspended = _book_loans_asks(state)

    assert suspended.status == TurnStatus.AWAITING_USER
    assert suspended.answer == "Which Dune?"
    assert ctx.awaiting_user
    assert ctx._pending_clarification_request == "Which Dune?"
    # The child is on top; the parent planning call and the turn frame stay under it, each suspended.
    frames = ctx._frame_stack()
    assert [f.command_name for f in frames] == [None, BOOK_LOANS, BOOK_BY_TITLE]
    assert all(f.agent.export_suspended() is not None for f in frames[1:])
    assert ctx._turn_frame().agent.export_suspended() is not None
    # Only the child's ask was taken: its find_book and its finish wait for the answer.
    assert len(state.nested.steps) == 2


def test_s_a_suspended_nested_stack_restores_on_a_fresh_context(
        tmp_path, initialized_fastworkflow, monkeypatch):
    origin = _open_find_state(tmp_path / "origin", initialized_fastworkflow, monkeypatch)
    try:
        _book_loans_asks(origin)
        origin_frames = list(origin.ctx._frame_stack())
        expected_aliases = [f.alias for f in origin_frames]
        expected_suspensions = [f.agent.export_suspended() for f in origin_frames[1:]]
        blob = json.loads(json.dumps(origin.ctx.serialize_state(channel_id="nested-restore")))
    finally:
        origin.ctx.pop_active_workflow()
    validate_state(blob)

    restored = _open_find_state(tmp_path / "restored", initialized_fastworkflow, monkeypatch)
    try:
        restored.ctx.apply_serialized_state(blob)

        frames = restored.ctx._frame_stack()
        assert [f.command_name for f in frames] == [None, BOOK_LOANS, BOOK_BY_TITLE]
        assert [f.alias for f in frames] == expected_aliases
        assert [f.agent.export_suspended() for f in frames[1:]] == expected_suspensions
        assert restored.ctx.awaiting_user
        assert restored.ctx._pending_clarification_request == "Which Dune?"
        # The restored nested agents count their steps against the turn's budget, as they did.
        assert all(f.agent.on_iteration is not None for f in frames[1:])

        restored.nested.run("find_book query=Dune")
        restored.nested.finish()
        restored.model.finish()
        result = restored.ctx.process_turn("the blue one")

        assert result.status == TurnStatus.COMPLETED
        assert len(restored.ctx._frame_stack()) == 1
        assert BOOK_LOANS in _output_names(restored.ctx)
    finally:
        restored.ctx.pop_active_workflow()


def test_t_the_answer_resumes_the_child_and_its_run_fills_the_parent_phase(open_find_env):
    state = open_find_env
    ctx = state.ctx
    _book_loans_asks(state)

    result = ctx.process_turn("the blue one")

    assert result.status == TurnStatus.COMPLETED
    assert not ctx.awaiting_user
    assert len(ctx._frame_stack()) == 1
    assert state.nested.steps == [] and state.model.steps == []
    # Each run, in order: the child's own, its output held under the step that called it, then
    # the parent's later run; the parent's output is the last.
    assert _output_names(ctx) == [
        "Library/find_book", "Library/open_book", BOOK_BY_TITLE, "Book/list_loans", BOOK_LOANS,
    ]
    child_output = next(o for o in ctx._turn_outputs if o.command_name == BOOK_BY_TITLE)
    assert child_output.success is True
    assert child_output.command_response.output.book_uid == "b1"
    loans_run = next(o for o in ctx._turn_outputs if o.command_name == BOOK_LOANS)
    assert loans_run.success is True
    assert loans_run.command_response.response == BOOK_LOANS_ANSWER
    # The parent's book phase resumed with the child's output as its run, so its facts come from it.
    book_span = [s for s in _phase_spans(state, BOOK_LOANS) if s.attributes["phase"] == "book"][-1]
    assert book_span.attributes["resumed"] is True
    assert book_span.attributes["facts"] == {
        "found": True, "book_uid": "b1", "title": "Dune", "author": "Frank Herbert", "copies": 3,
    }
    loans_span = [s for s in _phase_spans(state, BOOK_LOANS) if s.attributes["phase"] == "loans"][-1]
    assert loans_span.attributes["facts"] == {
        "borrowers": ["Ann Lee", "Raj Patel"], "due_dates": ["2026-11-02", "2026-10-20"], "total": 2,
    }


def test_u_a_direct_action_is_refused_while_a_question_waits_and_the_turn_is_kept(open_find_env):
    state = open_find_env
    ctx = state.ctx
    _book_loans_asks(state)
    frames = list(ctx._frame_stack())
    steps_used = ctx._turn_steps_used
    turn_key = ctx._turn_key
    outputs = list(ctx._turn_outputs)
    suspended_turn = ctx._turn_frame().agent.export_suspended()
    action = fastworkflow.Action(command_name=BOOK_BY_TITLE, command="book_by_title title=Dune")

    refusal = ctx.process_action(action)

    assert refusal.success is False
    assert refusal.command_response.response == AWAITING_ANSWER_REFUSAL_MESSAGE
    assert [id(f) for f in ctx._frame_stack()] == [id(f) for f in frames]
    assert ctx._turn_steps_used == steps_used
    assert ctx._turn_key == turn_key
    assert len(ctx._turn_outputs) == len(outputs)
    assert all(a is b for a, b in zip(ctx._turn_outputs, outputs))
    assert ctx._turn_frame().agent.export_suspended() == suspended_turn
    assert ctx.awaiting_user
    assert ctx._pending_clarification_request == "Which Dune?"

    turn = ctx.process_action_turn(action)

    assert turn.status == TurnStatus.AWAITING_USER
    assert turn.turn_key == turn_key
    assert turn.answer == AWAITING_ANSWER_REFUSAL_MESSAGE
    assert [id(f) for f in ctx._frame_stack()] == [id(f) for f in frames]
    assert ctx._turn_key == turn_key

    result = ctx.process_turn("the blue one")
    assert result.status == TurnStatus.COMPLETED
    assert len(ctx._frame_stack()) == 1


def test_v_a_nested_call_that_asks_twice_stays_suspended_until_its_last_answer(open_find_env):
    state = open_find_env
    ctx = state.ctx
    state.nested.steps.append(("ask", "ask_user", {"clarification_request": "Which Dune?"}))
    state.nested.steps.append(("ask", "ask_user", {"clarification_request": "Which edition?"}))
    state.nested.run("find_book query=Dune")
    state.nested.finish()

    first = _turn_calling(state, "book_loans title=Dune", query="who has Dune on loan")
    assert first.status == TurnStatus.AWAITING_USER
    assert first.answer == "Which Dune?"

    second = ctx.process_turn("the blue one")
    assert second.status == TurnStatus.AWAITING_USER
    assert second.answer == "Which edition?"
    assert ctx.awaiting_user
    assert ctx._pending_clarification_request == "Which edition?"
    assert [f.command_name for f in ctx._frame_stack()] == [None, BOOK_LOANS, BOOK_BY_TITLE]

    result = ctx.process_turn("the first edition")
    assert result.status == TurnStatus.COMPLETED
    assert len(ctx._frame_stack()) == 1
    assert state.nested.steps == []
    assert BOOK_LOANS in _output_names(ctx)


def test_w_a_nested_call_made_from_a_turn_step_suspends_and_resumes_under_the_turn(open_find_env):
    state = open_find_env
    ctx = state.ctx
    _ask_in_find(state)

    suspended = _turn_calling(state, "book_by_title title=Dune")
    assert suspended.status == TurnStatus.AWAITING_USER
    assert [f.command_name for f in ctx._frame_stack()] == [None, BOOK_BY_TITLE]

    result = ctx.process_turn("the blue one")

    assert result.status == TurnStatus.COMPLETED
    assert len(ctx._frame_stack()) == 1
    [child_output] = _book_by_title_calls(state, BOOK_BY_TITLE)
    assert child_output.command_response.output.book_uid == "b1"


def test_x_a_turn_level_phase_whose_code_step_asks_resumes_and_fills_its_facts(open_find_env):
    state = open_find_env
    ctx = state.ctx
    _ask_in_find(state)

    suspended = _run_book_loans(state)
    assert suspended.status == TurnStatus.AWAITING_USER
    assert [f.command_name for f in ctx._frame_stack()] == [None, BOOK_BY_TITLE]

    result = ctx.process_turn("the blue one")

    assert result.status == TurnStatus.COMPLETED
    assert len(ctx._frame_stack()) == 1
    assert state.nested.steps == []
    assert [o.command_response.response for o in _book_by_title_calls(state, BOOK_LOANS)] == [
        BOOK_LOANS_ANSWER]
    book_span = [s for s in _phase_spans(state, BOOK_LOANS) if s.attributes["phase"] == "book"][-1]
    assert book_span.attributes["resumed"] is True
    assert book_span.attributes["facts"]["book_uid"] == "b1"


def _last_phase_span(state, phase: str):
    return [s for s in _phase_spans(state, BOOK_LOANS) if s.attributes["phase"] == phase][-1]


def test_y_a_code_run_phase_whose_nested_call_asks_stays_run_by_code_after_the_answer(
        open_find_env):
    state = open_find_env
    ctx = state.ctx
    _ask_in_find(state)
    suspended = _run_book_loans(state)
    assert suspended.status == TurnStatus.AWAITING_USER

    result = ctx.process_turn("the blue one")

    assert result.status == TurnStatus.COMPLETED
    assert state.nested.steps == []
    # Code ran the book phase's step, which the answer completed, and the loans phase's step:
    # neither phase's agent was asked for a step (the loans phase's closing is not one).
    book_span = _last_phase_span(state, "book")
    assert book_span.attributes["resumed"] is True
    assert book_span.attributes["steps_by_code"] == 1
    assert book_span.attributes["code_stop"] == "done"
    loans_span = _last_phase_span(state, "loans")
    assert loans_span.attributes["steps_by_code"] == 1
    assert loans_span.attributes["code_stop"] == "done"
    assert state.model.asked == []


def _book_loans_by_title_argument(state):
    """A turn step that names book_loans with its title as a tool argument beside the command:
    the argument is folded into the command, and the observation says so."""
    state.model.steps.append(("next", "execute_workflow_query",
                              {"command": "book_loans", "title": "Dune"}))
    state.model.finish()
    return state.ctx.process_turn("who has Dune on loan")


def _turn_observation(state) -> str:
    """The observation the turn's first step was given, as the turn's model saw it next."""
    return next(a["trajectory"]["observation_0"] for a in state.model.asked
                if "observation_0" in a["trajectory"])


def _command_records(ctx) -> list[dict]:
    """The action-log records of command runs; the ask_user dialog records are left out."""
    return [record for record in ctx.action_log if "command" in record]


def test_z_a_restored_suspended_call_completes_as_an_uninterrupted_one(
        tmp_path, initialized_fastworkflow, monkeypatch):
    reference = _nested_state(tmp_path / "reference", initialized_fastworkflow, monkeypatch)
    try:
        assert _book_loans_by_title_argument(reference).status == TurnStatus.COMPLETED
        expected_observation = _turn_observation(reference)
        expected_records = _command_records(reference.ctx)
    finally:
        reference.ctx.pop_active_workflow()
    assert "(Ran as: book_loans title=Dune" in expected_observation

    origin = _open_find_state(tmp_path / "origin", initialized_fastworkflow, monkeypatch)
    try:
        _ask_in_find(origin)
        assert _book_loans_by_title_argument(origin).status == TurnStatus.AWAITING_USER
        blob = json.loads(json.dumps(origin.ctx.serialize_state(channel_id="nested-complete")))
    finally:
        origin.ctx.pop_active_workflow()
    validate_state(blob)

    restored = _open_find_state(tmp_path / "restored", initialized_fastworkflow, monkeypatch)
    try:
        restored.ctx.apply_serialized_state(blob)
        restored.nested.run("find_book query=Dune")
        restored.nested.finish()
        restored.model.finish()

        result = restored.ctx.process_turn("the blue one")

        assert result.status == TurnStatus.COMPLETED
        # The step the suspended call belongs to completes with the observation an uninterrupted
        # run gives: its handle line names the step, the folded note is there, and the run reads
        # the same. Its action-log records are those of the uninterrupted run too.
        observation = _turn_observation(restored)
        assert printed_alias(observation) == printed_alias(expected_observation) == "O0"
        assert observation == expected_observation
        assert _command_records(restored.ctx) == expected_records
    finally:
        restored.ctx.pop_active_workflow()



def _held_runs(ctx, command: str) -> list[dict]:
    """The archived runs of *command* in this turn, as the archive holds them."""
    rows = archive_for_path(observability_db_path(ctx)).command_runs(scope_for_host(ctx))
    return [row for row in rows if row["command"] == command]


def test_aa_the_child_is_held_once_under_its_step_with_the_context_it_ended_on(nested_env):
    result = _run_book_loans(nested_env)

    assert result.status == TurnStatus.COMPLETED
    [held] = _held_runs(nested_env.ctx, BOOK_BY_TITLE)
    child_output = next(o for o in nested_env.ctx._turn_outputs if o.command_name == BOOK_BY_TITLE)
    # Held under the book phase's step, the step that called the child.
    book = _last_phase_span(nested_env, "book")
    assert held["alias"] == f"O{book.attributes['first_step']}" == "O0"
    assert (held["step_index"], held["success"]) == (0, True)
    assert held["input"] == {"title": "Dune"}
    assert held["output"] == child_output.command_response.output.model_dump()
    # The child ended on book b1, in the Book context (unlabelled in the library): the run is held
    # on that context, not on the root it was called from.
    assert (held["context_name"], held["context_clause"]) == ("Book", "Book")


def test_ab_the_child_held_after_an_answer_is_held_the_same_way(open_find_env):
    state = open_find_env
    _book_loans_asks(state)

    result = state.ctx.process_turn("the blue one")

    assert result.status == TurnStatus.COMPLETED
    [held] = _held_runs(state.ctx, BOOK_BY_TITLE)
    book = _last_phase_span(state, "book")
    assert held["alias"] == f"O{book.attributes['first_step']}"
    assert (held["step_index"], held["success"]) == (book.attributes["first_step"], True)
    assert held["input"] == {"title": "Dune"}
    assert held["output"]["book_uid"] == "b1"
    assert (held["context_name"], held["context_clause"]) == ("Book", "Book")


def test_ac_the_parent_reads_leave_out_the_runs_inside_the_child_range(nested_env, monkeypatch):
    ctx = nested_env.ctx
    reads: list[tuple[list, list[str]]] = []
    original = type(ctx)._archived_command_runs

    def recording(self, **kwargs):
        rows = original(self, **kwargs)
        reads.append(([list(r) for r in self._top_frame().child_step_ranges],
                      [row["command"] for row in rows]))
        return rows

    monkeypatch.setattr(type(ctx), "_archived_command_runs", recording)
    result = _run_book_loans(nested_env)

    assert result.status == TurnStatus.COMPLETED
    # The parent's reads made after the child returned: the child's range is on its frame.
    during_parent = [commands for ranges, commands in reads if ranges]
    assert during_parent
    for commands in during_parent:
        assert "Library/find_book" not in commands
        assert "Library/open_book" not in commands
    # The held run of the child is the parent's own run, so the parent's reads see it.
    assert any(BOOK_BY_TITLE in commands for commands in during_parent)


def test_ad_facts_from_archive_leaves_out_a_child_range_when_given_it(nested_env):
    ctx = nested_env.ctx
    _run_book_loans(nested_env)
    store = archive_for_path(observability_db_path(ctx))
    scope = scope_for_host(ctx)
    # The child's own steps are its find (1) and its open (2), as its phase spans name them.
    child_steps = [s.attributes["first_step"] for s in nested_env.sink.named(tracing.SPAN_PLANNING_COMMAND_PHASE)
                   if s.attributes.get(tracing.ATTR_FRAME_ALIAS) == CHILD_ALIAS]
    ranges = [[min(child_steps), max(child_steps)]]

    assert ranges == [[1, 2]]
    runs = store.command_runs(scope)
    outside = runs_outside_ranges(runs, ranges)
    assert fill_typed_phase_output(BookByTitle.Output, outside).book_uid == "b1"
    # The child's find is in its range, so the parent's facts cannot read it.
    with pytest.raises(MissingFactError, match="find_book"):
        fill_typed_phase_output(FindBook.Output, outside)
    # Without dropping the ranges the same read sees the child's find.
    assert "b1" in fill_typed_phase_output(FindBook.Output, runs).book_uids


def test_af_the_parent_continues_in_the_context_its_child_ended_on(nested_env):
    ctx = nested_env.ctx
    result = _turn_calling(nested_env, "book_by_title title=Dune", "list_loans",
                           query="who has Dune on loan")

    assert result.status == TurnStatus.COMPLETED
    # The step that called the child is told where the workflow now is.
    first = next(a["trajectory"]["observation_0"] for a in nested_env.model.asked
                 if "observation_0" in a["trajectory"])
    assert "now in context 'Book'" in first
    [held] = _held_runs(ctx, BOOK_BY_TITLE)
    assert (held["context_name"], held["context_clause"]) == ("Book", "Book")
    # The parent's next step acts on the record the child left; nothing restored its context.
    [listed] = [o for o in ctx._turn_outputs if o.command_name == "Book/list_loans"]
    assert listed.success is True
    assert ctx.get_active_workflow().current_command_context_name == "Book"


def test_ag_a_resumed_step_whose_child_navigated_carries_the_now_in_note(open_find_env):
    state = open_find_env
    _ask_in_find(state)
    suspended = _turn_calling(state, "book_by_title title=Dune", "list_loans",
                              query="who has Dune on loan")
    assert suspended.status == TurnStatus.AWAITING_USER

    result = state.ctx.process_turn("the blue one")

    assert result.status == TurnStatus.COMPLETED
    assert "now in context 'Book'" in _turn_observation(state)
    [listed] = [o for o in state.ctx._turn_outputs if o.command_name == "Book/list_loans"]
    assert listed.success is True
    [held] = _held_runs(state.ctx, BOOK_BY_TITLE)
    assert (held["context_name"], held["context_clause"]) == ("Book", "Book")


#: A planning command that opens one book by its uid: a child a fan-out item's sub-plan calls.
OPEN_BY_UID = '''from pydantic import BaseModel, Field

from fastworkflow import CommandResponse
from fastworkflow.planning import run

from .open_book import Signature as OpenBook


class Signature:
    """Open one book, given its uid."""

    class Input(BaseModel):
        book_uid: str = Field(description="the uid of the book")

    class Output(BaseModel):
        book_uid: str = Field(description="the uid of the book opened")
        title: str = Field(description="its title")


class PlanGenerator:
    def book(self, input) -> OpenBook.Output:
        return [run("open_book", book_uid=input.book_uid)]

    def __result__(self, input, book):
        return CommandResponse(response=f"{book.title} is open.",
                               output=Signature.Output(book_uid=book.book_uid, title=book.title))
'''
OPEN_BY_UID_PLANNING_COMMAND = "Library/open_by_uid"


@pytest.fixture
def open_by_uid_env(tmp_path, initialized_fastworkflow, monkeypatch):
    monkeypatch.delenv(TURN_MAX_STEPS_ENV, raising=False)
    state = _make_library_runner_env(
        tmp_path,
        initialized_fastworkflow,
        monkeypatch,
        extra_planning_commands={OPEN_BY_UID_PLANNING_COMMAND: OPEN_BY_UID},
        extra_commands=NESTED_COMMANDS,
        real_dispatch=True,
    )
    yield state
    state.ctx.pop_active_workflow()


def _item_opened(state):
    """open_by_uid's Output, as the workflow loaded it: the model a fan-out phase fills."""
    from fastworkflow import planning_commands

    # A planning command is not routed: its Signature is in the field-source resolution.
    signatures = planning_commands.resolve_field_sources(state.wf.folderpath).signatures
    return signatures[OPEN_BY_UID_PLANNING_COMMAND].Output


def test_ae_a_fan_out_item_whose_sub_plan_calls_a_planning_command_is_bound_to_its_record(
        open_by_uid_env, monkeypatch):
    state = open_by_uid_env
    item_opened = _item_opened(state)

    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        @phase(over="find.book_uids", output=item_opened)
        def opened(self, uid):
            return [run("open_by_uid", book_uid=uid)]

        def __result__(self, input, find, opened):
            return ", ".join(f"{item.book_uid} {item.title}" for item in opened)

    entered = _record_planning_entries(monkeypatch, state.ctx)
    state.planning_command = _planning_command("Library/dune_books", Plan, TitleInput)
    state.args = {"title": "Dune Messiah"}
    result = state.ctx.process_turn("show the book Dune Messiah")

    assert result.status == TurnStatus.COMPLETED
    [item] = [s for s in _phase_spans(state, "Library/dune_books")
              if s.attributes["phase"] == "opened"]
    assert item.attributes["item"] == "b2"
    # The item's sub-plan called open_by_uid, which opened the item's record: the item is bound to
    # that record, is not flagged as never opened, and its output is filled from it.
    assert item.attributes["record_issues"] == []
    assert item.attributes["record"] == "Book b2"
    assert item.attributes["facts"] == {"book_uid": "b2", "title": "Dune Messiah"}
    assert entered == [("Library/open_by_uid", "open_by_uid book_uid=b2")]
    [held] = _held_runs(state.ctx, "Library/open_by_uid")
    assert (held["context_name"], held["context_clause"]) == ("Book", "Book")
    assert held["output"] == {"book_uid": "b2", "title": "Dune Messiah"}
    assert result.answer == "b2 Dune Messiah"


def test_ah_the_first_pushed_frame_is_current_under_the_childs_alias(nested_env, monkeypatch):
    ctx = nested_env.ctx
    pushed: list[tuple[str, str, str]] = []
    original = type(ctx)._attach_nested_agent

    def recording(self, frame):
        # Called once the frame is on the stack, before its first step runs.
        pushed.append((frame.command_name, frame.alias, self.current_frame_alias))
        return original(self, frame)

    monkeypatch.setattr(type(ctx), "_attach_nested_agent", recording)
    result = _run_book_loans(nested_env)

    assert result.status == TurnStatus.COMPLETED
    # The one nested call pushes one frame: the parent's alias, the command's name and its count.
    assert pushed == [(BOOK_BY_TITLE, CHILD_ALIAS, CHILD_ALIAS)]
    assert ctx.current_frame_alias == ""


def test_ai_a_facts_model_does_not_read_the_find_run_of_the_child_it_calls(nested_env):
    class Plan:
        def look(self, input) -> FindBook.Output:
            return [run("book_by_title", title=input.title)]

    nested_env.planning_command = _planning_command("Library/look_for_books", Plan, TitleInput)
    nested_env.args = {"title": "Dune"}
    result = nested_env.ctx.process_turn("which books are called Dune")

    # book_by_title's own find_book ran, inside the child's range: the parent's fill cannot read it.
    assert [o.success for o in _book_by_title_calls(nested_env, BOOK_BY_TITLE)] == [True]
    [span] = _phase_spans(nested_env, "Library/look_for_books")
    assert span.attributes["phase"] == "look"
    assert "find_book" in span.attributes["facts_error"]
    assert span.attributes.get("facts") is None
    assert "stopped at its part 'look'" in result.answer


def test_aj_the_same_call_fills_a_fact_from_the_child_output_it_names(nested_env):
    class Plan:
        def look(self, input) -> BookByTitle.Output:
            return [run("book_by_title", title=input.title)]

    nested_env.planning_command = _planning_command("Library/look_for_books", Plan, TitleInput)
    nested_env.args = {"title": "Dune"}
    result = nested_env.ctx.process_turn("which book is called Dune")

    assert result.status == TurnStatus.COMPLETED
    [span] = _phase_spans(nested_env, "Library/look_for_books")
    assert span.attributes.get("facts_error") is None
    assert span.attributes["facts"]["book_uid"] == "b1" and span.attributes["facts"]["found"] is True


def test_ak_two_fan_out_items_each_open_their_record_in_place(open_by_uid_env, monkeypatch):
    state = open_by_uid_env
    item_opened = _item_opened(state)

    class Plan:
        def find(self, input) -> FoundBooks:
            return [run("find_book", query=input.title)]

        @phase(over="find.book_uids", output=item_opened)
        def opened(self, uid):
            return [run("open_by_uid", book_uid=uid)]

        def __result__(self, input, find, opened):
            return ", ".join(f"{item.book_uid} {item.title}" for item in opened)

    entered = _record_planning_entries(monkeypatch, state.ctx)
    state.planning_command = _planning_command("Library/dune_books", Plan, TitleInput)
    state.args = {"title": "Dune"}
    result = state.ctx.process_turn("show the books called Dune")

    assert result.status == TurnStatus.COMPLETED
    items = [s for s in _phase_spans(state, "Library/dune_books")
             if s.attributes["phase"] == "opened"]
    assert [s.attributes["item"] for s in items] == ["b1", "b2"]
    assert [s.attributes["record_issues"] for s in items] == [[], []]
    assert [s.attributes["facts"] for s in items] == [
        {"book_uid": "b1", "title": "Dune"},
        {"book_uid": "b2", "title": "Dune Messiah"},
    ]
    assert entered == [("Library/open_by_uid", "open_by_uid book_uid=b1"),
                       ("Library/open_by_uid", "open_by_uid book_uid=b2")]
    # 3.9 (Calls by exact name): item 2's open_book is a Library command, on the lookup path from
    # the Book item 1 opened, so it runs in place, bound to the library. No go_up runs between the
    # items, and the second item's record is opened from the first one's.
    names = [o.command_name for o in state.ctx._turn_outputs]
    first_open = names.index("Library/open_by_uid")
    second_open = names.index("Library/open_by_uid", first_open + 1)
    assert "IntentDetection/go_up" not in names[first_open + 1:second_open]
    assert state.wf.current_command_context.uid == "b2"
    assert result.answer == "b1 Dune, b2 Dune Messiah"


class _OpenOne:
    def summary(self):
        return ["open_by_uid book_uid=b1"]


def test_a_library_command_and_a_library_planning_command_are_callable_from_the_book_record(
        open_by_uid_env):
    """After a planning call opens b1 the workflow stands on the Book record, and Library is its
    enclosing record. 3.9 (Calls by exact name, 3.4 Routing): a Library execution command
    (find_book) and a Library planning command (open_by_uid) are on the lookup path from there,
    so both are called by their names, in place, bound to the library. No refusal and no
    navigation. A command of the sibling Member context is off the path, and is still refused
    with the navigation sentence (fix-04jl: the way back to the root is named)."""
    state = open_by_uid_env
    state.planning_command = _planning_command("Library/open_one", _OpenOne)
    result = state.ctx.process_turn("open book b1")
    assert result.status == TurnStatus.COMPLETED
    wf = state.wf
    assert wf.current_command_context_name == "Book"

    # 3.9: accepted by name, and the current object does not move (no go_up is inserted).
    assert workflow_agent._explicit_agent_command("find_book", wf) == "find_book"
    assert workflow_agent._explicit_agent_command("open_by_uid", wf) == "open_by_uid"
    assert wf.current_command_context_name == "Book"

    # Off the path: Member is a sibling of Book, so its command is refused, as before.
    refused = _refusal_text(wf, "list_member_loans")
    assert refused.startswith(
        "Command 'list_member_loans' is not available in the current context 'Book'. "
        "It is available in: 'Member'.")


def test_a_refusal_from_a_record_under_the_root_names_the_way_back_through_the_root(
        open_by_uid_env):
    """fix-04jl: from the Book opened from the Library root, live_context_chain names the root
    ('Book', 'Library', '*'), so the planner sees go_up to Library and then open_member (which needs
    member_uid) to reach Member. The refusal offers that way, and does not say there is none."""
    state = open_by_uid_env
    state.planning_command = _planning_command("Library/open_one", _OpenOne)
    result = state.ctx.process_turn("open book b1")
    assert result.status == TurnStatus.COMPLETED
    wf = state.wf
    assert wf.current_command_context_name == "Book"

    refused = _refusal_text(wf, "list_member_loans")

    assert "No declared navigation path" not in refused
    assert "go_up" in refused
    assert "open_member" in refused
    assert "member_uid" in refused


def test_al_a_nested_step_naming_its_running_ancestor_is_refused_and_the_call_goes_on(
        open_find_env):
    state = open_find_env
    ctx = state.ctx
    # book_by_title's find is an open step: its agent names book_loans, which is already running.
    state.nested.run("book_loans title=Dune")
    state.nested.run("find_book query=Dune")
    state.nested.finish()

    result = _turn_calling(state, "book_loans title=Dune", query="who has Dune on loan")

    assert result.status == TurnStatus.COMPLETED
    assert state.nested.steps == []
    assert len(ctx._frame_stack()) == 1
    refusals = [o.command_response.response for o in ctx._turn_outputs
                if o.command_name == BOOK_LOANS and not o.success]
    assert refusals == [f"{BOOK_LOANS} is already running; call its steps directly."]
    # The nested agent was told why, then ran its find and finished the child.
    told = [value for a in state.nested.asked for key, value in a["trajectory"].items()
            if key.startswith("observation_")]
    assert any("is already running" in value for value in told)
    [find] = [o for o in ctx._turn_outputs if o.command_name == "Library/find_book"]
    assert find.success is True
    [child] = _book_by_title_calls(state, BOOK_BY_TITLE)
    assert child.command_response.output.book_uid == "b1"
    assert [o.command_response.response for o in ctx._turn_outputs
            if o.command_name == BOOK_LOANS and o.success] == [BOOK_LOANS_ANSWER]


def test_am_a_from_naming_the_planning_command_its_steps_call_asks_for_a_description(tmp_path):
    """A producer ``from_`` on a planning command's Input names the planning command its own
    steps run: the load error asks for a description, as an execution callee does."""
    calls_child = '''from pydantic import BaseModel, Field

from fastworkflow import from_
from fastworkflow.planning import run


class Signature:
    """Say the title of one book, given its title."""

    class Input(BaseModel):
        title: str = Field(description="the book's title")
        book_uid: str = from_("book_by_title.book_uid", description="the uid of the book")


class PlanGenerator:
    def steps(self, input):
        return [run("book_by_title", title=input.title), "say the title"]
'''
    described = calls_child.replace(
        'from_("book_by_title.book_uid", description="the uid of the book")',
        'Field(description="the uid of the book")').replace(
        "from fastworkflow import from_\n", "")
    folder = _library_workflow(
            tmp_path,
            extra_planning_commands={"Library/plan_calls_book": calls_child,
                          "Library/plan_described_book": described},
            extra_commands=(BOOK_BY_TITLE_FILE, "_commands/Library/_wording.py"),
        )
    res = planning_commands.resolve_field_sources(folder)

    assert [p.message for p in res.errors if p.command == "Library/plan_described_book"] == []
    assert [p.message for p in res.errors if p.command == "Library/plan_calls_book"] == [
        "Input.book_uid (from 'book_by_title.book_uid') tells the agent to run "
        "Library/book_by_title first, but this command's own steps call Library/book_by_title. "
        "To describe the value, write Field(description=...) instead"]


def test_an_the_now_in_note_names_the_record_the_child_opened(nested_env, monkeypatch):
    # The library's Book context declares no instance label, so its note names the context
    # alone; declaring the uid as the label (as the TodoList test does) lets it name the record.
    monkeypatch.setattr(context_class_for(nested_env.wf, "Book"), "instance_label_attr", "uid",
                        raising=False)
    result = _turn_calling(nested_env, "book_by_title title=Dune", "list_loans",
                           query="who has Dune on loan")

    assert result.status == TurnStatus.COMPLETED
    assert "now in context 'Book' and b1" in _turn_observation(nested_env)
    [listed] = [o for o in nested_env.ctx._turn_outputs if o.command_name == "Book/list_loans"]
    assert listed.command_response.output.borrowers == ["Ann Lee", "Raj Patel"]
    assert nested_env.ctx.get_active_workflow().current_command_context_name == "Book"


_PICK_LINE = '    return [pick("open_book", named=input.title)]'
_OPEN_OTHER_LINE = '    return [run("open_book", book_uid="b2")]'


def _book_by_title_ending_on_b2() -> str:
    """book_by_title, changed to open b2 (Dune Messiah) rather than the book the title names."""
    assert _BOOK_BY_TITLE_SOURCE.count(_PICK_LINE) == 1, "the library's book_by_title changed"
    return _BOOK_BY_TITLE_SOURCE.replace(_PICK_LINE, _OPEN_OTHER_LINE)


def test_ao_a_child_that_ends_on_another_record_is_acted_on_there_and_not_restored(
        tmp_path, initialized_fastworkflow, monkeypatch):
    state = _nested_state(tmp_path, initialized_fastworkflow, monkeypatch,
                          command_texts={BOOK_BY_TITLE_FILE: _book_by_title_ending_on_b2()})
    try:
        result = _turn_calling(state, "book_by_title title=Dune", "list_loans",
                               query="who has Dune on loan")

        assert result.status == TurnStatus.COMPLETED
        [child] = _book_by_title_calls(state, BOOK_BY_TITLE)
        assert child.command_response.output.book_uid == "b2"
        # The next step acted on b2: its loans are none, where b1 (the title's book) has two.
        [listed] = [o for o in state.ctx._turn_outputs if o.command_name == "Book/list_loans"]
        assert listed.command_response.output.borrowers == []
        assert listed.command_response.output.total == 0
        assert [h["output"]["book_uid"] for h in _held_runs(state.ctx, BOOK_BY_TITLE)] == ["b2"]
        # Nothing restored A's earlier context: the workflow stays on the record the child left.
        assert state.ctx.get_active_workflow().current_command_context_name == "Book"
        assert len(state.ctx._frame_stack()) == 1
    finally:
        state.ctx.pop_active_workflow()


@pytest.mark.xfail(strict=True, reason=(
    "fix-3rsq.5.9 gap, outside Phase 5 by the plan's step 6 (which no longer says the finish check "
    "flags it): the record check flags a run on another record only for fan-out items "
    "(planning_phases.item_record_issues). A non-fan-out phase whose planning child ended on another "
    "record gets no flag, and a record check for it is not part of Phase 5."))
def test_ap_the_finish_check_flags_a_step_run_on_another_record(
        tmp_path, initialized_fastworkflow, monkeypatch):
    state = _nested_state(tmp_path, initialized_fastworkflow, monkeypatch,
                          command_texts={BOOK_BY_TITLE_FILE: _book_by_title_ending_on_b2()})
    try:
        state.planning_command = _real_planning_command(BOOK_LOANS, "_commands/Library/book_loans.py").command
        state.args = {"title": "Dune"}
        state.ctx.process_turn("who has Dune on loan")

        # The loans phase acted on b2, the record its book phase did not name for "Dune".
        loans = _last_phase_span(state, "loans")
        assert loans.attributes["steps_by_code"] == 1
        assert loans.attributes.get("record_issues")
    finally:
        state.ctx.pop_active_workflow()


def test_ar_a_code_only_phase_makes_no_planner_call_of_its_own(nested_env, monkeypatch):
    planner_calls: list[bool] = []
    inner = workflow_agent.build_query_with_next_steps

    def spy(*args, **kwargs):
        planner_calls.append(kwargs.get("planning_command") is not None)
        return inner(*args, **kwargs)

    monkeypatch.setattr(workflow_agent, "build_query_with_next_steps", spy)

    class Plan:
        def summary(self):
            return ["find_book query=Dune"]

    nested_env.planning_command = _planning_command("Library/dune_count", Plan)
    result = nested_env.ctx.process_turn("how many books are called Dune")

    assert result.status == TurnStatus.COMPLETED
    # The turn's own planning call is the only one; the call's phase, which code ran, made none.
    assert planner_calls == [True]
    assert nested_env.model.asked == []


@pytest.fixture
def fastworkflow_records():
    """Every record the ``fastWorkflow`` logger emits while the test runs. That logger does not
    propagate, so a handler on it is the only place its ERROR lines can be read."""
    records: list[logging.LogRecord] = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Collect(level=logging.DEBUG)
    logger = logging.getLogger("fastWorkflow")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


@pytest.fixture
def real_library_env(tmp_path, initialized_fastworkflow, monkeypatch):
    """The library folder itself, not a copy, through the executor's real dispatch. The state
    root is in *tmp_path*, so the trace database is not written beside the library."""
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    state = _make_library_runner_env(
        tmp_path,
        initialized_fastworkflow,
        monkeypatch,
        real_dispatch=True,
        real_folder=True,
    )
    yield state
    state.ctx.pop_active_workflow()


def test_real_library_loads_with_no_error_and_book_loans_runs_book_by_title_nested(
        fastworkflow_records, real_library_env, monkeypatch):
    """The proof on the library folder itself (Phase 5, fix-3rsq.5.10): the folder loads with no
    field-source error and no ERROR line, and book_loans' book phase runs book_by_title in a
    frame of its own, held once under its step, with the answer book_loans gives."""
    ctx = real_library_env.ctx
    res = planning_commands.resolve_field_sources(LIBRARY)
    assert res.errors == ()
    assert {BOOK_LOANS, BOOK_BY_TITLE} <= set(res.planning_commands)
    assert res.callees[BOOK_LOANS] == frozenset({"Book/list_loans", BOOK_BY_TITLE})
    assert planning_commands.call_depth(BOOK_LOANS, res) == 2

    pushed: list[tuple[str, str]] = []
    original = type(ctx)._attach_nested_agent

    def recording(self, frame):
        pushed.append((frame.command_name, frame.alias))
        return original(self, frame)

    monkeypatch.setattr(type(ctx), "_attach_nested_agent", recording)
    entered = _record_planning_entries(monkeypatch, ctx)

    result = _run_book_loans(real_library_env, "Dune")

    assert result.status == TurnStatus.COMPLETED
    assert result.answer == BOOK_LOANS_ANSWER
    # The nested frame: one frame pushed, under the call's alias, for the step text book_loans wrote.
    assert pushed == [(BOOK_BY_TITLE, CHILD_ALIAS)]
    assert entered == [(BOOK_BY_TITLE, "book_by_title title=Dune")]
    # book_by_title's own phases ran inside that frame; its steps were run by code.
    child_phases = [s.attributes["phase"] for s in real_library_env.sink.named(tracing.SPAN_PLANNING_COMMAND_PHASE)
                    if s.attributes.get(tracing.ATTR_FRAME_ALIAS) == CHILD_ALIAS]
    assert child_phases == ["find", "book"]
    book_span, loans_span = _phase_spans(real_library_env, BOOK_LOANS)
    assert book_span.attributes["steps"] == "1. book_by_title title=Dune"
    assert book_span.attributes["code_stop"] == "done"
    assert loans_span.attributes["steps_by_code"] == 1
    # The held run: book_loans' book step ran book_by_title once, and it is held under that step.
    [held] = _held_runs(ctx, BOOK_BY_TITLE)
    assert held["alias"] == f"O{book_span.attributes['first_step']}" == "O0"
    assert (held["step_index"], held["success"]) == (0, True)
    assert held["output"]["book_uid"] == "b1"
    assert [o.command_name for o in ctx._turn_outputs] == [
        "Library/find_book", "Library/open_book", BOOK_BY_TITLE, "Book/list_loans", BOOK_LOANS,
    ]
    assert real_library_env.model.asked == [] and real_library_env.nested.asked == []
    assert len(ctx._frame_stack()) == 1
    assert [r.getMessage() for r in fastworkflow_records if r.levelno >= logging.ERROR] == []


# Runs in a child interpreter: the suspended turn is saved by one process and restored by another,
# so nothing the restore reads can come from the first process's memory.
_RESUME_CHILD = '''
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import fastworkflow
import tests.test_nested_planning_commands as nested
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

mode, workdir, blob_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
with pytest.MonkeyPatch.context() as mp:
    # What the initialized_fastworkflow fixture does, for the one interpreter.
    fastworkflow.init({})
    RoutingRegistry.clear_registry()
    mp.setattr(WorkflowExecutionContext, "_agent_dspy_context",
               lambda self: (SimpleNamespace(model="test-model"), None))
    # The review gate is off for these turns, as in the parent process (test_turn_result_capture.no_review_gate).
    mp.setattr(WorkflowExecutionContext, "_subtask_gate", lambda self, subtasks: None)
    state = nested._open_find_state(workdir, None, mp)
    try:
        ctx = state.ctx
        if mode == "suspend":
            nested._book_loans_asks(state)
            blob_path.write_text(json.dumps(ctx.serialize_state(channel_id="nested-new-process")))
            report = {"command_names": [f.command_name for f in ctx._frame_stack()],
                      "awaiting": ctx.awaiting_user}
        else:
            ctx.apply_serialized_state(json.loads(blob_path.read_text()))
            report = {"command_names": [f.command_name for f in ctx._frame_stack()],
                      "awaiting": ctx.awaiting_user,
                      "question": ctx._pending_clarification_request}
            state.nested.run("find_book query=Dune")
            state.nested.finish()
            state.model.finish()
            result = ctx.process_turn("the blue one")
            report.update(
                status=result.status.value,
                answer=result.answer,
                frames_after=len(ctx._frame_stack()),
                loans=[o.command_response.response for o in ctx._turn_outputs
                       if o.command_name == nested.BOOK_LOANS and o.success],
            )
    finally:
        ctx.pop_active_workflow()
print("REPORT " + json.dumps(report))
'''


def _run_resume_child(mode: str, workdir, blob_path) -> dict:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run(
        [sys.executable, "-c", _RESUME_CHILD, mode, str(workdir), str(blob_path)],
        cwd=repo_root, capture_output=True, text=True, timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    [line] = [ln for ln in done.stdout.splitlines() if ln.startswith("REPORT ")]
    return json.loads(line[len("REPORT "):])


def test_aq_a_suspended_call_is_resumed_by_a_new_process_with_its_stack_and_finishes_the_parent(
        tmp_path):
    blob = tmp_path / "suspended.json"
    suspended = _run_resume_child("suspend", tmp_path / "origin", blob)
    assert suspended["awaiting"] is True
    assert suspended["command_names"] == [None, BOOK_LOANS, BOOK_BY_TITLE]

    restored = _run_resume_child("resume", tmp_path / "restored", blob)

    assert restored["command_names"] == suspended["command_names"]
    assert restored["awaiting"] is True
    assert restored["question"] == "Which Dune?"
    assert restored["status"] == TurnStatus.COMPLETED.value
    assert restored["frames_after"] == 1
    assert restored["loans"] == [BOOK_LOANS_ANSWER]


#: A planning command of Book (a context of its own): from Library the agent tool refuses it.
LOAN_REPORT = "loan_report"
LOAN_REPORT_FILE = "_commands/Book/loan_report.py"

#: A global planning command: no context folder, so it belongs to the root context ('*').
_GLOBAL_DUNE_TOTAL = '''"""dune_total: how many books are called Dune, for the test."""
from fastworkflow.planning import run


class Signature:
    """Count the books called Dune."""


class PlanGenerator:
    def steps(self, input):
        return [run("find_book", query="Dune"), "say how many books match"]
'''


def _refusal_text(workflow, command: str) -> str:
    """The refusal the agent tool gives *command* in the workflow's current context, as text."""
    with pytest.raises(CommandNotFoundError) as refused:
        workflow_agent._explicit_agent_command(command, workflow)
    return str(refused.value)


def _observation_texts(trajectory) -> list[str]:
    return [trajectory[k] for k in sorted(
        (k for k in trajectory if k.startswith("observation_")),
        key=lambda k: int(k.rsplit("_", 1)[1]))]


def test_an_agent_step_naming_a_planning_command_of_another_context_is_refused(nested_book_env,
                                                                                monkeypatch):
    """The agent's step names loan_report (Book) from Library: the agent tool refuses it with the
    refusal an execution command of another context gets, and no planning frame is pushed."""
    entered = _record_planning_entries(monkeypatch, nested_book_env.ctx)
    wf = nested_book_env.ctx.get_active_workflow()
    expected = _refusal_text(wf, LOAN_REPORT)
    assert expected.startswith(
        "Command 'loan_report' is not available in the current context 'Library'.")

    result = _turn_calling(nested_book_env, LOAN_REPORT, query="who has this book on loan")

    assert result.status == TurnStatus.COMPLETED
    assert expected in _turn_observation(nested_book_env)
    assert entered == []
    assert not [o for o in nested_book_env.ctx._turn_outputs
                if o.command_name == "Book/loan_report"]
    assert wf.current_command_context_name == "Library"


class _PlanRefusedThenInContext:
    """Two steps: the first names loan_report, out of Library, and code refuses it; the second
    names book_by_title, in Library, which the agent runs (its nested call pushes a frame)."""

    def summary(self):
        return [LOAN_REPORT, "look the title up"]


def test_a_nested_step_naming_a_planning_command_of_another_context_is_refused_before_any_frame(
        nested_book_env, monkeypatch):
    """A planning command's step names loan_report, out of Library: the call is refused before
    run_planning_command is reached, with the refusal the agent tool gives. The next step names
    book_by_title, which is in context and does push a frame."""
    entered = _record_planning_entries(monkeypatch, nested_book_env.ctx)
    wf = nested_book_env.ctx.get_active_workflow()
    expected = _refusal_text(wf, LOAN_REPORT)
    nested_book_env.planning_command = _planning_command("Library/probe_loan_report", _PlanRefusedThenInContext)
    nested_book_env.model.run("book_by_title title=Dune")
    nested_book_env.model.finish()
    nested_book_env.model.finish()
    # The plan's second step never ran, so each of the first two finishes is reviewed and reopens the
    # loop; the third finish is past the review cap and ends the run.
    nested_book_env.model.finish()

    result = nested_book_env.ctx.process_turn("who has Dune on loan")

    assert result.status == TurnStatus.COMPLETED
    # Each asked trajectory carries every earlier observation: the refusal is there from the
    # first ask, which the code run's stop led to, before any frame was pushed.
    first_ask = nested_book_env.model.asked[0]["trajectory"]
    assert any(expected in text for text in _observation_texts(first_ask))
    assert [name for name, _ in entered] == [BOOK_BY_TITLE]
    assert not [o for o in nested_book_env.ctx._turn_outputs
                if o.command_name == "Book/loan_report"]


def test_a_global_planning_command_is_called_by_name_from_any_context(tmp_path,
                                                                   initialized_fastworkflow,
                                                                   monkeypatch):
    """dune_total belongs to '*'. 3.9: the global scope is on every lookup path, so from Library it is
    called by name, as the global execution command library_hours is; from the root it is too."""
    state = _make_library_runner_env(
        tmp_path, initialized_fastworkflow, monkeypatch,
        extra_planning_commands={"dune_total": _GLOBAL_DUNE_TOTAL})
    wf = state.wf
    assert fastworkflow.RoutingRegistry.get_definition(wf.folderpath).get_planning_command_names(
        "*") == ["dune_total"]

    assert wf.current_command_context_name == "Library"
    # 3.9 ("Planning commands use the same path", 3.4 Routing): the global scope is on every lookup
    # path, so a global planning command is called by its name from a context, as a global
    # execution command (library_hours) is. The call is bound to the root object (the global
    # scope), and the current context stays Library.
    assert workflow_agent._explicit_agent_command("dune_total", wf) == "dune_total"
    assert workflow_agent._explicit_agent_command("library_hours", wf) == "library_hours"
    walked = CommandExecutor.perform_action(
        state.ctx.cme_workflow,
        fastworkflow.Action(command_name="wildcard", command="dune_total"))
    assert walked.success
    assert walked.command_response.artifacts["command_name"] == "dune_total"
    # The root is the Library itself (the current object), so the run is not moved: no
    # "ran on" stamp is set, and the binding is the root object.
    assert "ran_on_context_name" not in walked.command_response.artifacts
    assert wf.command_context_for_response_generation is wf.root_command_context
    assert wf.current_command_context_name == "Library"

    wf.current_command_context = None
    assert wf.current_command_context_name == "*"
    assert workflow_agent._explicit_agent_command("dune_total", wf) == "dune_total"


# ---------------------------------------------------------------------------
# A fixed plan that calls one nested command twice (fix-3rsq.repeat-nested-finish)
# ---------------------------------------------------------------------------

class TitlesSignature:
    """Say who wrote two books, given their titles."""

    class Input(BaseModel):
        first: str = Field(description="the first title")
        second: str = Field(description="the second title")


def _two_book_by_title_command():
    """A planning command whose fixed plan calls book_by_title twice, with different titles."""
    def steps(input):
        return [f"book_by_title title={input.first}", f"book_by_title title={input.second}"]

    return planning_commands.planning_command_from_parts(
        "Library/two_books", TitlesSignature, os.path.join(LIBRARY, "_commands/Library/two_books.py"),
        steps_fn=steps).command


def _record_finish_notes(monkeypatch):
    """The finish check's real notes, recorded as the run reads them (the check itself runs)."""
    from fastworkflow.observation_offloading import finish_check

    notes: list = []
    real = finish_check.review_finish

    def recorded(**kwargs):
        note = real(**kwargs)
        notes.append(note)
        return note

    monkeypatch.setattr(finish_check, "review_finish", recorded)
    return notes


def test_a_fixed_plan_calling_one_nested_command_twice_passes_the_finish_check(nested_env, monkeypatch):
    notes = _record_finish_notes(monkeypatch)
    # The turn's agent numbers its steps from the context's next step, as _build_react_agent
    # sets it in production. Without it a nested call's steps reuse the parent's next O<k> alias
    # and archive_step refuses the parent's second row (the check then reads step 2 as not run).
    nested_env.ctx._workflow_tool_agent.step_floor = lambda: nested_env.ctx._next_step
    nested_env.planning_command = _two_book_by_title_command()
    nested_env.args = {"first": "Dune", "second": "Persuasion"}

    result = nested_env.ctx.process_turn("who wrote Dune and Persuasion")

    assert result.status == TurnStatus.COMPLETED
    # Both nested calls ran, each with its own title.
    runs = [o for o in nested_env.ctx._turn_outputs if o.command_name == BOOK_BY_TITLE]
    assert [o.command_parameters.title for o in runs] == ["Dune", "Persuasion"]
    assert all(o.success for o in runs)
    # The finish check read both steps as run: no note, and the model was not asked again.
    assert notes and all(note is None for note in notes)
    assert nested_env.model.asked == []
    assert nested_env.nested.asked == []


def test_a_second_call_of_the_same_nested_command_that_never_ran_is_named(nested_env):
    """Only the first of two calls of one command ran: the check names step 2, not step 1."""
    from fastworkflow.observation_offloading import finish_check

    plan = "1. book_by_title title=Dune\n2. book_by_title title=Persuasion"
    rows = [{"step_index": 0, "command": "book_by_title title=Dune",
             "text": BOOK_BY_TITLE_RESPONSE}]
    result = finish_check.check(plan, rows, [BOOK_BY_TITLE], values=True)

    assert [v.step for v in result.unmet] == [2]
    assert "Step 2 (book_by_title) was not run" in result.note
    assert "Step 1" not in result.note
