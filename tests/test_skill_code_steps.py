"""Steps run by code (fix-6llp): a phase's leading mechanical steps run without the model.

Offline, no model: the rule is read with the existing parsers, the runner is the real
``fastWorkflowReAct`` with a scripted model that fails the test when it is not
expected, and the end-to-end runs use the todo_list test workflow with its commands
run in code (the workflow ships no trained intent models). Neutral toy names only.
"""
from __future__ import annotations

from typing import Optional

import dspy
import pytest
from pydantic import BaseModel, Field

from fastworkflow import skill_phases
from fastworkflow.observation_offloading import finish_check
from fastworkflow.skill_phases import code_step, phase_query, plan_code_steps

CORE = ["IntentDetection/go_up", "IntentDetection/reset_context",
        "IntentDetection/what_can_i_do", "ErrorCorrection/abort"]
COMMANDS = ["Shelf/find_book", "Shelf/list_loans", "Shelf/open_book_by_uid",
            "Shelf/set_due", "Shelf/count_notes"]


class FindParams(BaseModel):
    query: Optional[str] = Field(default=None)
    start: Optional[int] = Field(default=None)


class LoanParams(BaseModel):
    scope_uid: Optional[str] = Field(default=None)
    filter: Optional[str] = Field(default=None)
    start: Optional[int] = Field(default=None)


class OpenParams(BaseModel):
    book_uid: str = Field(description="uid")


class DueParams(BaseModel):
    loan_uid: str = Field(description="uid")
    days: int = Field(description="days")


CLASSES = {"find_book": FindParams, "list_loans": LoanParams,
           "open_book_by_uid": OpenParams, "set_due": DueParams, "count_notes": None}


def _step(text):
    return finish_check.parse_plan(f"1. {text}", COMMANDS, CORE)[0]


def _read(text):
    """``(command, reason)`` as code reads one plan step."""
    [(_, command, reason)] = plan_code_steps(
        f"1. {text}", COMMANDS, CORE, lambda name: [CLASSES[name]] if name in CLASSES else [])
    return command, reason


# ----------------------------------------------------------------------
# The rule (fix-6llp.1)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text, command", [
    # bare: every field defaulted, or no parameters at all; the note is never sent
    ("list_loans (full list: next page with start=<n> until complete)", "list_loans"),
    ("count_notes", "count_notes"),
    # key=value, concrete
    ("open_book_by_uid book_uid=b1 (this opens it)", "open_book_by_uid book_uid=b1"),
    ("set_due loan_uid=l1 days=7", "set_due loan_uid=l1 days=7"),
    # a literal lookup, and a literal paging value: ordinary values
    ("find_book query=Dune Messiah (take a match whose name starts with Dune)",
     "find_book query=Dune Messiah"),
    ("find_book query=Dune start=20", "find_book query=Dune start=20"),
    ("list_loans filter=rare first editions; for its shelf", "list_loans filter=rare first editions"),
])
def test_a_mechanical_step_runs_its_chain_and_never_its_note(text, command):
    assert code_step(_step(text), CLASSES[command.split()[0]]) == command


@pytest.mark.parametrize("text", [
    "open_book_by_uid book_uid=<the uid of the match>",     # placeholder
    "open_book_by_uid with the uid of the Dune match",      # prose, no value
    "list_loans for the full list",                         # prose after the command
    "list_loans scope_uid=x1, no filter",                   # prose swallowed into the value
    "open_book_by_uid",                                     # required field missing
    "set_due loan_uid=l1",                                  # one required field missing
    "find_book query=Dune start=<n>",                       # paging placeholder
    "find_book query=Dune start=",                          # paging with no value
    "only if nothing was found, find_book query=Dune",      # conditional
    "find_book query=Dune (skip when it is already open)",  # conditional
    "then find_book query=Dune",                            # command not the first word
    "find_book query=Dune tags=[a]",                        # a word= that is no field
])
def test_an_open_step_is_left_to_the_agent(text):
    step = _step(text)
    assert code_step(step, CLASSES.get(step.command)) is None


def test_a_command_with_no_parameters_class_takes_no_chain():
    assert code_step(_step("count_notes on the book"), None) is None


@pytest.mark.parametrize("text, command", [
    ("go_up", "go_up"),
    ("reset_context (back to the top)", "reset_context"),
    ("go_up back to the shelf", None),            # prose after a core command
    ("what_can_i_do", None),                     # a core command code does not run
    ("go_up only if the book is not open", None),
    ("say which loans b1 has", ""),              # an answer step: passed over
    ("do not run list_loans again", ""),         # a negated step: passed over
    ("never go_up here", ""),
    ("next page with start=<n> and the same other arguments until done", None),
    ("open_shelf", None),                        # a command the workflow does not have
])
def test_steps_that_are_not_plain_command_steps(text, command):
    assert _read(text)[0] == command


def test_a_step_is_mechanical_only_under_every_context_that_defines_its_command():
    plan = "1. find_book query=Dune"
    one = plan_code_steps(plan, COMMANDS, CORE, lambda name: [FindParams])
    two = plan_code_steps(plan, COMMANDS, CORE, lambda name: [FindParams, OpenParams])
    assert one[0][1] == "find_book query=Dune"
    assert two[0][1] is None and two[0][2] == "differs by context"


def test_the_reason_is_given_for_the_replay():
    assert _read("list_loans (full list)")[1] == "bare"
    assert _read("count_notes")[1] == "no parameters"
    assert _read("find_book query=Dune")[1] == "key=value"
    assert _read("list_loans scope_uid=x1, no filter")[1] == "prose in value"
    assert _read("go_up")[1] == "no parameters"


def test_the_code_steps_framing_is_said_and_stripped_from_answers():
    query = phase_query("req", "1. find_book query=Dune", index=0, count=2, name="find",
                        code_steps=True)
    assert skill_phases._CODE_STEPS_DONE in query
    assert skill_phases._CODE_STEPS_DONE not in phase_query(
        "req", "1. x", index=0, count=2, name="find")
    last = phase_query("req", "1. x", index=1, count=2, name="report", code_steps=True)
    assert skill_phases._CODE_STEPS_DONE in last
    assert skill_phases.clean_answer(f"Dune is on loan. {skill_phases._CODE_STEPS_DONE}") \
        == "Dune is on loan."


@pytest.mark.parametrize("observation, more", [
    ("12 loans; showing 1-10. Next page: start=10, same arguments.", True),
    ("3 loans.", False),
    ("Nothing at start=0.", False),
    ("restart=5", False),
])
def test_more_pages_is_a_later_page_the_observation_names(observation, more):
    assert skill_phases._more_pages(observation, ["start"]) is more


# ----------------------------------------------------------------------
# The runner, end to end on the todo_list test workflow (fix-6llp.2 to .6)
# ----------------------------------------------------------------------

import json  # noqa: E402
import types  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import fastworkflow  # noqa: E402
from fastworkflow import tracing  # noqa: E402
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (  # noqa: E402
    ParameterExtraction)
from fastworkflow.command_executor import CommandExecutor  # noqa: E402
from fastworkflow.skill_commands import SkillCallPlan, command_from_module, phases_for  # noqa: E402
from fastworkflow.skill_phases import Phase  # noqa: E402
from fastworkflow.turn import TurnStatus  # noqa: E402
from fastworkflow.workflow_agent import initialize_workflow_tool_agent  # noqa: E402
from tests.test_skill_phases import RecordingSink  # noqa: E402
from tests.test_turn_result_capture import (  # noqa: E402,F401 - shared fixtures
    _make_agent_ctx,
    _set_agents,
    initialized_fastworkflow,
    todo_workflow_path,
)


def _from(source, **kwargs):
    return Field(json_schema_extra={"from": source}, **kwargs)


class Lists(BaseModel):
    list_ids: list[int] = _from("list_todo_lists.todolist_ids", min_length=1)


class Opened(BaseModel):
    description: str = _from("get_todo_list.description")


def lists_steps():
    return ["list_todo_lists (every list)"]


def open_steps(lists: Optional[Lists] = None):
    return [f"get_todo_list id={lists.list_ids[-1] if lists else '<the id>'}",
            "get_all_children (full list)"]


def report_steps(open: Optional[Opened] = None):
    return ["get_properties (its description)",
            f"say what {open.description if open else 'the list'} holds"]


class ShowSignature:
    """Show one todo list and what it holds."""

    class Input(BaseModel):
        name: str = Field(description="the list's name")


PLANS = {
    "three": lambda input: [
        Phase("lists", lists_steps, output=Lists),
        Phase("open", lambda lists: open_steps(lists), output=Opened),
        Phase("report", lambda open: report_steps(open)),
    ],
}


def _skill(plan):
    module = types.ModuleType("show_list_skill")
    module.Signature = ShowSignature
    module.plan = plan
    return command_from_module("show_list", module, "_skills/show_list.py").command


def _qualified(workflow, name):
    routing = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
    return next((n for n in routing.get_command_names(workflow.current_command_context_name)
                 if n.split("/")[-1] == name), None)


class Workflow:
    """Runs the todo workflow's commands in code, as the NLU would route them (the test
    workflow ships no trained intent models); *scripted* commands answer from a table."""

    def __init__(self, ctx, wf):
        self.ctx, self.wf = ctx, wf
        self.sent: list[str] = []
        self.scripted: dict = {}

    def __call__(self, session, command):
        self.sent.append(command)
        name = command.split()[0]
        wf = self.wf
        if name in self.scripted:
            reply = self.scripted[name]
            reply = reply(command) if callable(reply) else reply
            response, output, success = (reply + (True,))[:3] if isinstance(reply, tuple) else (
                reply, None, True)
            return fastworkflow.CommandOutput(
                command_name=f"Shelf/{name}",
                command_response=fastworkflow.CommandResponse(
                    response=response, success=success, output=output))
        if name in ("go_up", "reset_context"):
            if name == "reset_context" or wf.get_parent(wf.current_command_context) is None:
                wf.current_command_context = wf.root_command_context
            else:
                wf.current_command_context = wf.get_parent(wf.current_command_context)
            return fastworkflow.CommandOutput(
                command_name=f"IntentDetection/{name}",
                command_response=fastworkflow.CommandResponse(
                    response=f"Context is now '{wf.current_command_context_displayname}'"))
        qualified = _qualified(wf, name)
        routing = fastworkflow.RoutingRegistry.get_definition(wf.folderpath)
        cls = routing.get_command_class(qualified, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
        params = None
        if cls is not None:
            params = (ParameterExtraction._extract_parameters_from_key_values(command, cls, name)
                      or ParameterExtraction._extract_parameters_from_bare(command, cls, name))
        out = CommandExecutor.perform_action(wf, fastworkflow.Action(
            command_name=qualified, command=command,
            parameters=params.model_dump() if params is not None else {}))
        out.command_name = qualified
        text = out.command_response.response or ""
        if text.startswith("Response: "):
            out.command_response.output = json.loads(text[len("Response: "):])
        out.command_parameters = params
        return out


class Model:
    """The executor's model: scripted agent steps, and the answer call. Asked for a step
    nobody scripted, it fails the test."""

    def __init__(self, agent):
        self.agent = agent
        self.steps: list = []
        self.asked: list[dict] = []
        self.answers: list[str] = []

    def __call__(self, module, trajectory, **inputs):
        if module is self.agent.react:
            self.asked.append({"query": inputs.get("user_query"), "trajectory": dict(trajectory)})
            if not self.steps:
                raise AssertionError(f"the model was asked for a step after {trajectory}")
            thought, tool, args = self.steps.pop(0)
            return dspy.Prediction(next_thought=thought, next_tool_name=tool, next_tool_args=args)
        self.answers.append(inputs.get("user_query"))
        return dspy.Prediction(reasoning="", final_answer="the answer")

    def run(self, command):
        self.steps.append(("next", "execute_workflow_query", {"command": command}))

    def finish(self):
        self.steps.append(("done", "finish", {}))


@pytest.fixture
def todo(initialized_fastworkflow, todo_workflow_path, monkeypatch):
    """A phased skill call on the todo workflow, run by the real executor and tools."""
    monkeypatch.delenv(skill_phases.PHASE_MAX_STEPS_ENV, raising=False)
    ctx, wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    sink = RecordingSink()
    ctx.set_trace_sink(sink)
    ctx.push_active_workflow(wf)
    CommandExecutor.perform_action(wf, fastworkflow.Action(command_name="startup", command=""))
    # A second list, in memory only (the data file is never written). Its id is no
    # example value of get_todo_list, so a step that opens it binds in code.
    from tests.todo_list_workflow.application.todo_list import TodoList

    root = wf.root_command_context
    chores = TodoList(id=7, description="chores")
    chores.parent = root
    root.lists[7] = chores
    errands = TodoList(id=8, description="errands")
    errands.parent = root
    root.lists[8] = errands
    # A list prints its id as its instance label ("TodoList 8"), as a record does.
    from fastworkflow.context_identity import context_class_for

    monkeypatch.setattr(context_class_for(wf, "TodoList"), "instance_label_attr", "id",
                        raising=False)
    agent = initialize_workflow_tool_agent(ctx)
    _set_agents(ctx, agent)
    model = Model(agent)
    monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation", model)
    workflow = Workflow(ctx, wf)
    monkeypatch.setattr(CommandExecutor, "invoke_command",
                        classmethod(lambda cls, session, command: workflow(session, command)))
    state = SimpleNamespace(ctx=ctx, wf=wf, sink=sink, agent=agent, model=model,
                            workflow=workflow, plan=PLANS["three"], answered=[], toys={})
    # Toy commands (scripted in `workflow.scripted`) for what the todo workflow lacks:
    # their parameters classes, and their names among the commands a plan is read with.
    real_classes = skill_phases.parameter_classes
    monkeypatch.setattr(skill_phases, "parameter_classes", lambda routing: (
        lambda name: [state.toys[name]] if name in state.toys else real_classes(routing)(name)))
    real_names = ctx._phase_command_names

    def names():
        found = real_names()
        return {**found, "commands": found["commands"] + [f"Shelf/{n}" for n in state.toys]}

    monkeypatch.setattr(ctx, "_phase_command_names", names)

    def fake_plan(user_query, session, with_agent_inputs_and_trajectory=False,
                  planner_lm=None, skill_command=None, **kwargs):
        if skill_command is None:   # the reply to ask_user is planned too
            return user_query
        call = SkillCallPlan(plan="", args={"name": "groceries"}, notes=[], prediction=None,
                             phases=phases_for(skill_command, {"name": "groceries"}),
                             steps_after=list(getattr(state, "steps_after", [])),
                             steps_before=list(getattr(state, "steps_before", [])))
        session._phased_skill_call = (skill_command, call)
        session._initial_plan_text = ""
        return user_query

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", fake_plan)
    monkeypatch.setattr("fastworkflow.workflow_agent.planner_skill_for",
                        lambda text, folder: ("", _skill(state.plan)))
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks",
                        lambda *args, **kwargs: ["show the list groceries"])
    monkeypatch.setattr("fastworkflow.skill_phases.write_answer",
                        lambda query: state.answered.append(query) or "written")

    def turn(text="show the list groceries"):
        return ctx.process_turn(text)

    state.turn = turn
    yield state
    ctx.pop_active_workflow()


def _phases(sink):
    return sink.named(tracing.SPAN_SKILL_PHASE)


def test_a_phase_of_mechanical_steps_runs_with_no_model_call(todo):
    result = todo.turn()

    assert result.status == TurnStatus.COMPLETED
    # Every step ran in code, through the agent's own tool; the model was never asked
    # for a step, and only the last phase (it answers) made the one answer call.
    assert todo.model.asked == []
    assert len(todo.model.answers) == 1
    assert todo.workflow.sent == ["list_todo_lists", "get_todo_list id=8", "get_all_children",
                                  "get_properties"]
    spans = _phases(todo.sink)
    assert [s.attributes["phase"] for s in spans] == ["lists", "open", "report"]
    assert spans[0].attributes["facts"] == {"list_ids": [1, 7, 8]}
    assert spans[1].attributes["facts"] == {"description": "errands"}
    assert [s.attributes["steps_by_code"] for s in spans] == [1, 2, 1]
    assert [s.attributes["code_stop"] for s in spans] == ["done", "done", "done"]
    assert [s.attributes["steps_used"] for s in spans] == [1, 2, 1]
    assert all(s.status == tracing.STATUS_OK for s in spans)
    assert result.answer == "the answer"


def test_code_steps_are_ordinary_trajectory_steps_spans_and_archive_rows(todo):
    todo.turn()
    trajectory = todo.agent.trajectory
    first = min(int(k.rsplit("_", 1)[1]) for k in trajectory if k.startswith("thought_"))
    assert trajectory[f"thought_{first}"] == "Step 1 of the plan, run by code."
    assert trajectory[f"tool_name_{first}"] == "execute_workflow_query"
    assert trajectory[f"tool_args_{first}"] == {"command": "get_properties"}
    assert todo.agent.iteration_counter == 0
    steps = todo.sink.named(tracing.SPAN_AGENT_STEP)
    assert len(steps) == 4 and all(s.attributes["by_code"] is True for s in steps)
    calls = todo.sink.named(tracing.SPAN_AGENT_TOOL_CALL)
    assert [c.parent_span_id for c in calls] == [s.span_id for s in steps]
    runs = todo.ctx._archived_command_runs()
    assert [(r["step_index"], r["command"].split("/")[-1]) for r in runs] == [
        (0, "list_todo_lists"), (1, "get_todo_list"), (2, "get_all_children"),
        (3, "get_properties")]
    contract = tracing.SPAN_CONTRACTS[tracing.SPAN_AGENT_STEP].attributes
    assert all(set(s.attributes) - {tracing.ATTR_SPAN_CONTRACT_VERSION} <= contract for s in steps)
    phase_contract = tracing.SPAN_CONTRACTS[tracing.SPAN_SKILL_PHASE].attributes
    assert all(set(s.attributes) - {tracing.ATTR_SPAN_CONTRACT_VERSION} <= phase_contract
               for s in _phases(todo.sink))


def _plan(*phases):
    return lambda input: list(phases)


def _observations(trajectory):
    return [trajectory[k] for k in sorted(
        (k for k in trajectory if k.startswith("observation_")),
        key=lambda k: int(k.rsplit("_", 1)[1]))]


class ItemsParams(BaseModel):
    filter: Optional[str] = Field(default=None)
    start: Optional[int] = Field(default=None)


def test_the_retry_never_runs_a_code_step_twice(todo, monkeypatch):
    from dspy.utils.exceptions import AdapterParseError

    real = todo.model.__call__

    def answer_fails(module, trajectory, **inputs):
        if module is todo.agent.extract:
            raise AdapterParseError(adapter_name="ChatAdapter", signature=todo.agent.react.signature,
                                    lm_response="garbled")
        return real(module, trajectory, **inputs)

    monkeypatch.setattr(todo.agent, "_call_with_potential_trajectory_truncation", answer_fails)
    with pytest.raises(AdapterParseError):
        todo.turn()
    # The answer call failed after code ran its steps: no attempt runs them again.
    assert todo.workflow.sent.count("get_properties") == 1
    assert todo.workflow.sent.count("list_todo_lists") == 1


def test_a_mixed_phase_hands_over_after_the_code_prefix(todo):
    todo.plan = _plan(
        Phase("lists", lists_steps, output=Lists),
        Phase("open", lambda lists: ["get_todo_list id=8", "get_all_children for what it holds"],
              output=Opened),
        Phase("report", lambda open: ["get_properties"]))
    todo.model.run("get_all_children")
    todo.model.finish()
    result = todo.turn()

    assert result.status == TurnStatus.COMPLETED
    assert len(todo.model.asked) == 2
    first = todo.model.asked[0]
    assert skill_phases._CODE_STEPS_DONE in first["query"]
    # The agent sees the code-run step as its own earlier step, and goes on from it.
    assert first["trajectory"]["thought_1"] == "Step 1 of the plan, run by code."
    assert first["trajectory"]["tool_args_1"] == {"command": "get_todo_list id=8"}
    # The finish check counted the code-run step: finishing was not sent back (the
    # model would have been asked for a third step).
    spans = _phases(todo.sink)
    assert spans[1].attributes["code_stop"] == "open step 2"
    assert spans[1].attributes["steps_by_code"] == 1
    assert spans[1].attributes["steps_used"] == 3
    # Model steps = steps_used - steps_by_code.
    assert spans[1].attributes["steps_used"] - spans[1].attributes["steps_by_code"] == 2
    assert spans[1].attributes["facts"] == {"description": "errands"}


def test_a_failed_code_step_hands_over_with_the_failure_last(todo):
    todo.workflow.scripted["get_todo_list"] = ("No todo list has id 8.", None, False)
    todo.model.finish()
    todo.model.finish()  # the finish check sends it back once: step 1 has no run
    result = todo.turn()

    assert result.status == TurnStatus.COMPLETED
    asked = todo.model.asked[0]
    assert "No todo list has id 8." in _observations(asked["trajectory"])[-1]
    spans = _phases(todo.sink)
    assert spans[1].attributes["code_stop"] == "step 1 failed"
    # Code did not fail the phase; the fill did, as for any run that read nothing.
    assert "the planned run was not found" in spans[1].attributes["facts_error"]
    assert "stopped at its part 'open'" in result.answer


def test_an_answer_step_and_a_negated_step_do_not_stop_code(todo):
    todo.plan = _plan(Phase("report", lambda: [
        "get_todo_list id=8", "do not run get_all_children", "get_properties",
        "say what the list holds"]))
    todo.turn()
    assert todo.model.asked == []
    assert todo.workflow.sent == ["get_todo_list id=8", "get_properties"]
    assert len(todo.model.answers) == 1


def test_a_phase_whose_answer_is_not_shown_makes_no_call_and_the_last_one_call(todo):
    todo.turn()
    spans = todo.sink.named(tracing.SPAN_AGENT_EXECUTE)
    assert len(spans) == 3
    # One answer call in all: the last phase's.
    assert len(todo.model.answers) == 1 and "part 3 of 3 (report)" in todo.model.answers[0]


def test_the_framing_sentence_never_reaches_the_answer(todo, monkeypatch):
    real = todo.model.__call__

    def echoes(module, trajectory, **inputs):
        if module is todo.agent.extract:
            return dspy.Prediction(reasoning="", final_answer=(
                f"The list holds nothing. {skill_phases._CODE_STEPS_DONE}"))
        return real(module, trajectory, **inputs)

    monkeypatch.setattr(todo.agent, "_call_with_potential_trajectory_truncation", echoes)
    result = todo.turn()
    assert result.answer == "The list holds nothing."


def test_planner_steps_before_run_first_and_the_first_phase_runs_its_code_steps(todo):
    todo.steps_before = ["list_todo_lists"]
    todo.model.run("list_todo_lists")
    todo.model.finish()
    result = todo.turn()

    assert result.status == TurnStatus.COMPLETED
    spans = _phases(todo.sink)
    assert [s.attributes["phase"] for s in spans] == [
        skill_phases.STEPS_BEFORE_PHASE, "lists", "open", "report"]
    # The agent runs the planner's steps (code never does); the first phase then runs
    # wholly by code, as with no steps_before.
    assert spans[0].attributes["code_stop"] == "off"
    assert spans[0].attributes["steps_by_code"] == 0
    assert [s.attributes["code_stop"] for s in spans[1:]] == ["done", "done", "done"]
    assert len(todo.model.asked) == 2 and todo.model.steps == []
    assert todo.workflow.sent == ["list_todo_lists", "list_todo_lists", "get_todo_list id=8",
                                  "get_all_children", "get_properties"]
    # Facts fill from the first phase's own run and the next phase binds them.
    assert spans[1].attributes["facts"] == {"list_ids": [1, 7, 8]}
    assert spans[2].attributes["facts"] == {"description": "errands"}


def test_a_suspended_phase_resumes_without_running_code_steps_again(todo):
    todo.plan = _plan(
        Phase("lists", lambda: ["list_todo_lists (every list)",
                                "ask which list the user means when it is unclear"],
              output=Lists),
        Phase("report", lambda lists: ["get_todo_list id=8"]))
    todo.model.steps.append(("ask", "ask_user", {"clarification_request": "Which list?"}))
    todo.model.finish()
    first = todo.turn()
    assert first.status == TurnStatus.AWAITING_USER
    assert todo.workflow.sent == ["list_todo_lists"]

    second = todo.ctx.process_turn("the errands list")
    assert second.status == TurnStatus.COMPLETED
    assert todo.workflow.sent.count("list_todo_lists") == 1
    spans = _phases(todo.sink)
    assert [(s.attributes["phase"], s.attributes["resumed"]) for s in spans][:2] == [
        ("lists", False), ("lists", True)]
    assert "code_stop" not in spans[1].attributes
    assert spans[0].attributes["code_stop"] == "open step 2"


# ----------------------------------------------------------------------
# Navigation (fix-6llp.3)
# ----------------------------------------------------------------------

def test_code_reaches_the_context_of_a_step_by_a_no_value_path(todo, climbable):
    todo.plan = _plan(Phase("report", lambda: ["get_todo_list id=8", "list_todo_lists"]))
    todo.turn()

    assert todo.model.asked == []
    assert todo.workflow.sent == ["get_todo_list id=8", "go_up", "list_todo_lists"]
    trajectory = todo.agent.trajectory
    assert trajectory["thought_1"] == "Run by code to reach the context of step 2."
    assert trajectory["tool_args_1"] == {"command": "go_up"}
    runs = [r["command"].split("/")[-1] for r in todo.ctx._archived_command_runs()]
    assert runs[:3] == ["get_todo_list", "go_up", "list_todo_lists"]


def test_a_path_that_needs_a_value_stops_code_with_the_refusal(todo, monkeypatch):
    from fastworkflow import context_navigation
    from fastworkflow.context_navigation import DescendEdge

    monkeypatch.setattr(context_navigation, "collect_descend_edges", lambda path: (
        DescendEdge("TodoListManager", "get_todo_list", "TodoList", needs=("id",)),))
    todo.plan = _plan(Phase("report", lambda: ["get_all_children", "get_properties"]))
    todo.model.finish()
    todo.model.finish()
    todo.turn()

    asked = todo.model.asked[0]
    last = _observations(asked["trajectory"])[-1]
    assert "is not available in the current context" in last and "get_todo_list" in last
    assert _phases(todo.sink)[0].attributes["code_stop"] == "no path for step 1"
    assert "get_all_children" not in todo.workflow.sent


def test_opening_the_record_already_open_counts_as_done(todo, monkeypatch):
    from fastworkflow import context_navigation
    from fastworkflow.context_navigation import DescendEdge

    monkeypatch.setattr(context_navigation, "collect_descend_edges", lambda path: (
        DescendEdge("TodoListManager", "get_todo_list", "TodoList", needs=("id",)),))
    todo.plan = _plan(Phase("report", lambda: [
        "get_todo_list id=8", "get_todo_list id=8", "get_properties"]))
    todo.turn()

    assert todo.model.asked == []
    assert todo.workflow.sent == ["get_todo_list id=8", "get_properties"]
    assert "You are already in context" in todo.agent.trajectory["observation_1"]
    assert _phases(todo.sink)[0].attributes["code_stop"] == "done"


def test_code_path_takes_only_one_shortest_no_value_path(monkeypatch):
    from fastworkflow import context_navigation
    from fastworkflow.context_navigation import DescendEdge, code_path

    wf = SimpleNamespace(current_command_context_name="Shelf", folderpath="/x")
    monkeypatch.setattr(context_navigation, "live_context_chain",
                        lambda workflow: ("Shelf", "Library", "*"))
    homes = {"find": ["Catalog"], "two": ["Catalog", "Archive"], "deep": ["Book"]}
    monkeypatch.setattr(context_navigation, "home_contexts", lambda workflow, token: homes[token])
    edges = (DescendEdge("Library", "open_catalog", "Catalog"),
             DescendEdge("Library", "open_archive", "Archive"),
             DescendEdge("Catalog", "open_book_by_uid", "Book", needs=("book_uid",)))
    monkeypatch.setattr(context_navigation, "collect_descend_edges", lambda path: edges)

    assert code_path(wf, "find") == ["go_up", "open_catalog"]
    assert code_path(wf, "two") is None          # two equal homes
    assert code_path(wf, "deep") is None         # the path needs a uid


# ----------------------------------------------------------------------
# Paging and fan-out items (fix-6llp.4)
# ----------------------------------------------------------------------

def _paged_items(command):
    start = int((command.split("start=") + ["0"])[1].split()[0]) if "start=" in command else 0
    more = start < 20
    text = f"items {start}-{start + 9}." + (f" Next page: start={start + 10}." if more else "")
    return text, {"labels": [f"i{start}"]}


@pytest.fixture
def paged(todo):
    todo.toys["list_items"] = ItemsParams
    todo.workflow.scripted["list_items"] = _paged_items
    return todo


def test_a_literal_paging_value_runs_exactly_that_page(paged):
    paged.plan = _plan(Phase("report", lambda: ["list_items start=20", "list_todo_lists"]))
    paged.turn()
    assert paged.model.asked == []
    assert [c for c in paged.workflow.sent if c.startswith("list_items")] == ["list_items start=20"]


def test_a_literal_value_with_more_pages_goes_on(paged):
    paged.plan = _plan(Phase("report", lambda: ["list_items start=10", "get_todo_list id=8"]))
    paged.turn()
    assert paged.model.asked == []
    assert paged.workflow.sent == ["list_items start=10", "get_todo_list id=8"]


@pytest.mark.parametrize("steps", [
    ["list_items start=<n>"],
    ["list_items filter=rare", "next page with start=<n> and the same arguments until done"],
])
def test_a_paging_placeholder_stops_code_and_the_agent_pages(paged, steps):
    paged.workflow.scripted["list_items"] = ("all 3 items.", {"labels": ["a", "b", "c"]})
    paged.plan = _plan(Phase("report", lambda: list(steps)))
    paged.model.finish()
    paged.turn()
    assert len(paged.model.asked) == 1
    assert _phases(paged.sink)[0].attributes["code_stop"] == f"open step {len(steps)}"


def test_a_paging_request_in_a_note_runs_page_1_and_hands_over_when_more_remain(paged):
    paged.plan = _plan(Phase("report", lambda: [
        "list_items (full list: next page with start=<n> until complete)", "list_todo_lists"]))
    paged.model.run("list_items start=10")
    paged.model.run("list_todo_lists")
    paged.model.finish()
    paged.turn()

    # Code read page 1 only, and computed no next page: the agent paged.
    assert paged.workflow.sent[0] == "list_items"
    assert paged.agent.trajectory["thought_0"] == "Step 1 of the plan, run by code (its first page)."
    assert _phases(paged.sink)[0].attributes["code_stop"] == "more pages at step 1"
    assert len(paged.model.asked) == 3
    assert paged.workflow.sent == ["list_items", "list_items start=10", "list_todo_lists"]


def test_a_note_list_that_fits_one_page_goes_on(paged):
    paged.workflow.scripted["list_items"] = ("all 3 items.", {"labels": ["a", "b", "c"]})
    paged.plan = _plan(Phase("report", lambda: [
        "list_items (full list: next page with start=<n> until complete)", "get_todo_list id=8"]))
    paged.turn()
    assert paged.model.asked == []
    assert paged.workflow.sent == ["list_items", "get_todo_list id=8"]


@pytest.fixture
def climbable(monkeypatch):
    """The todo workflow's root object is a context of its own (TodoListManager): say
    so, so that go_up from a list is a path to it."""
    from fastworkflow import context_navigation

    real_chain = context_navigation.live_context_chain
    monkeypatch.setattr(context_navigation, "live_context_chain", lambda wf: (
        ("TodoList", "TodoListManager", "*") if wf.current_command_context_name == "TodoList"
        else real_chain(wf)))


def test_a_fan_out_of_mechanical_items_runs_every_item_with_no_model_call(todo, climbable):
    class Props(BaseModel):
        description: str = _from("get_todo_list.description")

    todo.plan = _plan(
        Phase("each", lambda item: [f"get_todo_list id={item}", "get_properties"],
              output=Props, over=[7, 8]),
        Phase("report", lambda each=None: [
            f"say {', '.join(p.description for p in each or [])}"]))
    result = todo.turn()

    assert result.status == TurnStatus.COMPLETED
    assert todo.model.asked == [] and todo.model.answers == []
    assert todo.workflow.sent == ["get_todo_list id=7", "get_properties",
                                  "go_up", "get_todo_list id=8", "get_properties"]
    items = [s for s in _phases(todo.sink) if s.attributes["phase"] == "each"]
    assert [s.attributes["facts"] for s in items] == [{"description": "chores"},
                                                      {"description": "errands"}]
    # The item record check ran on each item's span.
    assert all("record_issues" in s.attributes for s in items)
    assert [s.attributes["code_stop"] for s in items] == ["done", "done"]


@pytest.mark.parametrize("step", [
    "list_items (first page only; say how many of the total were shown)",
    "list_items",
    "list_items (the items it holds)",
])
def test_page_1_is_enough_unless_the_note_asks_for_the_full_list(paged, step):
    paged.plan = _plan(Phase("report", lambda: [step, "get_todo_list id=8"]))
    paged.turn()
    # More pages remain, but the note asks for none of them: code goes on.
    assert paged.model.asked == []
    assert paged.workflow.sent == ["list_items", "get_todo_list id=8"]
    assert paged.agent.trajectory["thought_0"] == "Step 1 of the plan, run by code."
    assert _phases(paged.sink)[0].attributes["code_stop"] == "done"


@pytest.mark.parametrize("note", ["(every page)", "(All pages of it)", "(page through it)",
                                  "(read until complete)"])
def test_each_full_list_phrase_hands_over_when_more_pages_remain(paged, note):
    paged.plan = _plan(Phase("report", lambda: [f"list_items {note}"]))
    paged.model.finish()
    paged.turn()
    assert _phases(paged.sink)[0].attributes["code_stop"] == "more pages at step 1"
    assert paged.workflow.sent == ["list_items"]


def test_a_planner_command_step_after_the_last_phase_is_left_to_the_agent(todo):
    todo.steps_after = ["get_all_children (what the list holds)"]
    todo.model.run("get_all_children")
    todo.model.finish()
    result = todo.turn()

    assert result.status == TurnStatus.COMPLETED
    report = _phases(todo.sink)[-1]
    assert report.attributes["steps"].endswith("3. get_all_children (what the list holds)")
    # Code ran the phase's own step, then handed the planner's step to the agent.
    assert report.attributes["code_stop"] == "open step 3"
    assert report.attributes["steps_by_code"] == 1
    assert len(todo.model.asked) == 2
    assert todo.workflow.sent[-2:] == ["get_properties", "get_all_children"]


def test_planner_answer_steps_after_the_last_phase_do_not_stop_code(todo):
    todo.steps_after = ["say how many items the list holds"]
    todo.turn()
    report = _phases(todo.sink)[-1]
    assert report.attributes["code_stop"] == "done"
    assert todo.model.asked == [] and len(todo.model.answers) == 1


# ----------------------------------------------------------------------
# Picks: an open step that opens the one match of the lookup before it (fix-6llp.5)
# ----------------------------------------------------------------------

def _pick(text, output, classes=(OpenParams,), lookup="find_book query=Dune"):
    return skill_phases.pick_step(_step(text), _step(lookup), output, list(classes))


@pytest.mark.parametrize("output, uid", [
    ({"labels": ["Dune"], "book_uids": ["b1"], "total": 1}, "b1"),
    ({"labels": ["Dune 1", " dune ", "Dune Messiah"], "book_uids": ["b1", "b2", "b3"]}, "b2"),
    ({"labels": ["Dune Messiah"], "book_uids": ["b3"]}, "b3"),        # the only row
    ({"labels": ["Dune", "Dune"], "book_uids": ["b1", "b2"]}, None),  # two exact matches
    ({"labels": ["Dune 1", "Dune 2"], "book_uids": ["b1", "b2"]}, None),  # none exact
    ({"labels": [], "book_uids": []}, None),                          # no rows
    ({"labels": ["Dune"]}, None),                                     # no uid column
    ({"labels": ["Dune"], "book_uids": ["b1"], "shelf_uids": ["s1"]}, None),  # two
    ("Dune: b1", None),                                               # no Output
])
def test_a_pick_opens_the_one_exact_match_or_the_only_row(output, uid):
    command, _ = _pick("open_book_by_uid with the uid of the Dune match", output)
    assert command == (f"open_book_by_uid book_uid={uid}" if uid else None)


def test_a_pick_names_the_lookup_value_and_has_one_required_uid():
    output = {"labels": ["Dune"], "book_uids": ["b1"]}
    assert _pick("open_book_by_uid with the uid of the match", output)[1] == \
        "names no lookup value"
    assert _pick("open_book_by_uid with the uid of the Emma match", output)[0] is None
    assert _pick("set_due for the Dune loan", output, (DueParams,))[1] == "not one required uid"
    assert _pick("open_book_by_uid with the uid of the Dune match", output,
                 (OpenParams, DueParams))[0] is None
    # A placeholder for the one uid is a pick like the prose form.
    assert _pick("open_book_by_uid book_uid=<the Dune match>", output)[0] == \
        "open_book_by_uid book_uid=b1"
    # A paging value of the lookup is never the named value.
    assert _pick("open_book_by_uid for Dune at 20", output,
                 lookup="find_book query=Dune start=20")[0] == "open_book_by_uid book_uid=b1"
    assert _pick("open_book_by_uid with the uid of the Dune match", output)[1] == "only row"


class FindListParams(BaseModel):
    query: Optional[str] = Field(default=None)
    start: Optional[int] = Field(default=None)


@pytest.fixture
def picks(todo):
    todo.toys["find_list"] = FindListParams
    todo.plan = _plan(Phase("open", lambda: [
        "find_list query=errands", "get_todo_list with the id of the errands match"]))
    return todo


def _found(picks, labels, ids, text="matches."):
    picks.workflow.scripted["find_list"] = (text, {"labels": labels, "list_ids": ids})


@pytest.mark.parametrize("labels, ids, opened, thought", [
    (["errands"], [8], 8, "the only row of step 1"),
    (["errands 1", "Errands", "weekly errands"], [7, 8, 9], 8,
     "the only exact match of step 1"),
    (["weekly errands"], [7], 7, "the only row of step 1"),
])
def test_code_runs_a_pick_of_the_one_match(picks, labels, ids, opened, thought):
    _found(picks, labels, ids)
    result = picks.turn()

    assert result.status == TurnStatus.COMPLETED
    assert picks.model.asked == []
    assert picks.workflow.sent == ["find_list query=errands", f"get_todo_list id={opened}"]
    span = _phases(picks.sink)[0]
    assert span.attributes["code_stop"] == "done" and span.attributes["steps_by_code"] == 2
    trajectory = picks.agent.trajectory
    assert f"Step 2 of the plan, run by code ({thought})." in trajectory.values()


@pytest.mark.parametrize("labels, ids, text", [
    (["errands", "Errands"], [7, 8], "matches."),                 # two exact matches
    (["chores", "weekly errands"], [7, 8], "matches."),           # none exact
    (["errands"], [8], "1-10 of 12. Next page: start=10."),       # more pages
])
def test_the_agent_picks_when_no_one_row_matches(picks, labels, ids, text):
    _found(picks, labels, ids, text)
    picks.model.run("get_todo_list id=8")
    picks.model.finish()
    result = picks.turn()

    assert result.status == TurnStatus.COMPLETED
    assert len(picks.model.asked) == 2
    assert _phases(picks.sink)[0].attributes["code_stop"] == "open step 2"
    assert picks.workflow.sent == ["find_list query=errands", "get_todo_list id=8"]
