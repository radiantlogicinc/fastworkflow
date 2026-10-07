"""Recorded pass identity and pass content for distillation (`fix-txxy`).

Distillation runs the agent TWICE for one user message — a teacher pass and a
student pass — inside ONE turn. Both passes therefore write to one trace id and
one turn row, and until now nothing recorded which activity was whose:
`comparison.PassSelector` resolves pass membership only from recorded spans, and
there was nothing recorded to resolve against, so `discover_pass_selectors`
answered `[]` for every real turn and the two passes could only be compared by a
caller hand-building a span list from out-of-band knowledge.

What is under test is the PRODUCER and the binding that carries it, not a helper
in isolation. Each test below drives the real `WorkflowExecutionContext` through
`process_turn`, which opens the real turn, routes into `_process_agent_message`,
and reaches `distill_message` exactly the way the CLI does — into a real
`SQLiteTraceSink`, from which the assertions read the spans back out of SQLite.
A test that called `_run_agent_pass` directly would pass just as happily against
a producer nothing in production ever binds to a recorder.

Scripting stops at the LLM boundary and nowhere else. The agent, the planner and
the insight extractor are scripted because calling them costs money; the command
dispatches they drive are real (`_execute_workflow_query` into the real
`CommandExecutor`, against the real `todo_list_workflow`), and the one
`fw.llm.call` span a scripted pass emits goes through the same `tracing`
emission path, host binding and parenting stack that DSPy's callback uses for a
real call. So the span tree these tests read is the shape a paid run records.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from contextlib import suppress
from pathlib import Path
from queue import Queue

import dspy
import pytest
from dspy.dsp.utils.utils import dotdict

import fastworkflow
from fastworkflow import state_paths, tracing
from fastworkflow.benchmark import setup
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.distillation import DistillationSession, PlanningStep
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import comparison as comparison_module
from fastworkflow.observability import store as obs
from fastworkflow.observability.comparison import (
    ATTRIBUTION_PASS,
    ATTRIBUTION_TURN,
    ExecutionRef,
    PassSelector,
    StoreExecutionReader,
    discover_pass_selectors,
    project_execution,
)
from fastworkflow.run_chatbot import selection_api
from fastworkflow.run_chatbot import server as run_chatbot_server
from fastworkflow.run_chatbot.server import cost_rollup, execution_ledger
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

# The recorder-side seeding helpers, reused rather than restated (the same
# import `tests/test_selection_api.py` makes): attempt 2 below is a turn a
# recorder wrote, not a fixture the reader is taught to recognise.
from tests.test_execution_comparison import (
    _execute_span,
    _output,
    _record,
    _turn_row,
    _write,
)
from tests.todo_list_workflow.application.todo_manager import TodoListManager

LIST_COMMAND = "TodoListManager/list_todo_lists"
SAVE_COMMAND = "TodoListManager/save_lists"

# What the scripted agent asks the workflow to do, and what the stand-in for the
# untrained CME wildcard resolves each one to. The agent tool takes command
# names (`_explicit_agent_command` resolves the token against the context's real
# command surface, which is why these are real names), and only the parameter
# extraction the test workflows cannot run is replaced.
DISPATCHES = {
    "list_todo_lists": (LIST_COMMAND, {}),
    "save_lists": (SAVE_COMMAND, {}),
}

TEACHER_ANSWER = "you have 0 lists; I saved them anyway"
STUDENT_ANSWER = "you have 0 lists"


# ----------------------------------------------------------------------
# A scripted pass: real dispatches, real spans, no paid calls
# ----------------------------------------------------------------------


class _PassScript:
    """One pass's behaviour: what it dispatches, plans, answers and costs.

    `exhausted` and `suspended` are the two ways the real agent stops WITHOUT
    raising: it ran out of iterations, or it stopped to ask the user. Both
    return a final answer object, so a producer that reads "no exception" as
    "completed" records them as clean runs.
    """

    def __init__(
        self,
        commands: list[str],
        plan: list[str],
        answer: str,
        cost: float,
        fails: bool = False,
        exhausted: bool = False,
        suspended: bool = False,
        dspy_call: bool = False,
    ) -> None:
        self.commands = commands
        self.plan = plan
        self.answer = answer
        self.cost = cost
        self.fails = fails
        self.exhausted = exhausted
        self.suspended = suspended
        self.dspy_call = dspy_call


class _ScriptedAgent:
    """Stands in for the DSPy ReAct agent for exactly one pass.

    The script is claimed when the agent is CALLED, not when it is built: a turn
    builds one agent before distillation starts (`_ensure_agent_initialized`)
    that distillation then replaces and never invokes, so binding by
    construction order would hand the teacher's script to an agent nobody runs.
    """

    def __init__(self, chat_session, pending: list[_PassScript]) -> None:
        self._chat_session = chat_session
        self._pending = pending
        self.current_trajectory: dict = {}

    def __call__(self, **_kwargs):
        from fastworkflow.workflow_agent import _execute_workflow_query

        script = self._pending.pop(0)
        if script.dspy_call:
            # A real DSPy call against a real local LM: the `fw.llm.call` span
            # is then written by DSPy's own callback, which the WEC installed
            # for this turn, rather than by this test.
            with dspy.context(lm=_LocalEchoLM()):
                dspy.Predict("question -> answer")(question="how many lists?")
        else:
            # The LLM call a real pass makes, emitted through the real seam:
            # same host, same parenting stack, same sink. This is the span whose
            # cost a pass roll-up is supposed to claim.
            span = tracing.start_span(
                self._chat_session,
                tracing.SPAN_LLM_CALL,
                kind=tracing.KIND_LLM,
                attributes={"model": "scripted-pass-model"},
            )
            tracing.end_span(
                self._chat_session, span, attributes={"cost": script.cost}
            )

        for command in script.commands:
            _execute_workflow_query(command, self._chat_session)

        self.current_trajectory = {"thought_0": f"scripted: {script.answer}"}
        if script.fails:
            raise RuntimeError("scripted student pass failure")
        return type(
            "AgentResult",
            (),
            {
                "final_answer": script.answer,
                "exhausted": script.exhausted,
                "suspended": script.suspended,
            },
        )()


def _script_the_llm_boundaries(monkeypatch, ctx, scripts: list[_PassScript]) -> None:
    """Replace every paid call on the distillation path, and nothing else."""
    pending = list(scripts)
    plans = [script.plan for script in scripts]

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.initialize_workflow_tool_agent",
        lambda chat_session, **_kwargs: _ScriptedAgent(chat_session, pending),
    )

    def scripted_planner(user_query, session, planning_insights=None, planner_lm=None):
        # The real planner's hook appends to this list as it plans; the capture
        # is what `_run_agent_pass` records as the pass's plan.
        capture = getattr(session, "_planning_steps_capture", None)
        if capture is not None and plans:
            capture.append(
                PlanningStep(
                    step_number=0,
                    user_query=user_query,
                    generated_plan=plans.pop(0),
                    reasoning="scripted",
                )
            )
        return user_query

    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps", scripted_planner
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do", lambda session: "commands"
    )
    monkeypatch.setattr(
        "fastworkflow.utils.dspy_utils.get_lm", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        ctx, "_call_agent_with_retry", lambda agent_call, lm=None: agent_call()
    )
    # Conversation summarization is an LLM call per pass; its span would be a
    # real part of the pass, but running it is not free.
    monkeypatch.setattr(
        ctx, "summarize_and_record_turn", lambda *args, **kwargs: ("summary", None)
    )
    monkeypatch.setattr(
        DistillationSession, "extract_planning_insights", lambda *a, **k: []
    )
    monkeypatch.setattr(DistillationSession, "extract_insights", _scripted_extraction)


def _scripted_extraction(self, *_args, **_kwargs) -> list[str]:
    """Insight extraction: one real LLM span, belonging to neither pass.

    Emitted through the same seam as the passes' own calls, so the only thing
    keeping it out of both passes is WHERE it happens — after both pass spans
    have closed — which is exactly the property under test.
    """
    span = tracing.start_span(
        self.chat_session,
        tracing.SPAN_LLM_CALL,
        kind=tracing.KIND_LLM,
        attributes={"model": "extraction-model"},
    )
    tracing.end_span(self.chat_session, span, attributes={"cost": 0.30})
    return []


def _cme_wildcard_stand_in(monkeypatch, app_workflow) -> None:
    """Resolve the agent's raw text the way the trained CME wildcard would.

    Same stand-in shape as `tests/test_span_contract_versioning.py`: the real
    hop resolves text into an Action and calls `perform_action` again, so the
    dispatch underneath — and every span it emits — stays real.
    """
    real_perform_action = CommandExecutor.perform_action

    def hop(cls, workflow, action):
        raw = (action.command or "").strip()
        command_name, parameters = DISPATCHES[raw.split(maxsplit=1)[0]]
        command_output = real_perform_action(
            app_workflow,
            fastworkflow.Action(
                command_name=command_name, command=raw, parameters=parameters
            ),
        )
        command_output.command_response.artifacts["command_handled"] = True
        command_output.command_name = command_name
        return command_output

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(hop))


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "observability.sqlite3")


def _distillation_context(
    todo_workflow_path: str, tmp_path: Path, sink
) -> WorkflowExecutionContext:
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"distill-pass-{uuid.uuid4().hex}"
    )
    ctx = WorkflowExecutionContext(
        run_as_agent=True, generate_insights=True, trace_sink=sink
    )
    ctx.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    # Distillation is guarded on a user_message_queue (Topology A / CLI only).
    ctx.set_transport_queues(user_message_queue=Queue())
    return ctx


TEACHER_SCRIPT = _PassScript(
    commands=["list_todo_lists", "save_lists"],
    plan=["list the lists", "save them"],
    answer=TEACHER_ANSWER,
    cost=0.10,
)
STUDENT_SCRIPT = _PassScript(
    commands=["list_todo_lists"],
    plan=["list the lists"],
    answer=STUDENT_ANSWER,
    cost=0.01,
)


def _run_distillation_turn(
    monkeypatch,
    todo_workflow_path: str,
    tmp_path: Path,
    db_path: str,
    scripts: list[_PassScript] | None = None,
    identity: dict | None = None,
) -> str:
    """One real distilled turn into a real store. Returns its turn_key.

    `identity` is the embedder's binding (channel, conversation, experiment
    labels), bound before the turn the way FastAPI and the in-process harness
    bind it, so the recorded turn is an experiment attempt's evidence and the
    selection API can be asked for it.
    """
    sink = obs.SQLiteTraceSink(db_path)
    ctx = _distillation_context(todo_workflow_path, tmp_path, sink)
    if identity:
        ctx.bind_observability_identity(**identity)
    try:
        _cme_wildcard_stand_in(monkeypatch, ctx.app_workflow)
        _script_the_llm_boundaries(
            monkeypatch, ctx, scripts or [TEACHER_SCRIPT, STUDENT_SCRIPT]
        )
        turn = ctx.process_turn("tidy up my lists")
        return turn.turn_key
    finally:
        with suppress(Exception):
            ctx.close()
        sink.close()  # drains pending writes


def _reader(db_path: str) -> StoreExecutionReader:
    return StoreExecutionReader("store-1", obs.ObservabilityStore(db_path))


def _project(ref: ExecutionRef, reader, **kwargs):
    return project_execution(
        ref, reader, ledger=execution_ledger, cost_rollup=cost_rollup, **kwargs
    )


def _pass_projection(reader, turn_key: str, pass_id: str):
    return _project(
        ExecutionRef(store_id="store-1", turn_keys=(turn_key,), pass_id=pass_id),
        reader,
        pass_selector=PassSelector(
            pass_id=pass_id,
            attribute_key=tracing.ATTR_PASS,
            attribute_value=pass_id,
        ),
    )


# ----------------------------------------------------------------------
# The producer records passes, in a turn a recorder is actually bound to
# ----------------------------------------------------------------------


def test_a_distilled_turn_records_one_span_per_pass(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """The stamp has to survive the whole production path into SQLite.

    `distill_message` is reached from `_process_agent_message`, inside the turn
    `_begin_turn` already opened, so the pass spans carry that turn's trace id
    and land in the store beside everything else the turn recorded.
    """
    turn_key = _run_distillation_turn(
        monkeypatch, todo_workflow_path, tmp_path, db_path
    )

    spans = _reader(db_path).trace("store-1", turn_key)
    pass_spans = [
        span for span in spans if span["name"] == tracing.SPAN_DISTILLATION_PASS
    ]
    assert len(pass_spans) == 2
    stamped = {json.loads(span["attributes"])[tracing.ATTR_PASS] for span in pass_spans}
    assert stamped == {tracing.PASS_TEACHER, tracing.PASS_STUDENT}
    # The stamp is on the pass span only. Membership below is resolved by
    # walking ancestry, so copying it onto every descendant would be a second
    # thing to keep in step for no gain.
    assert len(
        [span for span in spans if tracing.ATTR_PASS in json.loads(span["attributes"])]
    ) == 2


def test_both_passes_are_discovered_without_a_caller_supplied_span_list(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """`fix-txxy`'s acceptance criterion, read off recorded evidence.

    Before this, a caller had to hand-build `span_ids` from out-of-band
    knowledge of how distillation runs; `discover_pass_selectors` reports what
    was stamped and nothing else.
    """
    turn_key = _run_distillation_turn(
        monkeypatch, todo_workflow_path, tmp_path, db_path
    )
    reader = _reader(db_path)

    selectors = discover_pass_selectors(
        reader.trace("store-1", turn_key), attribute_key=tracing.ATTR_PASS
    )
    assert [selector.pass_id for selector in selectors] == [
        tracing.PASS_STUDENT,
        tracing.PASS_TEACHER,
    ]
    assert all(
        selector.attribute_key == tracing.ATTR_PASS and not selector.span_ids
        for selector in selectors
    )


def test_each_pass_projects_its_own_dispatches(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """Membership travels down the span tree: a dispatch is in the pass that
    ran it because its span sits under that pass's span, not because of when it
    happened."""
    turn_key = _run_distillation_turn(
        monkeypatch, todo_workflow_path, tmp_path, db_path
    )
    reader = _reader(db_path)

    teacher = _pass_projection(reader, turn_key, tracing.PASS_TEACHER)
    student = _pass_projection(reader, turn_key, tracing.PASS_STUDENT)

    assert [step.command_name for step in teacher.steps] == [
        LIST_COMMAND,
        SAVE_COMMAND,
    ]
    assert [step.command_name for step in student.steps] == [LIST_COMMAND]
    assert all(step.pass_id == tracing.PASS_TEACHER for step in teacher.steps)
    assert not (
        {step.command_call_id for step in teacher.steps}
        & {step.command_call_id for step in student.steps}
    )
    # The other pass's dispatches are left out of this view and counted, not
    # dropped.
    assert any(
        step.command_call_id in {s.command_call_id for s in student.steps}
        for step in teacher.unassigned_steps
    )

    # The partition is exact: every dispatch the span tree places in a pass is
    # in exactly one of the two views, and the ledger rows with no span of
    # their own -- the inner hops -- are in neither, because nothing recorded
    # which pass ran them.
    whole = _project(ExecutionRef(store_id="store-1", turn_keys=(turn_key,)), reader)
    span_less = [step for step in whole.steps if not step.span_recorded]
    assert span_less, "the inner CME hops are the unattributable case here"
    assert len(teacher.steps) + len(student.steps) + len(span_less) == len(whole.steps)


def test_each_pass_reports_the_answer_it_actually_produced(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """The point of the whole exercise: two different answers, each attributed.

    One turn row holds one answer. Quoting it under both headings — which is
    what a pass-scoped projection had to do before the producer recorded
    anything — hides the divergence a teacher/student comparison is read for.
    """
    turn_key = _run_distillation_turn(
        monkeypatch, todo_workflow_path, tmp_path, db_path
    )
    reader = _reader(db_path)

    teacher = _pass_projection(reader, turn_key, tracing.PASS_TEACHER)
    student = _pass_projection(reader, turn_key, tracing.PASS_STUDENT)

    assert teacher.answers()[0]["answer"] == TEACHER_ANSWER
    assert student.answers()[0]["answer"] == STUDENT_ANSWER
    for projection in (teacher, student):
        answer = projection.answers()[0]
        assert answer["attribution"] == ATTRIBUTION_PASS
        assert answer["pass_content_recorded"] is True
        assert projection.content_attribution == ATTRIBUTION_PASS
    # The shared turn row's answer is kept beside the pass's, not replaced by
    # it: the two passes agree about what the turn recorded and disagree about
    # what each of them said.
    assert teacher.turns[0].turn_answer == student.turns[0].turn_answer

    whole = _project(ExecutionRef(store_id="store-1", turn_keys=(turn_key,)), reader)
    assert whole.content_attribution == ATTRIBUTION_TURN
    assert whole.turns[0].pass_content_recorded is False


def test_each_pass_reports_the_plan_it_actually_generated(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """Planning is per pass too, and nothing else records it: the planner's
    steps live in process and the turn row has no column for them."""
    turn_key = _run_distillation_turn(
        monkeypatch, todo_workflow_path, tmp_path, db_path
    )
    reader = _reader(db_path)

    teacher = json.loads(_pass_projection(reader, turn_key, tracing.PASS_TEACHER).turns[0].plan)
    student = json.loads(_pass_projection(reader, turn_key, tracing.PASS_STUDENT).turns[0].plan)

    assert [step["generated_plan"] for step in teacher] == [TEACHER_SCRIPT.plan]
    assert [step["generated_plan"] for step in student] == [STUDENT_SCRIPT.plan]


def test_a_pass_reports_its_own_measured_wall_time(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """A pass span has a start and an end, so its duration is measured rather
    than apportioned out of the turn's. The shared turn figure stays beside
    it."""
    turn_key = _run_distillation_turn(
        monkeypatch, todo_workflow_path, tmp_path, db_path
    )
    reader = _reader(db_path)

    teacher = _pass_projection(reader, turn_key, tracing.PASS_TEACHER)
    assert teacher.timing["wall_ms"] is not None
    assert teacher.timing["wall_ms_attribution"] == ATTRIBUTION_PASS
    assert teacher.timing["pass_wall_ms_turns_recorded"] == 1
    assert teacher.timing["turn_wall_ms"] is not None
    # A pass cannot have taken longer than the turn that contains both passes.
    assert teacher.timing["wall_ms"] <= teacher.timing["turn_wall_ms"]


def test_insight_extraction_is_attributed_to_neither_pass(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """`fix-txxy`'s other acceptance criterion.

    Extraction analyses both passes, so charging it to either would make one
    model look more expensive for having been compared. It runs after both pass
    spans close, which puts it under the turn root and inside neither subtree.
    """
    turn_key = _run_distillation_turn(
        monkeypatch, todo_workflow_path, tmp_path, db_path
    )
    reader = _reader(db_path)

    teacher = _pass_projection(reader, turn_key, tracing.PASS_TEACHER)
    student = _pass_projection(reader, turn_key, tracing.PASS_STUDENT)
    whole = _project(ExecutionRef(store_id="store-1", turn_keys=(turn_key,)), reader)

    assert teacher.cost["total"] == pytest.approx(0.10)
    assert student.cost["total"] == pytest.approx(0.01)
    # The 0.30 extraction call is in the turn and in neither pass.
    assert whole.cost["total"] == pytest.approx(0.41)


def test_a_failed_pass_records_its_own_failure(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """A student pass that raises is caught and the turn carries on with the
    teacher's result — so the turn row describes the teacher. The failure is
    recorded where it belongs instead of disappearing into a successful turn."""
    failing_student = _PassScript(
        commands=["list_todo_lists"],
        plan=["list the lists"],
        answer=STUDENT_ANSWER,
        cost=0.01,
        fails=True,
    )
    turn_key = _run_distillation_turn(
        monkeypatch,
        todo_workflow_path,
        tmp_path,
        db_path,
        scripts=[TEACHER_SCRIPT, failing_student],
    )
    reader = _reader(db_path)

    student = _pass_projection(reader, turn_key, tracing.PASS_STUDENT).turns[0]
    assert student.status == "failed"
    assert student.failure_reason == "RuntimeError"
    # It produced no answer, and the turn's is not borrowed to fill the gap.
    assert student.answer is None
    whole = _project(ExecutionRef(store_id="store-1", turn_keys=(turn_key,)), reader)
    assert student.turn_answer == whole.turns[0].answer

    teacher = _pass_projection(reader, turn_key, tracing.PASS_TEACHER).turns[0]
    assert teacher.status == "completed"
    assert teacher.answer == TEACHER_ANSWER
    # Whole-turn success is a turn-level code; a pass-scoped view does not
    # re-label it as the pass's.
    assert teacher.success is None


def test_a_turn_with_no_distillation_records_no_pass_and_projects_whole(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """Ordinary turns are untouched: one pass-less execution, as before.

    This is also the shape of every trace already in an evidence store, so it is
    the compatibility half of the acceptance criteria.
    """
    sink = obs.SQLiteTraceSink(db_path)
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"distill-none-{uuid.uuid4().hex}"
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    try:
        turn = ctx.process_action_turn(
            fastworkflow.Action(
                command_name=LIST_COMMAND, command="list my todo lists", parameters={}
            )
        )
        turn_key = turn.turn_key
    finally:
        with suppress(Exception):
            ctx.close()
        sink.close()

    reader = _reader(db_path)
    assert (
        discover_pass_selectors(
            reader.trace("store-1", turn_key), attribute_key=tracing.ATTR_PASS
        )
        == []
    )
    whole = _project(ExecutionRef(store_id="store-1", turn_keys=(turn_key,)), reader)
    assert whole.readable
    assert whole.content_attribution == ATTRIBUTION_TURN
    assert whole.turns[0].pass_content_recorded is False
    assert whole.turns[0].plan is None
    assert whole.timing["wall_ms_attribution"] == ATTRIBUTION_TURN


# ----------------------------------------------------------------------
# The two spellings of one contract
# ----------------------------------------------------------------------


def test_the_reader_and_the_producer_agree_on_the_pass_contract():
    """`comparison.py` restates the span name and its content keys rather than
    importing `tracing` (the module reads stored evidence and stays off the
    runtime's import path). A restatement that drifts reads every recorded pass
    as an unrecorded one — silently, because "no pass content" is a legitimate
    answer — so the two spellings are checked against each other here."""
    assert comparison_module.SPAN_DISTILLATION_PASS == tracing.SPAN_DISTILLATION_PASS
    contract = tracing.SPAN_CONTRACTS[tracing.SPAN_DISTILLATION_PASS]
    assert set(comparison_module.PASS_CONTENT_KEYS) | {tracing.ATTR_PASS} == set(
        contract.attributes
    )


def test_an_unstamped_pass_id_records_nothing(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """A pass with no id is not a pass.

    `_run_agent_pass` is reachable without one, and inferring "teacher" from an
    LM role would be the confident wrong attribution this seam exists to
    prevent: the caller that did not name a pass gets no pass recorded.
    """
    sink = obs.SQLiteTraceSink(db_path)
    ctx = _distillation_context(todo_workflow_path, tmp_path, sink)
    try:
        _cme_wildcard_stand_in(monkeypatch, ctx.app_workflow)
        _script_the_llm_boundaries(monkeypatch, ctx, [TEACHER_SCRIPT])
        ctx.push_active_workflow(ctx.app_workflow)
        ctx._begin_turn("tidy up my lists")
        turn_key = ctx.current_turn_key
        DistillationSession(ctx)._run_agent_pass(
            "tidy up my lists",
            agent_lm_role="LLM_TEACHER_AGENT",
            agent_api_key_role="LITELLM_API_KEY_TEACHER_AGENT",
            planner_lm_role="LLM_TEACHER_PLANNER",
            planner_api_key_role="LITELLM_API_KEY_TEACHER_PLANNER",
        )
    finally:
        ctx.clear_workflow_stack()
        with suppress(Exception):
            ctx.close()
        sink.close()

    spans = _reader(db_path).trace("store-1", turn_key)
    assert spans, "the pass ran and its dispatches were recorded"
    assert not [
        span for span in spans if span["name"] == tracing.SPAN_DISTILLATION_PASS
    ]
    assert discover_pass_selectors(spans, attribute_key=tracing.ATTR_PASS) == []


# ----------------------------------------------------------------------
# What "this pass ended" is allowed to mean
# ----------------------------------------------------------------------


def test_a_pass_that_ran_out_of_iterations_is_not_recorded_as_completed(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """The agent's two silent stops, recorded as what they were.

    `exhausted` (out of iterations) and `suspended` (stopped to ask the user)
    both return a final answer and raise nothing. A producer that recorded
    "completed" whenever no exception escaped would report a truncated pass as
    a clean one — and a teacher/student comparison reading those statuses would
    see two completed passes with different answers and conclude the models
    disagreed, when one of them simply never finished.
    """
    exhausted_student = _PassScript(
        commands=["list_todo_lists"],
        plan=["list the lists"],
        answer="I was still working",
        cost=0.01,
        exhausted=True,
    )
    turn_key = _run_distillation_turn(
        monkeypatch,
        todo_workflow_path,
        tmp_path,
        db_path,
        scripts=[TEACHER_SCRIPT, exhausted_student],
    )
    reader = _reader(db_path)

    student = _pass_projection(reader, turn_key, tracing.PASS_STUDENT).turns[0]
    assert student.status == "failed"
    assert student.failure_reason == "max_iters_exhausted"
    # The answer it did produce is still the pass's own and still recorded:
    # "did not finish" is not "said nothing".
    assert student.answer == "I was still working"
    assert _pass_projection(reader, turn_key, tracing.PASS_TEACHER).turns[
        0
    ].status == "completed"


def test_a_suspended_pass_records_that_it_stopped_to_ask(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """`awaiting_user` is the turn vocabulary's word for it, deliberately reused
    so a pass and a turn cannot disagree about what happened."""
    asking_student = _PassScript(
        commands=["list_todo_lists"],
        plan=["ask which list"],
        answer="which list did you mean?",
        cost=0.01,
        suspended=True,
    )
    turn_key = _run_distillation_turn(
        monkeypatch,
        todo_workflow_path,
        tmp_path,
        db_path,
        scripts=[TEACHER_SCRIPT, asking_student],
    )

    student = _pass_projection(
        _reader(db_path), turn_key, tracing.PASS_STUDENT
    ).turns[0]
    assert student.status == "awaiting_user"
    assert student.failure_reason is None


def test_an_unclassified_pass_outcome_is_recorded_as_unknown(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """The pass wrapper does not fill in an outcome the body never observed.

    Reached here by running a pass whose body never gets as far as recording
    one — the agent factory itself fails, before the status write — which is
    the shape of every future path through this method that forgets to. The
    honest record is "unknown", not the optimistic default.
    """
    sink = obs.SQLiteTraceSink(db_path)
    ctx = _distillation_context(todo_workflow_path, tmp_path, sink)
    try:
        _script_the_llm_boundaries(monkeypatch, ctx, [TEACHER_SCRIPT])
        ctx.push_active_workflow(ctx.app_workflow)
        ctx._begin_turn("tidy up my lists")
        turn_key = ctx.current_turn_key
        session = DistillationSession(ctx)
        with session._pass_span(tracing.PASS_TEACHER, "scripted-pass-model"):
            pass
    finally:
        ctx.clear_workflow_stack()
        with suppress(Exception):
            ctx.close()
        sink.close()

    # Read off the span rather than a projection: this pass ran inside a turn
    # that was opened and never completed, so there is no turn row to project.
    spans = _reader(db_path).trace("store-1", turn_key)
    recorded = json.loads(
        next(
            span
            for span in spans
            if span["name"] == tracing.SPAN_DISTILLATION_PASS
        )["attributes"]
    )
    assert recorded["status"] == tracing.PASS_STATUS_UNKNOWN
    assert recorded["status"] not in {"completed", "failed", "awaiting_user"}
    assert "answer" not in recorded


# ----------------------------------------------------------------------
# What a recorded pass is allowed to carry
# ----------------------------------------------------------------------


def test_the_recorded_plan_is_the_user_visible_plan(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """A plan is the next-step sequence, not the planner's reasoning.

    `PlanningStep` carries `reasoning` as well, and the UI renders this field
    as "the plan": folding chain-of-thought into it would move agent-internal
    text onto a user-visible surface without anything having decided that, and
    a reader comparing two passes' plans would be reading two essays.
    """
    turn_key = _run_distillation_turn(
        monkeypatch, todo_workflow_path, tmp_path, db_path
    )
    recorded = json.loads(
        _pass_projection(_reader(db_path), turn_key, tracing.PASS_TEACHER)
        .turns[0]
        .plan
    )

    assert recorded == [{"step_number": 0, "generated_plan": TEACHER_SCRIPT.plan}]
    assert all("reasoning" not in step for step in recorded)
    # The scripted planner really did produce reasoning: the absence above is
    # the producer's choice, not an empty input.
    assert PlanningStep(
        step_number=0, user_query="q", generated_plan=[], reasoning="scripted"
    ).reasoning


def test_each_pass_reports_only_its_own_artifacts(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """Artifacts follow their dispatch, so pass-scoped steps scope them too.

    The teacher ran a command the student never did. Showing that command's
    output under both headings would credit the student with work it did not
    do — the artifact-level form of quoting one turn answer twice.
    """
    turn_key = _run_distillation_turn(
        monkeypatch, todo_workflow_path, tmp_path, db_path
    )
    reader = _reader(db_path)

    teacher = _pass_projection(reader, turn_key, tracing.PASS_TEACHER)
    student = _pass_projection(reader, turn_key, tracing.PASS_STUDENT)

    assert {artifact.command_name for artifact in teacher.artifacts} == {
        LIST_COMMAND,
        SAVE_COMMAND,
    }
    assert {artifact.command_name for artifact in student.artifacts} == {LIST_COMMAND}
    for projection in (teacher, student):
        calls = {step.command_call_id for step in projection.steps}
        assert projection.artifacts
        assert all(
            artifact.command_call_id in calls for artifact in projection.artifacts
        )
        assert all(
            artifact.attribution == ATTRIBUTION_PASS
            for artifact in projection.artifacts
        )
    assert not (
        {artifact.command_call_id for artifact in teacher.artifacts}
        & {artifact.command_call_id for artifact in student.artifacts}
    )
    # Whole-turn reading is unchanged: both passes' artifacts, attributed to
    # the turn, which is what every pre-`fix-txxy` trace still reads as.
    whole = _project(ExecutionRef(store_id="store-1", turn_keys=(turn_key,)), reader)
    assert len(whole.artifacts) >= len(teacher.artifacts) + len(student.artifacts)
    assert all(
        artifact.attribution == ATTRIBUTION_TURN for artifact in whole.artifacts
    )


def test_a_recorded_pass_answer_is_scrubbed_like_a_turn_answer(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """Pass content is a span attribute, and must not be a softer route.

    A turn's `answer` column is scrubbed of credential shapes at the sink
    boundary. Pass answers are agent text from the same source, so recording
    them somewhere that skipped that scrub would make the new field the way a
    leaked key reaches disk. Span attributes go through the same `Redactor`, and
    this pins it rather than trusting the read of the store's code.
    """
    leaky_teacher = _PassScript(
        commands=["list_todo_lists"],
        plan=["list the lists"],
        answer="I authenticated with sk-abcdefghijklmnopqrstuvwx and listed them",
        cost=0.10,
    )
    turn_key = _run_distillation_turn(
        monkeypatch,
        todo_workflow_path,
        tmp_path,
        db_path,
        scripts=[leaky_teacher, STUDENT_SCRIPT],
    )

    teacher = _pass_projection(
        _reader(db_path), turn_key, tracing.PASS_TEACHER
    ).turns[0]
    assert "sk-abcdefghijklmnopqrstuvwx" not in teacher.answer
    assert "[REDACTED]" in teacher.answer
    # Nowhere else in the database either: the span row is the only new place
    # this text is written, and the turn row's own scrub is the baseline.
    assert b"sk-abcdefghijklmnopqrstuvwx" not in Path(db_path).read_bytes()


# ----------------------------------------------------------------------
# The two fields that carry text
# ----------------------------------------------------------------------


def test_a_capped_pass_answer_and_plan_keep_a_visible_prefix(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """Over-limit text is cut, and the cut is stated rather than implied.

    The tracing attribute cap leaves its own envelope. Both fields have to come
    back as a quoted prefix plus a recorded original length, or a reader
    compares two passes on the first 64 bytes of each and never learns that is
    what they are looking at.
    """
    monkeypatch.setattr(tracing, "MAX_ATTR_BYTES", 64)
    long_answer = "I checked every list and here is the full rundown: " + "x" * 400
    long_plan = "step one, at considerable length: " + "y" * 400
    turn_key = _run_distillation_turn(
        monkeypatch,
        todo_workflow_path,
        tmp_path,
        db_path,
        scripts=[
            _PassScript(
                commands=["list_todo_lists"],
                plan=[long_plan],
                answer=long_answer,
                cost=0.10,
            ),
            STUDENT_SCRIPT,
        ],
    )

    teacher = _pass_projection(
        _reader(db_path), turn_key, tracing.PASS_TEACHER
    ).turns[0]
    for field, projected, produced in (
        # The plan is recorded as JSON text, so the value that was cut is that
        # serialization rather than the bare planning line.
        ("answer", teacher.answer, long_answer),
        ("plan", teacher.plan, json.dumps([{"step_number": 0, "generated_plan": [long_plan]}])),
    ):
        envelope = teacher.pass_content[field]
        assert envelope["truncated"] is True, field
        assert envelope["original_length"] > 64
        assert envelope["sha256"]
        # The prefix is recorded evidence and is quoted; it is not the whole
        # value, and the envelope beside it is how a reader knows.
        assert projected == envelope["value"]
        assert projected and produced.startswith(projected)
        assert len(projected) < len(produced)


# ----------------------------------------------------------------------
# Through the API a client actually reads
# ----------------------------------------------------------------------


def test_the_selection_api_serves_each_pass_its_own_recorded_answer(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """The whole point, end to end: producer to HTTP handler.

    The comparison UI reads passes through these two routes and nothing else.
    Recording pass identity is only useful if `.../runs/<n>/passes` can now
    NAME the two passes — it answered `[]` for every real turn before — and if
    `.../runs/<n>?left_pass=` then serves each one the answer and plan that
    pass actually produced.

    A real experiment attempt, declared and finished through the real
    `ExperimentController`, whose evidence is the distilled turn recorded
    above. Nothing about the attempt is fabricated: the API resolves the
    attempt, the store and the pass exactly as it does for a benchmark run.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    db_path = state_paths.observability_db(todo_workflow_path)
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(
        todo_workflow_path,
        store.store_identity(),
        external=False,
    )
    experiment_id = f"exp-{uuid.uuid4().hex}"
    task_id = f"task_{uuid.uuid4().hex}"
    channel = f"ch-pass-{uuid.uuid4().hex[:8]}"
    controller.create_experiment(
        experiment_id,
        "teacher/student pass capture",
        declared_tasks=1,
        declared_attempts=1,
        declarations=[(task_id, 1, channel)],
        workflow_name=setup.workflow_name_for(todo_workflow_path),
    )
    conversation = store.mint_conversation_id(
        channel, experiment_id=experiment_id, task_id=task_id, attempt=1
    )
    controller.start_attempt(
        experiment_id, task_id, 1, channel, conversation_id=conversation
    )
    _run_distillation_turn(
        monkeypatch,
        todo_workflow_path,
        tmp_path,
        db_path,
        identity={
            "channel_id": channel,
            "conversation_id": conversation,
            "experiment_id": experiment_id,
            "task_id": task_id,
            "attempt": 1,
        },
    )
    controller.finish_attempt(
        experiment_id, task_id, 1, outcome="pass", outcome_source="derived"
    )

    run = f"/api/experiments/{experiment_id}/tasks/{task_id}/runs/1"

    status, listing = selection_api.handle_get(todo_workflow_path, f"{run}/passes", {})
    assert status == 200
    assert listing["pass_attribute"] == tracing.ATTR_PASS
    assert {entry["pass_id"] for entry in listing["passes"]} == {
        tracing.PASS_TEACHER,
        tracing.PASS_STUDENT,
    }
    assert all(entry["turn_count"] == 1 for entry in listing["passes"])

    served = {}
    for pass_id in (tracing.PASS_TEACHER, tracing.PASS_STUDENT):
        status, body = selection_api.handle_get(
            todo_workflow_path,
            run,
            {"left_pass": [pass_id], "view": ["answers"]},
        )
        assert status == 200
        assert body["pass_id"] == pass_id
        assert body["pass_turns_omitted"] == []
        served[pass_id] = body["projection"]

    assert served[tracing.PASS_TEACHER]["answers"][0]["answer"] == TEACHER_ANSWER
    assert served[tracing.PASS_STUDENT]["answers"][0]["answer"] == STUDENT_ANSWER
    for pass_id, projection in served.items():
        answer = projection["answers"][0]
        assert answer["attribution"] == ATTRIBUTION_PASS
        assert answer["pass_content_recorded"] is True
        assert answer["pass_id"] == pass_id
        turn = projection["turns"][0]
        assert json.loads(turn["plan"])
        assert turn["turn_answer"] == served[tracing.PASS_TEACHER]["turns"][0][
            "turn_answer"
        ]
    assert (
        json.loads(served[tracing.PASS_TEACHER]["turns"][0]["plan"])
        != json.loads(served[tracing.PASS_STUDENT]["turns"][0]["plan"])
    )
    # `ref_id` is the pass's, not the attempt's: a comment written about the
    # teacher pass must not re-label the whole run.
    assert (
        served[tracing.PASS_TEACHER]["ref"]["ref_id"]
        != served[tracing.PASS_STUDENT]["ref"]["ref_id"]
    )
    assert served[tracing.PASS_TEACHER]["ref"]["pass_id"] == tracing.PASS_TEACHER

    # The unscoped read of the same attempt is untouched: one turn, the turn's
    # own answer, attributed to the turn.
    status, whole = selection_api.handle_get(
        todo_workflow_path, run, {"view": ["answers"]}
    )
    assert status == 200
    assert whole["pass_id"] is None
    assert whole["projection"]["answers"][0]["attribution"] == ATTRIBUTION_TURN
    assert whole["projection"]["turns"][0]["pass_content_recorded"] is False


# ----------------------------------------------------------------------
# The shipped page, against producer-recorded passes
# ----------------------------------------------------------------------


@pytest.fixture
def pass_world(initialized_fastworkflow, todo_workflow_path, tmp_path, monkeypatch):
    """Two attempts of one task, recorded two ways.

    Attempt 1 is a REAL distilled turn: `distillation.py` recorded a pass span
    per pass with that pass's own answer and plan. Attempt 2 is a turn whose
    dispatch spans are pass-STAMPED but which recorded no pass content at all —
    a producer that marks passes without describing them, and the shape every
    trace recorded before `fix-txxy` has. The page must not render them alike.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    db = state_paths.observability_db(todo_workflow_path)
    store = obs.ObservabilityStore(db)
    controller = ExperimentController(
        todo_workflow_path,
        store.store_identity(),
        external=False,
    )
    experiment_id = f"exp-{uuid.uuid4().hex}"
    task_id = f"task_{uuid.uuid4().hex}"
    channels = {n: f"ch-{uuid.uuid4().hex[:8]}" for n in (1, 2)}
    controller.create_experiment(
        experiment_id,
        "recorded passes",
        declared_tasks=1,
        declared_attempts=2,
        declarations=[(task_id, n, channels[n]) for n in (1, 2)],
        workflow_name=setup.workflow_name_for(todo_workflow_path),
    )

    conversations = {}
    for attempt in (1, 2):
        conversations[attempt] = store.mint_conversation_id(
            channels[attempt],
            experiment_id=experiment_id,
            task_id=task_id,
            attempt=attempt,
        )
        controller.start_attempt(
            experiment_id,
            task_id,
            attempt,
            channels[attempt],
            conversation_id=conversations[attempt],
        )

    _run_distillation_turn(
        monkeypatch,
        todo_workflow_path,
        tmp_path,
        db,
        identity={
            "channel_id": channels[1],
            "conversation_id": conversations[1],
            "experiment_id": experiment_id,
            "task_id": task_id,
            "attempt": 1,
        },
    )

    # Attempt 2, stamped but undescribed. Written the way a recorder writes,
    # through the store's own row upserts.
    stamped_key = "stamped-no-content-t1"
    spans = [
        _execute_span(
            f"{stamped_key}-span-{index}",
            stamped_key,
            call_id=f"{stamped_key}-call-{index}",
            command_name=command,
            start_ns=1_700_000_000_000_000_000 + index * 1_000_000,
            extra={tracing.ATTR_PASS: pass_id},
        )
        for index, (command, pass_id) in enumerate(
            [
                (LIST_COMMAND, tracing.PASS_TEACHER),
                (LIST_COMMAND, tracing.PASS_STUDENT),
            ]
        )
    ]
    row = _turn_row(
        stamped_key,
        record=_record(
            stamped_key,
            refs=[
                (f"{stamped_key}-call-{index}", index, f"{stamped_key}-span-{index}")
                for index in (0, 1)
            ],
            outputs=[
                _output(f"{stamped_key}-call-{index}", LIST_COMMAND, {})
                for index in (0, 1)
            ],
        ),
        answer="one answer for both passes",
        experiment_id=experiment_id,
        task_id=task_id,
        attempt=2,
    )
    row["conversation_id"] = conversations[2]
    row["ordinal"] = 1
    _write(store, row, spans)

    for attempt in (1, 2):
        controller.finish_attempt(
            experiment_id, task_id, attempt, outcome="pass", outcome_source="derived"
        )

    return {
        "folder": todo_workflow_path,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "db": db,
    }


@pytest.fixture
def pass_server(pass_world):
    # The same evidence DB the world wrote, so the experiment routes read the
    # recorded turns rather than an empty cold-start store.
    srv = run_chatbot_server.ChatbotServer(
        db_path=pass_world["db"],
        workflow_path=pass_world["folder"],
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def test_the_page_shows_each_pass_its_own_content_in_a_real_dom(
    pass_server, pass_world
):
    """What a person reads, driven through the real page against the real server.

    Exposing the new fields in the API only moves the problem: the page labelled
    every non-turn attribution "shared across passes", so a recorded pass would
    have been reported as quoting the turn. Both labels are checked here, on two
    attempts that differ only in whether the producer recorded pass content.
    """
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_pass_comparison_dom.cjs")
    result = subprocess.run(
        [
            "node",
            str(script),
            jsdom_root,
            f"http://127.0.0.1:{pass_server.port}/?token={pass_server.token}",
            pass_world["experiment_id"],
            pass_world["task_id"],
            TEACHER_ANSWER,
            STUDENT_ANSWER,
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_page_keeps_the_two_labels_distinct():
    """A cheap guard on the page source for the distinction the DOM run above
    drives, so a future edit that collapses the two labels back into one fails
    here too rather than only on a machine with jsdom installed."""
    source = run_chatbot_server.load_index_html().decode("utf-8")

    assert 'if (answer.attribution === "pass") {' in source
    assert '"recorded for this pass"' in source
    assert "shared across passes — this text is the turn's, not this pass's" in source
    assert "function appendPassPlan(block, turn)" in source
    assert "function appendRecordedEnvelope(block, turn)" in source
    # The plan is offered from the pass's recorded value and nowhere else.
    assert 'box.appendChild(el("summary", null, "Plan this pass generated"));' in source
    assert '[["answer", "answer"], ["plan", "plan"]].forEach' in source


# ----------------------------------------------------------------------
# The real DSPy callback, inside a pass
# ----------------------------------------------------------------------


class _LocalEchoLM(dspy.BaseLM):
    """A real DSPy LM that answers locally.

    Not a stand-in for one: it IS an LM on the same `forward_contract="legacy"`
    path `dspy.LM` uses, so DSPy normalizes the response and dispatches its
    callbacks exactly as it does for a paid provider. Same pattern as
    `tests/test_usage_and_cache_rollups.py::_EchoLM`, which is where the shape
    of this response is justified.
    """

    def __init__(self) -> None:
        super().__init__(model="local/echo")

    def forward(self, prompt=None, messages=None, **kwargs):
        return dotdict(
            choices=[
                dotdict(
                    message=dotdict(
                        content="[[ ## answer ## ]]\nyou have 0 lists",
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            model="local/echo",
            usage=dotdict(prompt_tokens=11, completion_tokens=5, total_tokens=16),
            id="resp-local-echo",
        )


def test_a_real_dspy_call_is_attributed_to_the_pass_that_made_it(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path, monkeypatch
):
    """Ancestry is automatic, and this is the proof with nothing fabricated.

    Every other test here emits its pass's `fw.llm.call` through `tracing`
    directly, which shows the emission path but not that DSPy's own callback
    finds the pass. Here the teacher pass makes a REAL `dspy.Predict` call
    against a real local LM, and the span that lands in SQLite is the one
    `dspy_logger`'s callback wrote: the WEC installs that callback for the whole
    turn (`observe_dspy_calls` on `_execute_message`), so nothing in the test
    binds it and nothing in the producer has to pass a pass id down to it.

    Costs nothing: the LM answers in-process and no model is downloaded.
    """
    dspy_teacher = _PassScript(
        commands=["list_todo_lists"],
        plan=["list the lists"],
        answer=TEACHER_ANSWER,
        cost=0.10,
        dspy_call=True,
    )
    turn_key = _run_distillation_turn(
        monkeypatch,
        todo_workflow_path,
        tmp_path,
        db_path,
        scripts=[dspy_teacher, STUDENT_SCRIPT],
    )

    spans = _reader(db_path).trace("store-1", turn_key)
    by_id = {span["span_id"]: span for span in spans}
    real_calls = [
        span
        for span in spans
        if span["name"] == tracing.SPAN_LLM_CALL
        and json.loads(span["attributes"]).get("model") == "local/echo"
    ]
    assert len(real_calls) == 1, "the real LM was called once, by the teacher"

    def ancestors(span):
        seen = []
        parent = span.get("parent_span_id")
        while parent in by_id:
            seen.append(by_id[parent])
            parent = by_id[parent].get("parent_span_id")
        return seen

    pass_ancestors = [
        json.loads(ancestor["attributes"])[tracing.ATTR_PASS]
        for ancestor in ancestors(real_calls[0])
        if ancestor["name"] == tracing.SPAN_DISTILLATION_PASS
    ]
    assert pass_ancestors == [tracing.PASS_TEACHER]
    # And the student's pass span is not on that chain, so the two passes'
    # calls are separable without anyone having stamped the call itself.
    assert tracing.PASS_STUDENT not in pass_ancestors
