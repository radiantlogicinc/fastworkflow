"""The policy verifier in the ReAct loop (utils/react.py + policy_runtime.AgentPolicy).

Real fastWorkflowReAct agents with scripted model replies, as in
tests/test_finish_check.py. The command tool here stands at the executor
boundary: it consults the guard exactly as ``command_executor._invoke_command_impl``
does and returns the rewrite as its output when blocked. The real executor path
is covered in tests/test_policy_guard.py.
"""

from __future__ import annotations

import json

import dspy

from fastworkflow import tracing
from fastworkflow.policy_decision import (
    ContractFacts,
    DecisionPhase,
    DecisionTable,
    PolicyDecisionPoint,
    PolicyMode,
    PolicyOutcome,
    PolicyRow,
)
from fastworkflow.policy_runtime import AgentPolicy, policy_of
from fastworkflow.utils.react import fastWorkflowReAct

SIDE = frozenset({DecisionPhase.BEFORE_SIDE_EFFECT})
ASK = frozenset({DecisionPhase.AFTER_OBSERVATION})
FINISH = frozenset({DecisionPhase.BEFORE_FINISH})

FACTS = ContractFacts(write_commands=frozenset({"close_ticket"}))


def _never_listed(inputs) -> bool:
    return not inputs.ran("list_tickets")


TABLE = DecisionTable("loop-test/1", [
    PolicyRow("READ-FIRST", SIDE, PolicyOutcome.REWRITE, "write before listing",
              lambda i: FACTS.is_write(i.command_name) and _never_listed(i),
              "List the tickets before closing one."),
    PolicyRow("NO-ASK", ASK, PolicyOutcome.REWRITE, "asks for what a listing gives",
              lambda i: i.pending_tool == "ask_user" and _never_listed(i),
              "Look the tickets up instead of asking."),
    PolicyRow("CLAIM", FINISH, PolicyOutcome.REWRITE, "claims a close that did not run",
              lambda i: "closed" in i.answer and not i.ran("close_ticket"),
              "Nothing was closed yet: close the ticket, then answer."),
])


def _agent(table=TABLE, facts=FACTS, *, max_iters=12):
    """A real agent with the policy attached, and a log of commands that really ran."""
    ran: list[str] = []
    holder: dict = {}

    def execute_workflow_query(command: str) -> str:
        """Runs a workflow command."""
        name = command.split()[0]
        rewrite = holder["agent"].policy.guard(name, {})
        if rewrite is not None:
            return rewrite
        ran.append(name)
        response = "T-1, T-2" if name == "list_tickets" else f"{name} done"
        holder["agent"].policy.record_command(name, {}, response, True)
        return response

    def ask_user(clarification_request: str) -> str:
        """Asks the user."""
        ran.append("ask_user")
        return "the user answered"

    agent = fastWorkflowReAct(
        "user_query -> final_answer", tools=[execute_workflow_query, ask_user],
        max_iters=max_iters)
    agent.policy = AgentPolicy(PolicyDecisionPoint(table, PolicyMode.ENFORCE), facts)
    holder["agent"] = agent
    return agent, ran


def _script(agent, monkeypatch, steps, answers=("All set.",)):
    replies = iter([
        dspy.Prediction(next_thought="", next_tool_name=name, next_tool_args=args)
        for name, args in steps])
    monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation",
                        lambda *a, **k: next(replies))
    answer_iter = iter(answers)
    extracted: list[str] = []

    def extract(trajectory, **kwargs):
        extracted.append(next(answer_iter))
        return dspy.Prediction(final_answer=extracted[-1])

    monkeypatch.setattr(agent, "_extract_prediction", extract)
    return extracted


def _cmd(text):
    return ("execute_workflow_query", {"command": text})


class _Sink:
    def __init__(self):
        self.spans = []

    def emit_span(self, span):
        self.spans.append(span)

    def emit_turn_record(self, record):
        return True

    def record_conversation_label(self, *a):
        pass


def test_a_blocked_write_never_runs_and_the_rewrite_is_the_observation(monkeypatch):
    agent, ran = _agent()
    _script(agent, monkeypatch, [_cmd("close_ticket T-1"), _cmd("list_tickets"),
                                 _cmd("close_ticket T-1"), ("finish", {})])
    trajectory = {}
    assert agent._run_loop(trajectory, 0, {}, 12, 0) is None
    assert trajectory["observation_0"] == "List the tickets before closing one."
    assert ran == ["list_tickets", "close_ticket"]


def test_a_rule_gets_two_blocks_then_the_step_goes_through(monkeypatch):
    agent, ran = _agent()
    _script(agent, monkeypatch, [_cmd("close_ticket T-1")] * 3 + [("finish", {})])
    trajectory = {}
    agent._run_loop(trajectory, 0, {}, 12, 0)
    assert [trajectory[f"observation_{i}"] for i in range(2)] == [
        "List the tickets before closing one."] * 2
    assert ran == ["close_ticket"]


def test_a_question_is_rewritten_before_it_is_sent(monkeypatch):
    agent, ran = _agent()
    _script(agent, monkeypatch, [("ask_user", {"clarification_request": "Which ticket?"}),
                                 _cmd("list_tickets"), ("finish", {})])
    trajectory = {}
    agent._run_loop(trajectory, 0, {}, 12, 0)
    assert trajectory["observation_0"] == "Look the tickets up instead of asking."
    assert "ask_user" not in ran


def test_a_false_claim_sends_the_agent_back_and_the_checked_answer_is_reused(monkeypatch):
    agent, ran = _agent()
    extracted = _script(
        agent, monkeypatch,
        [_cmd("list_tickets"), ("finish", {}), _cmd("close_ticket T-1"), ("finish", {})],
        answers=("T-1 is closed.", "T-1 is closed now."))
    prediction = agent.forward(user_query="close T-1")
    assert agent.trajectory["observation_1"] == (
        "Nothing was closed yet: close the ticket, then answer.")
    assert ran == ["list_tickets", "close_ticket"]
    # The answer the policy checked is the run's answer: no second extraction.
    assert prediction.final_answer == "T-1 is closed now." and len(extracted) == 2
    assert agent.policy.state.dialogue[-1] == ("assistant", "T-1 is closed now.")


def test_without_progress_a_finish_is_not_pushed_back_twice(monkeypatch):
    agent, ran = _agent()
    _script(agent, monkeypatch, [("finish", {}), ("finish", {})],
            answers=("T-1 is closed.", "T-1 is closed."))
    prediction = agent.forward(user_query="close T-1")
    assert agent.policy.finish_blocks == 1
    assert prediction.final_answer == "T-1 is closed."


def test_finish_pushes_are_capped_per_turn_even_with_progress():
    always = DecisionTable("cap/1", [PolicyRow(
        "ALWAYS", FINISH, PolicyOutcome.REWRITE, "never satisfied", lambda i: True,
        "Not yet.")])
    policy = AgentPolicy(PolicyDecisionPoint(always, PolicyMode.ENFORCE), FACTS)
    policy.observe_turn("close everything")
    blocks = 0
    for _ in range(AgentPolicy.MAX_FINISH_BLOCKS + 2):
        if policy.before_finish("All closed.") is not None:
            blocks += 1
        policy.record_command("close_ticket", {}, "ok", True)  # progress each time
    assert blocks == AgentPolicy.MAX_FINISH_BLOCKS
    policy.observe_turn("and the next one")  # a new turn starts a new budget
    assert policy.before_finish("All closed.") is not None


def test_a_run_that_is_not_the_turns_last_is_not_checked_at_finish(monkeypatch):
    agent, ran = _agent()
    agent.policy_final_run = False
    extracted = _script(agent, monkeypatch, [("finish", {})], answers=("T-1 is closed.",))
    prediction = agent.forward(user_query="close T-1 (first part)")
    assert prediction.final_answer == "T-1 is closed." and agent.policy.finish_blocks == 0
    assert len(extracted) == 1


def test_the_finish_reviewer_speaks_first(monkeypatch):
    agent, ran = _agent()
    # The first finish is reviewed and gets the note; the second is reviewed too and finds nothing.
    notes = iter(["Step 2 (list_tickets) was not run."])
    agent.finish_reviewer = lambda trajectory, left: next(notes, None)
    _script(agent, monkeypatch, [("finish", {}), _cmd("list_tickets"),
                                 _cmd("close_ticket T-1"), ("finish", {})],
            answers=("T-1 is closed.",))
    agent.forward(user_query="close T-1")
    assert agent.trajectory["observation_0"] == "Step 2 (list_tickets) was not run."
    assert agent.policy.finish_blocks == 0


def test_the_guard_only_checks_the_models_own_dispatches():
    agent, _ = _agent()
    policy = agent.policy
    # A command the user typed, or one a skill's code runs, is not dispatched by
    # the model, and the guard lets it through.
    assert policy.guard("close_ticket", {}) is None
    policy.dispatching = True
    assert policy.guard("close_ticket", {}) == "List the tickets before closing one."


def test_off_leaves_the_loop_untouched(monkeypatch):
    agent, ran = _agent()
    agent.policy = AgentPolicy(PolicyDecisionPoint(TABLE, PolicyMode.OFF), FACTS)
    assert policy_of(agent) is None
    _script(agent, monkeypatch, [_cmd("list_tickets"), ("finish", {})])
    # The executor stand-in still asks the (disabled) guard, which says nothing.
    trajectory = {}
    agent._run_loop(trajectory, 0, {}, 12, 0)
    assert ran == ["list_tickets"]


def test_policy_steps_are_traced_within_the_step_contract(monkeypatch):
    from types import SimpleNamespace

    agent, ran = _agent()
    sink = _Sink()
    _script(agent, monkeypatch, [_cmd("close_ticket T-1"), _cmd("list_tickets"),
                                 ("finish", {}), _cmd("close_ticket T-1"), ("finish", {})],
            answers=("T-1 is closed.", "T-1 is closed."))
    host = SimpleNamespace(trace_sink=sink, current_turn_key="turn-1", trace_span_stack=[])
    with tracing.host_scope(host):
        agent.forward(user_query="close T-1")
    steps = [s for s in sink.spans if s.name == tracing.SPAN_AGENT_STEP]
    assert len(steps) == 5
    contract = tracing.SPAN_CONTRACTS[tracing.SPAN_AGENT_STEP].attributes
    for span in steps:
        assert set(span.attributes) - tracing.ENVELOPE_ATTRIBUTES <= contract
    by_source = {s.attributes.get("policy_source"): s.attributes for s in steps}
    assert set(by_source) == {None, "READ-FIRST", "CLAIM"}
    claim = by_source["CLAIM"]
    assert claim["tool_name"] == "finish" and claim["answer"] == "T-1 is closed."
    assert claim["observation"] == "Nothing was closed yet: close the ticket, then answer."
    assert claim["policy_table_version"] == "loop-test/1"
    assert claim["policy_mode"] == "enforce"


def test_the_state_survives_export_and_import():
    agent, _ = _agent()
    policy = agent.policy
    policy.observe_turn("close all my tickets")
    policy.record_command("list_tickets", {"page": object()}, "T-1, T-2", True)
    policy.observe_reply("Close T-1 and T-2?", "yes")
    policy.finish_blocks, policy.finish_mark = 1, (0, 2)
    data = json.loads(json.dumps(policy.export_state()))
    other = AgentPolicy(PolicyDecisionPoint(TABLE, PolicyMode.ENFORCE), FACTS)
    other.import_state(data)
    assert [c.name for c in other.state.commands_run] == ["@user", "list_tickets", "ask_user"]
    assert other.state.dialogue == policy.state.dialogue
    assert (other.finish_blocks, other.finish_mark) == (1, (0, 2))
