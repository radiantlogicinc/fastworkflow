"""The policy guard at the command executor, against the real todo-list workflow.

The real WorkflowExecutionContext, CommandExecutor and command tool
(``workflow_agent._execute_workflow_query``) run the real ``create_todo_list``
command against a scratch todo store. The only stand-in is the NLU hop
(intent detection + parameter extraction), because the test workflows ship no
trained intent models: it returns what the real wildcard step returns - the
resolved command name and its validated parameters - so everything after it is
the real path, guard included.
"""

from __future__ import annotations

import uuid
from contextlib import suppress
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.policy_decision import (
    ContractFacts,
    DecisionPhase,
    DecisionTable,
    PolicyDecisionPoint,
    PolicyMode,
    PolicyOutcome,
    PolicyRow,
)
from fastworkflow.policy_runtime import AgentPolicy
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

CREATE = "TodoListManager/create_todo_list"
FACTS = ContractFacts(write_commands=frozenset({CREATE}))
TABLE = DecisionTable("guard-test/1", [PolicyRow(
    "NO-GROCERIES", frozenset({DecisionPhase.BEFORE_SIDE_EFFECT}), PolicyOutcome.REWRITE,
    "blocks one description, by its canonical parameter",
    lambda i: i.command_name == CREATE and i.command_args.get("description") == "groceries",
    "Lists named groceries are not allowed here.",
)])


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def ctx(todo_workflow_path, tmp_path, monkeypatch):
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"policy-guard-{uuid.uuid4().hex}")
    context = WorkflowExecutionContext(run_as_agent=False)
    context.bind_app_workflow(workflow)
    context.push_active_workflow(workflow)
    manager = TodoListManager(str(tmp_path / "todo_list.json"))
    workflow.root = manager
    context.manager = manager

    from tests.todo_list_workflow._commands.TodoListManager.create_todo_list import Signature

    real_perform_action = CommandExecutor.perform_action

    def nlu_hop(cls, wf, action):
        if wf is not context.cme_workflow:
            return real_perform_action(wf, action)
        description = action.command.split("<description>")[-1].split("</description>")[0]
        # As the real wildcard step does before it hands the command on.
        workflow._command_context_for_response_generation = workflow._current_command_context
        return fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(
            response="",
            artifacts={"command_name": CREATE,
                       "cmd_parameters": Signature.Input(description=description)},
        ))

    monkeypatch.setattr(CommandExecutor, "perform_action", classmethod(nlu_hop))
    yield context
    with suppress(Exception):
        context.close()
    RoutingRegistry.clear_registry()


def _with_policy(ctx, mode=PolicyMode.ENFORCE) -> AgentPolicy:
    agent = fastWorkflowReAct("user_query -> final_answer", tools=[], max_iters=5)
    agent.policy = AgentPolicy(PolicyDecisionPoint(TABLE, mode), FACTS, host=ctx)
    ctx._workflow_tool_agent = agent
    return agent.policy


def _run(ctx, description: str) -> str:
    from fastworkflow.workflow_agent import _execute_workflow_query

    return _execute_workflow_query(
        f"create_todo_list <description>{description}</description>", chat_session_obj=ctx)


def test_a_blocked_command_never_executes(ctx):
    policy = _with_policy(ctx)
    policy.dispatching = True
    output = CommandExecutor.invoke_command(
        ctx, "create_todo_list <description>groceries</description>")
    assert output.command_response.artifacts.get("policy_blocked") is True
    assert not output.success
    assert output.command_response.response == "Lists named groceries are not allowed here."
    assert output.command_name == CREATE
    assert ctx.manager.lists == {}
    assert policy.last_block.source_policy == "NO-GROCERIES"


def test_the_command_tool_returns_the_rewrite_and_records_nothing_as_run(ctx):
    policy = _with_policy(ctx)
    policy.dispatching = True
    assert _run(ctx, "groceries") == "Lists named groceries are not allowed here."
    assert ctx.manager.lists == {}
    assert not [r for r in ctx.action_log if r.get("command_name") == CREATE]
    assert [c.name for c in policy.state.commands_run] == []


def test_an_allowed_command_runs_and_is_recorded_in_canonical_form(ctx):
    policy = _with_policy(ctx)
    policy.dispatching = True
    response = _run(ctx, "trip")
    assert "new_context" in response
    assert [todo.description for todo in ctx.manager.lists.values()] == ["trip"]
    recorded = policy.state.commands_run[-1]
    assert (recorded.name, recorded.args, recorded.success) == (
        CREATE, {"description": "trip"}, True)


def test_a_command_the_model_did_not_dispatch_is_not_checked(ctx):
    policy = _with_policy(ctx)
    assert policy.dispatching is False
    _run(ctx, "groceries")
    assert [todo.description for todo in ctx.manager.lists.values()] == ["groceries"]


def test_with_the_policy_off_the_command_runs_unchanged(ctx):
    policy = _with_policy(ctx, mode=PolicyMode.OFF)
    policy.dispatching = True
    _run(ctx, "groceries")
    assert [todo.description for todo in ctx.manager.lists.values()] == ["groceries"]
    assert policy.state.commands_run == []


def test_the_policy_state_travels_with_the_serialized_session(ctx):
    policy = _with_policy(ctx)
    policy.observe_turn("make me a trip list")
    policy.dispatching = True
    _run(ctx, "trip")
    state = ctx.serialize_state(channel_id="c1")
    assert [c["name"] for c in state["policy_state"]["commands_run"]] == ["@user", CREATE]

    other = _with_policy(ctx)
    ctx.apply_serialized_state(state)
    assert [c.name for c in other.state.commands_run] == ["@user", CREATE]
