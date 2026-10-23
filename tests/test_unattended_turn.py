"""An unattended turn runs with no ask_user tool, for that turn only."""

from fastworkflow.experiment.runner import ExperimentTask
from fastworkflow.run_fastapi_mcp.utils import InvokeRequest
from tests.test_subtask_execution import _agent_with_steps, _fake_progress_check, _split
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    _make_agent_ctx,
    _set_agents,
    initialized_fastworkflow,
    todo_workflow_path,
)


def _recording_agent(answers, **kwargs):
    agent = _agent_with_steps(answers, **kwargs)
    agent.disabled_tools = frozenset()
    inner, seen = agent.side_effect, []

    def call(**call_kwargs):
        seen.append(frozenset(agent.disabled_tools))
        return inner(**call_kwargs)

    agent.side_effect = call
    return agent, seen


def test_every_run_of_an_unattended_turn_lacks_ask_user_and_the_next_turn_has_it(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _fake_progress_check(monkeypatch)
    _split(monkeypatch, ["find X", "find Y"])
    # Run 1 exhausts, so the turn also hands off to a continuation.
    agent, seen = _recording_agent(["x1", "x2", "y", "z1", "z2"], exhaust=lambda n: n == 1)
    _set_agents(ctx, agent)

    ctx.process_turn("find X and find Y", unattended=True)
    assert seen == [frozenset({"ask_user"})] * 3

    ctx.process_turn("find X and find Y")
    assert seen[3:] == [frozenset()] * 2


def test_an_unattended_single_request_lacks_ask_user(
    initialized_fastworkflow, todo_workflow_path, monkeypatch
):
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    _split(monkeypatch, ["find X"])
    agent, seen = _recording_agent(["x"])
    _set_agents(ctx, agent)

    ctx.process_turn("find X", unattended=True)

    assert seen == [frozenset({"ask_user"})]


def test_attended_is_the_default_everywhere():
    assert InvokeRequest(user_query="hi").unattended is False
    assert InvokeRequest(user_query="hi", unattended=True).unattended is True
    assert ExperimentTask(task_id="t").unattended is False
