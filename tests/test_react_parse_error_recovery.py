"""A reply no adapter can parse is a recovered step, not a failed turn."""
import dspy
from dspy.utils.exceptions import AdapterParseError

from fastworkflow.utils.react import fastWorkflowReAct


def a_tool(command: str) -> str:
    """Runs a command."""
    return f"ran {command}"


def test_an_unparseable_reply_is_recovered_and_the_loop_goes_on(monkeypatch):
    agent = fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=10)
    replies = iter([
        dspy.Prediction(next_thought="run it", next_tool_name="a_tool",
                        next_tool_args={"command": "x"}),
        AdapterParseError(adapter_name="JSONAdapter", signature=agent.react.signature,
                          lm_response="{garbled", message="cannot parse"),
        dspy.Prediction(next_thought="done", next_tool_name="finish", next_tool_args={}),
    ])

    def next_reply(*args, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation", next_reply)
    trajectory = {}

    assert agent._run_loop(trajectory, 0, {}, 10, 0) is None

    assert trajectory["observation_0"] == "ran x"
    assert trajectory["observation_1"].startswith("Agent failed to select a valid tool")
    assert trajectory["tool_name_3"] == "finish"


def test_a_timed_out_model_call_is_recovered_and_the_loop_goes_on(monkeypatch):
    from litellm import exceptions as litellm_exceptions

    agent = fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=10)
    replies = iter([
        litellm_exceptions.Timeout(message="no reply in 120s", model="m", llm_provider="p"),
        dspy.Prediction(next_thought="done", next_tool_name="finish", next_tool_args={}),
    ])

    def next_reply(*args, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation", next_reply)
    trajectory = {}

    assert agent._run_loop(trajectory, 0, {}, 10, 0) is None
    assert "no reply in 120s" in trajectory["observation_0"]
    assert trajectory["tool_name_2"] == "finish"
