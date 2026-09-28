"""The AdapterParseError retry does not re-run a trajectory whose tools already ran.

``_call_agent_with_retry`` is the one choke point both the fresh forward and the
resume pass through, and a retry re-enters ``forward()`` under the SAME turn
scope: a tool that already ran has already had its side effects and already owns
this turn's ``observation_`` keys and archived evidence rows. These cases drive
the real method on a stand-in host that overrides ``_agent_dspy_context``, so no
LM, credential or network is involved -- the parse failure is raised by the
stand-in ``agent_call``, which is also what counts the attempts.
"""

from __future__ import annotations

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

from fastworkflow.workflow_execution_context import WorkflowExecutionContext


class _Agent:
    def __init__(self, trajectory):
        self.current_trajectory = dict(trajectory)


class _Host:
    """Only the attributes ``_call_agent_with_retry`` reaches for."""

    def __init__(self, trajectory):
        self._workflow_tool_agent = _Agent(trajectory)

    def _agent_dspy_context(self):
        return dspy.LM("openai/gpt-4o-mini", api_key="unused"), dspy.ChatAdapter()

    _call_agent_with_retry = WorkflowExecutionContext._call_agent_with_retry


def _attempts_for(trajectory):
    """How many times the agent is called when every call fails to parse."""
    host = _Host(trajectory)
    attempts = []

    def agent_call():
        attempts.append(len(attempts) + 1)
        raise AdapterParseError(
            lm_response="junk",
            adapter_name="ChatAdapter",
            signature=dspy.Signature("q -> a"),
        )

    # The guard removes the second attempt, never the failure: the turn fails
    # either way, so the count is the only thing that distinguishes them.
    with pytest.raises(AdapterParseError):
        host._call_agent_with_retry(agent_call)
    return len(attempts)


def test_a_parse_failure_before_any_tool_ran_retries_once():
    """Nothing has executed, so re-running the trajectory costs nothing."""
    assert _attempts_for({}) == 2


def test_a_parse_failure_after_a_tool_ran_does_not_retry():
    """``observation_0`` means a tool returned; a retry would reuse its keys."""
    ran_a_tool = {
        "thought_0": "t",
        "tool_name_0": "execute_workflow_query",
        "tool_args_0": {"command": "q"},
        "observation_0": "O1 payload",
    }
    assert _attempts_for(ran_a_tool) == 1


def test_a_chosen_but_unobserved_step_still_retries():
    """A step picked but not yet observed has claimed no alias and no row."""
    chosen_only = {"thought_0": "t", "tool_name_0": "execute_workflow_query"}
    assert _attempts_for(chosen_only) == 2
