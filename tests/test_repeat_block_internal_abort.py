"""The framework's own abort is never refused by the repeat guard (fix-hia1.13).

After a parameter-extraction error the agent tool aborts the pending round
itself. Two such aborts in a row return the same text, and the guard used to
refuse the third, leaving the session in a stale parameter round.
"""

from fastworkflow import workflow_agent
from tests.test_multi_command_refusal import initialized_fastworkflow, session  # noqa: F401
from tests.test_repeat_block import agent, responses, run_step  # noqa: F401


def _internal_abort(session, agent):
    result = workflow_agent._execute_workflow_query(
        "abort", chat_session_obj=session, internal=True)
    agent.on_step_end()
    return result


def test_internal_abort_is_neither_refused_nor_recorded(session, agent, responses):
    for _ in range(3):
        assert "unavailable for this step" not in _internal_abort(session, agent)
    assert responses["calls"] == ["abort"] * 3

    # Nothing was recorded, so the agent's own abort is not blocked by them either.
    assert "unavailable for this step" not in run_step(session, agent, "abort")


def test_internal_abort_runs_while_the_agent_abort_is_blocked(session, agent, responses):
    run_step(session, agent, "abort")
    run_step(session, agent, "abort")  # same output: repeat, blocks the next step
    assert "unavailable for this step" not in _internal_abort(session, agent)
    assert responses["calls"] == ["abort"] * 3
