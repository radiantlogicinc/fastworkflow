import json

import dspy
import pytest
from fastworkflow.conversation_summary import bounded_summary_inputs
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


def test_large_history_is_bounded_without_losing_source_evidence():
    actions = [{"command": "list_permissions", "result": "rights " * 20000}] * 90
    original = json.dumps(actions)
    inputs = bounded_summary_inputs("audit", actions, "Completed answer")
    assert len(json.dumps(inputs, ensure_ascii=True).encode()) <= 64000
    assert inputs["final_agent_response"] == "Completed answer"
    assert "omitted" in inputs["workflow_actions"][0]["history_excerpt"]
    assert json.dumps(actions) == original


@pytest.mark.parametrize("text", ['😀\\"\n' * 100000, "x" * 300000])
def test_every_summary_field_is_bounded_even_with_escaping(text):
    inputs = bounded_summary_inputs(text, [{"result": text}], text)
    assert len(json.dumps(inputs, ensure_ascii=True).encode()) <= 64000


def test_small_turn_is_unchanged():
    actions = [{"command": "list_accounts", "result": "account123"}]
    assert bounded_summary_inputs("audit", actions, "answer") == dict(
        user_query="audit", workflow_actions=actions, final_agent_response="answer")


def test_summary_failure_records_memory_and_preserves_full_traces(monkeypatch):
    from fastworkflow.utils import dspy_utils
    # Deliberately unavailable LM setup exercises the production fallback path.
    def unavailable(*args, **kwargs):
        raise RuntimeError("summary provider unavailable")
    monkeypatch.setattr(dspy_utils, "get_lm", unavailable)
    host = object.__new__(WorkflowExecutionContext)
    host._conversation_history = dspy.History(messages=[])
    host._turn_outputs = []
    host._app_workflow = None
    host._keep_alive = False
    host._command_output_queue = None
    host._command_trace_queue = None
    actions = [{"result": "evidence " * 20000}]
    host._action_log = actions
    output = host._finalize_agent_output("audit", dspy.Prediction(final_answer="Completed answer"))
    assert output.command_response.response == "Completed answer"
    summary = host._conversation_history.messages[-1]["conversation summary"]
    traces = host._conversation_history.messages[-1]["conversation_traces"]
    assert "Completed answer" in summary
    assert json.loads(traces)["agent_workflow_interactions"] == actions
    assert json.loads(traces)["summary_fallback_error_type"] == "RuntimeError"
    assert host._conversation_history.messages[-1]["conversation summary"] == summary


def test_successful_summary_receives_bounded_input_and_retains_raw_traces(monkeypatch):
    from dspy.utils import DummyLM
    from fastworkflow.utils import dspy_utils
    lm = DummyLM([{"reasoning": "summarize", "conversation_summary": "audit complete"}])
    monkeypatch.setattr(dspy_utils, "get_lm", lambda *a: lm)
    host = object.__new__(WorkflowExecutionContext)
    actions = [{"result": "evidence " * 200000}]
    summary, traces = host._extract_conversation_summary("audit", actions, "answer")
    assert summary == "audit complete"
    assert len(json.dumps(lm.history[-1]["messages"])) < 70000
    assert json.loads(traces)["agent_workflow_interactions"] == actions

