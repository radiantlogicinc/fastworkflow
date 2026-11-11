"""The model boundary for FastAPI tests that run turns without a live provider (fix-3rsq.fastapi-live-test).

A turn reaches a model at four places: the agent's ReAct steps, the sub-task split, the query refinement and the
text planner; the topic label and parameter extraction are model calls too. ``install_turn_seam`` replaces all of them
for one test. The agent's steps come from ``Script.steps``. With ``finish_when_empty`` an empty queue answers
``finish``, so a turn ends without a model; otherwise an unscripted step fails the test.

Real components throughout: the app, the session, the turn registry and the agent loop. Only the model answers are
scripted.
"""
from __future__ import annotations

import dspy

from fastworkflow import workflow_agent
from fastworkflow.utils import react as react_module
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


class Script:
    """The scripted agent steps, and whether an empty queue ends the turn."""

    def __init__(self, finish_when_empty: bool):
        self.steps: list = []
        self.finish_when_empty = finish_when_empty


def install_turn_seam(monkeypatch, *, finish_when_empty: bool = True, app_module=None) -> Script:
    script = Script(finish_when_empty)

    def trajectory_call(self, module, trajectory, **inputs):
        if module is self.react:
            if script.steps:
                tool, args = script.steps.pop(0)
            elif script.finish_when_empty:
                tool, args = "finish", {}
            else:
                raise AssertionError(f"the model was asked for a step after {trajectory}")
            return dspy.Prediction(next_thought="scripted", next_tool_name=tool, next_tool_args=args)
        return dspy.Prediction(reasoning="", final_answer="the answer")

    monkeypatch.setattr(react_module.fastWorkflowReAct, "_call_with_potential_trajectory_truncation", trajectory_call)
    monkeypatch.setattr(workflow_agent, "split_into_subtasks", lambda user_query, **kwargs: [user_query])
    monkeypatch.setattr(WorkflowExecutionContext, "_refine_user_query", lambda self, message, history: message)
    monkeypatch.setattr(WorkflowExecutionContext, "_subtask_gate", lambda self, subtasks: None)
    monkeypatch.setattr(workflow_agent, "build_query_with_next_steps",
                        lambda user_query, chat_session_obj, **kwargs: user_query)
    if app_module is not None:
        monkeypatch.setattr(app_module, "generate_topic_and_summary", lambda turns: ("a topic", "a summary"))
    return script
