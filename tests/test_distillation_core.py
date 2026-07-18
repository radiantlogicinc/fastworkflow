"""Unit tests for distillation's pure comparison/formatting/persistence logic.

These cover the LLM-free core of fastworkflow/distillation.py — trajectory and
plan comparison, trajectory formatting for the insight LLM, and numbered-insight
persistence — without any network/LLM dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastworkflow.distillation import (
    DistillationSession,
    PlanningStep,
    DistillationResult,
)


@pytest.fixture
def session() -> DistillationSession:
    # The comparison/formatting methods don't touch WEC state, so build an
    # instance without running __init__ (no WEC needed).
    return DistillationSession.__new__(DistillationSession)


# ---------------------------------------------------------------------------
# _action_signature / _format_action
# ---------------------------------------------------------------------------

def test_action_signature_is_order_independent_for_params():
    a1 = {"command_name": "cancel_order", "parameters": {"id": "1", "reason": "x"}}
    a2 = {"command_name": "cancel_order", "parameters": {"reason": "x", "id": "1"}}
    # Same command + same params (different dict order) -> identical signature.
    assert DistillationSession._action_signature(a1) == DistillationSession._action_signature(a2)


def test_action_signature_differs_on_params():
    a1 = {"command_name": "cancel_order", "parameters": {"id": "1"}}
    a2 = {"command_name": "cancel_order", "parameters": {"id": "2"}}
    assert DistillationSession._action_signature(a1) != DistillationSession._action_signature(a2)


def test_format_action_with_and_without_params():
    assert DistillationSession._format_action(
        {"command_name": "get_order", "parameters": {"id": "1"}}
    ) == 'get_order({"id": "1"})'
    # No params -> bare command name.
    assert DistillationSession._format_action(
        {"command_name": "list_all", "parameters": {}}
    ) == "list_all"


# ---------------------------------------------------------------------------
# compare_trajectories (action-level divergence + is_valid_action filter)
# ---------------------------------------------------------------------------

def test_compare_trajectories_identical_no_divergence(session):
    actions = [
        {"command_name": "find_user", "parameters": {"email": "a@b.com"}},
        {"command_name": "get_order", "parameters": {"id": "1"}},
    ]
    diverged, summary = session.compare_trajectories(list(actions), list(actions))
    assert diverged is False
    assert summary == ""


def test_compare_trajectories_detects_extra_student_action(session):
    teacher = [{"command_name": "find_user", "parameters": {"email": "a@b.com"}}]
    student = [
        {"command_name": "find_user", "parameters": {"email": "a@b.com"}},
        {"command_name": "get_user_details", "parameters": {"user_id": "u1"}},
    ]
    diverged, summary = session.compare_trajectories(teacher, student)
    assert diverged is True
    assert "get_user_details" in summary


def test_compare_trajectories_filters_ask_user_and_error_correction(session):
    # ask_user records (agent_query key) and ErrorCorrection/* must be excluded
    # from the command-level divergence comparison.
    teacher = [
        {"agent_query": "email?", "user_response": "a@b.com"},           # ask_user
        {"command_name": "ErrorCorrection/abort", "parameters": {}},     # abort
        {"command_name": "find_user", "parameters": {"email": "a@b.com"}},
    ]
    student = [
        {"command_name": "find_user", "parameters": {"email": "a@b.com"}},
    ]
    # After filtering, both reduce to the same single find_user action.
    diverged, summary = session.compare_trajectories(teacher, student)
    assert diverged is False
    assert summary == ""


# ---------------------------------------------------------------------------
# compare_planning_traces
# ---------------------------------------------------------------------------

def test_compare_planning_traces_identical(session):
    t = [PlanningStep(0, "q", ["step a", "step b"])]
    s = [PlanningStep(0, "q", ["step a", "step b"])]
    diverged, summary = session.compare_planning_traces(t, s)
    assert diverged is False


def test_compare_planning_traces_detects_difference(session):
    t = [PlanningStep(0, "q", ["step a", "step b"])]
    s = [PlanningStep(0, "q", ["step a", "different"])]
    diverged, summary = session.compare_planning_traces(t, s)
    assert diverged is True
    assert "Step 0" in summary


def test_compare_planning_traces_empty_both(session):
    assert session.compare_planning_traces([], []) == (False, "")


# ---------------------------------------------------------------------------
# _format_trajectory_for_llm (must surface ask_user steps + observations)
# ---------------------------------------------------------------------------

def test_format_trajectory_includes_ask_user_and_observations():
    trajectory = {
        "thought_0": "need email",
        "tool_name_0": "ask_user",
        "tool_args_0": {"clarification_request": "email?"},
        "observation_0": "mia@example.com",       # user's answer
        "thought_1": "look up",
        "tool_name_1": "execute_workflow_query",
        "tool_args_1": {"command": "find_user"},
        "observation_1": "user id: u1",
    }
    out = DistillationSession._format_trajectory_for_llm(trajectory)
    # ask_user step and the user's answer must reach the insight LLM.
    assert "ask_user" in out
    assert "mia@example.com" in out
    assert "execute_workflow_query" in out
    assert "user id: u1" in out


def test_format_trajectory_truncates_long_observations():
    trajectory = {
        "thought_0": "t",
        "tool_name_0": "x",
        "tool_args_0": {},
        "observation_0": "A" * 900,
    }
    out = DistillationSession._format_trajectory_for_llm(trajectory)
    assert "[truncated]" in out


# ---------------------------------------------------------------------------
# _append_numbered_insights (numbering continuity across appends)
# ---------------------------------------------------------------------------

def test_append_numbered_insights_starts_at_one_then_continues(tmp_path: Path):
    f = tmp_path / "insights.md"
    fmt = "{num}. {insight}\n"
    pattern = r"^(\d+)\.\s"
    header = "# Insights\n\n"

    DistillationSession._append_numbered_insights(["first"], f, header, pattern, fmt)
    DistillationSession._append_numbered_insights(["second", "third"], f, header, pattern, fmt)

    content = f.read_text(encoding="utf-8")
    assert "1. first" in content
    assert "2. second" in content
    assert "3. third" in content
    # Header written exactly once.
    assert content.count("# Insights") == 1


def test_append_numbered_planning_insights_uses_its_own_pattern(tmp_path: Path):
    f = tmp_path / "planning.md"
    fmt = "## {num}. {insight}\n\n"
    pattern = r"^## (\d+)\."
    header = "# Planning\n\n"

    DistillationSession._append_numbered_insights(["a"], f, header, pattern, fmt)
    DistillationSession._append_numbered_insights(["b"], f, header, pattern, fmt)

    content = f.read_text(encoding="utf-8")
    assert "## 1. a" in content
    assert "## 2. b" in content


# ---------------------------------------------------------------------------
# DistillationResult
# ---------------------------------------------------------------------------

def test_distillation_result_total_insights():
    r = DistillationResult(
        command_output=None,
        planning_insights_extracted=2,
        execution_insights_extracted=3,
    )
    assert r.insights_extracted == 5
