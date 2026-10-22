"""Answering a capability question ("what can you do?") in code, in agent mode, when the
workflow has skills: the matcher, the renderer and the turn path."""
from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction._commands.IntentDetection.what_can_i_do import (
    Signature as WhatCanIDo)
from fastworkflow.capability_answer import is_capability_question, render_capability_answer
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

UTTERANCES = WhatCanIDo.plain_utterances


@pytest.mark.parametrize("message", [
    "what can you do?",
    "WHAT CAN YOU DO??",
    "  what can   you do  ",
    "what can u do",
    "hey, what can you do?",
    "ok so what can you do, please?",
    "what else can you do?",
    "what can you help me with?",
    "what sort of things can you help with?",
    "can you tell me what you can do?",
    "what are my options here?",
    "what're my options?",
    "what could we do here?",
    "wht can you do",
    "what cna you do?",
    "what are my opitons?",
    "now what?",
    "list the commands",
])
def test_a_capability_question_matches_up_to_minor_wording_deviations(message):
    assert is_capability_question(message, UTTERANCES)


@pytest.mark.parametrize("message", [
    "what can I do about Ann's logins?",
    "what are my options for fixing the open finding?",
    "what can you tell me about the billing app?",
    "what can you do with the billing app?",
    "help me with Ann's logins",
    "what can I access?",
    "what can I add?",
    "what are my rights?",
    "list the accounts",
    "help",
    "what",
    "hello",
    "",
])
def test_a_message_that_adds_a_topic_or_asks_something_else_does_not_match(message):
    assert not is_capability_question(message, UTTERANCES)


def test_the_answer_names_each_skill_with_two_examples_then_each_command_then_other_contexts():
    answer = render_capability_answer(
        {"close_out": ("close this record", "note what was done", "a third one"),
         "tag_record": ()},
        ["open_catalog", "who_am_i"],
        {"ControlFinding": ["explain_risk", "show_who_i_am"], "EmptyContext": []},
    )
    assert answer.splitlines() == [
        "Here is what I can do. Ask in your own words, for example:",
        "",
        '- **Close out**: "close this record"; "note what was done"',
        "- **Tag record**",
        "",
        "Or ask for a single action:",
        "",
        "- Open catalog",
        "- Who am I",
        "- In Control finding: explain risk, show who I am",
    ]


def test_commands_several_contexts_add_are_named_once_and_each_line_is_capped():
    shared = ["add_tag", "assess_risk", "propose_fix", "remove_tag", "show_metadata",
              "show_properties", "show_risk_score"]
    answer = render_capability_answer(
        {}, ["who_am_i"],
        {"Account": shared + ["list_permissions"],
         "Group": ["list_members"] + shared + ["show_nesting"],
         "Repository": shared,
         "Finding": ["explain_risk", "show_metadata"],
         "Directory": ["find_a", "find_b", "find_c", "find_d", "find_e", "find_f", "find_g"]},
    )
    assert answer.splitlines()[-6:] == [
        "- Who am I",
        "- In several places: add tag, assess risk, propose fix, remove tag, show metadata, "
        "show properties and 1 more",
        "- In Account: list permissions",
        "- In Group: list members, show nesting",
        "- In Finding: explain risk",
        "- In Directory: find a, find b, find c, find d, find e, find f and 1 more",
    ]


@pytest.fixture
def agent_ctx(monkeypatch):
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    ctx = WorkflowExecutionContext(run_as_agent=True)
    wf = fastworkflow.Workflow.create(
        str(Path(__file__).parent.joinpath("todo_list_workflow").resolve()),
        workflow_id_str=f"capability-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)
    agent = MagicMock(return_value=SimpleNamespace(final_answer="Agent answer"))
    ctx._workflow_tool_agent = agent
    planned = []
    monkeypatch.setattr(ctx, "_ensure_agent_initialized", lambda: None)
    monkeypatch.setattr(WorkflowExecutionContext, "_agent_dspy_context",
                        lambda self: (SimpleNamespace(model="test-model"), None))
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks",
                        lambda query, **kwargs: planned.append(query) or [query])
    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps",
                        lambda user_query, session, **kwargs: user_query)
    yield ctx, agent, planned
    RoutingRegistry.clear_registry()


def test_with_skills_the_turn_is_answered_in_code_without_the_planner_or_agent(agent_ctx, monkeypatch):
    ctx, agent, planned = agent_ctx
    monkeypatch.setattr("fastworkflow.workflow_agent.skill_utterances",
                        lambda folder: {"close_out": ("close this record",)})

    turn = ctx.process_turn("Hey, what else can you do?")

    assert planned == [] and not agent.called
    assert turn.success
    assert '- **Close out**: "close this record"' in turn.answer
    assert "Or ask for a single action:" in turn.answer
    other_lines = [line for line in turn.answer.splitlines() if line.startswith("- In ")]
    assert other_lines and all("go up" not in line and "wildcard" not in line for line in other_lines)
    assert ctx.conversation_history.messages  # recorded as any finished turn


def test_without_skills_the_turn_goes_the_normal_way(agent_ctx):
    ctx, agent, planned = agent_ctx

    turn = ctx.process_turn("what can you do?")

    assert planned == ["what can you do?"] and agent.called
    assert turn.answer == "Agent answer"
