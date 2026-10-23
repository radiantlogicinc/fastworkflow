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


def test_the_answer_names_each_skill_with_its_shortest_example_then_each_command_then_other_contexts():
    answer = render_capability_answer(
        {"close_out": ("close this record", "note what was done", "  ", "a third one"),
         "tag_record": ()},
        ["open_catalog", "who_am_i"],
        {"ControlFinding": ["explain_risk", "show_who_i_am"], "EmptyContext": []},
    )
    assert answer.splitlines() == [
        "Here is what I can do. Ask in your own words, for example:",
        "",
        '- **Close out**: "a third one"',
        "- **Tag record**",
        "",
        "Or ask for a single action:",
        "",
        "- Open catalog",
        "- Who am I",
        "- In Control finding: explain risk, show who I am",
    ]


def test_commands_more_than_half_of_the_other_contexts_add_are_named_once():
    answer = render_capability_answer(
        {}, ["who_am_i"],
        {"Account": ["add_tag", "show_metadata", "list_permissions"],
         "Group": ["add_tag", "show_metadata", "list_members"],
         "Repository": ["add_tag", "list_members"],
         "Finding": ["explain_risk", "show_metadata"]},
    )
    # add tag (3 of 4) and show metadata (3 of 4) are shared; list members (2 of 4) is not
    assert answer.splitlines()[-5:] == [
        "- In several places: add tag, show metadata",
        "- In Account: list permissions",
        "- In Group: list members",
        "- In Repository: list members",
        "- In Finding: explain risk",
    ]


def test_a_context_left_with_no_commands_of_its_own_gets_no_line():
    answer = render_capability_answer(
        {}, ["who_am_i"],
        {"Account": ["add_tag"], "Group": ["add_tag"], "Repository": ["add_tag", "list_accounts"]},
    )
    assert answer.splitlines()[-3:] == [
        "- Who am I",
        "- In several places: add tag",
        "- In Repository: list accounts",
    ]


def test_a_context_line_names_commands_while_it_fits_the_line_width(monkeypatch):
    monkeypatch.setattr("fastworkflow.capability_answer.LINE_WIDTH", 42)
    answer = render_capability_answer(
        {}, [],
        {"Directory": ["find_a", "find_b", "find_c", "find_d", "find_e", "find_f"],
         "Short": ["list_x", "list_y"],
         "Long": ["a_command_name_far_wider_than_the_line_itself", "show_z"]},
    )
    lines = answer.splitlines()[-3:]
    assert lines == [
        "- In Directory: find a, find b and 4 more",
        "- In Short: list x, list y",
        "- In Long: a command name far wider than the line itself, show z",
    ]
    assert len(lines[0]) <= 42 and len(lines[1]) <= 42


def test_of_equally_short_examples_the_first_is_shown():
    answer = render_capability_answer(
        {"review": ("review this one carefully", "check it", "audit it", "go")}, [])
    assert answer.splitlines()[2] == '- **Review**: "go"'
    answer = render_capability_answer({"review": ("review it", "check it", "audit it")}, [])
    assert answer.splitlines()[2] == '- **Review**: "check it"'


def test_a_line_never_leaves_out_just_one_command(monkeypatch):
    monkeypatch.setattr("fastworkflow.capability_answer.LINE_WIDTH", 50)
    answer = render_capability_answer(
        {}, [],
        {"Account": ["find_a", "find_b", "find_c", "show_account_manager"],
         "Group": ["find_a1", "find_b1", "find_c1", "find_d1", "find_e1", "find_f1"]},
    )
    lines = answer.splitlines()[-2:]
    # "find a, find b, find c and 1 more" would fit in 50; the last one is named instead
    assert lines == [
        "- In Account: find a, find b, find c, show account manager",
        "- In Group: find a1, find b1, find c1 and 3 more",
    ]
    assert len(lines[0]) > 50 and len(lines[1]) <= 50


def test_the_shared_line_may_fill_two_line_widths(monkeypatch):
    monkeypatch.setattr("fastworkflow.capability_answer.LINE_WIDTH", 30)
    shared = ["cmd_a", "cmd_b", "cmd_c", "cmd_d", "cmd_e", "cmd_f", "cmd_g", "cmd_h"]
    answer = render_capability_answer({}, [], {"One": shared, "Two": shared, "Three": shared})
    line = answer.splitlines()[-1]
    assert line == "- In several places: cmd a, cmd b, cmd c, cmd d and 4 more"
    assert 30 < len(line) <= 60


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
