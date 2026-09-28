"""The plain-text plan fallback (fix-4dsr): steps, optional and user-gated flags.

A text plan is what the finish check reads when the structured planner fails, so
a step the parser merges or mis-flags is a step the check never asks about.
"""
import json
import os

import pytest

import fastworkflow
from fastworkflow.turn_plan import (
    PlanPart,
    PlanStep,
    PlanSubject,
    TurnPlan,
    is_checked,
    parse_text_plan,
    step_commands,
)
# Structured planning disabled 2026-09-28 (owner decision); kept for reference.
# from fastworkflow.workflow_agent import _redacted_subject_names, _text_turn_plan
from fastworkflow.workflow_agent import _text_turn_plan

TODO_WORKFLOW = os.path.join(os.path.dirname(__file__), "todo_list_workflow")


def _flags(text):
    return [(step.optional, step.needs_user) for step in parse_text_plan(text).steps]


def test_a_one_line_plan_is_split_into_its_steps():
    plan = parse_text_plan("1. Find. 2. If needed, list groups. 3. List accounts")
    assert [step.text for step in plan.steps] == ["Find.", "If needed, list groups.", "List accounts"]
    assert [step.optional for step in plan.steps] == [False, True, False]
    assert [is_checked(step) for step in plan.steps] == [True, False, True]


def test_inline_numbers_that_do_not_continue_the_list_stay_in_the_step():
    plan = parse_text_plan("1. List the 5. accounts and check 7) of them\n2. Report")
    assert [step.text for step in plan.steps] == ["List the 5. accounts and check 7) of them", "Report"]


def test_a_value_that_looks_like_the_next_step_number_stays_in_the_step():
    plan = parse_text_plan("1. Set the page size to 2. Then list items\n2. Show the first item")
    assert [step.text for step in plan.steps] == ["Set the page size to 2. Then list items",
                                                  "Show the first item"]


def test_a_line_holding_later_steps_is_split_after_multi_line_steps():
    plan = parse_text_plan("1. Find Alan\n2. Open him. 3. List his accounts\n4. Report")
    assert [step.text for step in plan.steps] == ["Find Alan", "Open him.", "List his accounts", "Report"]


def test_user_gating_late_in_a_long_step_is_seen():
    text = ("1. Find the stale accounts with find_accounts.\n"
            "2. Delete each of the three stale accounts with delete_account for the identity "
            "that owns them after user confirmation.")
    assert _flags(text) == [(False, False), (False, True)]


def test_wider_approval_phrasings_mark_the_step_user_gated():
    for phrasing in ("Delete the account if the user approves.",
                     "Delete the account if approved.",
                     "Delete the account if the user confirmed.",
                     "Delete the account if the user agrees.",
                     "Ask the user to confirm, then delete the account.",
                     "Ask for approval before deleting the account.",
                     "Delete the account with the user's approval.",
                     "Delete the account with user confirmation."):
        assert _flags(f"1. {phrasing}") == [(False, True)], phrasing


def test_ordinary_steps_are_not_optional_or_user_gated():
    for text in ("List accounts with optional filters.",
                 "Find the person, optionally with a filter on department.",
                 "Use the optional parameter to narrow the search.",
                 "Confirm the account exists with find_account.",
                 "Delete the account the user named."):
        assert _flags(f"1. {text}") == [(False, False)], text


def test_optional_steps_are_recognised_anywhere_in_the_step():
    for text in ("Optional: list his groups.",
                 "**Optionally** list his groups.",
                 "List his groups (optional).",
                 "List his groups; this step is optional.",
                 "List his groups if necessary.",
                 "Open the identity and list his groups if needed."):
        assert _flags(f"1. {text}") == [(True, False)], text


def test_a_text_plan_the_parser_cannot_read_is_no_plan_not_a_failed_turn():
    assert _text_turn_plan(123, TODO_WORKFLOW) is None  # type: ignore[arg-type]
    plan = _text_turn_plan("1. Add a todo", TODO_WORKFLOW)
    assert plan is not None and plan.source == "text" and len(plan.steps) == 1


@pytest.mark.skip(reason="structured planning disabled 2026-09-28 (owner decision)")
def test_planner_span_subjects_pass_the_capture_policy():
    secret = "Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz123456"
    plan = TurnPlan(steps=[PlanStep(text="Find them")],
                    subjects=[PlanSubject(name="Alan Cooper", kind="person"),
                              PlanSubject(name=secret, kind="token")])
    names = _redacted_subject_names(plan)  # noqa: F821 - commented out with structured planning
    assert names[0] == "Alan Cooper"
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in json.dumps(names)
    assert _redacted_subject_names(None) == []  # noqa: F821 - commented out with structured planning


def test_a_command_written_with_its_context_is_recovered_by_its_bare_name():
    known = {"find_identity", "apply_remediation", "list_accounts"}
    plan = parse_text_plan("1. Find him with `DirectoryExplorer/find_identity`.\n"
                           "2. Run `Resource/apply_remediation` on each finding.\n"
                           "3. List `Identity/Account/list_accounts` and his `Identity/identity_uid`.\n"
                           "4. Report `Resource/NotACommand`.", known)
    assert [step.commands for step in plan.steps] == [
        ["find_identity"], ["apply_remediation"], ["list_accounts"], []]


def test_qualified_names_are_still_filtered_to_the_workflows_commands():
    fastworkflow.init({})
    plan = _text_turn_plan("1. Add it with `TodoList/add_child_todoitem`.\n"
                           "2. Read `TodoItem/description_text` and `TodoItem/mark_completed`.\n"
                           "3. Check `IntentDetection/what_can_i_do`.", TODO_WORKFLOW)
    assert [step.commands for step in plan.steps] == [
        ["add_child_todoitem"], ["mark_completed"], ["what_can_i_do"]]


def test_awaiting_confirmation_marks_the_step_user_gated():
    for phrasing in ("Await the user's confirmation, then delete the account.",
                     "Await user confirmation before deleting the account.",
                     "Await confirmation and delete the account.",
                     "Delete the account, awaiting the user's approval.",
                     "Wait for the user's confirmation, then delete the account."):
        assert _flags(f"1. {phrasing}") == [(False, True)], phrasing


def test_step_commands_are_the_step_and_every_part_once_each():
    step = PlanStep(text="Walk each person", commands=["find_identity"],
                    parts=[PlanPart(text="list accounts", commands=["list_accounts", "find_identity"]),
                           PlanPart(text="tag them", commands=["add_tag"], optional=True)])
    assert step_commands(step) == ["find_identity", "list_accounts", "add_tag"]
    assert step_commands(PlanStep(text="Report")) == []
