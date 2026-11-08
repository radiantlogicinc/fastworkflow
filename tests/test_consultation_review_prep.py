"""Review edits, user rendering, and re-draft hooks for the consultation workflow (fix-kk46.1–.3)."""
from __future__ import annotations

import dspy
import pytest
from dspy.utils import DummyLM

from fastworkflow import review_engine as re
from fastworkflow import review_proposals
from fastworkflow.command_writes import write_commands
from fastworkflow.review_engine import Change, apply_edits, reply_policy
from fastworkflow.review_proposals import (
    Means, PendingItemSlot, ProposedEdit, SubTask, SubTaskInput, SubTaskProposal, merge_subtask_input_edit,
    model_edit_reader, planning_match, redraft_for_look_up)

from pathlib import Path

import fastworkflow

from tests.test_approval_binding import annotated_library, todo  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"


def _tasks(*goals: str) -> SubTaskProposal:
    return SubTaskProposal(items=[SubTask(goal=g) for g in goals])


def test_optional_item_number_defaults_to_one_on_a_single_item_proposal():
    proposal = _tasks("only one")
    edits = review_proposals._proposed_edits_to_engine(
        proposal, [ProposedEdit(op="change", field="goal", value="revised")])
    applied = apply_edits(proposal, edits)
    assert applied.items[0].goal == "revised"


def test_single_item_merge_hallucination_becomes_goal_change():
    proposal = _tasks("only one")
    edits = review_proposals._proposed_edits_to_engine(
        proposal,
        [ProposedEdit(op="merge", indexes=[1], value="revised goal")],
    )
    applied = apply_edits(proposal, edits)
    assert applied.items[0].goal == "revised goal"


def test_single_item_lone_drop_raises_edit_parse_error():
    proposal = _tasks("only one")
    with pytest.raises(re.EditParseError, match="I could not read that as an edit"):
        review_proposals._proposed_edits_to_engine(
            proposal, [ProposedEdit(op="drop", index=1)],
        )


def test_single_item_empty_merge_raises_edit_parse_error():
    proposal = _tasks("only one")
    with pytest.raises(re.EditParseError, match="I could not read that as an edit"):
        review_proposals._proposed_edits_to_engine(
            proposal, [ProposedEdit(op="merge", indexes=[1], value="")],
        )


def test_goal_change_strips_model_rendering_suffix():
    raw = (
        "1. Which controls does Henderson Ramirez and Lewis and Sons trip? "
        "(inputs: name=Henry Ramirez, kind=unit, fact=controls; means: planning:record_fact); "
        "desired outcome: not stated"
    )
    goal, _, _, _ = review_proposals._clean_goal_value(raw)
    assert goal == "Which controls does Henderson Ramirez and Lewis and Sons trip?"
    proposal = _tasks("old goal")
    edits = review_proposals._proposed_edits_to_engine(
        proposal, [ProposedEdit(op="change", index=1, field="goal", value=raw)],
    )
    applied = apply_edits(proposal, edits)
    assert applied.items[0].goal == goal
    assert "means:" not in applied.items[0].goal
    assert "desired outcome" not in applied.items[0].goal


def test_infer_step_from_in_step_one_phrase():
    proposal = _tasks("step one", "step two", "step three")
    reply = (
        "In step 1, the unit is Mcclain Miller and Henderson and Sons 1—walk every person on that roster. "
        "Steps 2 and 3 look fine."
    )
    edits = review_proposals._proposed_edits_to_engine(
        proposal,
        [ProposedEdit(op="change", index=None, field="unit", value="Mcclain Miller and Henderson and Sons 1")],
        reply_text=reply,
    )
    assert len(edits) == 1
    assert edits[0].index == 1


def test_duplicate_reader_input_edits_deduplicate():
    proposal = _tasks("a", "b", "c")
    edits = review_proposals._proposed_edits_to_engine(
        proposal,
        [
            ProposedEdit(op="change", index=1, field="unit", value="A"),
            ProposedEdit(op="change", index=1, field="unit", value="A"),
        ],
    )
    apply_edits(proposal, edits)


def test_resolve_slot_item_steps_two_through_six():
    from fastworkflow._workflows.consultation import _item_validation as item_validation

    proposal = SubTaskProposal(items=[SubTask(goal=f"g{i}") for i in range(1, 7)])
    assert item_validation.resolve_slot_item(proposal, "steps 2 through 6") == "2,3,4,5,6"
    two_step = SubTaskProposal(items=[SubTask(goal="a"), SubTask(goal="b")])
    assert item_validation.resolve_slot_item(two_step, "both steps") == "all"
    assert item_validation.resolve_slot_item(
        SubTaskProposal(items=[SubTask(goal="a"), SubTask(goal="b"), SubTask(goal="c")]),
        "all three",
    ) == "all"


def test_extract_quoted_names_ignores_contractions():
    text = (
        "don't change person's record; \"Henderson Ramirez and Lewis and Sons\" "
        "and 'Mcclain Miller and Henderson and Sons 1' are units."
    )
    names = review_proposals.extract_quoted_names(text)
    assert names == [
        "Henderson Ramirez and Lewis and Sons",
        "Mcclain Miller and Henderson and Sons 1",
    ]


def test_drop_with_quoted_name_updates_surviving_step():
    proposal = SubTaskProposal(items=[
        SubTask(
            goal="Which controls does Henderson Ramirez trip?",
            inputs=[
                SubTaskInput(name="name", value="Henderson Ramirez"),
                SubTaskInput(name="kind", value="person"),
                SubTaskInput(name="fact", value="controls"),
            ],
            means=Means(kind="planning", name="Library/record_fact"),
        ),
        SubTask(goal="notify"),
    ])
    reply = (
        'Drop the second step; "Henderson Ramirez and Lewis and Sons" is one organization in step 1.'
    )
    edits = review_proposals._proposed_edits_to_engine(
        proposal,
        [ProposedEdit(op="drop", index=2)],
        reply_text=reply,
    )
    after = apply_edits(proposal, edits)
    assert len(after.items) == 1
    item = after.items[0]
    by_name = {inp.name: inp.value for inp in item.inputs}
    assert by_name["name"] == "Henderson Ramirez and Lewis and Sons"
    assert by_name["kind"] == "unit"
    assert by_name["fact"] == "controls"
    assert item.goal == "Which controls does Henderson Ramirez and Lewis and Sons trip?"


def test_truncated_name_repaired_from_quotes():
    proposal = SubTaskProposal(items=[
        SubTask(
            goal="Which controls does Henderson Ramirez trip?",
            inputs=[SubTaskInput(name="name", value="Henderson Ramirez")],
            means=Means(kind="planning", name="Library/record_fact"),
        ),
    ])
    reply = "'Henderson Ramirez and Lewis and Sons' is the full unit name."
    edits = review_proposals._proposed_edits_to_engine(
        proposal,
        [ProposedEdit(op="change", index=1, field="name", value="Henderson and Lewis and Sons")],
        reply_text=reply,
    )
    after = apply_edits(proposal, edits)
    assert after.items[0].inputs[0].value == "Henderson Ramirez and Lewis and Sons"


def test_function_call_goal_becomes_inputs_preserves_untouched_inputs(todo, monkeypatch):
    from fastworkflow.review_proposals import DRAFT_ARGUMENTS_ENV_VAR

    monkeypatch.setenv(DRAFT_ARGUMENTS_ENV_VAR, "on")

    def _stub_draft(workflow, item, sub_task, earlier, planner_lm=None):
        return item.model_copy(
            update={
                "drafted_means": item.means.name,
                "inputs": [
                    SubTaskInput(name="name", value="Henderson Ramirez and Lewis and Sons", drafted=True),
                    SubTaskInput(name="kind", value="unit", drafted=True),
                    SubTaskInput(name="fact", value="attributes", drafted=True),
                ],
            }
        )

    monkeypatch.setattr(review_proposals, "drafted_item", _stub_draft)
    before_item = SubTask(
        goal="Which controls does Henderson Ramirez trip?",
        inputs=[
            SubTaskInput(name="name", value="Henderson Ramirez"),
            SubTaskInput(name="kind", value="person"),
            SubTaskInput(name="fact", value="controls"),
        ],
        means=Means(kind="planning", name="Library/record_fact"),
    )
    proposal = SubTaskProposal(items=[before_item])
    edits = review_proposals._proposed_edits_to_engine(
        proposal,
        [
            ProposedEdit(
                op="change",
                index=1,
                field="goal",
                value="record_fact(name='Henderson Ramirez and Lewis and Sons', kind='unit')",
            ),
        ],
    )
    after = apply_edits(proposal, edits)
    after.items[0]._before_item = before_item
    host = type("Host", (), {"_app_workflow": todo.wf, "_current_planner_lm": None})()
    revised = after.revised(host, proposal)
    item = revised.items[0]
    assert item.goal == "Which controls does Henderson Ramirez and Lewis and Sons trip?"
    assert {inp.name: inp.value for inp in item.inputs} == {
        "name": "Henderson Ramirez and Lewis and Sons",
        "kind": "unit",
        "fact": "controls",
    }


def test_title_field_change_maps_to_goal():
    proposal = _tasks("old goal")
    goal = "Which controls does Henderson Ramirez and Lewis and Sons trip?"
    edits = review_proposals._proposed_edits_to_engine(
        proposal,
        [ProposedEdit(op="change", index=1, field="title", value=goal)],
    )
    assert len(edits) == 1
    assert isinstance(edits[0], Change)
    assert edits[0].field == "goal"
    assert edits[0].value == goal


def test_multi_clause_drop_and_change_infer_indexes():
    proposal = _tasks("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")
    reply = (
        "Drop steps 1 and 2. In steps 7 and 8, walk each person's first login listed."
    )
    edits = review_proposals._proposed_edits_to_engine(
        proposal,
        [
            ProposedEdit(op="drop", index=None),
            ProposedEdit(op="change", index=None, field="goal", value="first login listed"),
        ],
        reply_text=reply,
    )
    assert {edit.index for edit in edits if isinstance(edit, re.Drop)} == {1, 2}
    assert all(edit.index in (7, 8) for edit in edits if isinstance(edit, Change))


def test_expand_indexes_list_on_proposed_edit():
    proposal = _tasks("a", "b", "c")
    edits = review_proposals._proposed_edits_to_engine(
        proposal,
        [ProposedEdit(op="drop", index=None, indexes=[1, 2])],
    )
    assert {edit.index for edit in edits if isinstance(edit, re.Drop)} == {1, 2}


def test_engine_dedupe_drop_beats_change_when_reply_drops():
    proposal = _tasks("a", "b")
    edits = review_proposals._dedupe_engine_edits(
        [re.Drop(index=1), Change(index=1, field="goal", value="x")],
        reply_text="Drop step 1",
        proposal=proposal,
    )
    assert any(isinstance(edit, re.Drop) and edit.index == 1 for edit in edits)
    assert not any(isinstance(edit, Change) and edit.index == 1 for edit in edits)


def test_engine_dedupe_merge_conflicting_means_keeps_last():
    proposal = _tasks("a", "b", "c", "d", "e")
    edits = review_proposals._dedupe_engine_edits(
        [
            Change(index=5, field="means", value="planning:A"),
            Change(index=5, field="means", value="planning:B"),
        ],
        proposal=proposal,
    )
    means_edits = [edit for edit in edits if isinstance(edit, Change) and edit.field == "means"]
    assert len(means_edits) == 1
    assert means_edits[0].value == "planning:B"


def test_input_value_splits_and_scope():
    merged = review_proposals.merge_subtask_input_edit(
        [],
        "unit=Mcclain Miller and Henderson and Sons 1 and scope=every",
    )
    assert {item.name: item.value for item in merged} == {
        "unit": "Mcclain Miller and Henderson and Sons 1",
        "scope": "every",
    }


def test_resolve_slot_item_from_prose_clauses():
    from fastworkflow._workflows.consultation import _item_validation as item_validation

    proposal = SubTaskProposal(items=[SubTask(goal=f"g{i}") for i in range(1, 10)])
    assert item_validation.resolve_slot_item(
        proposal,
        "Steps 1 and 2—drop them. Steps 7 and 8—use each person's first login listed.",
    ) == "1,2,7,8"
    assert item_validation.resolve_slot_item(
        proposal,
        "All remaining steps—run 3 through 9. Drop 1 and 2; steps 7–8 use first login listed.",
    ) in ("3,4,5,6,7,8,9", "1,2,3,4,5,6,7,8,9", "all")


def test_inputs_edit_splits_semicolon_and_drops_means():
    merged = review_proposals.merge_subtask_input_edit(
        [], "name=Henderson Ramirez, kind=unit, fact=controls; means=planning:record_fact")
    assert [each.render() for each in merged] == [
        "name=Henderson Ramirez",
        "kind=unit",
        "fact=controls",
    ]


def test_a_missing_item_on_several_items_asks_which_step():
    proposal = _tasks("a", "b")
    with pytest.raises(re.EditRefused, match="Which of the 2 steps"):
        review_proposals._proposed_edits_to_engine(
            proposal, [ProposedEdit(op="change", field="goal", value="x")])


def test_changing_one_input_keeps_the_others():
    proposal = SubTaskProposal(items=[
        SubTask(goal="g", inputs=[
            SubTaskInput(name="title", value="Emma"),
            SubTaskInput(name="notes", value="true"),
        ], means=Means(kind="planning", name="Library/book_by_title")),
    ])
    edited = apply_edits(proposal, [Change(index=1, field="inputs", value="title=Dune")])
    assert [each.render() for each in edited.items[0].inputs] == ["title=Dune", "notes=true"]


def test_a_comma_inside_an_input_value_is_not_split():
    assert merge_subtask_input_edit([], "title=Lewis and Sons, Inc.") == [
        SubTaskInput(name="title", value="Lewis and Sons, Inc.")
    ]


def test_index_zero_and_desired_outcome_field_on_one_item():
    proposal = _tasks("only one")
    edits = review_proposals._proposed_edits_to_engine(
        proposal, [ProposedEdit(op="change", index=0, field="desired outcome", value="world says yes")])
    applied = apply_edits(proposal, edits)
    assert applied.items[0].outcome == "world says yes"


def test_index_zero_on_several_items_opens_the_item_slot():
    proposal = _tasks("a", "b")
    with pytest.raises(PendingItemSlot, match="Which of the 2 steps"):
        review_proposals._proposed_edits_to_engine(
            proposal, [ProposedEdit(op="change", index=0, field="goal", value="x")])


def test_a_named_input_field_becomes_an_inputs_change():
    proposal = SubTaskProposal(items=[
        SubTask(goal="first"),
        SubTask(goal="second", inputs=[
            SubTaskInput(name="name", value="Amy"),
            SubTaskInput(name="kind", value="group"),
            SubTaskInput(name="fact", value="old"),
        ]),
    ])
    edits = review_proposals._proposed_edits_to_engine(
        proposal, [ProposedEdit(op="change", index=2, field="kind", value="unit")])
    applied = apply_edits(proposal, edits)
    assert [each.render() for each in applied.items[1].inputs] == [
        "name=Amy", "kind=unit", "fact=old",
    ]


def test_several_named_inputs_in_one_edit_apply_together():
    merged = merge_subtask_input_edit(
        [SubTaskInput(name="name", value="old")], "name=Amy Walsh, kind=person")
    assert [each.render() for each in merged] == ["name=Amy Walsh", "kind=person"]


def test_add_with_index_zero_and_parts_fills_the_goal():
    proposal = _tasks("first")
    edits = review_proposals._proposed_edits_to_engine(
        proposal, [ProposedEdit(op="add", index=0, after=1, value="", parts=["Findings driving risk"])])
    applied = apply_edits(proposal, edits)
    assert applied.items[1].goal == "Findings driving risk"


def test_two_input_keys_on_one_item_in_one_reply_both_apply():
    proposal = SubTaskProposal(items=[
        SubTask(goal="g", inputs=[SubTaskInput(name="name", value="old")]),
    ])
    edited = apply_edits(proposal, [
        Change(index=1, field="inputs", value="name=Ann"),
        Change(index=1, field="inputs", value="kind=person"),
    ])
    assert [each.render() for each in edited.items[0].inputs] == ["name=Ann", "kind=person"]


def test_two_fields_of_one_item_in_one_reply_both_apply():
    proposal = _tasks("a", "b")
    edited = apply_edits(proposal, [
        Change(index=1, field="goal", value="alpha"),
        Change(index=1, field="outcome", value="done"),
    ])
    assert edited.items[0].goal == "alpha" and edited.items[0].outcome == "done"


def test_two_changes_to_the_same_field_still_conflict():
    proposal = _tasks("a")
    with pytest.raises(re.EditRefused, match="more than one edit"):
        apply_edits(proposal, [
            Change(index=1, field="goal", value="x"),
            Change(index=1, field="goal", value="y"),
        ])


def test_a_drop_still_conflicts_with_a_field_change_on_that_item():
    proposal = _tasks("a", "b")
    with pytest.raises(re.EditRefused, match="more than one edit"):
        apply_edits(proposal, [Change(index=1, field="goal", value="x"), re.Drop(index=1)])


def test_the_user_rendering_omits_means_keys_and_the_edit_hint(todo, monkeypatch):
    folder = todo.wf.folderpath
    monkeypatch.setattr(
        "fastworkflow.command_writes.write_commands",
        lambda path: frozenset({"Library/book_loans"}),
    )
    proposal = SubTaskProposal(items=[
        SubTask(goal="record a fact", means=Means(kind="planning", name="Library/record_fact"),
                inputs=[SubTaskInput(name="fact", value="Ann is on loan")]),
        SubTask(goal="add a loan", means=Means(kind="planning", name="Library/book_loans"),
                inputs=[SubTaskInput(name="title", value="Dune")]),
    ])
    text = proposal.user_rendering(folder)
    assert "means:" not in text and "desired outcome" not in text and "change " not in text.lower()
    assert "1. record a fact (fact: Ann is on loan)" in text
    assert " - changes data" not in text.split("2. add a loan")[0]
    assert "2. add a loan (title: Dune) - changes data" in text
    assert all(line.strip() != "- changes data" for line in text.splitlines())


def test_prompt_for_sub_tasks_uses_the_new_closing_line():
    proposal = _tasks("a")
    assert re.prompt_for(proposal).endswith(re.SUB_TASK_PROMPT_HINT)


def test_the_reader_gets_the_model_rendering_not_the_user_one():
    seen: list[str] = []

    class RecordingLM(DummyLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            seen.append(str(prompt or messages))
            return super().forward(prompt=prompt, messages=messages, **kwargs)

    proposal = SubTaskProposal(items=[
        SubTask(goal="g", means=Means(kind="planning", name="Library/book_loans"),
                inputs=[SubTaskInput(name="title", value="Dune")]),
    ])
    lm = RecordingLM([{"edits": [{"op": "change", "index": 1, "field": "goal", "value": "new"}]}])
    reader = model_edit_reader(planner_lm=lm)
    reader("make it about trains", proposal)
    assert seen and "means: planning:Library/book_loans" in seen[0]


def test_a_goal_edit_re_drafts_means_and_inputs(todo, monkeypatch):
    from fastworkflow.review_proposals import DRAFT_ARGUMENTS_ENV_VAR

    monkeypatch.setenv(DRAFT_ARGUMENTS_ENV_VAR, "on")

    def _stub_draft(workflow, item, sub_task, earlier, planner_lm=None):
        return item.model_copy(update={"drafted_means": item.means.name,
                                       "inputs": [SubTaskInput(name="title", value="stub", drafted=True)]})

    monkeypatch.setattr(review_proposals, "drafted_item", _stub_draft)
    before = SubTaskProposal(items=[SubTask(goal="find Emma by title")])
    after = SubTaskProposal(items=[SubTask(goal="find Persuasion by title",
                                           means=Means(kind="planning", name="Library/book_by_title"))])
    host = type("Host", (), {"_app_workflow": todo.wf, "_current_planner_lm": None})()
    revised = after.revised(host, before)
    assert revised.items[0].drafted_means == "Library/book_by_title"


def test_planning_match_exclude_changes_the_turn_cache_key(todo, monkeypatch):
    keys: list = []
    from fastworkflow import turn_cache

    real = turn_cache.memo

    def recorder(key, compute):
        keys.append(key)
        return real(key, compute)

    monkeypatch.setattr(turn_cache, "memo", recorder)
    text = "loan a book for the consultation cache key test"
    planning_match(todo.wf, text, exclude=frozenset())
    planning_match(todo.wf, text, exclude=frozenset({"Library/book_loans"}))
    assert len(keys) == 2 and keys[0] != keys[1]


def test_redraft_for_look_up_preserves_the_goal(todo, monkeypatch):
    writes = frozenset({"Library/book_loans"})
    monkeypatch.setattr("fastworkflow.command_writes.write_commands", lambda path: writes)
    item = SubTask(goal="add a loan for Ann", means=Means(kind="planning", name="Library/book_loans"),
                   inputs=[SubTaskInput(name="title", value="Dune")])
    redrafted = redraft_for_look_up(todo.wf, item, earlier=[], planner_lm=None)
    assert redrafted.goal == item.goal
    assert redrafted.means.name not in writes


def test_sub_task_policy_reprompt_uses_plain_unreadable_text():
    policy = reply_policy(re.KIND_SUB_TASKS)
    decision = policy.decide(_tasks("a"), "hmm")
    assert decision.verdict is re.Verdict.REPROMPT
    assert decision.message == re.SUB_TASK_UNREADABLE_REPLY


def test_prompt_for_sub_tasks_structured_uses_structured_hint():
    proposal = _tasks("a")
    assert re.prompt_for(proposal, structured=True).endswith(re.SUB_TASK_STRUCTURED_PROMPT_HINT)
    assert re.prompt_for(proposal, structured=False).endswith(re.SUB_TASK_PROMPT_HINT)


def test_structured_error_prompt_uses_current_steps_header():
    proposal = SubTaskProposal(items=[
        SubTask(goal="first", inputs=[SubTaskInput(name="kind", value="person")]),
        SubTask(goal="second"),
    ])
    text = re._structured_error_prompt(proposal, None, re.SUB_TASK_UNREADABLE_REPLY)
    assert text.startswith("I couldn't tell what to change.")
    assert "Current steps:" in text
    assert "Sub-tasks:" not in text
    assert text.endswith(re.SUB_TASK_STRUCTURED_PROMPT_HINT)


def test_normalize_parsed_subtask_edits_maps_kind_to_inputs():
    proposal = SubTaskProposal(items=[
        SubTask(
            goal="record",
            inputs=[
                SubTaskInput(name="name", value="Henderson Ramirez"),
                SubTaskInput(name="kind", value="person"),
            ],
        ),
        SubTask(goal="notify"),
    ])
    policy = reply_policy(re.KIND_SUB_TASKS)
    decision = policy.decide(proposal, "change 2: kind=application")
    assert decision.verdict is re.Verdict.EDIT
    edited = apply_edits(proposal, list(decision.edits))
    assert len(edited.items) == 2


def test_normalize_merges_several_input_keys_in_one_reply():
    proposal = SubTaskProposal(items=[
        SubTask(
            goal="Which controls does Henderson Ramirez trip?",
            inputs=[
                SubTaskInput(name="name", value="Henderson Ramirez"),
                SubTaskInput(name="kind", value="person"),
            ],
        ),
        SubTask(goal="notify"),
    ])
    policy = reply_policy(re.KIND_SUB_TASKS)
    text = (
        "change 1: inputs=name='Henderson Ramirez and Lewis and Sons', kind=unit; drop 2"
    )
    decision = policy.decide(proposal, text)
    assert decision.verdict is re.Verdict.EDIT
    edited = apply_edits(proposal, list(decision.edits))
    assert len(edited.items) == 1
    by_name = {inp.name: inp.value for inp in edited.items[0].inputs}
    assert by_name["name"] == "Henderson Ramirez and Lewis and Sons"
    assert by_name["kind"] == "unit"


def test_tag_action_remove_paraphrase_confirms_not_corrective():
    policy = reply_policy(re.KIND_SUB_TASKS)
    proposal = SubTaskProposal(items=[SubTask(goal="tag the HR feed")])
    decision = policy.decide(
        proposal,
        "Yes, go ahead and remove the audit-scope tag from the HR feed",
    )
    assert decision.verdict is re.Verdict.CONFIRM


def test_tag_action_add_paraphrase_confirms_not_corrective():
    policy = reply_policy(re.KIND_SUB_TASKS)
    proposal = SubTaskProposal(items=[SubTask(goal="tag a feed")])
    decision = policy.decide(
        proposal,
        "Yes, add audit-scope to the RLDEVENTRA feed",
    )
    assert decision.verdict is re.Verdict.CONFIRM
