"""End-to-end consultation reply policy (fix-kk46.7): trained head, real CME, scripted planner only."""
from __future__ import annotations

import os
from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import review_engine as re_
from fastworkflow import review_proposals
from fastworkflow.review_engine import CONFIRMED, DECLINED, REVISED
from fastworkflow.review_proposals import Means, ProposedEdit, SubTask, SubTaskInput, SubTaskProposal
from fastworkflow.turn import TurnStatus

from tests.test_approval_binding import _finish, annotated_library, no_live_model, todo  # noqa: F401
from tests.test_round1_gate import EMMA, JOKE, gated, _open_review, _split_into  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

CONSULTATION = fastworkflow.get_internal_workflow_path("consultation")
LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"
DROP = "ProposalReview/drop"


def _head_trained() -> bool:
    return os.path.isfile(os.path.join(CONSULTATION, "___command_info", "ProposalReview", "threshold.json"))


pytestmark = [
    pytest.mark.usefixtures("initialized_fastworkflow", "no_live_model", "no_draft_extraction"),
    pytest.mark.skipif(not _head_trained(), reason="consultation ProposalReview head is not trained"),
]


@pytest.fixture(autouse=True)
def no_draft_extraction(monkeypatch):
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")


@pytest.fixture(autouse=True)
def script_param_extraction(monkeypatch):
    from fastworkflow.utils import signatures

    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")
    param_lm = DummyLM(
        [{"reasoning": "n", "item": "2"}, {"reasoning": "n", "item": "1"}] * 12,
        adapter=dspy.JSONAdapter(),
    )

    def get_lm(role, *a, **k):
        if role == "LLM_PARAM_EXTRACTION":
            return param_lm
        if role == "LLM_PLANNER":
            return _PlannerLM([{"edits": []}] * 30)
        raise AssertionError(f"unscripted model role {role!r}")

    monkeypatch.setattr("fastworkflow.utils.dspy_utils.get_lm", get_lm)


class _PlannerLM(DummyLM):
    """Planner / edit-reader boundary (same shape as test_round1_handoff._reply_lm)."""

    def __init__(self, answers: list[dict]):
        super().__init__(answers)


@pytest.fixture
def planner(monkeypatch):
    """Script the planner-role LM (free-text edit reader)."""
    holder: dict = {}

    def bind(gated, *answers: dict) -> _PlannerLM:
        lm = _PlannerLM(list(answers))
        holder["lm"] = lm
        gated.ctx._current_planner_lm = lm
        param_lm = DummyLM(
            [
                {"reasoning": "n", "items": ["2"], "item": "2"},
                {"reasoning": "n", "item": "1"},
            ]
            * 10,
            adapter=dspy.JSONAdapter(),
        )

        def get_lm(role, *a, **k):
            if role == "LLM_PARAM_EXTRACTION":
                return param_lm
            return lm

        monkeypatch.setattr("fastworkflow.utils.dspy_utils.get_lm", get_lm)
        return lm

    bind.holder = holder
    return bind


def _wire_consultation(gated) -> None:
    gated.store.consultation_cme = gated.ctx._consultation_cme
    gated.store.consultation_workflow = gated.ctx._consultation_workflow


def _open_review_two_step(gated, monkeypatch) -> re_.ReviewRecord:
    _split_into(monkeypatch, [EMMA, JOKE])
    out = gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    assert out.status is TurnStatus.AWAITING_USER
    record = _open_review(gated)
    store = gated.store
    _wire_consultation(gated)
    return record


def test_policy_decide_confirm_paraphrase(gated, monkeypatch):
    record = _open_review_two_step(gated, monkeypatch)
    from fastworkflow.review_engine import Verdict, proposal_of, reply_policy

    decision = reply_policy(
        record.kind,
        gated.store.edit_reader,
        consultation_cme=gated.store.consultation_cme,
        consultation_workflow=gated.store.consultation_workflow,
        app_workflow=gated.wf,
    ).decide(proposal_of(record), "sounds good to me")
    assert decision.verdict is Verdict.CONFIRM


def test_confirm_paraphrase(gated, monkeypatch):
    from fastworkflow.review_engine import reply_to_proposal

    record = _open_review_two_step(gated, monkeypatch)
    review = reply_to_proposal(gated.ctx, gated.store, record, "sounds good to me")
    assert review.done and record.state == CONFIRMED


def test_decline_paraphrase(gated, monkeypatch):
    record = _open_review_two_step(gated, monkeypatch)
    gated.ctx.process_turn("never mind")
    assert record.state == DECLINED


def test_compound_edit_with_scripted_planner(gated, monkeypatch, planner):
    record = _open_review_two_step(gated, monkeypatch)
    planner(gated, {
        "edits": [
            {"op": "drop", "index": 2},
            {"op": "change", "index": 1, "field": "goal", "value": "find Persuasion instead"},
        ],
    })
    out = gated.ctx.process_turn("in step 1 make the branch Central and drop step 2")
    assert record.state == REVISED
    assert "Persuasion" in (out.answer or record.rendering)


def test_bare_value_correction_with_scripted_planner(gated, monkeypatch, planner):
    record = _open_review_two_step(gated, monkeypatch)
    planner(gated, {"edits": [{"op": "change", "index": 1, "field": "goal", "value": "Lewis and Sons is a company"}]})
    out = gated.ctx.process_turn("Lewis and Sons is a company")
    assert record.state == REVISED
    assert "Lewis and Sons" in (out.answer or record.rendering)


def test_item_slot_then_answered(gated, monkeypatch, planner):
    record = _open_review_two_step(gated, monkeypatch)
    planner(
        gated,
        {"edits": [{"op": "change", "field": "goal", "value": "make it about trains instead"}]},
        {"edits": [{"op": "change", "index": 2, "field": "goal", "value": "make it about trains instead"}]},
    )
    first = gated.ctx.process_turn("make it about trains instead")
    assert "Which of the 2 steps do you mean?" in (first.answer or "")
    assert gated.ctx._consultation_cme.context.get("command_name") == "ProposalReview/edit"
    second = gated.ctx.process_turn("2")
    assert record.state == REVISED
    assert "trains" in (second.answer or record.rendering).lower()


def _enter_drop_item_slot(gated):
    from fastworkflow.utils.signatures import NOT_FOUND

    routing = fastworkflow.RoutingRegistry.get_definition(gated.ctx._consultation_workflow.folderpath)
    params_class = routing.get_command_class(DROP, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS)
    cme = gated.ctx._consultation_cme.context
    cme["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    cme["command"] = "drop"
    cme["command_name"] = DROP
    cme["stored_parameters"] = params_class.model_construct(items=[NOT_FOUND])
    cme["missing_value_question"] = "Which of the 2 steps do you mean?"


@pytest.mark.parametrize("intent", ["wait", "abort", "misunderstood", "different_command"])
def test_item_slot_jev_escape(intent, gated, monkeypatch, planner):
    from fastworkflow._workflows.command_metadata_extraction._commands import wildcard as wildcard_mod

    record = _open_review_two_step(gated, monkeypatch)
    _enter_drop_item_slot(gated)
    cme = gated.ctx._consultation_cme

    def missing_value_intent(_workflow, _app_workflow, _command_name, _command):
        fields = ["item"] if intent != "different_command" else []
        return intent, fields

    monkeypatch.setattr(wildcard_mod, "_missing_value_reply_intent", missing_value_intent)
    if intent == "misunderstood":
        planner(gated, {"edits": [{"op": "change", "index": 1, "field": "goal", "value": "ok"}]})
    out = gated.ctx.process_turn("hold on")
    text = out.answer or ""
    if intent == "wait":
        assert "Which of the 2 steps do you mean?" in text
    elif intent == "abort":
        assert "OK, nothing changed." in text
        assert record.state == re_.AWAITING
    elif intent == "misunderstood":
        assert "you_misunderstood" not in text.lower()
    elif intent == "different_command":
        planner(gated, {"edits": [{"op": "change", "index": 1, "field": "goal", "value": "ok"}]})
        assert cme.context.get("NLU_Pipeline_Stage") == fastworkflow.NLUPipelineStage.INTENT_DETECTION


def test_cap_turn_confirm_paraphrase(gated, monkeypatch):
    from fastworkflow.review_engine import reply_to_proposal

    record = _open_review_two_step(gated, monkeypatch)
    gated.ctx._current_planner_lm = _PlannerLM([{"edits": "not-a-list"}] * 12)
    for _ in range(8):
        review = reply_to_proposal(gated.ctx, gated.store, record, "make it better somehow")
        assert not review.done
        record.turns_seen += 1
        assert record.state == re_.AWAITING
    assert record.turns_seen == 8
    reply_to_proposal(gated.ctx, gated.store, record, "sounds good to me", cap_turn=True)
    assert record.state == CONFIRMED


def test_restart_during_edit_item_slot(gated, monkeypatch, planner):
    import uuid
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    record = _open_review_two_step(gated, monkeypatch)
    planner(
        gated,
        {"edits": [{"op": "change", "field": "goal", "value": "make it about trains instead"}]},
        {"edits": [{"op": "change", "index": 2, "field": "goal", "value": "make it about trains instead"}]},
    )
    first = gated.ctx.process_turn("make it about trains instead")
    assert "Which of the 2 steps do you mean?" in (first.answer or "")
    channel = gated.ctx.session_key or f"kk46_{uuid.uuid4().hex[:8]}"
    blob = gated.ctx.serialize_state(channel_id=channel)
    wf_path = gated.wf.folderpath
    gated.ctx.close()

    restored = WorkflowExecutionContext(run_as_agent=False, session_key=channel)
    restored_wf = fastworkflow.Workflow.create(wf_path, workflow_id_str=channel)
    restored.bind_app_workflow(restored_wf)
    restored.apply_serialized_state(blob)
    turn = restored._turn_frame()
    turn.subtask_review_open = True
    restored_store = __import__("fastworkflow.approval_binding", fromlist=["store_for"]).store_for(restored_wf)
    restored_store.consultation_cme = restored._consultation_cme
    restored_store.consultation_workflow = restored._consultation_workflow
    restored._current_planner_lm = _PlannerLM(
        [{"edits": [{"op": "change", "index": 2, "field": "goal", "value": "make it about trains instead"}]}]
    )
    from fastworkflow import approval_binding as ab
    from fastworkflow.review_engine import reply_to_proposal

    restored_record = ab.store_for(restored_wf).active()
    assert restored_record is not None
    review = reply_to_proposal(restored, restored_store, restored_record, "2")
    assert not review.done
    assert restored_record.state == REVISED
    assert "trains" in (review.text or restored_record.rendering).lower()


def test_look_up_on_read_command_preserves_goal(gated, monkeypatch):
    from fastworkflow.review_engine import open_review

    item = SubTask(
        goal="record a fact",
        means=Means(kind="planning", name="Library/record_fact"),
        inputs=[SubTaskInput(name="fact", value="Ann is on loan")],
    )
    record = open_review(gated.ctx, gated.store, SubTaskProposal(items=[item, SubTask(goal="tell Ann")]),
                         surface=gated.store.surface)
    _wire_consultation(gated)
    before_goal = item.goal
    gated.ctx.process_turn("look up step 1")
    assert record.state == REVISED
    after = re_.proposal_of(record).items[0]
    assert after.goal == before_goal


def test_look_up_on_write_command_rematches_without_writes(gated, monkeypatch):
    from fastworkflow.review_engine import open_review

    item = SubTask(
        goal="add a loan for Ann",
        means=Means(kind="planning", name="Library/book_loans"),
        inputs=[SubTaskInput(name="title", value="Dune")],
    )
    record = open_review(gated.ctx, gated.store, SubTaskProposal(items=[item, SubTask(goal="notify")]),
                         surface=gated.store.surface)
    _wire_consultation(gated)
    gated.ctx.process_turn("look up step 1")
    assert record.state == REVISED
    after = re_.proposal_of(record).items[0]
    assert after.goal == item.goal


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("the last step", -1),
        ("the second step", 2),
        ("the first one", 1),
        ("the last part", -1),
        ("fifth", 5),
    ],
)
def test_parse_explicit_index_step_phrases(phrase, expected):
    from fastworkflow._workflows.consultation import _item_validation as item_validation

    assert item_validation._parse_explicit_index(phrase) == expected


def test_parse_explicit_index_step_prefix_with_explanation():
    from fastworkflow._workflows.consultation import _item_validation as item_validation

    text = "Step 1—I want the RLIQA login count in that step only, not step 2."
    assert item_validation._parse_explicit_index(text) == 1


@pytest.mark.parametrize(
    "phrase",
    [
        "Yes, list every right Brent Randolph holds for recertification.",
        "Yes, count how many logins arrive on the RLIQA feed.",
        "Yes, only the match is needed.",
        "Yes—tag exactly Davis and Sons unit with restructure-2026, not the LLC or PLC names.",
    ],
)
def test_affirmative_restate_goal_not_corrective(phrase):
    from fastworkflow.review_engine import _affirmative_accept_cue

    assert _affirmative_accept_cue(phrase)


def test_yes_shown_but_not_applied_is_accept_cue():
    from fastworkflow.review_engine import _affirmative_accept_cue

    text = (
        "Yes—finding summary first, then Catherine Taylor offboarding with corrections "
        "shown but not applied."
    )
    assert _affirmative_accept_cue(text)


def test_compound_drop_edit_plain_drop_only():
    from fastworkflow.review_engine import _compound_drop_edit

    assert not _compound_drop_edit("drop step 2", ["2"])


def test_compound_drop_edit_semicolon_and_extra_step():
    from fastworkflow.review_engine import _compound_drop_edit

    text = "Drop step 2; step 1 alone should return how many logins arrive on the RLIQA feed."
    assert _compound_drop_edit(text, ["2"])


def test_compound_drop_edit_is_one_organization_triggers_edit_reader(
    gated, monkeypatch, planner,
):
    from fastworkflow._workflows.consultation._commands.ProposalReview.drop import (
        Signature as DropSignature,
    )
    from fastworkflow.review_engine import Change, Drop, ProposalReplyPolicy, Verdict, proposal_of

    record = _open_review_two_step(gated, monkeypatch)
    proposal = proposal_of(record)
    planner(
        gated,
        {
            "edits": [
                {"op": "drop", "index": 2},
                {
                    "op": "change",
                    "index": 1,
                    "field": "goal",
                    "value": "Henderson Ramirez and Lewis and Sons is one organization",
                },
            ],
        },
    )
    policy = ProposalReplyPolicy(
        gated.store.edit_reader,
        consultation_cme=gated.store.consultation_cme,
        consultation_workflow=gated.store.consultation_workflow,
        app_workflow=gated.wf,
    )

    class _WildOut:
        command_name = DROP
        cmd_parameters = DropSignature.Input(items=["2"])

    text = "Drop step 2; Henderson Ramirez and Lewis and Sons is one organization in step 1."
    decision = policy._handle_routed_consultation(
        proposal,
        text,
        _WildOut(),
        "Which of the 2 steps do you mean?",
    )
    assert decision.verdict is Verdict.EDIT
    assert Drop(index=2) in decision.edits
    assert any(isinstance(edit, Change) and edit.index == 1 for edit in decision.edits)


def test_slot_whole_plan_confirm_clears_edit_slot(gated, monkeypatch, planner):
    from fastworkflow._workflows.consultation._commands.ProposalReview.edit import Signature as EditSignature
    from fastworkflow.review_engine import reply_to_proposal
    from fastworkflow.utils.signatures import NOT_FOUND

    record = _open_review_two_step(gated, monkeypatch)
    cme = gated.ctx._consultation_cme.context
    cme["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    cme["command_name"] = "ProposalReview/edit"
    cme["stored_parameters"] = EditSignature.SlotInput.model_construct(
        item=NOT_FOUND,
        edits=[{"op": "change", "field": "goal", "value": "x"}],
    )
    cme["missing_value_question"] = "Which of the 2 steps do you mean?"
    review = reply_to_proposal(
        gated.ctx,
        gated.store,
        record,
        "Both steps—I am confirming the whole plan, not changing either one.",
    )
    assert review.done and record.state == CONFIRMED


def test_yes_list_rights_confirms_single_step_review(gated, monkeypatch, planner):
    from fastworkflow.review_engine import open_review, reply_to_proposal

    goal = "List every right Brent Randolph holds for recertification."
    item = SubTask(
        goal=goal,
        means=Means(kind="planning", name="Library/record_fact"),
        inputs=[],
    )
    record = open_review(gated.ctx, gated.store, SubTaskProposal(items=[item]), surface=gated.store.surface)
    _wire_consultation(gated)
    planner(gated, {"edits": [{"op": "change", "index": 1, "field": "goal", "value": goal}]})
    review = reply_to_proposal(gated.ctx, gated.store, record, f"Yes, {goal}")
    assert review.done and record.state == CONFIRMED


def test_drop_fallback_keeps_valid_extraction_when_edit_reader_opens_slot(
    gated, monkeypatch, planner,
):
    from fastworkflow._workflows.consultation._commands.ProposalReview.drop import (
        Signature as DropSignature,
    )
    from fastworkflow.review_engine import Change, Drop, ProposalReplyPolicy, Verdict, proposal_of

    record = _open_review_two_step(gated, monkeypatch)
    proposal = proposal_of(record)
    planner(
        gated,
        {"edits": [{"op": "change", "field": "goal", "value": "x"}]},
    )
    policy = ProposalReplyPolicy(
        gated.store.edit_reader,
        consultation_cme=gated.store.consultation_cme,
        consultation_workflow=gated.store.consultation_workflow,
        app_workflow=gated.wf,
    )

    class _WildOut:
        command_name = DROP
        cmd_parameters = DropSignature.Input(items=["2"])

    text = "Drop step 2; step 1 alone should return how many logins arrive on the RLIQA feed."
    monkeypatch.setattr(re_, "_compound_drop_edit", lambda _t, _items=None: True)
    decision = policy._handle_routed_consultation(
        proposal,
        text,
        _WildOut(),
        "Which of the 2 steps do you mean?",
    )
    assert decision.verdict is Verdict.EDIT
    assert (
        any(isinstance(edit, Drop) and edit.index == 2 for edit in (decision.edits or ()))
        or any(isinstance(edit, Change) for edit in (decision.edits or ()))
    )


def test_item_slot_all_steps_expands_edits(gated, monkeypatch, planner):
    record = _open_review_two_step(gated, monkeypatch)
    planner(
        gated,
        {"edits": [{"op": "change", "field": "goal", "value": "same overview fix"}]},
        {"edits": [{"op": "change", "field": "goal", "value": "same overview fix"}]},
    )
    first = gated.ctx.process_turn("apply the same overview fix to each step")
    assert "Which of the 2 steps do you mean?" in (first.answer or "")
    second = gated.ctx.process_turn("All eight steps—the same overview fix applies to each record type.")
    assert record.state == REVISED
    for item in re_.proposal_of(record).items:
        assert "overview" in (item.goal or "").lower()


def test_resolve_proposal_step_last_and_ordinal_phrases():
    from fastworkflow._workflows.consultation import _item_validation as item_validation

    proposal = SubTaskProposal(items=[SubTask(goal="a"), SubTask(goal="b"), SubTask(goal="c")])
    assert item_validation.resolve_proposal_step(proposal, "the last step") == "3"
    assert item_validation.resolve_proposal_step(proposal, "the second step") == "2"
    assert item_validation.resolve_proposal_step(proposal, "the first one") == "1"


def test_yes_restate_goal_confirms_without_readback(gated, monkeypatch, planner):
    from fastworkflow.review_engine import open_review, reply_to_proposal

    goal = "Is DrChrono onboarded as an application here?"
    item = SubTask(
        goal=goal,
        means=Means(kind="planning", name="Library/record_fact"),
        inputs=[],
    )
    record = open_review(gated.ctx, gated.store, SubTaskProposal(items=[item]), surface=gated.store.surface)
    _wire_consultation(gated)
    planner(
        gated,
        {"edits": [{"op": "change", "index": 1, "field": "goal", "value": goal}]},
    )
    review = reply_to_proposal(
        gated.ctx,
        gated.store,
        record,
        "Yes, check whether DrChrono is onboarded as an application here.",
    )
    assert review.done and record.state == CONFIRMED
    assert "I read your reply as these edits" not in (review.text or "")


def test_noop_edit_non_affirmative_reprompts(gated, monkeypatch, planner):
    from fastworkflow.review_engine import AWAITING, open_review, reply_to_proposal

    goal = "Is DrChrono onboarded as an application here?"
    item = SubTask(
        goal=goal,
        means=Means(kind="planning", name="Library/record_fact"),
        inputs=[],
    )
    record = open_review(gated.ctx, gated.store, SubTaskProposal(items=[item]), surface=gated.store.surface)
    _wire_consultation(gated)
    planner(
        gated,
        {"edits": [{"op": "change", "index": 1, "field": "goal", "value": goal}]},
    )
    review = reply_to_proposal(gated.ctx, gated.store, record, "DrChrono onboarding status")
    assert not review.done
    assert record.state == AWAITING
    assert record.state != CONFIRMED
    assert record.state != REVISED
    assert review.text == re_.SUB_TASK_UNREADABLE_REPLY


def test_yes_but_corrective_uses_edit_reader_not_confirm(gated, monkeypatch, planner):
    record = _open_review_two_step(gated, monkeypatch)
    planner(
        gated,
        {
            "edits": [
                {
                    "op": "change",
                    "index": 1,
                    "field": "goal",
                    "value": "list which findings are driving her risk",
                },
            ],
        },
    )
    gated.ctx.process_turn("Yes, but also list which findings are driving her risk")
    assert record.state == REVISED
    assert record.state != CONFIRMED


def test_not_just_decline_routes_to_edit_reader(gated, monkeypatch, planner):
    record = _open_review_two_step(gated, monkeypatch)
    planner(
        gated,
        {
            "edits": [
                {
                    "op": "change",
                    "index": 1,
                    "field": "goal",
                    "value": "the kinds of records on file in the directory",
                },
            ],
        },
    )
    gated.ctx.process_turn(
        "Not just people: I want the kinds of records on file in the directory"
    )
    assert record.state == REVISED
    assert record.state != DECLINED


def test_q1_and_q2_sub_task_hints_stay_natural(gated, monkeypatch):
    from fastworkflow.review_engine import reply_to_proposal

    record = _open_review_two_step(gated, monkeypatch)
    assert re_.SUB_TASK_PROMPT_HINT in record.rendering
    assert re_.SUB_TASK_STRUCTURED_PROMPT_HINT not in record.rendering
    review = reply_to_proposal(gated.ctx, gated.store, record, "hmm not sure")
    assert not review.done
    assert review.text == re_.SUB_TASK_UNREADABLE_REPLY
    record.turns_seen += 1
    assert re_.SUB_TASK_PROMPT_HINT in record.rendering
    assert re_.SUB_TASK_STRUCTURED_PROMPT_HINT not in record.rendering


def test_q3_reprompt_uses_structured_fallback_and_clears_slot(gated, monkeypatch):
    from fastworkflow.review_engine import reply_to_proposal

    record = _open_review_two_step(gated, monkeypatch)
    reply_to_proposal(gated.ctx, gated.store, record, "unclear one")
    record.turns_seen += 1
    review = reply_to_proposal(gated.ctx, gated.store, record, "unclear two")
    record.turns_seen += 1
    assert not review.done
    assert re_.SUB_TASK_STRUCTURED_PROMPT_HINT in (review.text or "")
    assert "Current steps:" in (review.text or "")
    assert gated.ctx._consultation_cme.context.get("stored_parameters") is None


def test_q3_add_after_and_change_without_planner(gated, monkeypatch):
    from fastworkflow.review_engine import reply_to_proposal

    record = _open_review_two_step(gated, monkeypatch)
    gated.ctx._current_planner_lm = _PlannerLM([{"edits": "must not run"}])
    for text in ("unclear one", "unclear two"):
        reply_to_proposal(gated.ctx, gated.store, record, text)
        record.turns_seen += 1
    review = reply_to_proposal(
        gated.ctx,
        gated.store,
        record,
        "add after 1: goal=tell a story; change 1: goal=find Persuasion",
    )
    assert record.state == REVISED
    items = re_.proposal_of(record).items
    assert [item.goal for item in items] == ["find Persuasion", "tell a story", JOKE]
    assert "must not run" not in str(review.text or record.rendering)


def test_q3_answer_structured_edit_without_planner(gated, monkeypatch):
    from fastworkflow.review_engine import reply_to_proposal

    record = _open_review_two_step(gated, monkeypatch)
    gated.ctx._current_planner_lm = _PlannerLM([{"edits": "must not run"}])
    for text in ("unclear one", "unclear two"):
        reply_to_proposal(gated.ctx, gated.store, record, text)
        record.turns_seen += 1
    review = reply_to_proposal(
        gated.ctx,
        gated.store,
        record,
        "change 2: kind=application",
    )
    assert not review.done
    assert record.state == REVISED
    assert re_.SUB_TASK_STRUCTURED_PROMPT_HINT in (review.text or record.rendering)


def test_tag_feed_confirmations_on_q1_with_consultation(gated, monkeypatch):
    from fastworkflow.review_engine import open_review, reply_to_proposal

    item = SubTask(
        goal="Apply audit-scope tag to the HR feed",
        means=Means(kind="planning", name="Library/record_fact"),
        inputs=[SubTaskInput(name="tag", value="audit-scope"), SubTaskInput(name="feed", value="HR")],
    )
    record = open_review(
        gated.ctx, gated.store, SubTaskProposal(items=[item]), surface=gated.store.surface,
    )
    _wire_consultation(gated)
    review = reply_to_proposal(
        gated.ctx,
        gated.store,
        record,
        "Yes, go ahead and remove the audit-scope tag from the HR feed",
    )
    assert review.done and record.state == CONFIRMED


def test_unknown_kind_judgment_routes_to_edit_not_lookup_slot(gated, monkeypatch, planner):
    record = _open_review_two_step(gated, monkeypatch)
    planner(
        gated,
        {
            "edits": [
                {
                    "op": "change",
                    "index": 1,
                    "field": "inputs",
                    "value": "unit=use your best judgment",
                },
            ],
        },
    )
    out = gated.ctx.process_turn(
        "I don't know whether IT Support is a unit or a group; use your best judgment"
    )
    assert "Which of the 2 steps do you mean?" not in (out.answer or "")
    assert record.state == REVISED
