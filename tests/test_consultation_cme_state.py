"""WEC-owned consultation CME: second instance, convo_path, and session state (fix-kk46.6)."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import approval_binding
from fastworkflow.review_engine import open_review
from fastworkflow.review_proposals import SubTask, SubTaskProposal
from fastworkflow.utils.signatures import NOT_FOUND
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

CONSULTATION_DROP = "ProposalReview/drop"


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


def _make_ctx(workflow_path: str, channel_id: str) -> WorkflowExecutionContext:
    ctx = WorkflowExecutionContext(run_as_agent=False, session_key=channel_id)
    workflow = fastworkflow.Workflow.create(workflow_path, workflow_id_str=channel_id)
    ctx.bind_app_workflow(workflow)
    return ctx


def _drop_params_class(ctx: WorkflowExecutionContext):
    routing = fastworkflow.RoutingRegistry.get_definition(
        ctx._consultation_workflow.folderpath
    )
    return routing.get_command_class(
        CONSULTATION_DROP, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
    )


def _enter_consultation_extraction(
    ctx: WorkflowExecutionContext,
    *,
    missing_value_question: str | None = None,
) -> None:
    params_class = _drop_params_class(ctx)
    assert params_class is not None
    cme = ctx._consultation_cme.context
    cme["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    cme["command"] = "drop step 2"
    cme["command_name"] = CONSULTATION_DROP
    cme["stored_parameters"] = params_class.model_construct(items=[NOT_FOUND])
    if missing_value_question is not None:
        cme["missing_value_question"] = missing_value_question


def _tasks(*goals: str) -> SubTaskProposal:
    return SubTaskProposal(items=[SubTask(goal=g) for g in goals])


def test_wec_owns_consultation_workflow_and_cme(todo_workflow_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    assert ctx._consultation_workflow is not None
    assert ctx._consultation_cme is not None
    assert ctx._consultation_workflow._current_command_context_name == "ProposalReview"
    assert ctx._consultation_cme.context["app_workflow"] is ctx._consultation_workflow
    assert "run_as_agent" not in ctx._consultation_workflow.context
    assert "run_as_agent" not in ctx._consultation_cme.context


def test_bind_app_workflow_redirects_consultation_convo_cache(todo_workflow_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    expected = os.path.join(ctx.app_workflow.folderpath, "___convo_info")
    assert ctx._consultation_cme.context.get("convo_path") == expected


def test_intent_detection_honours_cme_convo_path_override(todo_workflow_path, tmp_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    custom = str(tmp_path / "custom_convo")
    ctx._consultation_cme.context["convo_path"] = custom
    from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
        CommandNamePrediction,
    )

    predictor = CommandNamePrediction(ctx._consultation_cme)
    assert predictor.convo_path == custom


def test_consultation_cme_round_trips_while_subtask_review_open(todo_workflow_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    _enter_consultation_extraction(
        ctx, missing_value_question="Which of the 2 steps do you mean?",
    )
    turn = ctx._turn_frame()
    turn.subtask_review_open = True

    blob = ctx.serialize_state(channel_id=channel)
    assert blob["consultation_cme"] is not None
    assert blob["consultation_cme"]["command_name"] == CONSULTATION_DROP
    assert blob["consultation_cme"]["missing_value_question"] == (
        "Which of the 2 steps do you mean?"
    )
    assert blob["consultation_nlu_stage"] == (
        fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION.value
    )

    ctx.close()
    restored = _make_ctx(todo_workflow_path, channel)
    restored.apply_serialized_state(blob)

    cme = restored._consultation_cme.context
    assert cme["command_name"] == CONSULTATION_DROP
    assert cme["missing_value_question"] == "Which of the 2 steps do you mean?"
    assert cme["NLU_Pipeline_Stage"] == fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    params = cme["stored_parameters"]
    assert type(params) is _drop_params_class(restored)
    assert params.items == [NOT_FOUND]


def test_consultation_cme_dropped_when_no_subtask_review(todo_workflow_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    _enter_consultation_extraction(ctx)
    turn = ctx._turn_frame()
    turn.subtask_review_open = False

    blob = ctx.serialize_state(channel_id=channel)
    assert blob["consultation_cme"] is not None

    restored = _make_ctx(todo_workflow_path, channel)
    restored.apply_serialized_state(blob)
    cme = restored._consultation_cme.context
    assert cme.get("command_name") is None
    assert cme.get("stored_parameters") is None
    assert cme["NLU_Pipeline_Stage"] == fastworkflow.NLUPipelineStage.INTENT_DETECTION


def test_active_subtask_record_restores_consultation_without_frame_flag(todo_workflow_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    store = approval_binding.store_for(ctx.app_workflow)
    open_review(ctx.app_workflow, store, _tasks("alpha", "beta"), surface=store.surface)
    _enter_consultation_extraction(ctx)
    turn = ctx._turn_frame()
    turn.subtask_review_open = False

    blob = ctx.serialize_state(channel_id=channel)
    restored = _make_ctx(todo_workflow_path, channel)
    restored.apply_serialized_state(blob)

    cme = restored._consultation_cme.context
    assert cme["command_name"] == CONSULTATION_DROP
    store = approval_binding.store_for(restored.app_workflow)
    assert store.consultation_cme is restored._consultation_cme
    assert store.consultation_workflow is restored._consultation_workflow


def test_clear_live_pending_clears_consultation_cme(todo_workflow_path):
    channel = f"kk46_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel)
    _enter_consultation_extraction(ctx)
    ctx._clear_live_pending()
    cme = ctx._consultation_cme.context
    assert cme.get("command_name") is None
    assert cme["NLU_Pipeline_Stage"] == fastworkflow.NLUPipelineStage.INTENT_DETECTION
