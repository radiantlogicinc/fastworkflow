"""The parameter-extraction round ordinal, against its real co-writers (fix-8ko2).

`retry_round_ordinal` counts attempts within one parameter-extraction error
state. The counter lives in the CME workflow context next to the
`stored_parameters` it describes -- and `parameter_extraction.py` is not the
only thing that writes that context:

- `Workflow.end_command_processing` deletes `stored_parameters` and knows
  nothing about the counter, so it leaves one behind pointing at a command that
  already finished.
- `WorkflowExecutionContext.serialize_state` / `apply_serialized_state` carry
  `stored_parameters` across a suspend and rebuild it, but do not carry the
  counter, so a restored session has genuine stored parameters whose round
  nobody recorded.

An ordinal that ignored either of those would be confidently wrong: a fresh
command reported as retry #4, or a resumed third attempt reported as the first.
Both writers are in files this change may not edit, so the rule is read-side
and these tests hold it to the real objects rather than to a description of
them -- a real `Workflow`, the real serializer, the real restore.

The honest outcome is sometimes "unknown", and unknown must stay unknown:
nothing here may turn it into 0, and no test here may assert that the counter
survives a restore, because it does not. `fix-7gp9` tracks persisting it.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
    ParameterExtraction,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


ROUND_KEY = "stored_parameter_round"


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield tmp_path
    RoutingRegistry.clear_registry()


def _make_ctx(workflow_path: str, channel_id: str) -> WorkflowExecutionContext:
    ctx = WorkflowExecutionContext(run_as_agent=False, session_key=channel_id)
    workflow = fastworkflow.Workflow.create(workflow_path, workflow_id_str=channel_id)
    ctx.bind_app_workflow(workflow)
    return ctx


def _params_class(ctx: WorkflowExecutionContext, command_name: str):
    routing = fastworkflow.RoutingRegistry.get_definition(ctx.app_workflow.folderpath)
    return routing.get_command_class(
        command_name, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
    )


def _enter_parameter_extraction(ctx: WorkflowExecutionContext) -> str:
    """The state a failed extraction leaves behind, built the way it is built.

    The stored instance comes from `model_construct` with a NOT_FOUND sentinel
    in the missing field, which is the shape ordinary validation refuses -- so
    a restore that validated would reject the very state it exists to carry.
    """
    command_name = "TodoListManager/create_todo_list"
    params_class = _params_class(ctx, command_name)
    assert params_class is not None, "test fixture needs a real Input class"

    cme = ctx._cme_workflow.context
    cme["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    cme["command"] = "create a todo list"
    cme["command_name"] = command_name
    return command_name


def _store_a_failed_attempt(ctx: WorkflowExecutionContext, command_name: str) -> None:
    """One round of "extraction ran and came up short", through the real writer."""
    params_class = _params_class(ctx, command_name)
    ParameterExtraction._store_parameters(
        ctx._cme_workflow, params_class.model_construct(description="NOT_FOUND")
    )


def _round(ctx: WorkflowExecutionContext):
    return ParameterExtraction._get_stored_round(ctx._cme_workflow)


# ---------------------------------------------------------------------------
# The ordinary sequence, so the regressions below are read against a baseline
# that is known to be right.
# ---------------------------------------------------------------------------

def test_a_clean_sequence_numbers_its_attempts_from_zero(
    initialized_fastworkflow, todo_workflow_path
):
    ctx = _make_ctx(todo_workflow_path, f"cme_{uuid.uuid4().hex[:8]}")
    command_name = _enter_parameter_extraction(ctx)
    try:
        assert _round(ctx) == 0, "the first attempt is attempt 0, not a retry"

        _store_a_failed_attempt(ctx, command_name)
        assert _round(ctx) == 1, "the attempt resuming from one stored round is #1"

        _store_a_failed_attempt(ctx, command_name)
        assert _round(ctx) == 2

        # Success ends the error state, and the count of its attempts ends too.
        ParameterExtraction._clear_parameters(ctx._cme_workflow)
        assert ROUND_KEY not in ctx._cme_workflow.context
        assert _round(ctx) == 0, "the next command starts over"
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# Co-writer 1: Workflow.end_command_processing.
# ---------------------------------------------------------------------------

def test_end_command_processing_leaves_a_stale_counter_and_it_is_not_believed(
    initialized_fastworkflow, todo_workflow_path
):
    """A real `end_command_processing`, not a simulation of one.

    It deletes `stored_parameters` and leaves the counter untouched, which is
    the whole hazard: read naively, the next command's FIRST extraction would
    report itself as retry #3 of a command that already finished.
    """
    ctx = _make_ctx(todo_workflow_path, f"cme_{uuid.uuid4().hex[:8]}")
    command_name = _enter_parameter_extraction(ctx)
    try:
        _store_a_failed_attempt(ctx, command_name)
        _store_a_failed_attempt(ctx, command_name)
        _store_a_failed_attempt(ctx, command_name)
        assert _round(ctx) == 3

        ctx._cme_workflow.end_command_processing()

        # The hazard is real and this test would be vacuous without it: the
        # counter IS still sitting there.
        assert "stored_parameters" not in ctx._cme_workflow.context
        assert ctx._cme_workflow.context.get(ROUND_KEY) == 3

        # And it is not believed. No stored parameters means no error state for
        # a round to be a round of.
        assert _round(ctx) == 0

        # Nor does the stale value seed the next sequence: the new command's
        # first stored round is 1, not 4.
        _store_a_failed_attempt(ctx, command_name)
        assert ctx._cme_workflow.context[ROUND_KEY] == 1
        assert _round(ctx) == 1
    finally:
        ctx.close()


# ---------------------------------------------------------------------------
# Co-writer 2: the real suspend/rehydrate path.
# ---------------------------------------------------------------------------

def test_the_serializer_does_not_carry_the_counter(
    initialized_fastworkflow, todo_workflow_path
):
    """Pinned so nobody can claim the round survives a restore while it does not.

    If `fix-7gp9` later persists it, this test fails loudly and is the place to
    record that it now does -- which is the point. A silent upgrade from
    "unknown" to "known" is the thing the ordinal exists to prevent.
    """
    channel_id = f"cme_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    command_name = _enter_parameter_extraction(ctx)
    _store_a_failed_attempt(ctx, command_name)
    _store_a_failed_attempt(ctx, command_name)
    assert _round(ctx) == 2

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    assert blob["cme"] is not None
    assert blob["cme"]["stored_parameters"] == {"description": "NOT_FOUND"}
    assert ROUND_KEY not in blob["cme"], (
        "if the counter is persisted now, update this test and fix-7gp9 rather "
        "than leaving the unknown-round handling in place unexamined"
    )


def test_a_round_restored_from_a_continuation_is_unknown_not_zero(
    initialized_fastworkflow, todo_workflow_path
):
    """The restored session really is mid-retry; it just cannot say which one.

    Reporting 0 here would say "first attempt" about an extraction resuming
    from parameters two rounds deep -- a wrong number, which is worse than no
    number, because a consumer cannot tell it apart from a real first attempt.
    """
    channel_id = f"cme_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    command_name = _enter_parameter_extraction(ctx)
    _store_a_failed_attempt(ctx, command_name)
    _store_a_failed_attempt(ctx, command_name)

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)
    try:
        cme = restored._cme_workflow.context
        # The error state itself came back -- this is a genuine retry.
        assert cme["stored_parameters"] is not None
        assert bool(cme["stored_parameters"]) is True, (
            "the boolean retry_round the producer records is still True here"
        )
        assert ROUND_KEY not in cme

        assert _round(restored) is None, "unknown, not 0"

        # And unknown stays unknown: a further failed attempt cannot count up
        # from a number nobody has.
        _store_a_failed_attempt(restored, command_name)
        assert ROUND_KEY not in restored._cme_workflow.context
        assert _round(restored) is None

        # It becomes knowable again only when the error state ends and a new
        # one starts from a round this module did write.
        ParameterExtraction._clear_parameters(restored._cme_workflow)
        assert _round(restored) == 0
        _store_a_failed_attempt(restored, command_name)
        assert _round(restored) == 1
    finally:
        restored.close()


# ---------------------------------------------------------------------------
# Corruption is unknown too, never a number that happens to parse.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad", [True, False, 2.0, "3", -1, 0, None, [], {"round": 1}]
)
def test_an_unreadable_counter_beside_stored_parameters_is_unknown(
    initialized_fastworkflow, todo_workflow_path, bad
):
    """`True` is not round 1 and `"3"` is not round 3.

    0 is in this list on purpose: the writer only ever stores >= 1, so a 0
    sitting beside stored parameters did not come from here and saying "first
    attempt" on the strength of it would be a guess.
    """
    ctx = _make_ctx(todo_workflow_path, f"cme_{uuid.uuid4().hex[:8]}")
    command_name = _enter_parameter_extraction(ctx)
    try:
        _store_a_failed_attempt(ctx, command_name)
        ctx._cme_workflow.context[ROUND_KEY] = bad

        assert _round(ctx) is None

        # And storing again does not launder it into a number.
        _store_a_failed_attempt(ctx, command_name)
        assert _round(ctx) is None
    finally:
        ctx.close()
