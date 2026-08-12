"""Round-trip for the logical-turn accumulator and the CME continuation keys.

Before schema 2 the pending snapshot carried neither, so a restored session
started a fresh logical turn and re-extracted parameters from the answer text
alone. These tests build the state with real objects -- a real Workflow, the
real command Input class, real CommandOutput instances -- rather than driving a
live LLM, because the intent classifier's trained artifacts are not in git and
a test that needs a provider is a latency flake waiting to happen (fix-wi3).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.run_fastapi_mcp import checkpoint
from fastworkflow.session_state_store import SCHEMA_VERSION
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


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
    workflow = fastworkflow.Workflow.create(
        workflow_path, workflow_id_str=channel_id
    )
    ctx.bind_app_workflow(workflow)
    return ctx


def _params_class(ctx: WorkflowExecutionContext, command_name: str):
    routing = fastworkflow.RoutingRegistry.get_definition(
        ctx.app_workflow.folderpath
    )
    return routing.get_command_class(
        command_name, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
    )


def _enter_parameter_extraction(ctx: WorkflowExecutionContext) -> str:
    """Put the CME workflow in the state a failed extraction leaves behind.

    parameter_extraction stores an instance built with model_construct whose
    missing fields hold the NOT_FOUND sentinel, which is exactly the shape
    ordinary validation refuses -- so a restore that validates would reject the
    very state it exists to carry.
    """
    command_name = "TodoListManager/create_todo_list"
    params_class = _params_class(ctx, command_name)
    assert params_class is not None, "test fixture needs a real Input class"

    cme = ctx._cme_workflow.context
    cme["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    cme["command"] = "create a todo list"
    cme["command_name"] = command_name
    cme["stored_parameters"] = params_class.model_construct(description="NOT_FOUND")
    return command_name


def _open_a_turn(ctx: WorkflowExecutionContext) -> None:
    ctx._begin_turn("create a todo list")
    ctx.append_turn_output(
        fastworkflow.CommandOutput(
            command_name="TodoListManager/create_todo_list",
            command_parameters="create a todo list",
            command_response=
                fastworkflow.CommandResponse(response="need a description", success=False),
            started_at=datetime.now(timezone.utc),
        )
    )


def test_cme_continuation_survives_a_round_trip(
    initialized_fastworkflow, todo_workflow_path
):
    channel_id = f"cme_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    command_name = _enter_parameter_extraction(ctx)

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    assert blob["cme"] is not None
    assert blob["cme"]["command_name"] == command_name
    assert blob["cme"]["stored_parameters"] == {"description": "NOT_FOUND"}

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    cme = restored._cme_workflow.context
    assert cme["command_name"] == command_name
    assert cme["command"] == "create a todo list"
    assert (
        cme["NLU_Pipeline_Stage"]
        == fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    )

    params = cme["stored_parameters"]
    # Rebuilt as the real Input class, not a dict: parameter_extraction merges
    # into it with getattr and type(...).model_fields, both of which a dict fails.
    assert type(params) is _params_class(restored, command_name)
    assert params.description == "NOT_FOUND"
    restored.close()


def test_sentinel_in_a_typed_field_survives_restore(
    initialized_fastworkflow, todo_workflow_path
):
    """The sentinel goes into the field whatever the field's declared type is.

    parameter_extraction writes NOT_FOUND into every missing field, so an int
    field holds the string "NOT_FOUND". That is unvalidatable by construction,
    which is exactly why restore rebuilds with model_construct: model_validate
    would raise on the very state the snapshot exists to carry, and the session
    would be stranded mid-extraction with no way back.
    """
    channel_id = f"typed_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)

    command_name = "TodoListManager/get_todo_list"
    params_class = _params_class(ctx, command_name)
    assert "id" in params_class.model_fields

    cme = ctx._cme_workflow.context
    cme["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
    cme["command"] = "get the todo list"
    cme["command_name"] = command_name
    cme["stored_parameters"] = params_class.model_construct(id="NOT_FOUND")

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    assert blob["cme"]["stored_parameters"] == {"id": "NOT_FOUND"}

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    params = restored._cme_workflow.context["stored_parameters"]
    assert type(params) is params_class
    assert params.id == "NOT_FOUND"
    restored.close()


def test_turn_accumulator_survives_a_round_trip(
    initialized_fastworkflow, todo_workflow_path
):
    channel_id = f"turn_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    _open_a_turn(ctx)
    original_key = ctx._turn_key
    assert original_key is not None

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    # The same logical turn, not a new one: resume deliberately skips
    # _begin_turn, so a lost key silently splits one turn's telemetry in two.
    assert restored._turn_key == original_key
    assert restored._turn_user_message == "create a todo list"
    assert restored._turn_entry_context == ctx._turn_entry_context

    assert len(restored._turn_outputs) == 1
    output = restored._turn_outputs[0]
    assert isinstance(output, fastworkflow.CommandOutput)
    assert output.command_name == "TodoListManager/create_todo_list"
    assert output.command_response.response == "need a description"
    assert output.command_response.success is False
    assert output.started_at is not None
    restored.close()


def test_turn_output_with_typed_params_survives_cold_rehydrate(
    initialized_fastworkflow, todo_workflow_path
):
    """CommandExecutor assigns a Pydantic params instance to command_parameters.

    The field was long declared ``str`` while the write path stored a model;
    ``model_dump(mode="json")`` emitted a dict and ``model_validate`` on
    cold-rehydrate rejected it (fix-fjh / A10 honesty). Typed in-memory,
    dict-on-wire, and restore must all agree.
    """
    channel_id = f"params_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    command_name = "TodoListManager/create_todo_list"
    params_class = _params_class(ctx, command_name)
    assert params_class is not None

    ctx._begin_turn("create a todo list called groceries")
    output = fastworkflow.CommandOutput(
        command_name=command_name,
        command_response=fastworkflow.CommandResponse(
            response="created", success=True
        ),
        started_at=datetime.now(timezone.utc),
    )
    # Mirror CommandExecutor.invoke: assign the typed instance (no re-validate).
    output.command_parameters = params_class(description="groceries")
    ctx.append_turn_output(output)

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    assert blob["turn"]["outputs"][0]["command_parameters"] == {
        "description": "groceries"
    }

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    restored_params = restored._turn_outputs[0].command_parameters
    assert restored_params == {"description": "groceries"}
    restored.close()


def test_ask_user_entry_is_still_completable_after_restore(
    initialized_fastworkflow, todo_workflow_path
):
    """The restored accumulator has to be live state, not an inert record.

    complete_ask_user_entry scans backwards for an unanswered ask_user and
    fills it in. If restore produced dicts, or dropped success=False, the
    user's answer would land nowhere and the entry would stay unanswered.
    """
    channel_id = f"ask_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("add something")
    ctx.append_ask_user_entry("Which list?")

    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)
    restored.complete_ask_user_entry("the groceries one")

    entry = restored._turn_outputs[-1]
    assert entry.command_name == "ask_user"
    assert entry.command_response.response == "the groceries one"
    assert entry.command_response.success is True
    restored.close()


def test_no_open_turn_or_command_serializes_as_absent(
    initialized_fastworkflow, todo_workflow_path
):
    """Absent must stay absent, or every idle session looks mid-command."""
    channel_id = f"idle_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)

    assert ctx.has_open_command() is False
    blob = ctx.serialize_state(channel_id=channel_id)
    assert blob["turn"] is None
    assert blob["cme"] is None

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)
    assert restored._turn_key is None
    assert restored._turn_outputs == []
    assert "stored_parameters" not in restored._cme_workflow.context
    ctx.close()
    restored.close()


def test_has_open_command_tracks_the_cme_keys(
    initialized_fastworkflow, todo_workflow_path
):
    """This predicate decides whether the blob is written at all."""
    channel_id = f"open_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    assert ctx.has_open_command() is False

    _enter_parameter_extraction(ctx)
    assert ctx.has_open_command() is True

    # end_command_processing is what a completed command runs.
    ctx._cme_workflow.end_command_processing()
    assert ctx.has_open_command() is False

    # It clears command and stored_parameters but leaves command_name behind,
    # which is why the predicate must not key off command_name: a session that
    # merely ran one command would otherwise look mid-extraction forever.
    assert ctx._cme_workflow.context.get("command_name") is not None
    assert ctx.serialize_state(channel_id=channel_id)["cme"] is None
    ctx.close()


def test_unresolvable_parameter_class_resets_rather_than_resuming(
    initialized_fastworkflow, todo_workflow_path
):
    """A command that no longer exists must not leave the session mid-extraction.

    Restoring PARAMETER_EXTRACTION without a resolvable class would strand the
    session: wildcard.py reads context["command_name"] unconditionally at that
    stage, and the next message would merge into a command that can never run.
    """
    channel_id = f"gone_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    _enter_parameter_extraction(ctx)
    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    blob["cme"]["command_name"] = "TodoListManager/command_deleted_in_a_later_release"

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)

    cme = restored._cme_workflow.context
    assert (
        cme["NLU_Pipeline_Stage"] == fastworkflow.NLUPipelineStage.INTENT_DETECTION
    )
    assert "command_name" not in cme
    assert "stored_parameters" not in cme
    restored.close()


def test_v1_blob_is_refused_rather_than_partly_restored(
    initialized_fastworkflow, todo_workflow_path
):
    """Schema 1 lacked exactly the fields whose absence made restore wrong."""
    from fastworkflow.session_state_store import IncompatibleSessionState

    channel_id = f"v1_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    _enter_parameter_extraction(ctx)
    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    legacy = {k: v for k, v in blob.items() if k not in ("turn", "cme")}
    legacy["schema_version"] = 1

    restored = _make_ctx(todo_workflow_path, channel_id)
    with pytest.raises(IncompatibleSessionState):
        restored.apply_serialized_state(legacy)
    assert "stored_parameters" not in restored._cme_workflow.context
    restored.close()


def test_schema_version_is_current(initialized_fastworkflow, todo_workflow_path):
    """Adding fields without bumping would let an old reader half-apply a new blob."""
    assert SCHEMA_VERSION == 3
    channel_id = f"ver_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    assert ctx.serialize_state(channel_id=channel_id)["schema_version"] == SCHEMA_VERSION
    ctx.close()


def test_restored_agent_result_carries_exhaustion(
    initialized_fastworkflow, todo_workflow_path
):
    """The finalize path reads .exhausted off the agent result to set FAILED."""
    channel_id = f"exh_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx._begin_turn("do something long")

    class _Exhausted:
        exhausted = True

    ctx._turn_agent_result = _Exhausted()
    blob = ctx.serialize_state(channel_id=channel_id)
    ctx.close()

    assert blob["turn"]["agent_result"] == {"exhausted": True}

    restored = _make_ctx(todo_workflow_path, channel_id)
    restored.apply_serialized_state(blob)
    assert restored._turn_agent_result is not None
    assert restored._turn_agent_result.exhausted is True
    restored.close()


def test_missing_cme_workflow_reads_as_nothing_in_flight(
    initialized_fastworkflow, todo_workflow_path
):
    """has_open_command() now runs after every turn, not only suspensions.

    close() treats a missing cme_workflow as reachable, so raising here would
    turn a tolerated state into a crash in the post-turn persist path.
    """
    channel_id = f"nocme_{uuid.uuid4().hex[:8]}"
    ctx = _make_ctx(todo_workflow_path, channel_id)
    ctx.close()
    ctx._cme_workflow = None

    assert ctx.has_open_command() is False
    assert ctx._is_extracting_parameters() is False
