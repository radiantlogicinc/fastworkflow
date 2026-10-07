"""`TurnResult.execution_records` and `.routing_events` (arch §12.2).

The fields are additive. ``execution_records`` populate once a trace sink is
present (fix-ajv.3); ``routing_events`` remain empty until the NLU routing
ledger lands. Tests below cover the two things that can go wrong with a field
before anything writes to it — its shape, and its blast radius.

**Shape.** Both element models are thin correlation records rather than copies
of architecture §6.6's `ExecutionRecord` field list, and the constraint that
makes that the right answer is testable: a new top-level list on `TurnResult`
reaches `record_json` as it stands. Every field is therefore an opaque id, a
closed vocabulary, or a count — asserted structurally below, not left to
review, since the failure it prevents produces a clean-looking record and no
error.

**Blast radius.** EXP-003 is a Phase 0 slice: adding a field must not change a
turn, a public projection, or an already-written record. The compatibility
half — a record serialized without the keys, and a reader meeting keys it does
not know — lives in `tests/test_public_shape_parity.py` with the other §12.2
shapes; what is here is the runtime half.

Real components per `.cursor/rules/testing_rules.mdc`: the real
`todo_list_workflow`, a real `WorkflowExecutionContext` over a real
`TodoListManager`, and a real `SQLiteTraceSink` writing real SQLite in
`tmp_path`.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Literal, Union, get_args, get_origin

import pytest
from pydantic import ValidationError

import fastworkflow
from fastworkflow import TurnResult, TurnStatus, mint_turn_key
from fastworkflow.observability import store as obs
from fastworkflow.turn import (
    TURN_CAPTURE_CONTRACT_VERSION,
    ExecutionRecordRef,
    RoutingEvent,
    RoutingOutcome,
    RoutingTier,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext

from tests.todo_list_workflow.application.todo_manager import TodoListManager

LIST_COMMAND = "TodoListManager/list_todo_lists"

# The two element models, as one list, so a new one added later cannot quietly
# skip the structural rules below.
CAPTURE_RECORD_MODELS = (ExecutionRecordRef, RoutingEvent)


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "observability.sqlite3")


class RecordingTraceSink:
    """Real TraceSink implementation that keeps what it is handed."""

    def __init__(self):
        self.spans: list = []
        self.turn_records: list = []

    def emit_span(self, span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> bool:
        self.turn_records.append(record)
        return True

    def record_conversation_label(self, channel_id, conversation_id, topic, summary):
        pass


def _run_one_real_turn(todo_workflow_path: str, tmp_path, sink=None):
    """One real command through the real runtime; returns (turn_output, ctx)."""
    workflow = fastworkflow.Workflow.create(
        todo_workflow_path, workflow_id_str=f"additive-{uuid.uuid4().hex}"
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(workflow)
    workflow.root_command_context = TodoListManager(str(tmp_path / "todo_list.json"))
    try:
        turn_output = ctx.process_action_turn(
            fastworkflow.Action(command_name=LIST_COMMAND, command="list them")
        )
    finally:
        with suppress(Exception):
            ctx.close()
    return turn_output


def _execution_record(**overrides) -> ExecutionRecordRef:
    kwargs = dict(command_call_id="c0ffee", command_ordinal=0, span_id="span-0")
    kwargs.update(overrides)
    return ExecutionRecordRef(**kwargs)


def _routing_event(**overrides) -> RoutingEvent:
    kwargs = dict(
        ordinal=0,
        tier="classifier",
        outcome="resolved",
        candidate_count=3,
        span_id="span-nlu",
        command_call_id="c0ffee",
    )
    kwargs.update(overrides)
    return RoutingEvent(**kwargs)


def _turn_result(**overrides) -> TurnResult:
    kwargs = dict(
        turn_output=fastworkflow.TurnOutput(
            turn_key=mint_turn_key(),
            status=TurnStatus.COMPLETED,
            answer="done",
            command_outputs=[
                fastworkflow.CommandOutput(
                    command_name=LIST_COMMAND,
                    command_response=fastworkflow.CommandResponse(response="3 lists"),
                )
            ],
        ),
        channel_id="chan",
        conversation_id=1,
        user_message="list my todo lists",
        conversation_summary="user listed their todo lists",
        conversation_traces="list_todo_lists -> ok",
    )
    kwargs.update(overrides)
    return TurnResult(**kwargs)


def _rows(path: str, sql: str) -> list[dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


# ----------------------------------------------------------------------
# The fields are genuinely optional
# ----------------------------------------------------------------------


def test_a_turn_result_still_constructs_the_old_way():
    """Every existing construction in the codebase looks like this."""
    turn_result = TurnResult(
        turn_output=fastworkflow.TurnOutput(
            turn_key=mint_turn_key(), status=TurnStatus.COMPLETED
        ),
        user_message="anything",
    )
    assert turn_result.execution_records == ()
    assert turn_result.routing_events == ()


def test_the_defaults_are_empty_rather_than_none():
    """Empty and absent are the same claim here, and None would be a third one.

    A consumer iterating the list must not have to guard for None, and there is
    no state this model can represent that means "capture was attempted and
    produced nothing" as distinct from "no capture ran" — inventing one with
    None would promise a distinction nothing produces.
    """
    dumped = _turn_result().model_dump(mode="json")
    assert dumped["execution_records"] == []
    assert dumped["routing_events"] == []


def test_populated_records_round_trip_through_dump_and_validate():
    original = _turn_result(
        execution_records=(
            _execution_record(),
            _execution_record(
                command_call_id="dec0de",
                parent_call_id="c0ffee",
                command_ordinal=1,
                span_id=None,
            ),
        ),
        routing_events=(_routing_event(),),
    )
    restored = TurnResult.model_validate(original.model_dump(mode="json"))

    assert restored.execution_records == original.execution_records
    assert restored.routing_events == original.routing_events
    # The child execution — no span of its own, filed under its parent — is the
    # case that exists nowhere else in a turn record.
    child = restored.execution_records[1]
    assert child.span_id is None
    assert child.parent_call_id == "c0ffee"


def test_every_record_carries_its_contract_version():
    """A reader joining runs needs to know an absent field means "this engine
    could not emit it" rather than "it did not occur"."""
    assert _execution_record().contract_version == TURN_CAPTURE_CONTRACT_VERSION
    assert _routing_event().contract_version == TURN_CAPTURE_CONTRACT_VERSION
    dumped = _turn_result(
        execution_records=(_execution_record(),), routing_events=(_routing_event(),)
    ).model_dump(mode="json")
    assert dumped["execution_records"][0]["contract_version"] >= 1
    assert dumped["routing_events"][0]["contract_version"] >= 1


# ----------------------------------------------------------------------
# Runtime population: execution_records yes, routing_events not yet
# ----------------------------------------------------------------------


def test_execution_records_populate_but_routing_events_stay_empty(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """fix-ajv.3 lands execution refs; routing_events wait on NLU ledger work."""
    sink = RecordingTraceSink()
    _run_one_real_turn(todo_workflow_path, tmp_path, sink)

    assert sink.turn_records, "the real turn emitted no turn record"
    for record in sink.turn_records:
        assert record.execution_records, "expected at least one execution ref"
        assert all(ref.command_call_id for ref in record.execution_records)
        assert record.routing_events == ()


def test_the_public_projection_gains_nothing(
    initialized_fastworkflow, todo_workflow_path, tmp_path
):
    """§12.2: old APIs continue their legacy projection.

    `process_turn` returns the `TurnOutput`, and these two fields live on the
    `TurnResult` around it — so a caller on the old API sees exactly what it saw
    before.
    """
    turn_output = _run_one_real_turn(todo_workflow_path, tmp_path)

    assert not hasattr(turn_output, "execution_records")
    assert not hasattr(turn_output, "routing_events")
    dumped = turn_output.model_dump()
    assert "execution_records" not in dumped
    assert "routing_events" not in dumped


# ----------------------------------------------------------------------
# The fields reach storage without a serializer change
# ----------------------------------------------------------------------


def test_the_fields_reach_record_json_through_the_existing_model_dump():
    """`serialize_turn_result` dumps the whole `TurnResult`, so additive fields
    flow to storage with no change to the store — verified rather than assumed,
    because the store is the half this bead does not own."""
    turn_row, _artifacts = obs.serialize_turn_result(
        _turn_result(
            execution_records=(_execution_record(),), routing_events=(_routing_event(),)
        )
    )
    record = json.loads(turn_row["record_json"])

    assert record["execution_records"] == [
        {
            "contract_version": TURN_CAPTURE_CONTRACT_VERSION,
            "command_call_id": "c0ffee",
            "parent_call_id": None,
            "command_ordinal": 0,
            "span_id": "span-0",
        }
    ]
    assert record["routing_events"][0]["tier"] == "classifier"
    assert record["routing_events"][0]["outcome"] == "resolved"


def test_a_stored_record_reads_back_into_the_model(db_path):
    """Through real SQLite, because `record_json` is TEXT and the tuples have to
    survive both the dump and the parse to be worth writing."""
    sink = obs.SQLiteTraceSink(db_path)
    try:
        sink.emit_turn_record(
            _turn_result(
                execution_records=(_execution_record(),),
                routing_events=(_routing_event(),),
            )
        )
        assert sink.flush()
    finally:
        sink.close()

    stored = json.loads(_rows(db_path, "SELECT record_json FROM turns")[0]["record_json"])
    restored = TurnResult.model_validate(stored)

    assert restored.execution_records[0].command_call_id == "c0ffee"
    assert restored.routing_events[0].candidate_count == 3


def test_an_empty_pair_still_appears_in_a_stored_record(
    initialized_fastworkflow, todo_workflow_path, tmp_path, db_path
):
    """Present-and-empty, not absent.

    An absent key would make a record written by this version indistinguishable
    from one written before the fields existed, which is exactly the distinction
    a reader joining runs across a release needs.
    """
    sink = obs.SQLiteTraceSink(db_path)
    try:
        _run_one_real_turn(todo_workflow_path, tmp_path, sink)
        assert sink.flush()
    finally:
        sink.close()

    rows = _rows(db_path, "SELECT record_json FROM turns")
    assert rows, "the real turn stored no row"
    record = json.loads(rows[0]["record_json"])
    assert "execution_records" in record
    assert "routing_events" in record
    assert record["execution_records"], "execution refs populate under a real sink"
    assert record["routing_events"] == []


# ----------------------------------------------------------------------
# Why the records are thin: they are retained whole
# ----------------------------------------------------------------------


def _is_id_or_vocabulary_or_count(annotation) -> bool:
    """`int`, `str`, `Optional` of either, or a `Literal` of strings."""
    if get_origin(annotation) is Literal:
        return all(isinstance(value, str) for value in get_args(annotation))
    if annotation in (int, str):
        return True
    if get_origin(annotation) is Union:
        return set(get_args(annotation)) <= {int, str, type(None)}
    return False


def _admits_free_str(annotation) -> bool:
    """Whether the annotation accepts an arbitrary string (a Literal does not)."""
    if get_origin(annotation) is Literal:
        return False
    return annotation is str or str in get_args(annotation)


@pytest.mark.parametrize("model", CAPTURE_RECORD_MODELS)
def test_every_field_is_an_identifier_a_vocabulary_or_a_count(model):
    """No field may hold free text or entity content.

    Nothing filters these records — see the test above — so the guarantee has to
    come from the shape. An `Any`, a `dict`, or a plain `str` field that is not
    an id would let a well-meant emitter put a user's utterance, a command
    response, or a context display label into the one part of a turn record no
    capture profile governs.
    """
    for name, field in model.model_fields.items():
        annotation = field.annotation
        assert _is_id_or_vocabulary_or_count(annotation), (
            f"{model.__name__}.{name} is annotated {annotation!r}, which can "
            "hold free text"
        )
        if _admits_free_str(annotation):
            assert name.endswith("_id"), (
                f"{model.__name__}.{name} is a bare string but is not an "
                "identifier; only framework-minted ids may be strings here"
            )


# ----------------------------------------------------------------------
# The records refuse what they cannot mean
# ----------------------------------------------------------------------


def test_a_routing_tier_outside_the_vocabulary_is_refused():
    """The vocabulary is closed on purpose: a tier the runtime cannot perform
    would make a record claim more than the engine knows."""
    with pytest.raises(ValidationError):
        RoutingEvent(ordinal=0, tier="telepathy", outcome="resolved")


def test_a_routing_outcome_outside_the_vocabulary_is_refused():
    with pytest.raises(ValidationError):
        RoutingEvent(ordinal=0, tier="classifier", outcome="probably")


def test_the_vocabularies_are_the_layers_the_runtime_actually_has():
    """Pinned against `intent_detection.py`'s `matcher_layer` values plus the
    direct-action path, so a member added without an emitter is visible."""
    assert set(get_args(RoutingTier)) == {
        "direct_action",
        "exact_prefix",
        "fuzzy_prematch",
        "embedding_cache",
        "classifier",
        "clarification_default",
        "unknown",
    }
    assert set(get_args(RoutingOutcome)) == {
        "resolved",
        "unresolved",
        "ambiguous",
        "error",
    }


def test_an_unresolved_attempt_needs_no_call_id_and_no_candidates():
    """The ordinary case: the utterance matched nothing in this context and the
    CME wildcard command walks up the parent chain. Not a failure, and not a
    record with holes in it."""
    event = RoutingEvent(ordinal=1, tier="exact_prefix", outcome="unresolved")
    assert event.candidate_count is None
    assert event.command_call_id is None
    assert event.span_id is None


def test_negative_ordinals_and_counts_are_refused():
    with pytest.raises(ValidationError):
        ExecutionRecordRef(command_call_id="x", command_ordinal=-1)
    with pytest.raises(ValidationError):
        RoutingEvent(ordinal=-1, tier="classifier", outcome="resolved")
    with pytest.raises(ValidationError):
        RoutingEvent(
            ordinal=0, tier="classifier", outcome="ambiguous", candidate_count=-2
        )


def test_a_record_missing_its_join_key_is_refused():
    """An execution record with no `command_call_id` joins nothing, which is the
    only thing it exists to do."""
    with pytest.raises(ValidationError):
        ExecutionRecordRef(command_ordinal=0)


@pytest.mark.parametrize("model", CAPTURE_RECORD_MODELS)
def test_the_records_are_frozen_and_reject_unknown_keys(model):
    """`decision_signals._Strict`'s posture: a captured record must not be
    editable by the code being measured, and a typo'd field that parses is a
    field nobody notices is missing."""
    assert model.model_config["frozen"] is True
    assert model.model_config["extra"] == "forbid"

    with pytest.raises(ValidationError):
        model.model_validate({"totally_unexpected": 1})
