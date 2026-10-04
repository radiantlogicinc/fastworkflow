"""Shared comparison projection over recorded executions (`fix-9eg.4`).

Integration throughout: a real `ObservabilityStore` on disk, real span rows
through `upsert_span_rows`, real turn rows through `upsert_turn_row`, the real
execution ledger from `run_chatbot/server.py`, a real sealed workspace archive
and the real `add_human_feedback` validator. No mocks and no fixtures that
stand in for a component -- the whole point of this module is that what the
browser shows and what an agent reads come from the same recorded evidence, and
a fake store would not test that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastworkflow import tracing
from fastworkflow.observability import store as obs
from fastworkflow.observability.comparison import (
    ATTRIBUTION_PASS,
    ATTRIBUTION_SHARED,
    ATTRIBUTION_TURN,
    ATTRIBUTION_UNATTRIBUTED,
    BASIS_COMMAND,
    BASIS_COMMAND_CONTEXT,
    BASIS_COMMAND_CONTEXT_PARAMETERS,
    BASIS_RECORDED,
    BASIS_UNKNOWN,
    PAIR_LEFT_ONLY,
    PAIR_MATCHED,
    PAIR_RIGHT_ONLY,
    ExecutionRef,
    ExecutionScopeMismatch,
    InvalidExecutionRef,
    InvalidRecordedAlignment,
    PassSelector,
    StoreExecutionReader,
    UnknownRecordedPass,
    WorkspaceExecutionReader,
    align_steps,
    anchor_for_step,
    anchors_for_pair,
    compare_executions,
    comparison_digest,
    discover_pass_selectors,
    project_execution,
    review_pair_key,
)
from fastworkflow.observability.pair_review import (
    STATE_NOT_REVIEWED,
    STATE_REVIEWED,
    PairReviewIdentityMismatch,
    PairReviewModeMismatch,
    PairReviewStore,
    UnauthorizedEvidenceSource,
    open_shared_pair_review,
    pair_review_db_path_for,
    shared_pair_review_db_path_for,
)
from fastworkflow.observability.workspace import (
    WORKSPACE_SCHEMA,
    load_observability_workspace,
)
from fastworkflow.run_chatbot.server import cost_rollup, execution_ledger

T0 = 1_700_000_000_000_000_000


# ----------------------------------------------------------------------
# Seeding real evidence
# ----------------------------------------------------------------------


def _execute_span(
    span_id: str,
    turn_key: str,
    *,
    call_id: str,
    command_name: str,
    context: str = "global",
    start_ns: int,
    duration_ns: int = 1_000_000,
    status: str = tracing.STATUS_OK,
    success: bool = True,
    parameters: dict | None = None,
    parent_span_id: str | None = None,
    parent_call_id: str | None = None,
    child_calls: list | None = None,
    extra: dict | None = None,
) -> tracing.Span:
    attributes: dict = {
        tracing.ATTR_COMMAND_CALL_ID: call_id,
        "success": success,
    }
    if parent_call_id:
        attributes[tracing.ATTR_PARENT_CALL_ID] = parent_call_id
    if parameters is not None:
        attributes["parameters"] = parameters
    if child_calls:
        attributes[tracing.ATTR_CHILD_CALLS] = child_calls
    if extra:
        attributes.update(extra)
    return tracing.Span(
        span_id=span_id,
        trace_id=turn_key,
        parent_span_id=parent_span_id,
        name=tracing.SPAN_COMMAND_EXECUTE,
        kind=tracing.KIND_INTERNAL,
        channel_id="channel-1",
        command_name=command_name,
        context=context,
        start_ns=start_ns,
        end_ns=start_ns + duration_ns,
        status=status,
        attributes=attributes,
    )


def _llm_span(
    span_id: str,
    turn_key: str,
    *,
    start_ns: int,
    cost: float | None,
    parent_span_id: str | None = None,
) -> tracing.Span:
    return tracing.Span(
        span_id=span_id,
        trace_id=turn_key,
        parent_span_id=parent_span_id,
        name=tracing.SPAN_LLM_CALL,
        kind=tracing.KIND_LLM,
        channel_id="channel-1",
        start_ns=start_ns,
        end_ns=start_ns + 500_000,
        status=tracing.STATUS_OK,
        attributes={} if cost is None else {"cost": cost},
    )


def _turn_row(
    turn_key: str,
    *,
    record: dict,
    answer: str = "done",
    status: str = "completed",
    success: bool = True,
    experiment_id: str | None = None,
    task_id: str | None = None,
    attempt: int | None = None,
    started_at: str = "2026-09-19T00:00:00+00:00",
    completed_at: str = "2026-09-19T00:00:02+00:00",
    user_message: str = "do the thing",
) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": "channel-1",
        "conversation_id": None,
        "ordinal": None,
        "user_message": user_message,
        "refined_user_message": None,
        "entry_workflow_name": "comparison-test",
        "entry_context": "global",
        "status": status,
        "success": 1 if success else 0,
        "failure_reason": None,
        "answer": answer,
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": started_at,
        "completed_at": completed_at,
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "attempt": attempt,
        "claim_epoch": None,
        "server_incarnation": None,
        "record_json": json.dumps(record),
    }


def _record(
    turn_key: str,
    *,
    refs: list[tuple[str, int, str | None]],
    outputs: list[dict] | None = None,
    success: bool = True,
) -> dict:
    return {
        "turn_output": {
            "turn_key": turn_key,
            "success": success,
            "command_outputs": outputs or [],
        },
        "execution_records": [
            {
                "command_call_id": call_id,
                "parent_call_id": None,
                "command_ordinal": ordinal,
                "span_id": span_id,
            }
            for call_id, ordinal, span_id in refs
        ],
        "routing_events": [],
    }


def _output(
    call_id: str,
    command_name: str,
    parameters: dict,
    *,
    response: str = "ok",
    success: bool = True,
    artifacts: dict | None = None,
) -> dict:
    return {
        "command_response": {
            "response": response,
            "success": success,
            "artifacts": artifacts or {},
        },
        "workflow_name": "comparison-test",
        "context": "global",
        "command_name": command_name,
        "command_parameters": parameters,
        "command_call_id": call_id,
    }


def _output_without_call_id(command_name: str, *, artifacts: dict) -> dict:
    """A recorded `CommandOutput` with no `command_call_id`.

    Real and unremarkable: the join to a dispatch is what is missing, not the
    output, so anything it produced can be attributed to the turn and to no
    finer scope.
    """
    output = _output("unused", command_name, {}, artifacts=artifacts)
    output.pop("command_call_id")
    return output


def _write(store: obs.ObservabilityStore, turn_row: dict, spans: list) -> None:
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(conn, turn_row, [], store._store_redactor())
        if spans:
            store.upsert_span_rows(conn, spans, store._store_redactor())
        conn.commit()


@pytest.fixture
def store(tmp_path: Path) -> obs.ObservabilityStore:
    return obs.ObservabilityStore(str(tmp_path / "evidence.sqlite3"))


@pytest.fixture
def reader(store: obs.ObservabilityStore) -> StoreExecutionReader:
    return StoreExecutionReader("store-1", store)


def _project(ref: ExecutionRef, reader: StoreExecutionReader, **kwargs):
    """Always inject the real ledger and cost roll-up the server uses."""
    return project_execution(
        ref, reader, ledger=execution_ledger, cost_rollup=cost_rollup, **kwargs
    )


def _compare(left: ExecutionRef, right: ExecutionRef, reader, **kwargs):
    return compare_executions(
        left, right, reader, ledger=execution_ledger, cost_rollup=cost_rollup, **kwargs
    )


def seed_winner(store: obs.ObservabilityStore) -> str:
    """A three-dispatch run, one of them with a span-less inner hop."""
    turn_key = "turn-winner"
    spans = [
        _execute_span(
            "w-ex1",
            turn_key,
            call_id="w1",
            command_name="open_directory",
            start_ns=T0,
            parameters={"path": "/a"},
            child_calls=[
                {"call_id": "w1a", "parent_call_id": "w1", "command_name": "wildcard"}
            ],
        ),
        _execute_span(
            "w-ex2",
            turn_key,
            call_id="w2",
            command_name="add_todo",
            start_ns=T0 + 10_000_000,
            parameters={"title": "write tests"},
        ),
        _llm_span("w-llm1", turn_key, start_ns=T0 + 1_000, cost=0.02),
        _llm_span("w-llm2", turn_key, start_ns=T0 + 2_000, cost=None),
    ]
    record = _record(
        turn_key,
        refs=[("w1", 0, "w-ex1"), ("w2", 1, "w-ex2")],
        outputs=[
            _output("w1", "open_directory", {"path": "/a"}),
            _output(
                "w2",
                "add_todo",
                {"title": "write tests"},
                artifacts={
                    "summary": "a short inline artifact",
                    "report": {
                        "__fw_artifact_ref__": "artifact-1",
                        "size": 999_999,
                        "content_type": "application/json",
                        "content_encoding": None,
                        "error": None,
                    },
                },
            ),
        ],
    )
    _write(store, _turn_row(turn_key, record=record, answer="winner answer"), spans)
    return turn_key


def seed_candidate(store: obs.ObservabilityStore) -> str:
    """Same first call, a changed second call, and one extra call."""
    turn_key = "turn-candidate"
    spans = [
        _execute_span(
            "c-ex1",
            turn_key,
            call_id="c1",
            command_name="open_directory",
            start_ns=T0,
            parameters={"path": "/a"},
        ),
        _execute_span(
            "c-ex2",
            turn_key,
            call_id="c2",
            command_name="add_todo",
            start_ns=T0 + 5_000_000,
            parameters={"title": "write DIFFERENT tests"},
            status=tracing.STATUS_ERROR,
            success=False,
        ),
        _execute_span(
            "c-ex3",
            turn_key,
            call_id="c3",
            command_name="list_todos",
            start_ns=T0 + 9_000_000,
            parameters={},
        ),
        _llm_span("c-llm1", turn_key, start_ns=T0 + 1_000, cost=0.05),
    ]
    record = _record(
        turn_key,
        refs=[("c1", 0, "c-ex1"), ("c2", 1, "c-ex2"), ("c3", 2, "c-ex3")],
        outputs=[
            _output("c1", "open_directory", {"path": "/a"}),
            _output(
                "c2",
                "add_todo",
                {"title": "write DIFFERENT tests"},
                success=False,
                response="nope",
            ),
            _output("c3", "list_todos", {}),
        ],
    )
    _write(store, _turn_row(turn_key, record=record, answer="candidate answer"), spans)
    return turn_key


# ----------------------------------------------------------------------
# The reference
# ----------------------------------------------------------------------


class TestExecutionRef:
    def test_a_reference_needs_a_store_and_at_least_one_turn(self):
        with pytest.raises(InvalidExecutionRef):
            ExecutionRef(store_id="", turn_keys=("t",))
        with pytest.raises(InvalidExecutionRef):
            ExecutionRef(store_id="store-1", turn_keys=())
        with pytest.raises(InvalidExecutionRef):
            ExecutionRef(store_id="store-1", turn_keys=("t", "t"))

    def test_ref_id_is_derived_so_two_callers_agree_without_coordinating(self):
        one = ExecutionRef(store_id="s", turn_keys=("t1", "t2"), task_id="task-a")
        two = ExecutionRef.from_mapping(one.as_dict())
        assert one.ref_id() == two.ref_id()
        assert one.ref_id().startswith("xr-")

    def test_the_display_label_does_not_change_the_identity(self):
        pinned = ExecutionRef(store_id="s", turn_keys=("t",), label="Reference")
        renamed = ExecutionRef(store_id="s", turn_keys=("t",), label="Best run")
        # Relabelling "Reference" to "Best run" must not orphan review
        # progress or feedback recorded against the pair.
        assert pinned.ref_id() == renamed.ref_id()

    def test_two_passes_of_one_turn_are_two_references(self):
        teacher = ExecutionRef(store_id="s", turn_keys=("t",), pass_id="teacher")
        student = ExecutionRef(store_id="s", turn_keys=("t",), pass_id="student")
        whole = ExecutionRef(store_id="s", turn_keys=("t",))
        assert len({teacher.ref_id(), student.ref_id(), whole.ref_id()}) == 3

    def test_the_workspace_turn_ref_field_names_parse(self):
        ref = ExecutionRef.from_mapping(
            {"store_id": "s", "logical_turn_keys": ["t1", "t2"]}
        )
        assert ref.turn_keys == ("t1", "t2")
        assert ExecutionRef.from_mapping(
            {"store_id": "s", "logical_turn_key": "t1"}
        ).turn_keys == ("t1",)

    def test_a_reader_refuses_a_store_it_does_not_serve(self, reader):
        seed_winner(reader._store)
        ref = ExecutionRef(store_id="other-store", turn_keys=("turn-winner",))
        with pytest.raises(InvalidExecutionRef):
            _project(ref, reader)

    def test_an_attempt_is_an_exact_integer_or_a_refusal(self):
        # `int(attempt)` would have made each of these name attempt 1, which is
        # a DIFFERENT attempt's evidence than the caller asked for.
        for bogus in (True, False, 1.9, 2.0, "2", "two", object()):
            with pytest.raises(InvalidExecutionRef):
                ExecutionRef(store_id="s", turn_keys=("t",), attempt=bogus)
        assert ExecutionRef(store_id="s", turn_keys=("t",), attempt=0).attempt == 0
        with pytest.raises(InvalidExecutionRef):
            ExecutionRef(store_id="s", turn_keys=("t",), attempt=-1)

    def test_the_wire_boundary_parses_only_the_exact_integer_text(self):
        # A query parameter arrives as text; "2" is attempt 2 and nothing else
        # is silently truncated into one.
        parsed = ExecutionRef.from_mapping(
            {"store_id": "s", "turn_keys": ["t"], "attempt": "2"}
        )
        assert parsed.attempt == 2
        for bogus in ("2.0", "2.5", " ", "abc", 2.5, True):
            with pytest.raises(InvalidExecutionRef):
                ExecutionRef.from_mapping(
                    {"store_id": "s", "turn_keys": ["t"], "attempt": bogus}
                )


class TestDeclaredScope:
    """A reference's declared experiment/task/attempt is checked, not trusted.

    A forged or stale scope over real evidence would render that evidence under
    a false heading and carry the claim into every feedback anchor and
    review-pair key derived from the reference.
    """

    def _seed_attempt(self, store: obs.ObservabilityStore) -> str:
        store.create_experiment(
            "exp-scope", "scoped run", declared_tasks=1, declared_attempts=1
        )
        store.start_attempt("exp-scope", "task-a", 1, "channel-1")
        turn_key = "turn-scoped"
        _write(
            store,
            _turn_row(
                turn_key,
                record=_record(
                    turn_key,
                    refs=[("p1", 0, "p-ex1")],
                    outputs=[_output("p1", "add_todo", {"title": "scoped"})],
                ),
                experiment_id="exp-scope",
                task_id="task-a",
                attempt=1,
            ),
            [
                _execute_span(
                    "p-ex1",
                    turn_key,
                    call_id="p1",
                    command_name="add_todo",
                    start_ns=T0,
                    parameters={"title": "scoped"},
                )
            ],
        )
        return turn_key

    def test_a_declared_scope_that_matches_the_turn_row_projects(self, store, reader):
        turn_key = self._seed_attempt(store)
        projection = _project(
            ExecutionRef(
                store_id="store-1",
                turn_keys=(turn_key,),
                experiment_id="exp-scope",
                task_id="task-a",
                attempt=1,
            ),
            reader,
        )
        assert projection.readable is True
        assert projection.turns[0].experiment_id == "exp-scope"
        assert projection.turns[0].attempt == 1

    @pytest.mark.parametrize(
        "field,value",
        [
            ("experiment_id", "exp-somebody-else"),
            ("task_id", "task-b"),
            ("attempt", 2),
        ],
    )
    def test_a_scope_the_evidence_contradicts_is_refused(
        self, store, reader, field, value
    ):
        turn_key = self._seed_attempt(store)
        scope = {"experiment_id": "exp-scope", "task_id": "task-a", "attempt": 1}
        scope[field] = value
        with pytest.raises(ExecutionScopeMismatch) as raised:
            _project(
                ExecutionRef(store_id="store-1", turn_keys=(turn_key,), **scope), reader
            )
        assert field in str(raised.value)

    def test_a_scope_the_turn_records_nothing_for_is_refused(self, store, reader):
        # `seed_winner` is an ad-hoc chat turn: it belongs to no experiment, so
        # naming one over it is a claim the evidence does not support.
        seed_winner(store)
        with pytest.raises(ExecutionScopeMismatch):
            _project(
                ExecutionRef(
                    store_id="store-1",
                    turn_keys=("turn-winner",),
                    experiment_id="exp-scope",
                ),
                reader,
            )

    def test_declaring_no_scope_still_reports_the_recorded_one(self, store, reader):
        turn_key = self._seed_attempt(store)
        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=(turn_key,)), reader
        )
        # Omitting a field is not a claim about it; the recorded scope is still
        # what the projection reports.
        assert projection.ref.experiment_id is None
        assert projection.turns[0].experiment_id == "exp-scope"
        assert projection.turns[0].task_id == "task-a"


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------


class TestProjection:
    def test_steps_come_from_the_ledger_including_the_span_less_inner_hop(
        self, store, reader
    ):
        seed_winner(store)
        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner",)), reader
        )
        assert [step.command_call_id for step in projection.steps] == ["w1", "w1a", "w2"]
        wrapper = projection.steps[1]
        assert wrapper.child_call is True
        assert wrapper.span_recorded is False
        assert wrapper.status is None      # no span: no status, never invented
        assert wrapper.duration_ns is None

    def test_recorded_parameters_are_read_from_the_durable_record(self, store, reader):
        seed_winner(store)
        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner",)), reader
        )
        first = projection.steps[0]
        assert first.parameters == {"path": "/a"}
        assert first.parameters_source == "record"
        assert first.parameters_digest is not None
        assert projection.steps[1].parameters_digest is None

    def test_parameters_fall_back_to_the_span_when_the_record_has_no_output(
        self, store, reader
    ):
        turn_key = "turn-span-only"
        spans = [
            _execute_span(
                "s-ex1",
                turn_key,
                call_id="s1",
                command_name="add_todo",
                start_ns=T0,
                parameters={"title": "from the span"},
            )
        ]
        _write(
            store,
            _turn_row(turn_key, record=_record(turn_key, refs=[("s1", 0, "s-ex1")])),
            spans,
        )
        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=(turn_key,)), reader
        )
        assert projection.steps[0].parameters == {"title": "from the span"}
        assert projection.steps[0].parameters_source == "span"

    def test_answers_and_artifacts_are_the_default_view(self, store, reader):
        seed_winner(store)
        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner",)), reader
        )
        assert projection.answers() == [
            {
                "turn_index": 0,
                "turn_key": "turn-winner",
                "answer": "winner answer",
                "status": "completed",
                "success": True,
                "failure_reason": None,
                # A whole-turn projection owns its answer; a pass-scoped one
                # does not, and says so in the same field.
                "pass_id": None,
                "attribution": ATTRIBUTION_TURN,
                "pass_content_recorded": False,
            }
        ]
        by_key = {artifact.key: artifact for artifact in projection.artifacts}
        assert set(by_key) == {"report", "summary"}
        # An offloaded artifact is reachable by id; an inline one is reachable
        # from the record. Neither is missing.
        assert by_key["report"].artifact_id == "artifact-1"
        assert by_key["report"].inline is False
        assert by_key["report"].size_bytes == 999_999
        assert by_key["summary"].artifact_id is None
        assert by_key["summary"].inline is True
        assert by_key["report"].command_call_id == "w2"

    def test_a_nested_wrapper_is_not_counted_twice_in_the_timing_rollup(
        self, store, reader
    ):
        turn_key = "turn-nested"
        # An outer dispatch of 5ms whose inner dispatch of 3ms has its own
        # span: summing both rows would report 8ms of work that took 5.
        spans = [
            _execute_span(
                "n-ex1",
                turn_key,
                call_id="n1",
                command_name="outer",
                start_ns=T0,
                duration_ns=5_000_000,
            ),
            _execute_span(
                "n-ex2",
                turn_key,
                call_id="n2",
                command_name="inner",
                start_ns=T0 + 1_000_000,
                duration_ns=3_000_000,
                parent_span_id="n-ex1",
                parent_call_id="n1",
            ),
        ]
        record = _record(
            turn_key, refs=[("n1", 0, "n-ex1"), ("n2", 1, "n-ex2")]
        )
        _write(store, _turn_row(turn_key, record=record), spans)
        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=(turn_key,)), reader
        )
        assert len(projection.steps) == 2
        assert projection.timing["root_step_duration_ns"] == 5_000_000

    def test_recorded_cost_only_and_never_a_zero(self, store, reader):
        seed_winner(store)
        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner",)), reader
        )
        assert projection.cost == {
            "calls": 2,
            "recorded": 1,
            "unrecorded": 1,
            "total": pytest.approx(0.02),
        }

        turn_key = "turn-no-cost"
        _write(
            store,
            _turn_row(turn_key, record=_record(turn_key, refs=[])),
            [_llm_span("z-llm", turn_key, start_ns=T0, cost=None)],
        )
        no_cost = _project(
            ExecutionRef(store_id="store-1", turn_keys=(turn_key,)), reader
        )
        assert no_cost.cost["total"] is None       # not 0.0
        assert no_cost.cost["unrecorded"] == 1

    def test_a_missing_turn_is_reported_and_the_rest_still_projects(
        self, store, reader
    ):
        seed_winner(store)
        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner", "turn-pruned")),
            reader,
        )
        assert projection.readable is True
        assert len(projection.turns) == 1
        assert len(projection.unavailable) == 1
        assert "turn-pruned" in projection.unavailable[0]

    def test_an_execution_with_no_readable_turn_is_inspectable_not_fatal(
        self, store, reader
    ):
        projection = _project(
            ExecutionRef(store_id="store-1", turn_keys=("nothing-here",)), reader
        )
        assert projection.readable is False
        assert projection.steps == ()
        assert projection.timing["wall_ms"] is None
        assert projection.as_dict()["unavailable"]

    def test_a_multi_turn_attempt_stays_navigable_in_full(self, store, reader):
        # A real experiment attempt, because the reference declares that scope
        # and the projection now checks the declaration against the turn rows.
        store.create_experiment(
            "exp-multi", "multi-turn attempt", declared_tasks=1, declared_attempts=1
        )
        store.start_attempt("exp-multi", "task-1", 1, "channel-1")
        for index in (1, 2, 3):
            turn_key = f"turn-multi-{index}"
            spans = [
                _execute_span(
                    f"m-ex{index}",
                    turn_key,
                    call_id=f"m{index}",
                    command_name=f"step_{index}",
                    start_ns=T0 + index * 1_000_000,
                )
            ]
            _write(
                store,
                _turn_row(
                    turn_key,
                    record=_record(
                        turn_key, refs=[(f"m{index}", 0, f"m-ex{index}")]
                    ),
                    answer=f"answer {index}",
                    experiment_id="exp-multi",
                    task_id="task-1",
                    attempt=1,
                ),
                spans,
            )
        ref = ExecutionRef(
            store_id="store-1",
            turn_keys=("turn-multi-1", "turn-multi-2", "turn-multi-3"),
            experiment_id="exp-multi",
            task_id="task-1",
            attempt=1,
        )
        projection = _project(ref, reader)
        assert [turn.turn_key for turn in projection.turns] == [
            "turn-multi-1",
            "turn-multi-2",
            "turn-multi-3",
        ]
        assert [answer["answer"] for answer in projection.answers()] == [
            "answer 1",
            "answer 2",
            "answer 3",
        ]
        assert [step.turn_index for step in projection.steps] == [0, 1, 2]
        assert projection.timing["turns"] == 3
        # Three turns of two recorded seconds each.
        assert projection.timing["wall_ms"] == 6000


# ----------------------------------------------------------------------
# Recorded passes within one trace
# ----------------------------------------------------------------------


def seed_teacher_student(store: obs.ObservabilityStore) -> str:
    """One turn, one trace, two recorded passes plus extraction activity.

    This is the shape `distillation.py` produces: the agent runs twice for a
    single user message, so both passes share a trace id, and the insight
    extraction that follows belongs to neither.

    It is also the LIMIT of what the shipped producer records, and the seed is
    deliberately built that way: ONE turn row with one answer, one status and one
    started_at/completed_at. `distillation.py` stamps no pass marker (the
    `fw.pass` attributes below are what `fix-txxy` would add), writes no per-pass
    turn row, and keeps each pass's answer, plan and action log in process --
    `summarize_and_record_turn` appends to in-memory conversation history and
    `ObservabilityStore.list_distillation_runs` is a stub returning `[]`. So no
    per-pass answer exists to project, and the tests below assert that the turn's
    answer is reported as SHARED rather than as either pass's own.

    The artifacts are the same point: one is joined to a teacher dispatch by
    `command_call_id`, and one rides on a `CommandOutput` that recorded no call
    id, so no pass can claim it.
    """
    turn_key = "turn-distill"
    spans = [
        _execute_span(
            "t-ex1",
            turn_key,
            call_id="t1",
            command_name="open_directory",
            start_ns=T0,
            parameters={"path": "/a"},
            extra={"fw.pass": "teacher"},
        ),
        _execute_span(
            "t-ex2",
            turn_key,
            call_id="t2",
            command_name="add_todo",
            start_ns=T0 + 1_000_000,
            parameters={"title": "correct"},
            extra={"fw.pass": "teacher"},
        ),
        _execute_span(
            "s-ex1",
            turn_key,
            call_id="s1",
            command_name="open_directory",
            start_ns=T0 + 10_000_000,
            parameters={"path": "/a"},
            extra={"fw.pass": "student"},
        ),
        _execute_span(
            "s-ex2",
            turn_key,
            call_id="s2",
            command_name="add_todo",
            start_ns=T0 + 11_000_000,
            parameters={"title": "wrong"},
            extra={"fw.pass": "student"},
        ),
        _llm_span("t-llm", turn_key, start_ns=T0 + 500, cost=0.10, parent_span_id="t-ex1"),
        _llm_span("s-llm", turn_key, start_ns=T0 + 10_500, cost=0.01, parent_span_id="s-ex1"),
        # Insight extraction: after both passes, belonging to neither.
        _llm_span("x-llm", turn_key, start_ns=T0 + 20_000_000, cost=0.30),
    ]
    record = _record(
        turn_key,
        refs=[
            ("t1", 0, "t-ex1"),
            ("t2", 1, "t-ex2"),
            ("s1", 2, "s-ex1"),
            ("s2", 3, "s-ex2"),
        ],
        outputs=[
            _output("t1", "open_directory", {"path": "/a"}),
            _output(
                "t2",
                "add_todo",
                {"title": "correct"},
                artifacts={"teacher_note": "produced by a teacher dispatch"},
            ),
            _output("s1", "open_directory", {"path": "/a"}),
            _output("s2", "add_todo", {"title": "wrong"}),
            _output_without_call_id(
                "summarize", artifacts={"shared_note": "no call id to join on"}
            ),
        ],
    )
    _write(store, _turn_row(turn_key, record=record, answer="student answer"), spans)
    return turn_key


class TestRecordedPasses:
    def test_selectors_are_discovered_from_what_a_producer_stamped(self, store, reader):
        seed_teacher_student(store)
        selectors = discover_pass_selectors(
            reader.trace("store-1", "turn-distill"), attribute_key="fw.pass"
        )
        assert [selector.pass_id for selector in selectors] == ["student", "teacher"]

    def test_a_turn_with_no_stamped_pass_discovers_nothing(self, store, reader):
        seed_winner(store)
        assert (
            discover_pass_selectors(
                reader.trace("store-1", "turn-winner"), attribute_key="fw.pass"
            )
            == []
        )

    def test_two_passes_of_one_trace_project_separately(self, store, reader):
        seed_teacher_student(store)
        teacher = PassSelector(
            pass_id="teacher", attribute_key="fw.pass", attribute_value="teacher"
        )
        student = PassSelector(
            pass_id="student", attribute_key="fw.pass", attribute_value="student"
        )
        left = _project(
            ExecutionRef(
                store_id="store-1", turn_keys=("turn-distill",), pass_id="teacher"
            ),
            reader,
            pass_selector=teacher,
        )
        right = _project(
            ExecutionRef(
                store_id="store-1", turn_keys=("turn-distill",), pass_id="student"
            ),
            reader,
            pass_selector=student,
        )
        assert [step.command_call_id for step in left.steps] == ["t1", "t2"]
        assert [step.command_call_id for step in right.steps] == ["s1", "s2"]
        assert all(step.pass_id == "teacher" for step in left.steps)
        # The other pass's steps are withheld from this view but counted, not
        # silently dropped.
        assert {step.command_call_id for step in left.unassigned_steps} == {"s1", "s2"}

    def test_extraction_activity_lands_in_neither_pass(self, store, reader):
        seed_teacher_student(store)
        teacher = _project(
            ExecutionRef(
                store_id="store-1", turn_keys=("turn-distill",), pass_id="teacher"
            ),
            reader,
            pass_selector=PassSelector(
                pass_id="teacher", attribute_key="fw.pass", attribute_value="teacher"
            ),
        )
        student = _project(
            ExecutionRef(
                store_id="store-1", turn_keys=("turn-distill",), pass_id="student"
            ),
            reader,
            pass_selector=PassSelector(
                pass_id="student", attribute_key="fw.pass", attribute_value="student"
            ),
        )
        # The 0.30 extraction call is in neither pass's cost.
        assert teacher.cost["total"] == pytest.approx(0.10)
        assert student.cost["total"] == pytest.approx(0.01)
        whole = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-distill",)), reader
        )
        assert whole.cost["total"] == pytest.approx(0.41)

    def test_an_explicit_exclusion_removes_activity_from_a_subtree_rule(
        self, store, reader
    ):
        seed_teacher_student(store)
        # A subtree rule that would otherwise sweep the extraction call in.
        selector = PassSelector(
            pass_id="teacher",
            root_span_ids=frozenset({"t-ex1"}),
            exclude_span_ids=frozenset({"t-llm"}),
        )
        projection = _project(
            ExecutionRef(
                store_id="store-1", turn_keys=("turn-distill",), pass_id="teacher"
            ),
            reader,
            pass_selector=selector,
        )
        assert [step.command_call_id for step in projection.steps] == ["t1"]
        assert projection.cost["total"] is None

    def test_excluding_a_span_excludes_everything_under_it(self, store, reader):
        seed_teacher_student(store)
        # Exclusion is checked against a span's whole ancestry, which is why
        # there is no separate subtree-exclusion rule: excluding the teacher's
        # first dispatch also excludes the LLM call recorded beneath it.
        projection = _project(
            ExecutionRef(
                store_id="store-1", turn_keys=("turn-distill",), pass_id="teacher"
            ),
            reader,
            pass_selector=PassSelector(
                pass_id="teacher",
                attribute_key="fw.pass",
                attribute_value="teacher",
                exclude_span_ids=frozenset({"t-ex1"}),
            ),
        )
        assert [step.command_call_id for step in projection.steps] == ["t2"]
        assert projection.cost["total"] is None

    def test_a_selector_must_disagree_with_the_reference_loudly(self, store, reader):
        seed_teacher_student(store)
        with pytest.raises(InvalidExecutionRef):
            _project(
                ExecutionRef(
                    store_id="store-1", turn_keys=("turn-distill",), pass_id="teacher"
                ),
                reader,
                pass_selector=PassSelector(
                    pass_id="student",
                    attribute_key="fw.pass",
                    attribute_value="student",
                ),
            )

    def test_a_selector_with_no_membership_rule_is_refused(self):
        with pytest.raises(ValueError):
            PassSelector(pass_id="teacher")

    def test_a_pass_id_with_no_selector_is_refused(self, store, reader):
        seed_teacher_student(store)
        # Without a selector nothing can resolve membership, so the projection
        # would be the WHOLE turn shown under the teacher's name.
        with pytest.raises(InvalidExecutionRef):
            _project(
                ExecutionRef(
                    store_id="store-1", turn_keys=("turn-distill",), pass_id="teacher"
                ),
                reader,
            )

    def test_a_selector_with_no_pass_id_on_the_reference_is_refused(
        self, store, reader
    ):
        seed_teacher_student(store)
        # The steps would be the teacher's while ref_id() -- and therefore every
        # anchor and review-pair key -- would be the whole turn's.
        with pytest.raises(InvalidExecutionRef):
            _project(
                ExecutionRef(store_id="store-1", turn_keys=("turn-distill",)),
                reader,
                pass_selector=PassSelector(
                    pass_id="teacher", attribute_key="fw.pass", attribute_value="teacher"
                ),
            )

    def test_a_pass_the_evidence_does_not_record_is_refused(self, store, reader):
        seed_teacher_student(store)
        for selector in (
            PassSelector(
                pass_id="tutor", attribute_key="fw.pass", attribute_value="tutor"
            ),
            PassSelector(pass_id="tutor", span_ids=frozenset({"no-such-span"})),
            PassSelector(pass_id="tutor", root_span_ids=frozenset({"no-such-root"})),
        ):
            with pytest.raises(UnknownRecordedPass):
                _project(
                    ExecutionRef(
                        store_id="store-1", turn_keys=("turn-distill",), pass_id="tutor"
                    ),
                    reader,
                    pass_selector=selector,
                )

    def test_the_turns_answer_and_wall_time_are_shared_not_the_passs_own(
        self, store, reader
    ):
        seed_teacher_student(store)
        projections = {}
        for pass_id in ("teacher", "student"):
            projections[pass_id] = _project(
                ExecutionRef(
                    store_id="store-1", turn_keys=("turn-distill",), pass_id=pass_id
                ),
                reader,
                pass_selector=PassSelector(
                    pass_id=pass_id, attribute_key="fw.pass", attribute_value=pass_id
                ),
            )
        for pass_id, projection in projections.items():
            answer = projection.answers()[0]
            # The one recorded answer, quoted as the turn's and labelled as
            # shared: the producer records no per-pass answer to show instead.
            assert answer["answer"] == "student answer"
            assert answer["attribution"] == ATTRIBUTION_SHARED
            assert answer["pass_content_recorded"] is False
            assert answer["pass_id"] == pass_id
            assert projection.content_attribution == ATTRIBUTION_SHARED
            assert projection.turns[0].content_attribution == ATTRIBUTION_SHARED
            # Two passes share one turn row's timestamps, so neither owns the
            # wall time; the shared figure is reported beside it, labelled.
            assert projection.timing["wall_ms"] is None
            assert projection.timing["wall_ms_attribution"] == ATTRIBUTION_SHARED
            assert projection.timing["turn_wall_ms"] == 2000
        # What IS pass-scoped stays pass-scoped: steps and recorded LLM cost.
        assert projections["teacher"].cost["total"] == pytest.approx(0.10)
        assert projections["student"].cost["total"] == pytest.approx(0.01)

        whole = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-distill",)), reader
        )
        assert whole.answers()[0]["attribution"] == ATTRIBUTION_TURN
        assert whole.timing["wall_ms"] == 2000
        assert whole.timing["wall_ms_attribution"] == ATTRIBUTION_TURN

    def test_an_unjoinable_artifact_is_claimed_by_neither_pass(self, store, reader):
        seed_teacher_student(store)
        teacher = _project(
            ExecutionRef(
                store_id="store-1", turn_keys=("turn-distill",), pass_id="teacher"
            ),
            reader,
            pass_selector=PassSelector(
                pass_id="teacher", attribute_key="fw.pass", attribute_value="teacher"
            ),
        )
        student = _project(
            ExecutionRef(
                store_id="store-1", turn_keys=("turn-distill",), pass_id="student"
            ),
            reader,
            pass_selector=PassSelector(
                pass_id="student", attribute_key="fw.pass", attribute_value="student"
            ),
        )
        # The artifact joined to a teacher dispatch is the teacher's.
        assert [artifact.key for artifact in teacher.artifacts] == ["teacher_note"]
        assert teacher.artifacts[0].attribution == ATTRIBUTION_PASS
        assert student.artifacts == ()
        # The one with no call id is reported apart on BOTH sides rather than
        # shown under each pass as if each had produced it.
        for projection in (teacher, student):
            assert [
                artifact.key for artifact in projection.unattributed_artifacts
            ] == ["shared_note"]
            assert (
                projection.unattributed_artifacts[0].attribution
                == ATTRIBUTION_UNATTRIBUTED
            )
        whole = _project(
            ExecutionRef(store_id="store-1", turn_keys=("turn-distill",)), reader
        )
        assert {artifact.key for artifact in whole.artifacts} == {
            "teacher_note",
            "shared_note",
        }
        assert all(
            artifact.attribution == ATTRIBUTION_TURN for artifact in whole.artifacts
        )
        assert whole.unattributed_artifacts == ()

    def test_the_two_passes_compare_through_the_same_view(self, store, reader):
        seed_teacher_student(store)
        comparison = _compare(
            ExecutionRef(
                store_id="store-1", turn_keys=("turn-distill",), pass_id="teacher"
            ),
            ExecutionRef(
                store_id="store-1", turn_keys=("turn-distill",), pass_id="student"
            ),
            reader,
            left_pass=PassSelector(
                pass_id="teacher", attribute_key="fw.pass", attribute_value="teacher"
            ),
            right_pass=PassSelector(
                pass_id="student", attribute_key="fw.pass", attribute_value="student"
            ),
        )
        kinds = [pair.kind for pair in comparison.alignment.pairs]
        assert kinds == [PAIR_MATCHED, PAIR_MATCHED]
        bases = [pair.basis for pair in comparison.alignment.pairs]
        assert bases == [BASIS_COMMAND_CONTEXT_PARAMETERS, BASIS_COMMAND_CONTEXT]
        # The changed parameters are the difference, and they are visible.
        differences = comparison.differences()
        assert len(differences) == 1
        assert differences[0].left.command_name == "add_todo"


# ----------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------


class TestAlignment:
    def test_winner_versus_candidate_shows_changed_and_inserted_calls(
        self, store, reader
    ):
        seed_winner(store)
        seed_candidate(store)
        comparison = _compare(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner",), label="Winner"),
            ExecutionRef(store_id="store-1", turn_keys=("turn-candidate",)),
            reader,
        )
        pairs = comparison.alignment.pairs
        described = [
            (
                pair.kind,
                (pair.left.command_name if pair.left else None),
                (pair.right.command_name if pair.right else None),
            )
            for pair in pairs
        ]
        assert described == [
            (PAIR_MATCHED, "open_directory", "open_directory"),
            # The span-less inner hop exists only on the winner's side.
            (PAIR_LEFT_ONLY, "wildcard", None),
            (PAIR_MATCHED, "add_todo", "add_todo"),
            (PAIR_RIGHT_ONLY, None, "list_todos"),
        ]
        summary = comparison.summary()
        assert summary["matched"] == 2
        assert summary["left_only"] == 1
        assert summary["right_only"] == 1

    def test_a_matched_pair_states_which_recorded_fields_agreed(self, store, reader):
        seed_winner(store)
        seed_candidate(store)
        comparison = _compare(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner",)),
            ExecutionRef(store_id="store-1", turn_keys=("turn-candidate",)),
            reader,
        )
        matched = [p for p in comparison.alignment.pairs if p.kind == PAIR_MATCHED]
        assert matched[0].basis == BASIS_COMMAND_CONTEXT_PARAMETERS
        # Same command and context, different parameters: a weaker claim, and
        # it says so rather than asserting the calls were identical.
        assert matched[1].basis == BASIS_COMMAND_CONTEXT
        assert matched[1].left.parameters_digest != matched[1].right.parameters_digest

    def test_alignment_is_never_by_list_position(self, store, reader):
        left_key, right_key = "turn-order-a", "turn-order-b"
        _write(
            store,
            _turn_row(
                left_key,
                record=_record(
                    left_key,
                    refs=[("a1", 0, "a-ex1"), ("a2", 1, "a-ex2")],
                    outputs=[
                        _output("a1", "alpha", {"n": 1}),
                        _output("a2", "beta", {"n": 2}),
                    ],
                ),
            ),
            [
                _execute_span("a-ex1", left_key, call_id="a1", command_name="alpha", start_ns=T0),
                _execute_span(
                    "a-ex2", left_key, call_id="a2", command_name="beta", start_ns=T0 + 1_000_000
                ),
            ],
        )
        # The same two calls, with an unrelated call wedged in front.
        _write(
            store,
            _turn_row(
                right_key,
                record=_record(
                    right_key,
                    refs=[("b0", 0, "b-ex0"), ("b1", 1, "b-ex1"), ("b2", 2, "b-ex2")],
                    outputs=[
                        _output("b0", "gamma", {"n": 0}),
                        _output("b1", "alpha", {"n": 1}),
                        _output("b2", "beta", {"n": 2}),
                    ],
                ),
            ),
            [
                _execute_span("b-ex0", right_key, call_id="b0", command_name="gamma", start_ns=T0),
                _execute_span(
                    "b-ex1", right_key, call_id="b1", command_name="alpha", start_ns=T0 + 1_000_000
                ),
                _execute_span(
                    "b-ex2", right_key, call_id="b2", command_name="beta", start_ns=T0 + 2_000_000
                ),
            ],
        )
        comparison = _compare(
            ExecutionRef(store_id="store-1", turn_keys=(left_key,)),
            ExecutionRef(store_id="store-1", turn_keys=(right_key,)),
            reader,
        )
        described = [
            (pair.kind, pair.left.command_name if pair.left else None,
             pair.right.command_name if pair.right else None)
            for pair in comparison.alignment.pairs
        ]
        # Position 1 on the left is `alpha` and position 1 on the right is
        # `gamma`; the insertion is shown as an insertion, not as a change.
        assert described == [
            (PAIR_RIGHT_ONLY, None, "gamma"),
            (PAIR_MATCHED, "alpha", "alpha"),
            (PAIR_MATCHED, "beta", "beta"),
        ]

    def test_cross_run_call_ids_are_not_a_correspondence_rule(self, store, reader):
        # The same call id in both runs naming different commands must not be
        # matched by that id.
        for key, command in (("turn-id-a", "alpha"), ("turn-id-b", "omega")):
            _write(
                store,
                _turn_row(
                    key,
                    record=_record(
                        key,
                        refs=[("shared-call", 0, f"{key}-ex")],
                        outputs=[_output("shared-call", command, {})],
                    ),
                ),
                [
                    _execute_span(
                        f"{key}-ex",
                        key,
                        call_id="shared-call",
                        command_name=command,
                        start_ns=T0,
                    )
                ],
            )
        comparison = _compare(
            ExecutionRef(store_id="store-1", turn_keys=("turn-id-a",)),
            ExecutionRef(store_id="store-1", turn_keys=("turn-id-b",)),
            reader,
        )
        assert [pair.kind for pair in comparison.alignment.pairs] == [
            PAIR_LEFT_ONLY,
            PAIR_RIGHT_ONLY,
        ]

    def test_repeated_identical_calls_are_matched_but_labelled_ambiguous(self):
        left = _steps("alpha", "alpha")
        right = _steps("alpha", "alpha")
        alignment = align_steps(left, right)
        assert [pair.kind for pair in alignment.pairs] == [PAIR_MATCHED, PAIR_MATCHED]
        assert all(pair.ambiguous for pair in alignment.pairs)
        assert {pair.ambiguity_reason for pair in alignment.pairs} == {"repeated-key"}

    def test_a_command_name_only_match_is_labelled_ambiguous(self):
        left = [_step("alpha", context="ctx-a", params={"n": 1})]
        right = [_step("alpha", context="ctx-b", params={"n": 2})]
        alignment = align_steps(left, right)
        pair = alignment.pairs[0]
        assert pair.kind == PAIR_MATCHED
        assert pair.basis == BASIS_COMMAND
        assert pair.ambiguous is True
        assert pair.ambiguity_reason == "command-name-only"

    def test_a_step_with_no_recorded_command_name_is_unknown_not_matched(self):
        left = [_step(None)]
        right = [_step(None)]
        alignment = align_steps(left, right)
        assert [pair.kind for pair in alignment.pairs] == [
            PAIR_LEFT_ONLY,
            PAIR_RIGHT_ONLY,
        ]
        assert all(pair.basis == BASIS_UNKNOWN for pair in alignment.pairs)
        assert alignment.summary()["unknown"] == 2

    def test_a_wrapper_never_absorbs_a_top_level_dispatch(self):
        left = [_step("wildcard", child_call=True)]
        right = [_step("wildcard", child_call=False)]
        alignment = align_steps(left, right)
        assert [pair.kind for pair in alignment.pairs] == [
            PAIR_LEFT_ONLY,
            PAIR_RIGHT_ONLY,
        ]

    def test_a_recorded_alignment_is_reused_rather_than_recomputed(self):
        left = [_step("alpha", call_id="L1"), _step("beta", call_id="L2")]
        right = [_step("gamma", call_id="R1"), _step("delta", call_id="R2")]
        alignment = align_steps(
            left,
            right,
            recorded_alignment=[
                {"left_command_call_id": "L1", "right_command_call_id": "R1"},
                {"left_command_call_id": "L2", "right_command_call_id": "R2"},
            ],
        )
        # Nothing about `alpha`/`gamma` would match deterministically; the
        # stored alignment is authoritative and says they correspond.
        assert [pair.kind for pair in alignment.pairs] == [PAIR_MATCHED, PAIR_MATCHED]
        assert all(pair.basis == BASIS_RECORDED for pair in alignment.pairs)
        assert alignment.summary()["recorded_matches"] == 2

    def test_steps_a_partial_recorded_alignment_omits_are_still_aligned(self):
        left = [
            _step("alpha", call_id="L1"),
            _step("beta", call_id="L2", params={"n": 1}),
        ]
        right = [
            _step("gamma", call_id="R1"),
            _step("beta", call_id="R2", params={"n": 1}),
        ]
        alignment = align_steps(
            left,
            right,
            recorded_alignment=[
                {"left_command_call_id": "L1", "right_command_call_id": "R1"}
            ],
        )
        assert [pair.basis for pair in alignment.pairs] == [
            BASIS_RECORDED,
            BASIS_COMMAND_CONTEXT_PARAMETERS,
        ]

    def test_a_supplied_alignment_naming_unknown_steps_is_refused(self):
        left = [_step("alpha", call_id="L1")]
        right = [_step("gamma", call_id="R1")]
        # Nothing in this repo writes a structured alignment, so this input is
        # external by definition. Skipping the entries that do not resolve would
        # report a rejected alignment as an accepted one.
        with pytest.raises(InvalidRecordedAlignment) as raised:
            align_steps(
                left,
                right,
                recorded_alignment=[
                    {"left_command_call_id": "L1", "right_command_call_id": "R-gone"}
                ],
            )
        assert "R-gone" in str(raised.value)
        with pytest.raises(InvalidRecordedAlignment):
            align_steps(left, right, recorded_alignment=["not an object"])

    def test_a_recorded_alignment_that_crosses_itself_is_kept_and_labelled(self):
        left = [_step("alpha", call_id="L1"), _step("beta", call_id="L2")]
        right = [_step("beta", call_id="R1"), _step("alpha", call_id="R2")]
        alignment = align_steps(
            left,
            right,
            recorded_alignment=[
                {"left_command_call_id": "L1", "right_command_call_id": "R2"},
                {"left_command_call_id": "L2", "right_command_call_id": "R1"},
            ],
        )
        matched = [p for p in alignment.pairs if p.kind == PAIR_MATCHED]
        assert len(matched) == 2
        reasons = {p.ambiguity_reason for p in matched if p.ambiguous}
        assert reasons == {"recorded-alignment-out-of-order"}

    def test_an_execution_compared_with_itself_matches_throughout(self, store, reader):
        seed_winner(store)
        ref = ExecutionRef(store_id="store-1", turn_keys=("turn-winner",))
        comparison = _compare(ref, ref, reader)
        assert all(
            pair.kind == PAIR_MATCHED for pair in comparison.alignment.pairs
        )
        assert comparison.differences() == []

    def test_one_missing_side_still_produces_a_readable_comparison(
        self, store, reader
    ):
        seed_winner(store)
        comparison = _compare(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner",)),
            ExecutionRef(store_id="store-1", turn_keys=("turn-gone",)),
            reader,
        )
        assert comparison.summary()["left_readable"] is True
        assert comparison.summary()["right_readable"] is False
        assert comparison.summary()["right_unavailable"] == 1
        assert all(
            pair.kind == PAIR_LEFT_ONLY for pair in comparison.alignment.pairs
        )


def _step(
    command_name,
    *,
    call_id: str | None = None,
    context: str = "global",
    params: dict | None = None,
    child_call: bool = False,
):
    """A projected step, built directly for the pure alignment cases.

    These exercise `align_steps` itself, which takes projected steps and no
    store; the store-backed cases above cover the projection that feeds it.
    """
    from fastworkflow.observability.comparison import ExecutionStep
    from fastworkflow.observability.comparison import _canonical_json, _digest

    _step.counter = getattr(_step, "counter", 0) + 1
    return ExecutionStep(
        turn_index=0,
        turn_key="turn-x",
        position=_step.counter,
        command_call_id=call_id or f"call-{_step.counter}",
        parent_call_id=None,
        command_ordinal=None,
        span_id=f"span-{_step.counter}",
        command_name=command_name,
        context=context,
        status="ok",
        success=True,
        start_ns=T0,
        duration_ns=1_000,
        in_record=True,
        span_recorded=True,
        child_call=child_call,
        asked_user=0,
        parameters=params,
        parameters_source="record" if params is not None else None,
        parameters_digest=(
            _digest(_canonical_json(params)) if params is not None else None
        ),
    )


def _steps(*names):
    return [_step(name, params={"same": True}) for name in names]


# ----------------------------------------------------------------------
# Evidence anchors and the machine projection
# ----------------------------------------------------------------------


class TestAnchorsAndDigest:
    def test_a_step_anchor_is_accepted_by_the_real_feedback_validator(
        self, store, reader
    ):
        seed_winner(store)
        ref = ExecutionRef(store_id="store-1", turn_keys=("turn-winner",))
        projection = _project(ref, reader)
        step = projection.steps[0]
        anchor = anchor_for_step(ref, step)
        assert anchor.anchorable is True
        # The real validator: target kind, span membership and provenance are
        # all checked against the recorded turn.
        store.add_human_feedback(
            anchor.turn_key,
            target_kind=anchor.target_kind,
            span_ids=list(anchor.span_ids),
            target_label="open_directory",
            provenance="human",
            comment="matched the reference",
            category="conclusions",
            subcategory="what_went_right",
        )
        recorded = store.list_human_feedback("turn-winner")
        assert len(recorded) == 1
        assert recorded[0]["span_ids"] == ["w-ex1"]

    def test_a_span_less_hop_can_only_be_anchored_at_turn_scope(self, store, reader):
        seed_winner(store)
        ref = ExecutionRef(store_id="store-1", turn_keys=("turn-winner",))
        wrapper = _project(ref, reader).steps[1]
        anchor = anchor_for_step(ref, wrapper)
        assert anchor.span_ids == ()
        assert anchor.target_kind == "turn"
        assert anchor.fallback_target_kind == "turn"
        store.add_human_feedback(
            anchor.turn_key,
            target_kind=anchor.target_kind,
            span_ids=[],
            target_label="wildcard hop",
            provenance="coding_agent",
            comment="no span to anchor to",
            category="conclusions",
            subcategory="what_went_wrong",
        )
        assert len(store.list_human_feedback("turn-winner")) == 1

    def test_a_matched_pair_anchors_to_both_sides(self, store, reader):
        seed_winner(store)
        seed_candidate(store)
        comparison = _compare(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner",)),
            ExecutionRef(store_id="store-1", turn_keys=("turn-candidate",)),
            reader,
        )
        pair = comparison.alignment.pairs[0]
        anchors = anchors_for_pair(comparison, pair)
        assert anchors["left"]["turn_key"] == "turn-winner"
        assert anchors["right"]["turn_key"] == "turn-candidate"
        assert anchors["left"]["ref_id"] != anchors["right"]["ref_id"]

    def test_the_digest_carries_counts_and_ids_but_no_recorded_text(
        self, store, reader
    ):
        seed_winner(store)
        seed_candidate(store)
        comparison = _compare(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner",)),
            ExecutionRef(store_id="store-1", turn_keys=("turn-candidate",)),
            reader,
        )
        digest = comparison_digest(comparison)
        blob = json.dumps(digest)
        for secret in (
            "winner answer",
            "candidate answer",
            "do the thing",
            "write tests",
            "/a",
        ):
            assert secret not in blob
        assert digest["left"]["steps"] == 3
        assert digest["right"]["steps"] == 3
        assert digest["left"]["artifacts"] == 2
        assert digest["left"]["artifacts_offloaded"] == 1
        assert digest["differences"] >= 1

    def test_the_human_and_machine_projections_are_the_same_object(
        self, store, reader
    ):
        seed_winner(store)
        seed_candidate(store)
        comparison = _compare(
            ExecutionRef(store_id="store-1", turn_keys=("turn-winner",)),
            ExecutionRef(store_id="store-1", turn_keys=("turn-candidate",)),
            reader,
        )
        wire = comparison.as_dict()
        # The SPA and an agent read the same alignment and the same anchors;
        # neither recomputes a diff of its own.
        assert wire["alignment"]["summary"] == comparison.alignment.summary()
        assert [pair["kind"] for pair in wire["alignment"]["pairs"]] == [
            pair.kind for pair in comparison.alignment.pairs
        ]
        assert wire["review_pair_key"] == review_pair_key(
            comparison.left.ref, comparison.right.ref
        )
        assert json.loads(json.dumps(wire))["summary"] == comparison.summary()


# ----------------------------------------------------------------------
# Sealed archives read through the workspace
# ----------------------------------------------------------------------


class TestWorkspaceReader:
    def test_a_sealed_archive_projects_through_the_workspace_reader(
        self, tmp_path: Path
    ):
        source = tmp_path / "live.sqlite3"
        live = obs.ObservabilityStore(str(source))
        live.create_experiment(
            "exp-sealed", "sealed run", declared_tasks=1, declared_attempts=1
        )
        live.start_attempt("exp-sealed", "task-1", 1, "channel-1")
        live.finish_attempt(
            "exp-sealed", "task-1", 1, outcome="pass", outcome_source="test"
        )
        turn_key = "turn-sealed"
        row = _turn_row(
            turn_key,
            record=_record(
                turn_key,
                refs=[("q1", 0, "q-ex1")],
                outputs=[_output("q1", "add_todo", {"title": "sealed"})],
            ),
            experiment_id="exp-sealed",
            task_id="task-1",
            attempt=1,
        )
        _write(
            live,
            row,
            [
                _execute_span(
                    "q-ex1",
                    turn_key,
                    call_id="q1",
                    command_name="add_todo",
                    start_ns=T0,
                    parameters={"title": "sealed"},
                )
            ],
        )
        archive = obs.ObservabilityStore(str(source), migrate=False).archive_to(
            str(tmp_path / "sealed.sqlite3")
        )
        manifest = tmp_path / "workspace.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": WORKSPACE_SCHEMA,
                    "workspace_id": "ws-1",
                    "label": "sealed",
                    "stores": [
                        {
                            "store_id": "sealed-1",
                            "label": "sealed-1",
                            "path": Path(archive["path"]).name,
                            "mode": "sealed",
                            "sha256": archive["sha256"],
                            "store_identity": archive["store_identity"],
                        }
                    ],
                    "experiments": [],
                    "projected_attempts": [],
                }
            ),
            encoding="utf-8",
        )
        workspace = load_observability_workspace(manifest)
        before = Path(archive["path"]).stat()
        projection = _project(
            ExecutionRef(
                store_id="sealed-1",
                turn_keys=(turn_key,),
                experiment_id="exp-sealed",
                task_id="task-1",
                attempt=1,
            ),
            WorkspaceExecutionReader(workspace),
        )
        assert [step.command_name for step in projection.steps] == ["add_todo"]
        assert projection.steps[0].parameters == {"title": "sealed"}
        after = Path(archive["path"]).stat()
        # Comparison is a read: the sealed bytes are untouched.
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)

    def test_the_workspace_reader_still_refuses_an_unknown_store(self, tmp_path: Path):
        source = tmp_path / "live2.sqlite3"
        live = obs.ObservabilityStore(str(source))
        _write(live, _turn_row("turn-x", record=_record("turn-x", refs=[])), [])
        archive = obs.ObservabilityStore(str(source), migrate=False).archive_to(
            str(tmp_path / "sealed2.sqlite3")
        )
        manifest = tmp_path / "workspace2.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": WORKSPACE_SCHEMA,
                    "workspace_id": "ws-2",
                    "label": "sealed",
                    "stores": [
                        {
                            "store_id": "sealed-2",
                            "label": "sealed-2",
                            "path": Path(archive["path"]).name,
                            "mode": "sealed",
                            "sha256": archive["sha256"],
                            "store_identity": archive["store_identity"],
                        }
                    ],
                    "experiments": [],
                    "projected_attempts": [],
                }
            ),
            encoding="utf-8",
        )
        workspace = load_observability_workspace(manifest)
        from fastworkflow.observability.workspace import UnknownWorkspaceStore

        with pytest.raises(UnknownWorkspaceStore):
            _project(
                ExecutionRef(store_id="not-registered", turn_keys=("turn-x",)),
                WorkspaceExecutionReader(workspace),
            )


# ----------------------------------------------------------------------
# Pair-review progress
# ----------------------------------------------------------------------


@pytest.fixture
def review(store: obs.ObservabilityStore) -> PairReviewStore:
    return PairReviewStore.for_evidence(store)


def _ref(turn_key: str, **kwargs) -> ExecutionRef:
    return ExecutionRef(store_id="store-1", turn_keys=(turn_key,), **kwargs)


class TestPairReview:
    def test_the_sidecar_sits_beside_the_evidence_and_is_bound_to_it(
        self, store, review, tmp_path: Path
    ):
        expected = pair_review_db_path_for(store.db_path)
        assert review.control_db_path == expected
        assert Path(expected).exists()
        assert review.evidence_store_identity() == store.store_identity()
        other = obs.ObservabilityStore(str(tmp_path / "other.sqlite3"))
        with pytest.raises(PairReviewIdentityMismatch):
            PairReviewStore(expected, evidence_store_identity=other.store_identity())

    def test_a_pair_reviewed_with_no_comment_is_reviewed(self, review):
        best, other = _ref("turn-best"), _ref("turn-other")
        state = review.state(review_pair_key(best, other), "dhar")
        assert state["state"] == STATE_NOT_REVIEWED
        assert state["recorded"] is False
        review.mark_reviewed(best, other, reviewer="dhar", reviewer_kind="human")
        # Zero comments does not imply unreviewed.
        assert review.state(review_pair_key(best, other), "dhar")["state"] == (
            STATE_REVIEWED
        )

    def test_three_runs_are_walked_against_one_reference_and_resumed(self, review):
        best = _ref("turn-best")
        others = [_ref("turn-b"), _ref("turn-c")]
        keys = [review_pair_key(best, other) for other in others]
        progress = review.progress("dhar", keys)
        assert progress["reviewed"] == 0
        assert progress["next_pair_key"] == keys[0]
        review.mark_reviewed(best, others[0], reviewer="dhar", reviewer_kind="human")
        progress = review.progress("dhar", keys)
        assert progress["reviewed"] == 1
        assert progress["remaining"] == 1
        # Where "resume" goes after a reload.
        assert progress["next_pair_key"] == keys[1]
        review.mark_reviewed(best, others[1], reviewer="dhar", reviewer_kind="human")
        assert review.progress("dhar", keys)["next_pair_key"] is None

    def test_progress_is_per_reviewer(self, review):
        best, other = _ref("turn-best"), _ref("turn-b")
        review.mark_reviewed(best, other, reviewer="dhar", reviewer_kind="human")
        key = review_pair_key(best, other)
        assert review.state(key, "dhar")["state"] == STATE_REVIEWED
        assert review.state(key, "agent-1")["state"] == STATE_NOT_REVIEWED

    def test_choosing_a_new_reference_leaves_the_old_pairs_alone(self, review):
        first, second = _ref("turn-best-1"), _ref("turn-best-2")
        other = _ref("turn-b")
        review.mark_reviewed(first, other, reviewer="dhar", reviewer_kind="human")
        old_key = review_pair_key(first, other)
        new_key = review_pair_key(second, other)
        assert old_key != new_key
        # The new reference starts unreviewed; the old pair keeps its record
        # and is not relabelled as being about the new reference.
        assert review.state(new_key, "dhar")["state"] == STATE_NOT_REVIEWED
        assert review.state(old_key, "dhar")["state"] == STATE_REVIEWED
        kept = review.reviews_for_reference(first.ref_id())
        assert [row["pair_key"] for row in kept] == [old_key]
        assert review.reviews_for_reference(second.ref_id()) == []

    def test_a_pass_scoped_pair_is_a_different_pair(self, review):
        teacher = _ref("turn-distill", pass_id="teacher")
        student = _ref("turn-distill", pass_id="student")
        whole = _ref("turn-distill")
        assert review_pair_key(teacher, student) != review_pair_key(whole, whole)
        review.mark_reviewed(teacher, student, reviewer="dhar", reviewer_kind="human")
        assert review.state(review_pair_key(whole, whole), "dhar")["state"] == (
            STATE_NOT_REVIEWED
        )

    def test_unmarking_keeps_the_history_of_both_marks(self, review):
        best, other = _ref("turn-best"), _ref("turn-b")
        review.mark_reviewed(best, other, reviewer="dhar", reviewer_kind="human")
        review.clear_reviewed(best, other, reviewer="dhar", reviewer_kind="human")
        key = review_pair_key(best, other)
        current = review.state(key, "dhar")
        assert current["state"] == STATE_NOT_REVIEWED
        # "Nobody marked this" and "somebody unmarked it" are different facts.
        assert current["recorded"] is True
        events = review.history(key, reviewer="dhar")
        assert [event["state"] for event in events] == [
            STATE_NOT_REVIEWED,
            STATE_REVIEWED,
        ]
        assert events[-1]["previous_state"] is None

    def test_an_invalid_reviewer_kind_is_refused(self, review):
        best, other = _ref("turn-best"), _ref("turn-b")
        with pytest.raises(ValueError):
            review.mark_reviewed(best, other, reviewer="dhar", reviewer_kind="robot")
        with pytest.raises(ValueError):
            review.mark_reviewed(best, other, reviewer="  ", reviewer_kind="human")

    def test_a_pair_spanning_two_stores_is_recordable(self, tmp_path: Path):
        # The real product shape: the winner and the candidate were registered
        # in different evidence stores, so a sidecar bound to one of them could
        # not record that anybody reviewed the pair.
        left_store = obs.ObservabilityStore(str(tmp_path / "left.sqlite3"))
        right_store = obs.ObservabilityStore(str(tmp_path / "right.sqlite3"))
        control_root = tmp_path / "workspace"
        control_root.mkdir()
        review = open_shared_pair_review(
            str(control_root),
            sources={"store-left": left_store, "store-right": right_store},
        )
        assert review.control_db_path == shared_pair_review_db_path_for(
            str(control_root)
        )
        assert {source["source_id"] for source in review.list_sources()} == {
            "store-left",
            "store-right",
        }
        left = ExecutionRef(store_id="store-left", turn_keys=("turn-winner",))
        right = ExecutionRef(store_id="store-right", turn_keys=("turn-candidate",))
        review.mark_reviewed(left, right, reviewer="dhar", reviewer_kind="human")
        key = review_pair_key(left, right)
        assert review.state(key, "dhar")["state"] == STATE_REVIEWED
        assert review.progress("dhar", [key])["next_pair_key"] is None

    def test_a_shared_control_store_refuses_a_store_it_was_not_told_to_trust(
        self, tmp_path: Path
    ):
        known = obs.ObservabilityStore(str(tmp_path / "known.sqlite3"))
        control_root = tmp_path / "ws2"
        control_root.mkdir()
        review = open_shared_pair_review(
            str(control_root), sources={"store-known": known}
        )
        with pytest.raises(UnauthorizedEvidenceSource):
            review.mark_reviewed(
                ExecutionRef(store_id="store-known", turn_keys=("t1",)),
                ExecutionRef(store_id="store-unknown", turn_keys=("t2",)),
                reviewer="dhar",
                reviewer_kind="human",
            )
        # Authorizing it makes the same pair recordable, and re-authorizing the
        # same store under the same id is a no-op.
        later = obs.ObservabilityStore(str(tmp_path / "later.sqlite3"))
        review.authorize_source("store-unknown", later)
        review.authorize_source("store-unknown", later)
        review.mark_reviewed(
            ExecutionRef(store_id="store-known", turn_keys=("t1",)),
            ExecutionRef(store_id="store-unknown", turn_keys=("t2",)),
            reviewer="dhar",
            reviewer_kind="human",
        )
        assert len(review.list_sources()) == 2

    def test_a_source_id_names_one_store_for_the_life_of_the_progress(
        self, tmp_path: Path
    ):
        first = obs.ObservabilityStore(str(tmp_path / "first.sqlite3"))
        second = obs.ObservabilityStore(str(tmp_path / "second.sqlite3"))
        control_root = tmp_path / "ws3"
        control_root.mkdir()
        review = open_shared_pair_review(str(control_root), sources={"s": first})
        with pytest.raises(PairReviewIdentityMismatch):
            review.authorize_source("s", second)

    def test_the_two_control_shapes_are_never_confused(self, tmp_path: Path):
        evidence = obs.ObservabilityStore(str(tmp_path / "single.sqlite3"))
        single = PairReviewStore.for_evidence(evidence)
        with pytest.raises(PairReviewModeMismatch):
            open_shared_pair_review(
                str(tmp_path), control_db_path=single.control_db_path
            )
        shared_root = tmp_path / "ws4"
        shared_root.mkdir()
        shared = open_shared_pair_review(str(shared_root))
        with pytest.raises(PairReviewModeMismatch):
            PairReviewStore(shared.control_db_path)

    def test_refs_round_trip_through_the_stored_pair(self, review):
        best, other = _ref("turn-best", task_id="task-1", attempt=3), _ref("turn-b")
        review.mark_reviewed(best, other, reviewer="dhar", reviewer_kind="human")
        row = review.reviews_for_reference(best.ref_id())[0]
        restored = ExecutionRef.from_mapping(json.loads(row["left_ref_json"]))
        assert restored.ref_id() == best.ref_id()
        assert restored.task_id == "task-1"
        assert restored.attempt == 3
