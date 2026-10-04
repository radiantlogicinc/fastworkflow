"""Consistency between the repeated runs of one task (`fix-9eg.17.5`).

Integration throughout, per `.cursor/rules/testing_rules.mdc`: a real workflow
folder, a real benchmark manifest, real `ObservabilityStore` databases, attempts
written through the real `ExperimentController`, real recorded spans -- including
real `fw.planner.plan` spans, which is where the plan text this module compares
actually lives -- the real selection control, the real `ChatbotServer` over a
socket, and the real locally-installed embedding model. No Mock fixtures and
nothing paid: the similarity numbers below come from the checkpoint already in
this machine's HuggingFace cache, loaded with `local_files_only=True`.

Three groups of claims, and the split is deliberate.

**Arithmetic.** `cosine`, `population_sd` and the normalized step difference are
checked against values worked out by hand, on vectors chosen so the answer does
not depend on a model. If the embedding stack ever changes, these still pin what
the numbers MEAN.

**Honesty about absence.** The tests that matter most here are the ones where
there is nothing to compare: absent plan text, a withheld value, a bounded one,
an attempt with no recorded turns, one recorded run. Every one of them has a
tempting wrong answer -- 1.0, 0, or "consistent" -- and each is checked to
produce an explicit unknown instead.

**Independence.** Choosing a different best run must move the reference rows and
nothing else, and two experiments measured differently must not be subtracted.
Both are properties of the whole report, so both are checked over real HTTP.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from fastworkflow import state_paths, tracing
from fastworkflow.benchmark import setup
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability import best_run as best_run_module
from fastworkflow.observability import comparison, consistency
from fastworkflow.observability import feedback as fb
from fastworkflow.observability import store as obs
from fastworkflow.observability.capture_policy import CapturedValue
from fastworkflow.run_chatbot import selection_api
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests.test_chatbot_benchmarks import _request
from tests.test_execution_comparison import _execute_span, _output, _record, _write
from tests.test_execution_comparison import _turn_row as _evidence_turn_row

T0 = 1_700_000_000_000_000_000

HUMAN = {"actor": "dhar", "actor_kind": "human"}

# Two plans that are about the same job and two that are not. Nothing here
# depends on a particular cosine VALUE between different texts -- only that
# identical text scores ~1 and that a spread exists -- because pinning "0.62"
# would be pinning the checkpoint rather than the metric.
PLAN_OPEN = "1. open the roster\n2. add the missing member\n3. list the roster"
PLAN_FINISH = "4. confirm the roster is complete\n5. report back"
PLAN_OTHER = "1. delete the stale export\n2. re-run the nightly import job"
ANSWER_DONE = "The roster now lists three members."
ANSWER_FAILED = "I could not finish; the roster is unchanged."


# ----------------------------------------------------------------------
# Seeding: real turns, real spans, real planner spans
# ----------------------------------------------------------------------


def _plan_span(span_id, turn_key, *, text, start_ns, replan_trigger=None):
    """A real `fw.planner.plan` / `fw.planner.replan` span.

    The same shape `workflow_agent.build_query_with_next_steps` ends: the plan
    text goes on the span at close, and the planner's chain-of-thought does
    NOT -- which is exactly why this module can compare plans without touching
    hidden reasoning.
    """
    return tracing.Span(
        span_id=span_id,
        trace_id=turn_key,
        parent_span_id=None,
        name=(
            tracing.SPAN_PLANNER_REPLAN if replan_trigger
            else tracing.SPAN_PLANNER_PLAN
        ),
        kind=tracing.KIND_LLM,
        channel_id="channel-1",
        start_ns=start_ns,
        end_ns=start_ns + 400_000,
        status=tracing.STATUS_OK,
        attributes={
            "model": "local/test",
            "replan_trigger": replan_trigger,
            "plan": text,
        },
    )


def _seed_turn(
    store,
    turn_key,
    *,
    experiment_id,
    task_id,
    attempt,
    conversation,
    ordinal,
    commands,
    plan=None,
    replan=None,
    answer=ANSWER_DONE,
    status="completed",
    success=True,
    child_of_first=None,
    duplicate_span_for_first=False,
):
    """One recorded turn: a span per dispatch, plus the planner's own span.

    `child_of_first` names a span-less inner hop of the FIRST command, recorded
    both in the parent's `child_calls` ledger and as a ref in the turn record --
    which is the real double-report the execution ledger is built to collapse.
    `duplicate_span_for_first` adds a SECOND execute span carrying the first
    dispatch's call id, which is what a re-recorded open/close update looks
    like on disk.
    """
    refs, spans, outputs = [], [], []
    for index, command in enumerate(commands):
        call_id = f"{turn_key}-call-{index}"
        span_id = f"{turn_key}-span-{index}"
        refs.append((call_id, index, span_id))
        outputs.append(_output(call_id, command, {"n": index}))
        children = None
        if index == 0 and child_of_first:
            children = [
                {"call_id": f"{turn_key}-{child_of_first}",
                 "command_name": child_of_first,
                 "parent_call_id": call_id}
            ]
        spans.append(
            _execute_span(
                span_id,
                turn_key,
                call_id=call_id,
                command_name=command,
                start_ns=T0 + index * 1_000_000,
                child_calls=children,
            )
        )
        if index == 0 and duplicate_span_for_first:
            spans.append(
                _execute_span(
                    f"{span_id}-again",
                    turn_key,
                    call_id=call_id,
                    command_name=command,
                    start_ns=T0 + index * 1_000_000 + 10_000,
                )
            )
    if child_of_first:
        # Also in the record, so the inner hop is reported twice by two
        # independent sources and must still be one row.
        refs.append((f"{turn_key}-{child_of_first}", len(commands), None))
    if plan is not None:
        spans.append(
            _plan_span(f"{turn_key}-plan", turn_key, text=plan, start_ns=T0 - 500_000)
        )
    if replan is not None:
        spans.append(
            _plan_span(
                f"{turn_key}-replan",
                turn_key,
                text=replan,
                start_ns=T0 + 9_000_000,
                replan_trigger="tool_error",
            )
        )
    row = _evidence_turn_row(
        turn_key,
        record=_record(turn_key, refs=refs, outputs=outputs, success=success),
        answer=answer,
        status=status,
        success=success,
        experiment_id=experiment_id,
        task_id=task_id,
        attempt=attempt,
    )
    row["conversation_id"] = conversation
    row["ordinal"] = ordinal
    _write(store, row, spans)


def _run_attempt(
    store, controller, experiment_id, task_id, attempt, turns,
    *, outcome="pass", execution_status=None, finish=True,
):
    """`turns` is a list of kwargs dicts, one per turn, in order."""
    channel = f"ch-{experiment_id[-4:]}-{attempt}"
    conversation = store.mint_conversation_id(
        channel, experiment_id=experiment_id, task_id=task_id, attempt=attempt
    )
    controller.start_attempt(
        experiment_id, task_id, attempt, channel, conversation_id=conversation
    )
    for ordinal, spec in enumerate(turns, start=1):
        _seed_turn(
            store,
            f"{experiment_id[-6:]}-a{attempt}-t{ordinal}",
            experiment_id=experiment_id,
            task_id=task_id,
            attempt=attempt,
            conversation=conversation,
            ordinal=ordinal,
            **spec,
        )
    if finish:
        kwargs = {"outcome": outcome, "outcome_source": "derived"}
        if execution_status is not None:
            kwargs["execution_status"] = execution_status
        controller.finish_attempt(experiment_id, task_id, attempt, **kwargs)


@pytest.fixture
def repeats(tmp_path, monkeypatch):
    """One task, repeated, under two experiments of one contest.

    BASELINE is the arithmetic fixture. Its two readable attempts dispatched
    exactly 2 and 4 steps, so the distribution is the acceptance criterion's
    own: mean 3, population SD 1, and the pair is +/-2 and 50%. Its second
    attempt records NO planner span, which is how "absent plan text is unknown,
    not identical" gets a real case; both attempts answer with the SAME text,
    which is how "identical nonempty text is ~1" gets one. Its third attempt
    finished having recorded nothing at all.

    CANDIDATE is the repetition fixture: three multi-turn attempts, two of them
    byte-identical in plan and answer and the third genuinely different, one of
    them failed and still counted. Its first turn reports one inner hop twice
    and one dispatch's span twice, which is the count-inflation case.
    """
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.delenv("FASTWORKFLOW_WORKFLOW_ID", raising=False)
    monkeypatch.delenv("FASTWORKFLOW_CONSISTENCY_EMBEDDING_MODEL", raising=False)
    consistency.reset_shared_embedders()

    folder = tmp_path / "roster_workflow"
    folder.mkdir()
    (folder / "_commands").mkdir()

    # Before anything is written: every derived path this test can produce is
    # under the temporary root, so nothing here can touch a real workflow.
    resolved = Path(state_paths.workflow_state_dir(str(folder))).resolve()
    assert resolved.is_relative_to((tmp_path / "state").resolve()), resolved

    benchmark = setup.save_benchmark(
        folder, {"title": "Roster review", "tasks": [{"prompt": "Review the roster"}]}
    )
    benchmark_id = benchmark["benchmark_id"]

    baseline = setup.create_experiment(folder, benchmark_id, "v1", runs_per_task=3)
    baseline_id = baseline["experiment_id"]
    task_id = baseline["task_ids"][0]

    db_one = str(tmp_path / "evidence-baseline.sqlite3")
    store_one = obs.ObservabilityStore(db_one)
    controller_one = ExperimentController(
        db_one, store_one.store_identity(), external=False,
        workflow_folderpath=str(folder),
    )
    controller_one.create_experiment(
        baseline_id, baseline["description"], declared_tasks=1, declared_attempts=3,
        declarations=[(task_id, n, f"ch-{baseline_id[-4:]}-{n}") for n in (1, 2, 3)],
        workflow_name=setup.workflow_name_for(folder),
    )
    _run_attempt(
        store_one, controller_one, baseline_id, task_id, 1,
        [{"commands": ["add_item", "list_items"], "plan": PLAN_OPEN,
          "answer": ANSWER_DONE}],
    )
    _run_attempt(
        store_one, controller_one, baseline_id, task_id, 2,
        [{"commands": ["add_item", "list_items", "sort_items", "remove_item"],
          "plan": None, "answer": ANSWER_DONE}],
    )
    _run_attempt(store_one, controller_one, baseline_id, task_id, 3, [])

    candidate = setup.create_experiment(folder, benchmark_id, "v1", runs_per_task=3)
    candidate_id = candidate["experiment_id"]
    db_two = str(tmp_path / "evidence-candidate.sqlite3")
    store_two = obs.ObservabilityStore(db_two)
    controller_two = ExperimentController(
        db_two, store_two.store_identity(), external=False,
        workflow_folderpath=str(folder),
    )
    controller_two.create_experiment(
        candidate_id, candidate["description"], declared_tasks=1,
        declared_attempts=3,
        declarations=[(task_id, n, f"ch-{candidate_id[-4:]}-{n}") for n in (1, 2, 3)],
        workflow_name=setup.workflow_name_for(folder),
    )
    repeated = [
        {"commands": ["add_item", "list_items"], "plan": PLAN_OPEN,
         "answer": "working on it", "child_of_first": "validate_item",
         "duplicate_span_for_first": True},
        {"commands": ["complete_item"], "plan": PLAN_FINISH,
         "answer": ANSWER_DONE},
    ]
    _run_attempt(store_two, controller_two, candidate_id, task_id, 1,
                 [dict(spec) for spec in repeated])
    _run_attempt(store_two, controller_two, candidate_id, task_id, 2,
                 [dict(spec) for spec in repeated])
    _run_attempt(
        store_two, controller_two, candidate_id, task_id, 3,
        [
            {"commands": ["open_export", "delete_export"], "plan": PLAN_OTHER,
             "answer": "removing the export"},
            {"commands": ["import_roster", "verify_roster", "report"],
             "plan": PLAN_OTHER, "replan": PLAN_OTHER,
             "answer": ANSWER_FAILED, "status": "failed", "success": False},
        ],
        outcome="fail", execution_status="failed",
    )

    return {
        "folder": str(folder),
        "task_id": task_id,
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "store_one": store_one,
        "store_two": store_two,
        "db_one": db_one,
        "db_two": db_two,
        "state_root": tmp_path / "state",
    }


@pytest.fixture
def server(repeats):
    srv = run_chatbot_server.ChatbotServer(
        db_path="",
        workflow_path=repeats["folder"],
        port=0,
        spawn_options={"no_server": True},
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _consistency(repeats, experiment=None, **params):
    query = {key: [str(value)] for key, value in params.items() if value is not None}
    return selection_api.handle_get(
        repeats["folder"],
        f"/api/experiments/{experiment or repeats['baseline_id']}"
        f"/tasks/{repeats['task_id']}/consistency",
        query,
    )


def _get_live_comparison(repeats, *, left, right, experiment=None):
    """The shared comparison route over the LIVE store, for parity checks."""
    status, payload = selection_api.handle_get(
        repeats["folder"],
        f"/api/experiments/{experiment or repeats['candidate_id']}"
        f"/tasks/{repeats['task_id']}/comparison",
        {
            "left_attempt": [str(left)],
            "right_attempt": [str(right)],
            "view": ["steps"],
        },
    )
    assert status == 200, payload
    return payload


def _by_attempt(report, attempt):
    return next(row for row in report["runs"] if row["attempt"] == attempt)


def _pair(report, left, right, key="pairs"):
    return next(
        row for row in report[key]
        if row["left_attempt"] == left and row["right_attempt"] == right
    )


# ----------------------------------------------------------------------
# The arithmetic, pinned without a model
# ----------------------------------------------------------------------


class TestArithmetic:
    def test_cosine_of_known_vectors_needs_no_embedding_model(self):
        """Orthogonal, identical and 45 degrees apart, worked out by hand."""
        assert consistency.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
        assert consistency.cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
        assert consistency.cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0
        assert consistency.cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(
            0.7071067811865476
        )
        # Magnitude must not matter; only direction.
        assert consistency.cosine([3.0, 4.0], [30.0, 40.0]) == pytest.approx(1.0)

    def test_cosine_refuses_what_it_cannot_answer(self):
        with pytest.raises(consistency.ConsistencyError):
            consistency.cosine([1.0, 0.0], [1.0, 0.0, 0.0])
        with pytest.raises(consistency.ConsistencyError):
            consistency.cosine([0.0, 0.0], [1.0, 0.0])
        with pytest.raises(consistency.ConsistencyError):
            consistency.cosine([], [])

    def test_population_sd_of_two_and_four_is_one(self):
        assert consistency.population_sd([2.0, 4.0]) == 1.0
        assert consistency.population_sd([3.0]) == 0.0
        assert consistency.population_sd([]) is None

    def test_the_normalized_step_difference_is_the_documented_formula(self):
        assert consistency.normalized_step_difference(2, 4) == 50.0
        assert consistency.normalized_step_difference(4, 2) == 50.0
        # Both known zero is 0%, not a division by zero and not "unknown".
        assert consistency.normalized_step_difference(0, 0) == 0.0
        assert consistency.normalized_step_difference(0, 5) == 100.0

    def test_the_plan_span_names_still_match_the_emitter(self):
        """The restated constants are checked against the ones spans carry."""
        assert consistency.SPAN_PLANNER_PLAN == tracing.SPAN_PLANNER_PLAN
        assert consistency.SPAN_PLANNER_REPLAN == tracing.SPAN_PLANNER_REPLAN

    def test_the_canonical_plan_text_keeps_order_and_turn_boundaries(self):
        first = consistency.PlanSegment(0, "t1", tracing.SPAN_PLANNER_PLAN, None, "A")
        second = consistency.PlanSegment(1, "t2", tracing.SPAN_PLANNER_PLAN, None, "B")
        forwards = consistency.canonical_plan_text([first, second])
        backwards = consistency.canonical_plan_text([second, first])

        assert forwards != backwards, "order is content, not presentation"
        assert "[turn 1 plan]" in forwards and "[turn 2 plan]" in forwards
        # Nothing volatile rides along: two runs of the same plan differ in
        # turn keys and span ids, and neither appears in what is embedded.
        assert "t1" not in forwards.replace("[turn 1 plan]", "")


# ----------------------------------------------------------------------
# The local embedding facility
# ----------------------------------------------------------------------


class TestLocalEmbedder:
    def test_the_installed_model_loads_offline_and_describes_itself(self):
        embedder, why = consistency.load_embedder()
        if embedder is None:
            pytest.skip(f"no local embedding model on this machine: {why}")

        described = embedder.describe()
        assert described["available"] is True
        assert described["local_only"] is True
        assert described["dimension"] > 0
        assert described["pooling"] == "mean_over_attention_mask"
        assert described["fingerprint"]

    def test_identical_nonempty_text_scores_about_one(self):
        embedder, why = consistency.load_embedder()
        if embedder is None:
            pytest.skip(f"no local embedding model on this machine: {why}")

        left = embedder.embed(PLAN_OPEN)
        right = embedder.embed(PLAN_OPEN)
        unrelated = embedder.embed(PLAN_OTHER)

        assert consistency.cosine(left.vector, right.vector) == pytest.approx(
            1.0, abs=1e-6
        )
        assert consistency.cosine(left.vector, unrelated.vector) < 0.95
        assert left.truncated is False

    def test_a_long_text_is_embedded_whole_rather_than_by_its_opening(self):
        """The difference has to be AFTER the first window, or this proves
        nothing. Two plans that share their first 400 lines and then diverge
        completely are the exact case a prefix-only embedding calls identical.
        """
        embedder, why = consistency.load_embedder()
        if embedder is None:
            pytest.skip(f"no local embedding model on this machine: {why}")

        shared = "\n".join(f"{n}. inspect roster record {n}" for n in range(40))
        tail_one = "\n".join(
            f"{n}. confirm the roster is complete and report back"
            for n in range(40)
        )
        tail_two = "\n".join(
            f"{n}. delete the stale export and re-run the nightly import"
            for n in range(40)
        )
        one = embedder.embed(shared + "\n" + tail_one)
        two = embedder.embed(shared + "\n" + tail_two)

        assert one.token_count > embedder.window_tokens
        assert one.windows > 1, "the text needed more than one encoder window"
        assert one.truncated is False, (
            "a text within the work bound is embedded whole, so nothing about "
            "it is partial"
        )
        whole = consistency.cosine(one.vector, two.vector)

        # What the same two texts score when only their first window is read.
        # This is the number this rule exists to not report.
        prefix_only = consistency.LocalTextEmbedder(
            embedder.model_id, window_tokens=embedder.window_tokens, max_windows=1
        )
        opening = consistency.cosine(
            prefix_only.embed(shared + "\n" + tail_one).vector,
            prefix_only.embed(shared + "\n" + tail_two).vector,
        )

        assert whole < 0.8, f"two plans that diverge entirely scored {whole}"
        assert opening > whole + 0.2, (
            "reading only the opening makes two plans that end up doing "
            f"opposite things look alike: {opening} against {whole}"
        )

    def test_one_window_of_text_is_untouched_by_the_long_text_rule(self):
        """A short text must get what the plain single-pass recipe gives it."""
        embedder, why = consistency.load_embedder()
        if embedder is None:
            pytest.skip(f"no local embedding model on this machine: {why}")

        embedded = embedder.embed(PLAN_OPEN)

        assert embedded.windows == 1
        assert embedded.token_count < embedder.window_tokens
        assert embedded.truncated is False
        # L2-normalized, which is what makes a dot product a cosine.
        assert sum(value * value for value in embedded.vector) == pytest.approx(
            1.0, abs=1e-5
        )

    def test_text_past_the_work_bound_is_labelled_truncated(self):
        """The only cut left, and it says so rather than reporting full text."""
        embedder, why = consistency.load_embedder()
        if embedder is None:
            pytest.skip(f"no local embedding model on this machine: {why}")
        # A deliberately tiny bound rather than a 16k-token fixture: the
        # behaviour under test is the bound, not the size of the constant.
        small = consistency.LocalTextEmbedder(
            embedder.model_id, window_tokens=16, max_windows=2
        )
        embedded = small.embed(
            "\n".join(f"{n}. inspect record {n}" for n in range(60))
        )

        assert embedded.windows == 2
        assert embedded.token_count > 2 * 16
        assert embedded.truncated is True, (
            "a similarity over a cut text must never be reported as one over "
            "the whole text"
        )

    def test_the_long_text_rule_is_part_of_the_model_identity(self):
        """Two cosines from different aggregation rules must not be subtracted."""
        embedder, why = consistency.load_embedder()
        if embedder is None:
            pytest.skip(f"no local embedding model on this machine: {why}")

        other = consistency.LocalTextEmbedder(embedder.model_id, window_tokens=128)

        assert embedder.fingerprint() != other.fingerprint()
        assert embedder.describe()["aggregation"] == consistency.AGGREGATION_METHOD
        assert embedder.describe()["long_text_rule"] == consistency.LONG_TEXT_RULE

    def test_a_model_whose_revision_cannot_be_pinned_is_refused(
        self, monkeypatch
    ):
        """Loadable is not enough: the weights behind the name must be named.

        Without a resolved revision, a cached vector from one checkpoint could
        be served to another and two experiments' cosines could be subtracted
        across a silent model swap.
        """
        embedder, why = consistency.load_embedder()
        if embedder is None:
            pytest.skip(f"no local embedding model on this machine: {why}")

        monkeypatch.setattr(consistency, "_resolved_revision", lambda _id: None)
        refused, why = consistency.load_embedder()

        assert refused is None
        assert "revision" in why["reason"]
        assert why["remedy"]

    def test_a_model_this_machine_lacks_is_actionable_not_a_download(self, tmp_path):
        embedder, why = consistency.load_embedder("acme/not-a-real-checkpoint")

        assert embedder is None
        assert "not installed" in why["reason"]
        assert why["remedy"], "a refusal without a remedy is a dead end"
        assert why["model_id"] == "acme/not-a-real-checkpoint"

    def test_the_configured_model_id_is_overridable_and_not_silently_replaced(
        self, monkeypatch
    ):
        monkeypatch.setenv(
            "FASTWORKFLOW_CONSISTENCY_EMBEDDING_MODEL", "acme/not-a-real-checkpoint"
        )
        consistency.reset_shared_embedders()
        embedder, why = consistency.shared_embedder()

        assert embedder is None
        assert why["model_id"] == "acme/not-a-real-checkpoint", (
            "an override that is missing must not fall back to the default "
            "model under the operator's chosen name"
        )
        consistency.reset_shared_embedders()


class TestVectorCache:
    def test_a_vector_round_trips_by_content_and_model_identity(self, tmp_path):
        cache = consistency.VectorCache(tmp_path / "vectors")
        embedded = consistency.EmbeddedText(
            vector=(0.5, 0.5), token_count=4, truncated=False
        )

        cache.put("fp-1", "some plan", embedded)

        assert cache.get("fp-1", "some plan") == embedded
        # A different model, or a different text, is a miss rather than a
        # stale hit: the key IS the identity.
        assert cache.get("fp-2", "some plan") is None
        assert cache.get("fp-1", "some other plan") is None

    def test_the_cache_lives_where_it_was_told_and_nowhere_else(self, tmp_path):
        root = tmp_path / "vectors"
        cache = consistency.VectorCache(root)
        cache.put(
            "fp-1", "text",
            consistency.EmbeddedText(vector=(1.0,), token_count=1, truncated=False),
        )

        written = list(root.rglob("*.json"))
        assert written, "the cache wrote nothing"
        for path in written:
            assert path.resolve().is_relative_to(root.resolve())


# ----------------------------------------------------------------------
# One run's evidence, from real recorded turns
# ----------------------------------------------------------------------


class TestRunEvidence:
    def test_a_wrapper_and_a_re_recorded_span_do_not_inflate_the_count(self, repeats):
        """Candidate attempt 1's first turn reports three dispatches twice over.

        `add_item` has a span recorded twice, and `validate_item` appears both
        in the parent's `child_calls` and as a ref in the record. Six reports,
        three dispatches.
        """
        report = _consistency(repeats, repeats["candidate_id"])[1]
        run = _by_attempt(report, 1)

        assert run["step_count"] == 4, run["step_detail"]
        assert run["step_count_state"] == consistency.COUNT_KNOWN
        assert run["step_detail"]["child"] == 1
        assert run["step_detail"]["root"] == 3
        assert run["step_detail"]["turns"] == 2

    def test_an_attempt_with_no_recorded_turns_has_an_unknown_count_not_zero(
        self, repeats
    ):
        report = _consistency(repeats)[1]
        empty = _by_attempt(report, 3)

        assert empty["step_count"] is None
        assert empty["step_count_state"] == consistency.COUNT_UNKNOWN
        assert empty["plan"]["state"] == consistency.TEXT_ABSENT
        assert empty["answer"]["state"] == consistency.TEXT_ABSENT
        assert 3 in report["summary"]["step_counts"]["attempts_unknown"]

    def test_a_multi_turn_plan_keeps_every_emission_and_its_boundary(self, repeats):
        report = _consistency(repeats, repeats["candidate_id"])[1]
        failed = _by_attempt(report, 3)

        # Two turns, and the second replanned: three emissions, in order.
        kinds = [segment["span_name"] for segment in failed["plan"]["segments"]]
        assert kinds == [
            tracing.SPAN_PLANNER_PLAN,
            tracing.SPAN_PLANNER_PLAN,
            tracing.SPAN_PLANNER_REPLAN,
        ]
        boundaries = [segment["boundary"] for segment in failed["plan"]["segments"]]
        assert boundaries == [
            "[turn 1 plan]", "[turn 2 plan]", "[turn 2 replan after tool_error]"
        ]

    def test_the_final_answer_is_the_last_turn_and_not_a_borrowed_earlier_one(
        self, repeats
    ):
        report = _consistency(repeats, repeats["candidate_id"])[1]
        run = _by_attempt(report, 1)

        # Turn 1 answered "working on it" and turn 2 answered ANSWER_DONE.
        assert run["answer"]["state"] == consistency.TEXT_PRESENT
        assert run["answer"]["chars"] == len(ANSWER_DONE)

    def test_a_failed_attempt_is_measured_rather_than_dropped(self, repeats):
        """No success-only bias: the failed run is in the population."""
        report = _consistency(repeats, repeats["candidate_id"])[1]
        failed = _by_attempt(report, 3)

        assert failed["outcome"] == "fail"
        assert failed["step_count_state"] == consistency.COUNT_KNOWN
        assert any(
            pair["left_attempt"] == 1 and pair["right_attempt"] == 3
            for pair in report["pairs"]
        )

    def test_a_pruned_turn_makes_the_count_partial_rather_than_wrong(self, repeats):
        """A reference naming a turn the store no longer holds.

        This is what retention leaves behind. The run still projects, its count
        is over what survived, and it is labelled `partial` so it stays out of
        the distribution instead of dragging the mean down as if the agent had
        taken fewer steps.
        """
        summary = best_run_module.task_run_summary(
            _open_control(repeats), repeats["baseline_id"], repeats["task_id"]
        )
        row = dict(next(r for r in summary["attempts"] if r["attempt"] == 1))
        ref = dict(row["execution_ref"])
        ref["turn_keys"] = list(ref["turn_keys"]) + ["turn-that-was-pruned"]
        row["execution_ref"] = ref

        reader = selection_api._AuthorizedReader(_open_control(repeats))
        run = consistency.collect_run_evidence(row, reader)

        assert run.step_count == 2
        assert run.step_count_state == consistency.COUNT_PARTIAL
        assert run.evidence_complete is False
        assert any("turn-that-was-pruned" in item for item in run.unavailable)

    def test_a_withheld_plan_is_withheld_and_a_bounded_one_is_labelled(
        self, repeats, tmp_path
    ):
        """The capture policy's own envelopes, not a stand-in for them.

        A removed plan must not read as "this run planned nothing", and a
        bounded one must not be presented as the whole plan.
        """
        store = obs.ObservabilityStore(str(tmp_path / "envelopes.sqlite3"))
        removed = CapturedValue(
            classification="user-text", disposition="omit",
            original_bytes=120, digest="sha256:deadbeef",
            reason="omitted by evidence profile default",
        ).to_envelope()
        bounded = CapturedValue(
            classification="controlled-vocabulary", disposition="bounded-text",
            original_bytes=900, digest="sha256:cafe",
            reason="bounded by evidence profile default", prefix=PLAN_OPEN,
        ).to_envelope()

        for turn_key, value in (("withheld-turn", removed), ("bounded-turn", bounded)):
            row = _evidence_turn_row(
                turn_key,
                record=_record(turn_key, refs=[], outputs=[]),
                answer=ANSWER_DONE,
            )
            span = _plan_span(f"{turn_key}-plan", turn_key, text="", start_ns=T0)
            span.attributes["plan"] = value
            _write(store, row, [span])

        reader = comparison.StoreExecutionReader("envelopes", store)

        withheld = consistency.collect_run_evidence(
            _row_for("withheld-turn"), reader
        )
        assert withheld.plan.state == consistency.TEXT_WITHHELD
        assert withheld.plan.withheld_segments == 1
        assert "capture policy" in withheld.plan.reason

        partial = consistency.collect_run_evidence(_row_for("bounded-turn"), reader)
        assert partial.plan.state == consistency.TEXT_PRESENT
        assert partial.plan.truncated_source is True

    def test_a_missing_last_turn_makes_the_final_answer_unknown(self, repeats):
        """The run's ENDING is what is missing, so nothing stands in for it.

        The turn before the last one is the middle of the run. Substituting it
        would compare one run's conclusion against another run's penultimate
        step and publish the result as final-answer similarity.
        """
        summary = best_run_module.task_run_summary(
            _open_control(repeats), repeats["candidate_id"], repeats["task_id"]
        )
        row = dict(next(r for r in summary["attempts"] if r["attempt"] == 1))
        ref = dict(row["execution_ref"])
        surviving = list(ref["turn_keys"])
        ref["turn_keys"] = surviving + ["last-turn-that-was-pruned"]
        row["execution_ref"] = ref

        reader = selection_api._AuthorizedReader(_open_control(repeats))
        run = consistency.collect_run_evidence(row, reader)

        assert run.answer.state == consistency.TEXT_UNREADABLE
        assert "last-turn-that-was-pruned" in run.answer.reason
        assert ANSWER_DONE not in (run.answer.text or ""), (
            "an earlier turn's answer was borrowed as the final one"
        )

    def test_a_surviving_plan_from_an_incomplete_run_is_not_full_coverage(
        self, repeats
    ):
        """Some of the plan sequence is unread, so what is left is a fragment.

        The text that survived is worth comparing -- it is just not the run's
        plan, and a cosine over it is not a similarity of the run's planning.
        """
        summary = best_run_module.task_run_summary(
            _open_control(repeats), repeats["candidate_id"], repeats["task_id"]
        )
        row = dict(next(r for r in summary["attempts"] if r["attempt"] == 1))
        ref = dict(row["execution_ref"])
        ref["turn_keys"] = list(ref["turn_keys"]) + ["turn-that-was-pruned"]
        row["execution_ref"] = ref

        reader = selection_api._AuthorizedReader(_open_control(repeats))
        incomplete = consistency.collect_run_evidence(row, reader)
        whole = consistency.collect_run_evidence(
            next(r for r in summary["attempts"] if r["attempt"] == 2), reader
        )

        assert incomplete.plan.state == consistency.TEXT_PRESENT
        assert incomplete.plan.partial_source is True
        assert "could not be read" in incomplete.plan.incomplete_reason
        assert whole.plan.partial_source is False

        embedder, why = consistency.load_embedder()
        if embedder is None:
            pytest.skip(f"no local embedding model on this machine: {why}")
        session = consistency._EmbeddingSession(embedder, None)
        metric = consistency._similarity(
            session, incomplete.plan, whole.plan, None
        )

        assert metric["state"] == consistency.METRIC_COMPUTED
        assert metric["coverage"] == consistency.COVERAGE_PARTIAL
        assert metric["source_incomplete"] is True
        assert "not a similarity of the full recorded text" in (
            metric["coverage_note"]
        )

    def test_a_withheld_sibling_emission_makes_the_surviving_plan_partial(
        self, tmp_path
    ):
        """One plan removed by the capture policy, one kept, in one turn."""
        store = obs.ObservabilityStore(str(tmp_path / "mixed.sqlite3"))
        removed = CapturedValue(
            classification="user-text", disposition="omit",
            original_bytes=120, digest="sha256:deadbeef",
            reason="omitted by evidence profile default",
        ).to_envelope()
        row = _evidence_turn_row(
            "mixed-turn",
            record=_record("mixed-turn", refs=[], outputs=[]),
            answer=ANSWER_DONE,
        )
        kept = _plan_span("mixed-a", "mixed-turn", text=PLAN_OPEN, start_ns=T0)
        gone = _plan_span("mixed-b", "mixed-turn", text="", start_ns=T0 + 1)
        gone.attributes["plan"] = removed
        _write(store, row, [kept, gone])

        reader = comparison.StoreExecutionReader("mixed", store)
        run = consistency.collect_run_evidence(
            _row_for("mixed-turn", "mixed"), reader
        )

        assert run.plan.state == consistency.TEXT_PRESENT
        assert run.plan.withheld_segments == 1
        assert run.plan.partial_source is True, (
            "half a plan sequence compared as if it were the whole one"
        )


class TestRecordedOrder:
    """Which turn ended a run is read off the evidence, or it is unknown.

    A reference's turn vector is an identity and not a chronology -- an
    archived one names its turns newest-first (`fix-6v3n`) -- so the order the
    content is read in comes from what the turns themselves recorded. The
    authority is narrow on purpose: `ordinal` is dense from 1 WITHIN a
    conversation, and conversation ids are minted per channel, so ordinals from
    two conversations say nothing about each other and an interleaved or
    resumed run has to be ordered by its timestamps or not at all.
    """

    def _seed(self, tmp_path, turns):
        """`turns` is (turn_key, channel, conversation, ordinal, started_at,
        answer)."""
        store = obs.ObservabilityStore(str(tmp_path / "order.sqlite3"))
        for key, channel, conversation, ordinal, started, answer in turns:
            row = _evidence_turn_row(
                key,
                record=_record(key, refs=[], outputs=[]),
                answer=answer,
                started_at=started,
                completed_at=started,
            )
            row["channel_id"] = channel
            row["conversation_id"] = conversation
            row["ordinal"] = ordinal
            _write(
                store,
                row,
                [_plan_span(f"{key}-p", key, text=PLAN_OPEN, start_ns=T0)],
            )
        return store

    def _project(self, store, turn_keys):
        return comparison.project_execution(
            comparison.ExecutionRef.from_mapping(
                {"store_id": "order", "turn_keys": list(turn_keys),
                 "label": "attempt 1"}
            ),
            comparison.StoreExecutionReader("order", store),
        )

    def test_one_conversation_reads_in_its_recorded_ordinals(self, tmp_path):
        """Named backwards, read forwards: the ordinals are the record."""
        store = self._seed(tmp_path, [
            ("order-t1", "ch-a", 1, 1, "2026-09-19T00:00:00+00:00", "first"),
            ("order-t2", "ch-a", 1, 2, "2026-09-19T00:00:00+00:00", "last"),
        ])

        projection = self._project(store, ["order-t2", "order-t1"])

        assert [turn.turn_key for turn in projection.turns] == [
            "order-t1", "order-t2"
        ]
        assert projection.turn_order == comparison.TURN_ORDER_CHRONOLOGICAL
        assert projection.answers()[-1]["answer"] == "last"

    def test_ordinals_of_two_conversations_do_not_order_each_other(
        self, tmp_path
    ):
        """A resumed run: ordinal 5 of one conversation ran BEFORE ordinal 1
        of the next, and only the timestamps say so."""
        store = self._seed(tmp_path, [
            ("order-early", "ch-a", 7, 5, "2026-09-19T09:00:00+00:00", "first"),
            ("order-late", "ch-b", 8, 1, "2026-09-19T10:00:00+00:00", "last"),
        ])

        projection = self._project(store, ["order-late", "order-early"])

        assert [turn.turn_key for turn in projection.turns] == [
            "order-early", "order-late"
        ], "restarted ordinals were treated as one sequence"
        assert projection.turn_order == comparison.TURN_ORDER_CHRONOLOGICAL
        assert projection.answers()[-1]["answer"] == "last"

    def test_recorded_times_are_compared_as_instants_not_as_text(
        self, tmp_path
    ):
        """02:00+02:00 is BEFORE 01:00+00:00, and sorts after it as a string."""
        store = self._seed(tmp_path, [
            ("order-utc", "ch-a", 7, 1, "2026-09-19T02:00:00+02:00", "first"),
            ("order-cet", "ch-b", 8, 1, "2026-09-19T01:00:00+00:00", "last"),
        ])

        projection = self._project(store, ["order-cet", "order-utc"])

        assert [turn.turn_key for turn in projection.turns] == [
            "order-utc", "order-cet"
        ]
        assert projection.answers()[-1]["answer"] == "last"

    def test_an_unresolvable_order_is_said_rather_than_guessed(self, tmp_path):
        """Two conversations, colliding ordinals, one recorded instant.

        There is no fact here about which turn ended the run, so the
        projection keeps the reference's order and says the chronology is not
        recoverable -- and the final-answer metric declines to name an ending
        rather than quoting whichever turn the vector happened to end with.
        """
        store = self._seed(tmp_path, [
            ("order-x", "ch-a", 7, 1, "2026-09-19T09:00:00+00:00", "one"),
            ("order-y", "ch-b", 8, 1, "2026-09-19T09:00:00+00:00", "two"),
        ])

        projection = self._project(store, ["order-y", "order-x"])

        assert projection.turn_order == comparison.TURN_ORDER_REFERENCE
        assert "more than one conversation" in projection.turn_order_reason
        assert [turn.turn_key for turn in projection.turns] == [
            "order-y", "order-x"
        ], "the reference's order is kept, not resorted on a guess"

        run = consistency.collect_run_evidence(
            {
                "attempt": 1,
                "comparable": True,
                "execution_ref": {
                    "store_id": "order",
                    "turn_keys": ["order-y", "order-x"],
                    "label": "attempt 1",
                },
            },
            comparison.StoreExecutionReader("order", store),
        )

        assert run.answer.state == consistency.TEXT_UNREADABLE
        assert "not derivable" in run.answer.reason
        assert run.turn_order == comparison.TURN_ORDER_REFERENCE
        # The plan text survived whole; what is unknown is the order of its
        # parts, which is content, so it is partial coverage rather than a
        # refusal.
        assert run.plan.state == consistency.TEXT_PRESENT
        assert run.plan.partial_source is True
        assert "recorded order" in run.plan.incomplete_reason

    def test_a_single_turn_run_has_nothing_to_order(self, tmp_path):
        """One turn is its own ending, whatever it recorded about itself."""
        store = self._seed(tmp_path, [
            ("order-only", "ch-a", None, None, None, "done"),
        ])

        projection = self._project(store, ["order-only"])

        assert projection.turn_order == comparison.TURN_ORDER_CHRONOLOGICAL
        assert projection.answers()[-1]["answer"] == "done"


def _open_control(repeats):
    from fastworkflow.benchmark import setup as benchmark_setup

    return benchmark_setup.open_workflow_control(repeats["folder"], create=False)


def _evidence_digest(path):
    """A hash over every recorded row in a live evidence store.

    Opened read-only so the probe itself cannot write or checkpoint.
    """
    digest = hashlib.sha256()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            digest.update(table.encode())
            rows = sorted(repr(row) for row in connection.execute(
                f"SELECT * FROM {table}"  # noqa: S608 - names come from sqlite_master
            ))
            for row in rows:
                digest.update(row.encode())
    finally:
        connection.close()
    return digest.hexdigest()


def _choose_best(repeats, experiment_id, attempt, reason=None):
    """Pin an attempt as the best run, the way a screen would.

    `expected_selection_id` is read first because the real API refuses a
    decision made against a selection somebody has already replaced; these
    tests replace one deliberately, so they have to carry what they last read.
    """
    control = _open_control(repeats)
    summary = best_run_module.task_run_summary(
        control, experiment_id, repeats["task_id"]
    )
    return best_run_module.select_best_run(
        control, experiment_id, repeats["task_id"], attempt,
        expected_selection_id=summary["expected_selection_id"],
        provenance="human", reason=reason, **HUMAN,
    )


def _row_for(turn_key, store_id="envelopes"):
    """An attempt row naming one seeded turn, in the shape the routes emit."""
    return {
        "attempt": 1,
        "comparable": True,
        "execution_ref": {
            "store_id": store_id,
            "turn_keys": [turn_key],
            "label": "attempt 1",
        },
    }


# ----------------------------------------------------------------------
# The report over real recorded repeats
# ----------------------------------------------------------------------


class TestTaskConsistency:
    def test_the_step_distribution_is_the_documented_arithmetic(self, repeats):
        """Counts 2 and 4 -> mean 3, min 2, max 4, population SD 1."""
        status, report = _consistency(repeats)

        assert status == 200
        steps = report["summary"]["step_counts"]
        assert steps["rule"] == consistency.STEP_COUNT_RULE
        assert steps["min"] == 2 and steps["max"] == 4
        assert steps["mean"] == pytest.approx(3.0)
        assert steps["population_sd"] == pytest.approx(1.0)
        assert steps["runs_with_known_count"] == 2
        assert "sqrt" in steps["sd_formula"]

    def test_a_pair_carries_the_signed_delta_and_the_normalized_difference(
        self, repeats
    ):
        report = _consistency(repeats)[1]
        pair = _pair(report, 1, 2)

        assert pair["steps"]["left"] == 2 and pair["steps"]["right"] == 4
        # Signed as right minus left, the direction the cross-experiment
        # deltas use: counts 2 and 4 are +2, and 50% of the larger.
        assert pair["steps"]["delta"] == 2
        assert pair["steps"]["normalized_difference_pct"] == pytest.approx(50.0)
        assert "100 * abs(a - b) / max(a, b)" in pair["steps"]["formula"]

    def test_the_delta_flips_with_the_sides_and_the_percentage_does_not(
        self, repeats
    ):
        """Two runs, both orders. The direction is information; 50% is not."""
        report = _consistency(repeats)[1]
        _choose_best(repeats, repeats["baseline_id"], 2)
        reversed_row = _pair(
            _consistency(repeats)[1], 2, 1, key="reference_rows"
        )
        forwards = _pair(report, 1, 2)

        assert forwards["steps"]["delta"] == 2
        assert reversed_row["steps"]["delta"] == -2
        assert forwards["steps"]["normalized_difference_pct"] == pytest.approx(
            reversed_row["steps"]["normalized_difference_pct"]
        )

    def test_identical_recorded_answers_score_about_one(self, repeats):
        """Both baseline attempts answered with the same text."""
        report = _consistency(repeats)[1]
        if not report["metric_identity"]["embedding"]["available"]:
            pytest.skip("no local embedding model on this machine")
        pair = _pair(report, 1, 2)

        assert pair["final_answer"]["state"] == consistency.METRIC_COMPUTED
        assert pair["final_answer"]["cosine"] == pytest.approx(1.0, abs=1e-6)
        assert pair["final_answer"]["coverage"] == consistency.COVERAGE_FULL

    def test_an_absent_plan_makes_the_pair_unknown_rather_than_identical(
        self, repeats
    ):
        """Baseline attempt 2 recorded no planner span at all.

        Two runs with no plan text are not two equal empty plans, and one run
        with a plan against one without is not a similarity of 0 either -- both
        answers would be inventions. It is unknown, and it says which side.
        """
        report = _consistency(repeats)[1]
        pair = _pair(report, 1, 2)

        assert pair["planning"]["state"] == consistency.METRIC_UNKNOWN
        assert pair["planning"]["cosine"] is None
        assert pair["planning"]["right_state"] == consistency.TEXT_ABSENT
        assert "right: absent" in pair["planning"]["reason"]
        planning = report["summary"]["planning_similarity"]
        assert planning["pairs_computed"] == 0
        assert planning["mean"] is None, "no data is not a score of zero"

    def test_repeated_identical_runs_have_a_spread_the_formula_explains(
        self, repeats
    ):
        report = _consistency(repeats, repeats["candidate_id"])[1]
        if not report["metric_identity"]["embedding"]["available"]:
            pytest.skip("no local embedding model on this machine")
        planning = report["summary"]["planning_similarity"]

        assert planning["pairs_considered"] == 3
        assert planning["pairs_computed"] == 3
        assert planning["max"] == pytest.approx(1.0, abs=1e-6), (
            "attempts 1 and 2 recorded the same plan sequence"
        )
        assert planning["min"] < planning["max"], "attempt 3 planned something else"
        assert planning["spread"] == pytest.approx(
            planning["max"] - planning["min"]
        )
        assert planning["spread_formula"] == consistency.SPREAD_FORMULA

    def test_one_recorded_run_is_insufficient_and_never_perfect(self, repeats):
        """A task with a single eligible attempt.

        The tempting answer is 1.0 everywhere. The right one is that there is
        nothing to be consistent with.
        """
        runs = [
            consistency.RunEvidence(
                attempt=1, experiment_id="exp", task_id="t",
                ref=comparison.ExecutionRef(store_id="s", turn_keys=("k",)),
                plan=consistency.ProjectedText(
                    "plan", consistency.TEXT_PRESENT, text=PLAN_OPEN
                ),
                answer=consistency.ProjectedText(
                    "final_answer", consistency.TEXT_PRESENT, text=ANSWER_DONE
                ),
                step_count=3, step_count_state=consistency.COUNT_KNOWN,
            )
        ]
        report = consistency.task_consistency(
            experiment_id="exp", task_id="t", runs=runs
        )

        assert report["insufficient_repeats"] is True
        assert report["pairs"] == []
        assert report["summary"]["planning_similarity"]["mean"] is None
        assert report["summary"]["final_answer_similarity"]["mean"] is None
        # The one run's own count is still reported; a distribution of one is
        # a distribution of one, which is why the flag above exists.
        assert report["summary"]["step_counts"]["mean"] == 3.0

    def test_the_report_names_how_it_was_measured(self, repeats):
        identity = _consistency(repeats)[1]["metric_identity"]

        assert identity["metrics_version"] == consistency.METRICS_VERSION
        assert identity["text_projection_version"] == (
            consistency.TEXT_PROJECTION_VERSION
        )
        assert identity["step_count_rule"] == consistency.STEP_COUNT_RULE

    def test_the_report_refuses_to_claim_consistency_means_correct(self, repeats):
        report = _consistency(repeats)[1]

        assert "not correctness" in report["interpretation"]

    def test_the_pairwise_work_is_bounded_and_says_so(self, repeats):
        status, report = _consistency(
            repeats, repeats["candidate_id"], max_pairs=1
        )

        assert status == 200
        assert report["coverage"]["pairs_possible"] == 3
        assert report["coverage"]["pairs_reported"] == 1
        assert report["coverage"]["pairs_capped"] is True
        assert report["summary"]["planning_similarity"]["coverage"] == (
            consistency.COVERAGE_CAPPED
        )

    def test_a_pair_carries_the_identity_the_existing_comparison_uses(self, repeats):
        """So a comment written from a consistency row is counted on THAT pair."""
        report = _consistency(repeats, repeats["candidate_id"])[1]
        pair = _pair(report, 1, 2)

        control = _open_control(repeats)
        summary = best_run_module.task_run_summary(
            control, repeats["candidate_id"], repeats["task_id"]
        )
        refs = {
            row["attempt"]: comparison.ExecutionRef.from_mapping(row["execution_ref"])
            for row in summary["attempts"] if row.get("execution_ref")
        }
        expected = comparison.review_pair_key(refs[1], refs[2])

        assert pair["review_pair_key"] == expected


# ----------------------------------------------------------------------
# Choosing a best run moves the reference rows and nothing else
# ----------------------------------------------------------------------


class TestBestRunInvariance:
    def test_choosing_a_best_run_does_not_change_the_all_run_summary(
        self, repeats
    ):
        before = _consistency(repeats, repeats["candidate_id"])[1]

        _choose_best(repeats, repeats["candidate_id"], 1, "the quickest one")
        first = _consistency(repeats, repeats["candidate_id"])[1]

        _choose_best(
            repeats, repeats["candidate_id"], 3, "the interesting failure"
        )
        second = _consistency(repeats, repeats["candidate_id"])[1]

        assert json.dumps(before["summary"], sort_keys=True) == json.dumps(
            first["summary"], sort_keys=True
        )
        assert json.dumps(first["summary"], sort_keys=True) == json.dumps(
            second["summary"], sort_keys=True
        )
        assert json.dumps(first["pairs"], sort_keys=True) == json.dumps(
            second["pairs"], sort_keys=True
        )

    def test_the_reference_rows_do_follow_the_choice(self, repeats):
        before = _consistency(repeats, repeats["candidate_id"])[1]
        assert before["reference"]["usable"] is False
        assert before["reference_rows"] == []

        _choose_best(repeats, repeats["candidate_id"], 1)
        first = _consistency(repeats, repeats["candidate_id"])[1]

        _choose_best(repeats, repeats["candidate_id"], 3)
        second = _consistency(repeats, repeats["candidate_id"])[1]

        assert {row["left_attempt"] for row in first["reference_rows"]} == {1}
        assert {row["left_attempt"] for row in second["reference_rows"]} == {3}
        assert first["reference"]["attempt"] == 1
        assert second["reference"]["attempt"] == 3

    def test_the_reference_row_reads_from_the_best_run_outwards(self, repeats):
        """With attempt 2 chosen, the 4-step run is the LEFT side, and the row
        reads "the other run took two fewer steps than the best one"."""
        _choose_best(repeats, repeats["baseline_id"], 2)
        report = _consistency(repeats)[1]
        row = _pair(report, 2, 1, key="reference_rows")

        assert row["steps"]["left"] == 4 and row["steps"]["right"] == 2
        assert row["steps"]["delta"] == -2
        assert row["steps"]["normalized_difference_pct"] == pytest.approx(50.0)


# ----------------------------------------------------------------------
# Two experiments, one task
# ----------------------------------------------------------------------


class TestAcrossExperiments:
    def test_the_same_task_under_two_experiments_is_compared_with_its_n(
        self, repeats
    ):
        status, report = _consistency(
            repeats, repeats["candidate_id"],
            compare_experiment=repeats["baseline_id"],
        )

        assert status == 200
        block = report["comparison"]
        assert block["task_id"] == repeats["task_id"]
        assert block["baseline"]["experiment_id"] == repeats["baseline_id"]
        assert block["candidate"]["experiment_id"] == repeats["candidate_id"]
        # Different n, shown rather than averaged away.
        assert block["baseline"]["runs"] == 2
        assert block["candidate"]["runs"] == 3
        assert block["metric_identity_matches"] is True
        assert "not a significance test" in block["note"]

    def test_the_step_count_delta_is_real_arithmetic(self, repeats):
        block = _consistency(
            repeats, repeats["candidate_id"],
            compare_experiment=repeats["baseline_id"],
        )[1]["comparison"]
        delta = block["deltas"]["step_count_mean"]

        assert delta["state"] == consistency.METRIC_COMPUTED
        assert delta["baseline"] == pytest.approx(3.0)
        assert delta["delta"] == pytest.approx(
            delta["candidate"] - delta["baseline"]
        )

    def test_two_different_tasks_are_never_pooled(self, repeats):
        left = _consistency(repeats)[1]
        right = dict(_consistency(repeats, repeats["candidate_id"])[1])
        right["task_id"] = "some-other-task"

        with pytest.raises(consistency.ConsistencyError) as caught:
            consistency.compare_task_consistency(left, right)

        assert "different tasks" in str(caught.value)

    def test_a_different_embedding_model_refuses_the_similarity_delta(
        self, repeats
    ):
        """The case the identity block exists for.

        Same task, same runs, one side measured with another model. The
        similarity deltas are refused with the mismatch named; the step-count
        deltas survive, because counting dispatches needs no model.
        """
        baseline = _consistency(repeats)[1]
        candidate = json.loads(json.dumps(_consistency(repeats)[1]))
        candidate["metric_identity"]["embedding"]["fingerprint"] = "another-model"

        block = consistency.compare_task_consistency(baseline, candidate)

        assert block["metric_identity_matches"] is False
        assert any("embedding fingerprint" in item for item in block["mismatches"])
        for name in ("planning_similarity_mean", "final_answer_similarity_mean"):
            assert block["deltas"][name]["state"] == consistency.METRIC_UNKNOWN
            assert block["deltas"][name]["delta"] is None
            assert "not measured the same way" in block["deltas"][name]["reason"]
        assert block["deltas"]["step_count_mean"]["state"] == (
            consistency.METRIC_COMPUTED
        )

    def test_a_different_metric_version_refuses_every_delta(self, repeats):
        baseline = _consistency(repeats)[1]
        candidate = json.loads(json.dumps(baseline))
        candidate["metric_identity"]["metrics_version"] = "consistency/metrics/99"

        block = consistency.compare_task_consistency(baseline, candidate)

        assert block["metric_identity_matches"] is False
        for delta in block["deltas"].values():
            assert delta["state"] == consistency.METRIC_UNKNOWN


# ----------------------------------------------------------------------
# The same metrics out of sealed evidence
# ----------------------------------------------------------------------


SEALED_EXPERIMENT = "sealed-candidate"


def _sealed(repeats, tmp_path):
    """The candidate experiment's evidence, archived and reopened read-only.

    The manifest gives it a LOGICAL id of its own, which is how a workspace
    addresses an experiment; the reference the route builds still carries the
    local id the evidence itself records. Exercising that split is the point
    of reading the archive through the real routing rather than the module.
    """
    from fastworkflow.observability.workspace import load_observability_workspace

    from tests.test_observability_workspace import _manifest, _store_decl

    archived = obs.ObservabilityStore(repeats["db_two"], migrate=False).archive_to(
        str(tmp_path / "sealed-candidate.sqlite3")
    )
    manifest = _manifest(
        tmp_path,
        [_store_decl(archived, "sealed")],
        experiments=[
            {
                "experiment_id": SEALED_EXPERIMENT,
                "label": "the candidate, sealed",
                "segments": [
                    {
                        "segment_id": "only",
                        "store_id": "sealed",
                        "local_experiment_id": repeats["candidate_id"],
                    }
                ],
            }
        ],
    )
    return load_observability_workspace(manifest), archived


class TestSealedWorkspace:
    """A coding agent reading an archive and a person reading the live page
    must be quoting the same figures. An archive that answered differently --
    or not at all -- would make every consistency number a statement about
    which copy of the evidence somebody happened to open."""

    def _read(self, workspace, experiment_id, task_id, **params):
        return selection_api.handle_workspace_get(
            workspace,
            f"/api/experiments/{experiment_id}/tasks/{task_id}/consistency",
            {key: [str(value)] for key, value in params.items()},
        )

    def test_sealed_evidence_gives_the_figures_the_live_route_gives(
        self, repeats, tmp_path
    ):
        live = _consistency(repeats, repeats["candidate_id"])[1]
        workspace, _archive = _sealed(repeats, tmp_path)

        status, sealed = self._read(
            workspace, SEALED_EXPERIMENT, repeats["task_id"]
        )

        assert status == 200
        assert sealed["sealed"] is True
        assert sealed["summary"]["step_counts"] == live["summary"]["step_counts"]
        # Same identity, so the two are comparable rather than merely equal.
        assert sealed["metric_identity"] == live["metric_identity"]
        for metric in ("planning_similarity", "final_answer_similarity"):
            assert (
                sealed["summary"][metric]["pairs_computed"]
                == live["summary"][metric]["pairs_computed"]
            )
            assert (
                sealed["summary"][metric]["pairs_considered"]
                == live["summary"][metric]["pairs_considered"]
            )

    def test_an_archived_run_reads_in_the_order_it_ran(self, repeats, tmp_path):
        """The archive's answer, plan and steps are the live ones (fix-6v3n).

        `ObservabilityWorkspace.attempts` builds `turn_refs` from `list_turns`,
        which documents itself as newest-first, so an archived attempt's
        reference NAMES turn 2 before turn 1. Content read off that vector ran
        backwards: the run's final answer was the answer it opened with, a
        replan in turn 2 was labelled turn 1, and the aligned steps were
        aligned end-first.

        `project_execution` now orders content by the recorded chronology
        (`ordinal` within a conversation), so both copies of one run answer
        identically no matter which order their references were filed in.
        """
        live = _consistency(repeats, repeats["candidate_id"])[1]
        workspace, _archive = _sealed(repeats, tmp_path)
        sealed = self._read(
            workspace, SEALED_EXPERIMENT, repeats["task_id"]
        )[1]

        for attempt in (1, 2, 3):
            here, there = _by_attempt(live, attempt), _by_attempt(sealed, attempt)
            assert there["answer"] == here["answer"], (
                f"attempt {attempt}: the archive's final answer is not the "
                "live one"
            )
            assert there["plan"] == here["plan"], (
                f"attempt {attempt}: the archive's plan sequence differs"
            )
            assert there["step_detail"] == here["step_detail"]
            assert there["evidence"]["turn_order"] == (
                comparison.TURN_ORDER_CHRONOLOGICAL
            )
            assert here["evidence"]["turn_order"] == (
                comparison.TURN_ORDER_CHRONOLOGICAL
            )

    def test_the_archived_steps_run_in_the_order_the_archive_recorded(
        self, repeats, tmp_path
    ):
        """Through the shared comparison route, which is what a reader opens.

        The consistency figures above are counts, and a count survives being
        read backwards. The step SEQUENCE does not, so it is asserted where it
        shows: the archive's own alignment, against the live one.
        """
        workspace, _archive = _sealed(repeats, tmp_path)

        def _steps(payload):
            return [
                (step["turn_index"], step["command_name"])
                for step in payload["left"]["steps"]
            ]

        status, sealed = selection_api.handle_workspace_get(
            workspace,
            f"/api/experiments/{SEALED_EXPERIMENT}"
            f"/tasks/{repeats['task_id']}/comparison",
            {"left_attempt": ["1"], "right_attempt": ["3"], "view": ["steps"]},
        )
        live = _get_live_comparison(repeats, left=1, right=3)

        assert status == 200
        assert _steps(sealed) == _steps(live)
        assert [turn["turn_key"] for turn in sealed["left"]["turns"]] == [
            turn["turn_key"] for turn in live["left"]["turns"]
        ]
        assert sealed["left"]["answers"][-1]["answer"] == (
            live["left"]["answers"][-1]["answer"]
        )

    def test_correcting_the_order_did_not_re_key_one_recorded_comment(
        self, repeats, tmp_path
    ):
        """Content moved; identity did not, which is the whole constraint.

        A comparison comment is anchored by `ref_id` and filed under a
        `review_pair_key` derived from the two references AS THEY WERE FILED.
        Sorting the reference vectors would have re-keyed every comment ever
        written against an archived pair. So the vectors are untouched and only
        the projected content is ordered: a reference and its reverse still
        have DIFFERENT identities, and now project the SAME run.
        """
        workspace, _archive = _sealed(repeats, tmp_path)
        sealed = self._read(
            workspace, SEALED_EXPERIMENT, repeats["task_id"]
        )[1]
        reader = selection_api._AuthorizedReader(_open_control(repeats))
        row = _by_attempt(_consistency(repeats, repeats["candidate_id"])[1], 1)
        ref = comparison.ExecutionRef.from_mapping(row["execution_ref"])
        assert len(ref.turn_keys) > 1, "needs a multi-turn run to have an order"
        backwards = replace(ref, turn_keys=tuple(reversed(ref.turn_keys)))

        forward = comparison.project_execution(ref, reader)
        reverse = comparison.project_execution(backwards, reader)

        # Same run, either way it was named.
        assert [turn.turn_key for turn in forward.turns] == [
            turn.turn_key for turn in reverse.turns
        ]
        assert forward.answers()[-1] == reverse.answers()[-1]
        # Different identity, because that is what was filed and what old
        # comments are anchored by.
        assert ref.ref_id() != backwards.ref_id()
        assert forward.ref.turn_keys == ref.turn_keys
        assert reverse.ref.turn_keys == backwards.turn_keys

        # And the archive's pair keys are still the archive's: an anchor built
        # from its alignment rows resolves to the key its own pair row carries.
        # Attempts 1 and 2 ran the same commands, so their alignment has rows
        # with BOTH sides filled -- which is what a paired comment is written
        # on.
        status, compared = selection_api.handle_workspace_get(
            workspace,
            f"/api/experiments/{SEALED_EXPERIMENT}"
            f"/tasks/{repeats['task_id']}/comparison",
            {"left_attempt": ["1"], "right_attempt": ["2"], "view": ["steps"]},
        )
        assert status == 200
        anchored = [
            fb.FeedbackAnchors(
                primary=fb.FeedbackTarget.from_mapping(
                    dict(pair["anchors"]["left"], target_label="left")
                ),
                paired=fb.FeedbackTarget.from_mapping(
                    dict(pair["anchors"]["right"], target_label="right")
                ),
            ).pair_key
            for pair in compared["alignment"]["rows"]
            if pair["anchors"]["left"] and pair["anchors"]["right"]
        ]
        assert anchored, "a two-sided row is what a paired comment is written on"
        assert set(anchored) == {_pair(sealed, 1, 2)["review_pair_key"]}

    def test_an_archive_records_no_best_run_and_says_so(self, repeats, tmp_path):
        workspace, _archive = _sealed(repeats, tmp_path)

        sealed = self._read(
            workspace, SEALED_EXPERIMENT, repeats["task_id"]
        )[1]

        assert sealed["reference"]["usable"] is False
        assert "selection control" in sealed["reference"]["reason"]
        assert sealed["reference_rows"] == []
        # The pair table is still there: the comparison is evidence, not a
        # decision, and it is what a reader came for.
        assert len(sealed["pairs"]) == 3

    def test_reading_an_archive_writes_nothing_beside_it(self, repeats, tmp_path):
        """Byte-for-byte, which a sealed file can be held to."""
        workspace, archive = _sealed(repeats, tmp_path)
        path = Path(archive["path"])
        before = (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())

        assert self._read(
            workspace, SEALED_EXPERIMENT, repeats["task_id"]
        )[0] == 200

        after = (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
        assert before == after
        assert not list(path.parent.glob("*consistency-vectors*")), (
            "a read of sealed evidence created a derived cache beside it"
        )

    def test_an_archived_pair_is_the_pair_the_archive_compares(
        self, repeats, tmp_path
    ):
        """The row's key is the key the comparison it links to was filed under.

        Checked against the ARCHIVE's own `/comparison`, which is where the
        row deep-links; the live route's keys are a separate question, and
        today a separate answer, because of the turn-order defect above.
        """
        workspace, _archive = _sealed(repeats, tmp_path)
        sealed = self._read(
            workspace, SEALED_EXPERIMENT, repeats["task_id"]
        )[1]

        status, comparison_payload = selection_api.handle_workspace_get(
            workspace,
            f"/api/experiments/{SEALED_EXPERIMENT}"
            f"/tasks/{repeats['task_id']}/comparison",
            {"left_attempt": ["1"], "right_attempt": ["3"]},
        )

        assert status == 200
        assert _pair(sealed, 1, 3)["review_pair_key"] == (
            comparison_payload["review_pair_key"]
        )


# ----------------------------------------------------------------------
# Over real HTTP, against the real server
# ----------------------------------------------------------------------


class TestOverHttp:
    def _path(self, repeats, experiment=None, query=""):
        return (
            f"/api/experiments/{experiment or repeats['baseline_id']}"
            f"/tasks/{repeats['task_id']}/consistency{query}"
        )

    def test_a_client_gets_the_same_values_the_page_shows(self, server, repeats):
        status, payload = _request(server, self._path(repeats))
        direct = _consistency(repeats)[1]

        assert status == 200
        assert payload["summary"]["step_counts"]["population_sd"] == (
            direct["summary"]["step_counts"]["population_sd"]
        )
        assert payload["metric_identity"] == direct["metric_identity"]

    def test_the_route_needs_the_token(self, server, repeats):
        status, _payload = _request(server, self._path(repeats), token=None)

        assert status == 401

    def test_an_unknown_experiment_is_not_found(self, server, repeats):
        status, payload = _request(
            server,
            f"/api/experiments/exp-nobody/tasks/{repeats['task_id']}/consistency",
        )

        assert status == 404
        assert "exp-nobody" in payload["error"]

    def test_a_bad_bound_is_refused_rather_than_clamped_to_nothing(
        self, server, repeats
    ):
        status, payload = _request(
            server, self._path(repeats, query="?max_pairs=0")
        )

        assert status == 400
        assert "max_pairs" in payload["error"]

    def test_reading_consistency_does_not_touch_the_evidence(self, server, repeats):
        """A measurement that modifies what it measures is not a measurement.

        Checked on CONTENT rather than on file bytes. These stores are live and
        in WAL mode, where merely opening and closing a connection can fold WAL
        frames into the main file: the bytes move while not one recorded row
        does. Hashing the rows asserts the thing actually worth asserting, and
        would still fail if the route wrote, updated or deleted anything.
        """
        watched = [repeats["db_one"], repeats["db_two"]]
        before = [_evidence_digest(path) for path in watched]

        assert _request(server, self._path(repeats))[0] == 200
        assert _request(server, self._path(repeats, repeats["candidate_id"]))[0] == 200

        assert [_evidence_digest(path) for path in watched] == before

    def test_derived_vectors_are_cached_outside_the_evidence(self, server, repeats):
        cache_root = (
            Path(state_paths.workflow_state_dir(repeats["folder"]))
            / selection_api._CONSISTENCY_CACHE_DIRNAME
        )
        first = _request(server, self._path(repeats, repeats["candidate_id"]))[1]
        if not first["metric_identity"]["embedding"]["available"]:
            pytest.skip("no local embedding model on this machine")

        assert first["coverage"]["vectors_computed"] > 0
        assert cache_root.exists(), "no derived cache was written"
        assert cache_root.resolve().is_relative_to(
            Path(repeats["state_root"]).resolve()
        )

        second = _request(server, self._path(repeats, repeats["candidate_id"]))[1]
        assert second["coverage"]["vectors_from_cache"] > 0
        assert second["coverage"]["vectors_computed"] == 0, (
            "the second read recomputed vectors the first one cached"
        )

    def test_the_cross_experiment_block_comes_back_over_the_wire(
        self, server, repeats
    ):
        status, payload = _request(
            server,
            self._path(
                repeats, repeats["candidate_id"],
                query=f"?compare_experiment={repeats['baseline_id']}",
            ),
        )

        assert status == 200
        assert payload["compare_experiment_id"] == repeats["baseline_id"]
        assert payload["comparison"]["metric_identity_matches"] is True


# ----------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------


def test_the_consistency_panel_is_wired_into_both_task_views():
    """Pinned beside the DOM test below, never instead of it."""
    page = run_chatbot_server.load_index_html().decode("utf-8")

    assert "function renderConsistency(" in page
    # Runs and Compare both call it; nothing else does.
    assert page.count("renderConsistency(") == 3
    runs = page[page.index("function renderTaskRuns("):
                page.index("function renderTaskRunList(")]
    assert "renderConsistency(consistency, experimentId, taskId, label, nav, null)" \
        in runs
    # The click sets the same compare state the attempt list's own control
    # sets, so a pair opens in the existing comparison rather than a new view.
    button = page[page.index("function consistencyPairButton("):
                  page.index("function pairMetricText(")]
    assert "taskCompare.left = String(pair.left_attempt);" in button
    assert 'taskView = "compare";' in button
    assert "showExperimentTask(experimentId, taskId, label)" in button


def test_the_dom(server, repeats):
    """Clicked in a real DOM against the real server."""
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_consistency_dom.cjs")
    result = subprocess.run(
        [
            "node",
            str(script),
            dependency,
            f"http://127.0.0.1:{server.port}/?token={server.token}",
            json.dumps({
                "baseline": repeats["baseline_id"],
                "candidate": repeats["candidate_id"],
                "task": repeats["task_id"],
            }),
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
