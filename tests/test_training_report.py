"""Integration tests for the per-command training-data report (spec R3b, bd fix-551.4).

Per `.cursor/rules/testing_rules.mdc` these are integration tests: no Mock fixtures
and no patching of fastWorkflow internals. Where a failure condition is needed it is
**induced** — real `litellm` exception objects raised by locally defined callables, a
locally defined persona source, a real `ProvenanceRecorder` saving a real file — so
nothing below needs an API key or the network except the end-to-end test, which skips
cleanly without one following the convention in `test_train_modern_stack.py`.

The build inputs are the *real* `command_directory.json` and `routing_definition.json`
from the bundled `fastworkflow/examples/hello_world`, copied into a tmp workflow, so
the tests exercise the field names the trainer actually writes rather than a
hand-written approximation of them.
"""

import importlib.util
import json
import os
import shutil

import pytest
from dotenv import dotenv_values

import litellm

import fastworkflow
from fastworkflow.model_pipeline_training import TrainingDataError, split_training_data
from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, WILDCARD_LABEL
from fastworkflow.train import heldout_evaluation
from fastworkflow.train import training_report as tr
from fastworkflow.train.determinism import (
    COMMAND_INFO_FOLDERNAME,
    ContextTrainingStatus,
    PROVENANCE_FILENAME,
    ProvenanceRecorder,
    UtteranceProvenance,
)
from fastworkflow.train.generate_synthetic import (
    generate_diverse_utterances_with_provenance,
)
from fastworkflow.train.heldout_evaluation import (
    MIN_TRAINING_ROWS_PER_LABEL,
    HeldoutReport,
    RoutingScore,
)

HELLO_WORLD_PATH = os.path.join("fastworkflow", "examples", "hello_world")
EXAMPLE_COMMAND_INFO = os.path.join(HELLO_WORLD_PATH, COMMAND_INFO_FOLDERNAME)

# A real application command and a real framework command from the bundled example.
APP_COMMAND = "add_two_numbers"
FRAMEWORK_COMMAND = "IntentDetection/go_up"

SEED_UTTERANCES = ["add 2 and 3", "sum these numbers"]

# Stand-in for PersonaHub: `len()` plus `[i]['persona']` is the whole interface
# generate_synthetic uses. Real personas, no download. Mirrors the local dataset in
# test_training_determinism.py.
LOCAL_PERSONAS = [{"persona": f"A person who is persona number {i}."} for i in range(16)]


def _local_persona_dataset():
    return LOCAL_PERSONAS


def _rate_limit_error() -> litellm.exceptions.RateLimitError:
    """A real litellm RateLimitError, constructed locally rather than mocked."""
    return litellm.exceptions.RateLimitError(
        message="rate limited by the test", llm_provider="test", model="test/model"
    )


@pytest.fixture
def fast_retry_options():
    """Zero-delay retry budget through the generator's private test seam."""
    return {"_max_retries": 1, "_retry_base_seconds": 0.0}


@pytest.fixture
def workflow(tmp_path):
    """A tmp workflow carrying the bundled example's REAL build artifacts.

    Only the two JSON snapshots are copied: they are what `build_report` reads for
    command ownership, training participation and context membership. No provenance
    is written, so each test decides what state the run ended in.
    """
    folder = tmp_path / "workflow"
    info = folder / COMMAND_INFO_FOLDERNAME
    info.mkdir(parents=True)
    for name in (tr.COMMAND_DIRECTORY_FILENAME, tr.ROUTING_DEFINITION_FILENAME):
        shutil.copy2(os.path.join(EXAMPLE_COMMAND_INFO, name), info / name)
    return str(folder)


def _save_provenance(workflow_folderpath: str, *records: UtteranceProvenance) -> str:
    """Persist *records* through the real recorder the trainer uses."""
    recorder = ProvenanceRecorder(workflow_folderpath)
    for record in records:
        recorder.record(record)
    return recorder.save()


def _provenance_for(command_name: str, seeds: int, generated: int) -> UtteranceProvenance:
    """A healthy record shaped exactly as generation produces one.

    `final_count` is the command-name token plus seeds plus generated, matching
    `generate_diverse_utterances_with_provenance`.
    """
    return UtteranceProvenance(
        command_name=command_name,
        seed=42,
        seed_utterance_count=seeds,
        generated_count=generated,
        final_count=1 + seeds + generated,
    )


def _all_generating_commands(workflow_folderpath: str) -> list[str]:
    """Every command in the real command directory that generates utterances.

    Reserved labels are excluded: `wildcard`'s command file returns fixed literals and
    never calls the generator, so it legitimately has no provenance record.
    """
    path = os.path.join(
        workflow_folderpath, COMMAND_INFO_FOLDERNAME, tr.COMMAND_DIRECTORY_FILENAME
    )
    with open(path, encoding="utf-8") as f:
        directory = json.load(f)
    return [
        name
        for name in directory["map_command_2_metadata"]
        if name.split("/")[-1] not in {WILDCARD_LABEL, PARAMETER_VALUE_LABEL}
    ]


def _save_healthy_provenance_for_all(workflow_folderpath: str) -> None:
    """Persist a healthy record for every generating command, i.e. a clean run."""
    _save_provenance(
        workflow_folderpath,
        *[
            _provenance_for(name, seeds=9, generated=20)
            for name in _all_generating_commands(workflow_folderpath)
        ],
    )


def _row(report: tr.TrainingReport, command_name: str) -> tr.CommandRow:
    matches = [row for row in report.rows if row.command_name == command_name]
    assert matches, f"{command_name} is missing from the report"
    return matches[0]


# ---------------------------------------------------------------------------
# Degradation: a broken input must never produce a traceback
# ---------------------------------------------------------------------------

def test_absent_provenance_explains_itself_instead_of_raising(workflow):
    report = tr.build_report(workflow)

    assert report.problems, "an untrained workflow must say why the report is empty"
    assert any("never been trained" in problem for problem in report.problems)
    assert any(PROVENANCE_FILENAME in problem for problem in report.problems)
    # The build artifacts are still readable, so every command is still listed.
    assert {row.command_name for row in report.rows} >= {APP_COMMAND, FRAMEWORK_COMMAND}
    assert tr.format_report(report)


def test_corrupt_provenance_is_described_not_raised(workflow):
    path = os.path.join(workflow, COMMAND_INFO_FOLDERNAME, PROVENANCE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{ this is not json")

    report = tr.build_report(workflow)

    assert any("could not be read" in problem for problem in report.problems)
    assert "INCOMPLETE" in tr.format_report(report)


def test_a_single_unusable_record_does_not_discard_the_others(workflow):
    """A half-written provenance file is exactly when this report is most wanted."""
    path = os.path.join(workflow, COMMAND_INFO_FOLDERNAME, PROVENANCE_FILENAME)
    payload = {
        APP_COMMAND: _provenance_for(APP_COMMAND, seeds=9, generated=20).model_dump(),
        FRAMEWORK_COMMAND: {"command_name": FRAMEWORK_COMMAND, "seed": "not-an-int"},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    report = tr.build_report(workflow)

    assert _row(report, APP_COMMAND).status is tr.RowStatus.OK
    assert any(FRAMEWORK_COMMAND in problem for problem in report.problems)


def test_legacy_flat_schema_is_read_with_an_explicit_count_limitation(workflow):
    path = os.path.join(workflow, COMMAND_INFO_FOLDERNAME, PROVENANCE_FILENAME)
    legacy = {
        APP_COMMAND: _provenance_for(
            APP_COMMAND, seeds=9, generated=20
        ).model_dump()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(legacy, f, indent=2)

    report = tr.build_report(workflow)

    assert _row(report, APP_COMMAND).row_count == legacy[APP_COMMAND]["final_count"]
    assert any("legacy flat schema" in problem for problem in report.problems)
    assert any("per-context row" in problem for problem in report.problems)


def test_report_training_data_never_raises_on_a_nonexistent_workflow(tmp_path):
    """The entry point the trainer calls must not be what a developer sees instead
    of the models they waited hours for."""
    result = tr.report_training_data(
        str(tmp_path / "no-such-workflow"), print_report=False, write=False
    )
    assert result is not None
    assert result.problems


# ---------------------------------------------------------------------------
# The two things that must be impossible to miss
# ---------------------------------------------------------------------------

def test_a_real_rate_limit_fallback_is_impossible_to_miss(
    workflow, fast_retry_options
):
    """Induce the R3a fallback for real, then check the report shouts about it.

    This is the end of the chain F3 describes: generation is rate-limited, wave 1
    degrades it to seeds instead of `[]`, and R3b is what finally tells a human.
    """
    def always_rate_limited(**_kwargs):
        raise _rate_limit_error()

    utterances, provenance = generate_diverse_utterances_with_provenance(
        SEED_UTTERANCES,
        APP_COMMAND,
        num_personas=2,
        utterances_per_persona=3,
        personas_per_batch=1,
        seed=42,
        completion_fn=always_rate_limited,
        persona_dataset_loader=_local_persona_dataset,
        **fast_retry_options,
    )
    assert provenance.fell_back is True, "precondition: R3a must have fallen back"
    _save_provenance(workflow, provenance)

    report = tr.build_report(workflow)
    row = _row(report, APP_COMMAND)

    assert row.status is tr.RowStatus.FELL_BACK
    assert row.fell_back is True
    assert row.row_count == len(utterances)
    assert all(
        blocking.status is not tr.RowStatus.FELL_BACK
        for blocking in report.blocking_rows
    )
    assert row not in report.blocking_rows

    rendered = tr.format_report(report)
    assert "!!" in rendered, "a degraded command must carry the R3a-style banner"
    assert "FELL BACK" in rendered
    assert APP_COMMAND in rendered
    assert "RateLimitError" in rendered, "the reason must survive into the report"
    # The banner must appear before the detail so a developer reading three lines
    # still learns something is wrong.
    assert rendered.index("!!") < rendered.index("FELL BACK")


def test_fallback_context_use_is_explicit_and_uses_actual_row_count(workflow):
    degraded = _provenance_for(APP_COMMAND, seeds=2, generated=0)
    degraded.fell_back = True
    degraded.fallback_reason = "RateLimitError after 1 retries"
    recorder = ProvenanceRecorder(workflow)
    recorder.record(degraded)
    recorder.record_context(
        context_name="Child",
        command_name=APP_COMMAND,
        status=ContextTrainingStatus.INCLUDED_FALLBACK,
        row_count=4,
        reason=degraded.fallback_reason,
    )
    recorder.save()

    row = _row(tr.build_report(workflow), APP_COMMAND)
    assert row.status is tr.RowStatus.FELL_BACK
    assert row.rows_by_context == {"Child": 4}
    assert row.row_count == 4
    assert row.fallback_reason == degraded.fallback_reason


def test_below_the_row_floor_is_flagged_separately_from_a_fallback(workflow):
    starved = _provenance_for(APP_COMMAND, seeds=0, generated=0)
    assert starved.final_count < tr.DEFAULT_MIN_TRAINING_ROWS
    healthy = _provenance_for(FRAMEWORK_COMMAND, seeds=9, generated=20)
    _save_provenance(workflow, starved, healthy)

    report = tr.build_report(workflow)

    assert _row(report, APP_COMMAND).status is tr.RowStatus.BELOW_FLOOR
    assert _row(report, FRAMEWORK_COMMAND).status is tr.RowStatus.OK
    assert report.has_blocking_problems is True

    rendered = tr.format_report(report)
    assert "BELOW ROW FLOOR" in rendered
    # The floor's derivation travels with the number, so a reader is never left to
    # assume it was measured.
    assert "class-aware" in rendered


def test_multi_context_rows_are_counted_without_duplicate_generation_records(workflow):
    """One inherited command has one draw, but contributes rows to two models."""
    recorder = ProvenanceRecorder(workflow)
    generation = _provenance_for(APP_COMMAND, seeds=2, generated=3)
    recorder.record(generation)
    recorder.record_context(
        context_name="Parent",
        command_name=APP_COMMAND,
        status=ContextTrainingStatus.INCLUDED,
        row_count=7,
    )
    recorder.record_context(
        context_name="Child",
        command_name=APP_COMMAND,
        status=ContextTrainingStatus.INCLUDED,
        row_count=7,
    )
    recorder.save()

    report = tr.build_report(workflow, min_rows=4, min_seeds=8)
    row = _row(report, APP_COMMAND)

    assert len(ProvenanceRecorder.load(workflow)) == 1
    assert row.seed_count == generation.seed_utterance_count
    assert row.generated_count == generation.generated_count
    assert row.rows_by_context == {"Child": 7, "Parent": 7}
    assert row.row_count == 14
    assert row.contexts == ["Child", "Parent"]
    assert row.status is tr.RowStatus.THIN_SEEDS
    rendered = tr.format_report(report)
    assert "Child:7" in rendered
    assert "Parent:7" in rendered


def test_row_floor_is_applied_per_context_not_to_the_aggregate(workflow):
    recorder = ProvenanceRecorder(workflow)
    recorder.record(_provenance_for(APP_COMMAND, seeds=2, generated=0))
    for context_name in ("Parent", "Child"):
        recorder.record_context(
            context_name=context_name,
            command_name=APP_COMMAND,
            status=ContextTrainingStatus.INCLUDED,
            row_count=3,
        )
    recorder.save()

    row = _row(tr.build_report(workflow, min_rows=4, min_seeds=2), APP_COMMAND)
    assert row.row_count == 6
    assert row.status is tr.RowStatus.BELOW_FLOOR


def test_a_command_that_should_have_generated_but_did_not_is_reported_missing(workflow):
    """Distinct from EXCLUDED: this is a gap, not a design choice."""
    _save_provenance(workflow, _provenance_for(APP_COMMAND, seeds=9, generated=20))

    report = tr.build_report(workflow)
    row = _row(report, FRAMEWORK_COMMAND)

    assert row.status is tr.RowStatus.MISSING
    assert row.is_blocking is True
    assert "NO PROVENANCE" in tr.format_report(report)


# ---------------------------------------------------------------------------
# Reserved labels are not commands
# ---------------------------------------------------------------------------

def test_reserved_labels_are_never_reported_as_starved_commands(workflow):
    """`wildcard` has no provenance BY DESIGN — its command file returns the fixed
    `PARAMETER_VALUE_PLACEHOLDERS` and never calls the generator. Reporting that as a
    missing command would put a non-command at the top of a list of broken ones."""
    _save_healthy_provenance_for_all(workflow)

    report = tr.build_report(workflow)
    wildcard_row = _row(report, WILDCARD_LABEL)

    assert wildcard_row.kind is tr.CommandKind.RESERVED
    assert wildcard_row.status is tr.RowStatus.NOT_APPLICABLE
    assert wildcard_row.is_blocking is False
    assert report.has_blocking_problems is False

    rendered = tr.format_report(report)
    assert "reserved labels" in rendered
    # Named explicitly even though it produces no provenance, so a reader can tell it
    # was considered rather than lost.
    assert PARAMETER_VALUE_LABEL in rendered


def test_a_reserved_label_that_actually_fell_back_is_still_surfaced(workflow):
    """The exemption covers absence and thinness, not degradation: a wildcard class
    that fell back degrades escalation for the whole context."""
    degraded = _provenance_for(WILDCARD_LABEL, seeds=7, generated=0)
    degraded.fell_back = True
    degraded.fallback_reason = "RateLimitError after 1 retries"
    _save_provenance(workflow, degraded)

    report = tr.build_report(workflow)
    row = _row(report, WILDCARD_LABEL)

    assert row.kind is tr.CommandKind.RESERVED
    assert row.status is tr.RowStatus.FELL_BACK
    assert report.has_blocking_problems is True


def test_commands_without_signature_input_are_excluded_not_missing(workflow, tmp_path):
    """`_requires_utterances` keys off `input_for_param_extraction_class`; a command
    without one is dispatched via perform_action and is deliberately not a label."""
    info = os.path.join(workflow, COMMAND_INFO_FOLDERNAME)
    path = os.path.join(info, tr.COMMAND_DIRECTORY_FILENAME)
    with open(path, encoding="utf-8") as f:
        directory = json.load(f)
    directory["map_command_2_metadata"][APP_COMMAND]["input_for_param_extraction_class"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(directory, f)

    recorder = ProvenanceRecorder(workflow)
    recorder.record(_provenance_for(FRAMEWORK_COMMAND, seeds=9, generated=20))
    recorder.record_context(
        context_name="*",
        command_name=APP_COMMAND,
        status=ContextTrainingStatus.SKIPPED_NO_INPUT,
        reason="command has no Signature.Input",
    )
    recorder.save()
    report = tr.build_report(workflow)

    row = _row(report, APP_COMMAND)
    assert row.status is tr.RowStatus.EXCLUDED
    assert row.is_blocking is False, "a design choice is not a defect"
    assert "not intent-routed" in tr.format_report(report)


def test_explicit_no_utterances_skip_is_not_reported_as_lost_provenance(workflow):
    recorder = ProvenanceRecorder(workflow)
    recorder.record_context(
        context_name="Child",
        command_name=APP_COMMAND,
        status=ContextTrainingStatus.SKIPPED_NO_UTTERANCES,
        reason="utterance generator returned no rows",
    )
    recorder.save()

    report = tr.build_report(workflow)
    row = _row(report, APP_COMMAND)

    assert row.status is tr.RowStatus.NO_UTTERANCES
    assert row.skipped_contexts == {
        "Child": "utterance generator returned no rows"
    }
    assert row.is_blocking is True
    assert "NO UTTERANCES" in tr.format_report(report)


# ---------------------------------------------------------------------------
# Seed advisory (F13) — never blocking
# ---------------------------------------------------------------------------

def test_thin_seeds_are_advisory_and_labelled_as_one_workflows_observation(workflow):
    thin = _provenance_for(APP_COMMAND, seeds=3, generated=20)
    _save_provenance(workflow, thin)

    report = tr.build_report(workflow)
    row = _row(report, APP_COMMAND)

    assert row.status is tr.RowStatus.THIN_SEEDS
    assert row.is_blocking is False, "F13's curve is too weak a basis for blocking"
    assert tr.RowStatus.THIN_SEEDS not in tr.BLOCKING_STATUSES

    rendered = tr.format_report(report)
    assert "THIN SEEDS" in rendered
    assert "ONE workflow" in rendered, "the evidence base must travel with the number"
    assert "not a constant" in rendered


def test_floors_are_configurable_defaults(workflow):
    _save_provenance(workflow, _provenance_for(APP_COMMAND, seeds=3, generated=20))

    default_report = tr.build_report(workflow)
    assert default_report.min_rows == tr.DEFAULT_MIN_TRAINING_ROWS
    assert default_report.min_seeds == tr.DEFAULT_MIN_SEED_UTTERANCES
    assert _row(default_report, APP_COMMAND).status is tr.RowStatus.THIN_SEEDS

    relaxed = tr.build_report(workflow, min_rows=1, min_seeds=2)
    assert _row(relaxed, APP_COMMAND).status is tr.RowStatus.OK

    strict = tr.build_report(workflow, min_rows=100, min_seeds=2)
    assert _row(strict, APP_COMMAND).status is tr.RowStatus.BELOW_FLOOR


def test_class_aware_split_keeps_every_label_in_train_and_evaluation():
    dataset = [
        ("alpha one", 0),
        ("alpha two", 0),
        ("alpha three", 0),
        ("beta one", 1),
        ("beta two", 1),
    ]

    train_rows, evaluation_rows = split_training_data(dataset)

    assert {label for _, label in train_rows} == {0, 1}
    assert {label for _, label in evaluation_rows} == {0, 1}


def test_class_aware_split_fails_before_fitting_a_one_row_label():
    # Matched against the constant rather than the spelled-out "two": the floor is
    # now shared with the persona split (heldout_evaluation.MIN_TRAINING_ROWS_PER_LABEL)
    # so the two modules cannot disagree, and the message interpolates it. Hard-coding
    # the English here would break the moment the floor is retuned.
    with pytest.raises(
        TrainingDataError, match=f"at least {MIN_TRAINING_ROWS_PER_LABEL} rows"
    ):
        split_training_data([
            ("alpha one", 0),
            ("alpha two", 0),
            ("beta only", 1),
        ])


def test_class_aware_split_error_names_the_command_not_its_encoded_id():
    """A developer cannot map label id 1 to a command without the LabelEncoder."""
    with pytest.raises(TrainingDataError) as excinfo:
        split_training_data(
            [("alpha one", 0), ("alpha two", 0), ("beta only", 1)],
            lambda encoded: f"Ctx/command_{encoded}",
        )

    assert "Ctx/command_1" in str(excinfo.value)


def test_floor_environment_variables_are_ignored(workflow):
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({
            **previous,
            "TRAINING_REPORT_MIN_SEEDS": "2",
            "TRAINING_REPORT_MIN_ROWS": "1",
        })
        _save_provenance(workflow, _provenance_for(APP_COMMAND, seeds=3, generated=20))
        report = tr.build_report(workflow)
        assert report.min_seeds == tr.DEFAULT_MIN_SEED_UTTERANCES
        assert report.min_rows == tr.DEFAULT_MIN_TRAINING_ROWS
        assert _row(report, APP_COMMAND).status is tr.RowStatus.THIN_SEEDS
    finally:
        fastworkflow.init(previous)


def test_report_floors_are_fixed_trainer_policy(workflow):
    """Malformed environment values cannot alter trainer policy."""
    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init({**previous, "TRAINING_REPORT_MIN_ROWS": "not-a-number"})
        assert tr.get_min_training_rows() == tr.DEFAULT_MIN_TRAINING_ROWS
        assert tr.get_min_seed_utterances() == tr.DEFAULT_MIN_SEED_UTTERANCES
    finally:
        fastworkflow.init(previous)


# ---------------------------------------------------------------------------
# Held-out join (R1)
# ---------------------------------------------------------------------------

def test_heldout_filename_agrees_with_its_owning_module():
    """`HELDOUT_REPORT_FILENAME` is duplicated to keep this module cheap to import;
    this is the assertion that stops the copies drifting apart."""
    assert tr.HELDOUT_REPORT_FILENAME == heldout_evaluation.REPORT_FILENAME


def test_per_command_heldout_scores_are_joined_in_and_summed_across_contexts(workflow):
    _save_provenance(
        workflow,
        _provenance_for(APP_COMMAND, seeds=9, generated=30),
        _provenance_for(FRAMEWORK_COMMAND, seeds=9, generated=30),
    )
    # Real HeldoutReport objects written by R1's own writer, not a hand-rolled file.
    heldout_evaluation.write_report(
        workflow,
        [
            HeldoutReport(
                context="*",
                routing=RoutingScore(
                    total=8,
                    top1_correct=5,
                    in_list_correct=6,
                    per_command={
                        APP_COMMAND: {"total": 4, "top1_correct": 3, "in_list_correct": 3},
                        FRAMEWORK_COMMAND: {
                            "total": 4, "top1_correct": 0, "in_list_correct": 1
                        },
                    },
                ),
            ),
            HeldoutReport(
                context="IntentDetection",
                routing=RoutingScore(
                    total=2,
                    top1_correct=1,
                    in_list_correct=1,
                    per_command={
                        FRAMEWORK_COMMAND: {
                            "total": 2, "top1_correct": 0, "in_list_correct": 0
                        },
                    },
                ),
            ),
        ],
    )

    report = tr.build_report(workflow)
    assert report.heldout_available is True

    app_row = _row(report, APP_COMMAND)
    assert (app_row.heldout_total, app_row.heldout_top1_correct) == (4, 3)
    assert app_row.heldout_top1 == pytest.approx(0.75)

    framework_row = _row(report, FRAMEWORK_COMMAND)
    assert (framework_row.heldout_total, framework_row.heldout_top1_correct) == (6, 0)

    # Healthy row count, zero held-out hits: the report must not summarise this away.
    assert framework_row.status is tr.RowStatus.OK
    assert framework_row in report.never_routed_rows
    assert app_row not in report.never_routed_rows
    assert "NEVER ROUTED" in tr.format_report(report)


def test_the_heldout_column_is_absent_rather_than_wrong_when_r1_has_not_run(workflow):
    _save_provenance(workflow, _provenance_for(APP_COMMAND, seeds=9, generated=20))

    report = tr.build_report(workflow)

    assert report.heldout_available is False
    assert _row(report, APP_COMMAND).heldout_top1 is None
    assert report.never_routed_rows == []
    assert "not available" in tr.format_report(report)


# ---------------------------------------------------------------------------
# Output surfaces
# ---------------------------------------------------------------------------

def test_healthy_commands_are_summarised_rather_than_enumerated(workflow):
    """A report that lists every fine command buries the ones that are not — which is
    the failure mode F4 describes."""
    many = [
        _provenance_for(f"bulk_command_{index:02d}", seeds=9, generated=20)
        for index in range(20)
    ]
    _save_provenance(workflow, *many)

    report = tr.build_report(workflow)
    rendered = tr.format_report(report)

    healthy = report.with_status(tr.RowStatus.OK)
    assert len(healthy) >= 20
    named = sum(1 for row in healthy if row.command_name in rendered)
    assert named <= tr.HEALTHY_SAMPLE_LIMIT, (
        "healthy commands must be summarised, not listed"
    )
    assert "more" in rendered


def test_write_report_produces_a_text_and_a_json_artifact(workflow):
    _save_provenance(workflow, _provenance_for(APP_COMMAND, seeds=0, generated=0))
    report = tr.build_report(workflow)

    text_path, json_path = tr.write_report(workflow, report)

    assert os.path.basename(text_path) == tr.REPORT_FILENAME
    assert os.path.basename(json_path) == tr.REPORT_JSON_FILENAME
    with open(text_path, encoding="utf-8") as f:
        assert "TRAINING DATA REPORT" in f.read()

    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["summary"]["has_blocking_problems"] is True
    assert payload["summary"]["below_floor"] == 1
    assert payload["min_rows"] == tr.DEFAULT_MIN_TRAINING_ROWS
    assert any(row["command_name"] == APP_COMMAND for row in payload["rows"])


def test_report_training_data_writes_and_returns(workflow, capsys):
    _save_provenance(workflow, _provenance_for(APP_COMMAND, seeds=9, generated=20))

    report = tr.report_training_data(workflow)

    assert report is not None
    assert "TRAINING DATA REPORT" in capsys.readouterr().out
    info = os.path.join(workflow, COMMAND_INFO_FOLDERNAME)
    assert os.path.isfile(os.path.join(info, tr.REPORT_FILENAME))
    assert os.path.isfile(os.path.join(info, tr.REPORT_JSON_FILENAME))


def test_rows_are_ordered_worst_first(workflow):
    fallen = _provenance_for("a_fallen_command", seeds=3, generated=0)
    fallen.fell_back = True
    fallen.fallback_reason = "RateLimitError after 1 retries"
    _save_provenance(
        workflow,
        _provenance_for("z_healthy_command", seeds=9, generated=20),
        _provenance_for("m_thin_command", seeds=2, generated=20),
        _provenance_for("b_starved_command", seeds=0, generated=0),
        fallen,
    )

    report = tr.build_report(workflow)
    statuses = [row.status for row in report.rows]
    severities = [tr._STATUS_SEVERITY[status] for status in statuses]

    assert severities == sorted(severities), "worst problems must come first"
    assert statuses[0] is tr.RowStatus.FELL_BACK


# ---------------------------------------------------------------------------
# End to end against a real training run
# ---------------------------------------------------------------------------

def _datasets_available() -> bool:
    return importlib.util.find_spec("datasets") is not None


def _looks_like_real_key(value) -> bool:
    """Reject empty / placeholder keys like ``<API KEY ...>``."""
    return bool(value) and "<" not in value and "your-" not in value.lower()


def _resolve_env_vars() -> dict:
    """Build the training env, matching test_train_modern_stack.py."""
    example_env = os.path.join("fastworkflow", "examples", "fastworkflow.env")
    example_pwd = os.path.join("fastworkflow", "examples", "fastworkflow.passwords.env")
    env_vars = {**dotenv_values(example_env), **dotenv_values(example_pwd)}

    local_env = os.path.join("env", ".env")
    local_pwd = os.path.join("passwords", ".env")
    if os.path.exists(local_env):
        env_vars.update(dotenv_values(local_env))
    if os.path.exists(local_pwd):
        env_vars.update(dotenv_values(local_pwd))

    for key in (
        "LLM_SYNDATA_GEN",
        "LITELLM_API_KEY_SYNDATA_GEN",
        "LITELLM_PROXY_API_BASE",
        "LITELLM_PROXY_API_KEY",
    ):
        val = os.environ.get(key)
        if val and "<" not in val:
            env_vars[key] = val
    return env_vars


def test_report_describes_a_real_training_run(tmp_path_factory):
    """Train the bundled hello_world for real and report on what came out.

    Skips without `datasets` and a real synthetic-generation key, following
    `test_train_modern_stack.py`. Trains into an isolated COPY so it cannot destroy the
    real example's checked-in artifacts, which other tests depend on being present.
    """
    if not _datasets_available():
        pytest.skip("datasets package not installed; training cannot run.")
    env_vars = _resolve_env_vars()
    if not _looks_like_real_key(env_vars.get("LITELLM_API_KEY_SYNDATA_GEN")):
        pytest.skip(
            "No real LITELLM_API_KEY_SYNDATA_GEN available; cannot run synthetic "
            "utterance generation required for model training."
        )

    # Imported here rather than at module scope: `train_workflow` pulls in torch and
    # transformers, and every other test in this file must stay importable and fast
    # when training is skipped.
    from fastworkflow.train.__main__ import train_workflow

    workflow_path = str(tmp_path_factory.mktemp("report_e2e") / "hello_world")
    shutil.copytree(
        HELLO_WORLD_PATH,
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"
        ),
    )

    previous = dict(fastworkflow._env_vars)
    try:
        fastworkflow.init(env_vars=env_vars)
        train_workflow(workflow_path)

        report = tr.build_report(workflow_path)

        # Every command the run generated for must appear with a real row count.
        recorded = ProvenanceRecorder.load(workflow_path)
        context_records = ProvenanceRecorder.load_context_records(workflow_path)
        assert recorded, "a completed training run must leave provenance behind"
        wildcard_record = context_records[("*", WILDCARD_LABEL)]
        parameter_value_record = context_records[("*", PARAMETER_VALUE_LABEL)]
        assert wildcard_record.status is ContextTrainingStatus.SKIPPED_NO_UTTERANCES
        assert wildcard_record.own_row_count
        assert wildcard_record.raw_candidate_count == 0
        assert wildcard_record.deduplicated_candidate_count == 0
        assert wildcard_record.selected_budget is None
        assert parameter_value_record.status is ContextTrainingStatus.INCLUDED
        assert parameter_value_record.row_count > 0
        assert parameter_value_record.raw_candidate_count == 7
        assert parameter_value_record.deduplicated_candidate_count == (
            parameter_value_record.row_count
        )
        for command_name, record in recorded.items():
            row = _row(report, command_name)
            expected_rows = sum(
                context_record.row_count
                for (_context, recorded_command), context_record
                in context_records.items()
                if recorded_command == command_name
                and context_record.status
                in {
                    ContextTrainingStatus.INCLUDED,
                    ContextTrainingStatus.INCLUDED_FALLBACK,
                }
            )
            assert row.row_count == expected_rows
            assert row.seed_count == record.seed_utterance_count
            assert row.contexts, f"{command_name} should be a label in some context"

        # The application command is the developer's, the CME ones are not.
        assert _row(report, APP_COMMAND).kind is tr.CommandKind.APPLICATION
        assert _row(report, FRAMEWORK_COMMAND).kind is tr.CommandKind.FRAMEWORK
        assert _row(report, WILDCARD_LABEL).kind is tr.CommandKind.RESERVED
        assert _row(report, WILDCARD_LABEL).row_count == 0
        assert _row(report, PARAMETER_VALUE_LABEL).row_count == (
            parameter_value_record.row_count
        )

        # A successful run produces rows well above the structural floor.
        assert _row(report, APP_COMMAND).row_count >= tr.DEFAULT_MIN_TRAINING_ROWS

        text_path, json_path = tr.write_report(workflow_path, report)
        assert os.path.isfile(text_path) and os.path.isfile(json_path)
        assert "TRAINING DATA REPORT" in tr.format_report(report)
    finally:
        fastworkflow.init(previous)
        if os.path.isdir("./___workflow_contexts"):
            shutil.rmtree("./___workflow_contexts")
