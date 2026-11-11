"""The 'not here' class of each head must leave an artifact a human can read (bd fix-k0i.34).

Phase 9 retired the ancestor construction, so the old per-context ESCALATION BUDGET table
(ancestor rows, budget, coverage floor) described nothing the shipped trainer does. The report
now shows what the shipped class is: per head, the 'not here' rows trained and the foreign
commands they were sampled from. A head with no 'not here' class (the internal workflow) says
so instead of printing an empty table.

No mocks (`.cursor/rules/testing_rules.mdc`): the provenance these tests read is written by
the real `train.determinism.ProvenanceRecorder` through the same `record_context` call the
trainer makes in `model_pipeline_training._record_wildcard_context_training`, and read back
by the real `training_report.build_report`.
"""

import json
from pathlib import Path

import pytest

from fastworkflow.nlu_labels import WILDCARD_LABEL
from fastworkflow.train import determinism, training_report
from fastworkflow.train.determinism import ContextTrainingStatus

INTERNAL_CME_REASON = "no escalation class (the internal CME has no wildcard class)"


def _record_a_trained_context(
    recorder: determinism.ProvenanceRecorder,
    context_name: str,
    command_name: str,
    row_count: int,
) -> None:
    """Record one ordinary command's labelled rows, so the report has a table to sit in."""
    recorder.record(
        determinism.UtteranceProvenance(
            command_name=command_name,
            seed=1234,
            seed_utterance_count=9,
            generated_count=row_count - 9,
            final_count=row_count,
        )
    )
    recorder.record_context(
        context_name=context_name,
        command_name=command_name,
        status=ContextTrainingStatus.INCLUDED,
        row_count=row_count,
    )


@pytest.fixture
def workflow_with_not_here_provenance(tmp_path: Path) -> Path:
    """A workflow whose provenance has one head with a 'not here' class and one without.

    `TodoItem` is a sparse head: nine 'not here' rows, one sampled from each of nine foreign
    commands. The root context is the internal-workflow shape, with no 'not here' class.
    """
    workflow = tmp_path / "not_here_workflow"
    workflow.mkdir()

    recorder = determinism.ProvenanceRecorder(str(workflow))
    _record_a_trained_context(recorder, "TodoItem", "TodoItem/description", 40)
    _record_a_trained_context(recorder, "*", "add_two_numbers", 40)

    recorder.record_context(
        context_name="TodoItem",
        command_name=WILDCARD_LABEL,
        status=ContextTrainingStatus.INCLUDED,
        row_count=9,
        reason="reserved escalation class",
        sampled_commands=9,
    )
    recorder.record_context(
        context_name="*",
        command_name=WILDCARD_LABEL,
        status=ContextTrainingStatus.SKIPPED_NO_UTTERANCES,
        row_count=0,
        reason=INTERNAL_CME_REASON,
    )
    recorder.save()
    return workflow


def test_the_not_here_summary_reaches_the_report_model(
    workflow_with_not_here_provenance: Path,
):
    """Each head's 'not here' rows and sampled foreign commands come back from the report."""
    report = training_report.build_report(str(workflow_with_not_here_provenance))
    heads = {head.context_name: head for head in report.not_here_heads}

    assert set(heads) == {"TodoItem", "*"}
    todo = heads["TodoItem"]
    assert todo.included is True
    assert todo.not_here_rows == 9
    assert todo.sampled_commands == 9


def test_a_head_with_no_not_here_class_is_reported_as_such(
    workflow_with_not_here_provenance: Path,
):
    """Absent is a legitimate answer; the internal workflow must say why it has none."""
    report = training_report.build_report(str(workflow_with_not_here_provenance))
    root = next(h for h in report.not_here_heads if h.context_name == "*")

    assert root.included is False
    assert root.not_here_rows == 0
    assert root.sampled_commands is None
    assert "internal CME" in (root.reason or "")


def test_the_human_report_prints_the_not_here_table(
    workflow_with_not_here_provenance: Path,
):
    """A developer reading the text report can see each head's not-here rows and commands."""
    report = training_report.build_report(str(workflow_with_not_here_provenance))
    rendered = training_report.format_report(report)

    assert "NOT-HERE SAMPLES (2)" in rendered
    todo_line = next(line for line in rendered.splitlines() if line.strip().startswith("TodoItem"))
    assert todo_line.split() == ["TodoItem", "9", "9"]
    assert "no 'not here' class" in rendered
    assert "internal CME" in rendered
    assert "ESCALATION BUDGET" not in rendered
    assert "ancestor" not in rendered.split("NOT-HERE SAMPLES", 1)[1]


def test_the_json_report_carries_the_not_here_heads_for_a_ci_reader(
    workflow_with_not_here_provenance: Path,
):
    report = training_report.build_report(str(workflow_with_not_here_provenance))
    payload = report.to_dict()

    assert payload["summary"]["not_here_heads"] == 2
    by_context = {entry["context_name"]: entry for entry in payload["not_here_heads"]}
    assert by_context["TodoItem"]["not_here_rows"] == 9
    assert by_context["TodoItem"]["sampled_commands"] == 9
    assert by_context["*"]["included"] is False


def test_a_workflow_without_not_here_provenance_prints_no_section(tmp_path: Path):
    """Workflows with no wildcard records must not grow an empty table."""
    workflow = tmp_path / "flat_workflow"
    workflow.mkdir()
    recorder = determinism.ProvenanceRecorder(str(workflow))
    _record_a_trained_context(recorder, "*", "add_two_numbers", 40)
    recorder.save()

    report = training_report.build_report(str(workflow))
    assert report.not_here_heads == []
    assert "NOT-HERE SAMPLES" not in training_report.format_report(report)


def test_the_sampled_count_survives_provenance_written_by_the_real_recorder(
    workflow_with_not_here_provenance: Path,
):
    """Guard the coupling: the report reads a field the recorder persists.

    `record_context` serialises with `exclude_none`, so an unset field is absent from the file
    rather than null. Reading it back through the shipped loader is what proves the two agree.
    """
    path = training_report.provenance_path(str(workflow_with_not_here_provenance))
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert persisted["context_training"]["TodoItem"][WILDCARD_LABEL]["sampled_commands"] == 9
    assert "sampled_commands" not in persisted["context_training"]["*"][WILDCARD_LABEL]

    report = training_report.build_report(str(workflow_with_not_here_provenance))
    root = next(h for h in report.not_here_heads if h.context_name == "*")
    assert root.sampled_commands is None
