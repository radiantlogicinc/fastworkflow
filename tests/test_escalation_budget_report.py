"""The escalation-budget decision must leave an artifact a human can read (bd fix-k0i.34).

"Why does this context's `wildcard` class have 87 rows?" is the question AR6 spent a spec
section on, because the reference workflow's numbers could not be reconstructed after the
run. The budget fields already reached `training_provenance.json`; nothing put them in
front of a person, so every other pipeline decision was auditable and this one was not.

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
def workflow_with_escalation_provenance(tmp_path: Path) -> Path:
    """A workflow whose provenance describes one budgeted context and one skipped one.

    The numbers are the shape the trainer produces: a context whose coverage floor bound
    the budget above its own row count, and a root context with no ancestors, where no
    escalation class is emitted at all.
    """
    workflow = tmp_path / "escalation_workflow"
    workflow.mkdir()

    recorder = determinism.ProvenanceRecorder(str(workflow))
    _record_a_trained_context(recorder, "TodoItem", "TodoItem/description", 40)
    _record_a_trained_context(recorder, "*", "add_two_numbers", 40)

    recorder.record_context(
        context_name="TodoItem",
        command_name=WILDCARD_LABEL,
        status=ContextTrainingStatus.INCLUDED,
        row_count=87,
        reason="reserved escalation class",
        own_row_count=40,
        raw_candidate_count=412,
        deduplicated_candidate_count=203,
        always_include_count=7,
        selected_budget=87,
        coverage_floor=87,
        coverage_floor_applied=True,
    )
    recorder.record_context(
        context_name="*",
        command_name=WILDCARD_LABEL,
        status=ContextTrainingStatus.SKIPPED_NO_UTTERANCES,
        row_count=0,
        reason="context has no non-local ancestor utterances",
        own_row_count=40,
        raw_candidate_count=0,
        deduplicated_candidate_count=0,
        always_include_count=0,
        selected_budget=None,
        coverage_floor=0,
        coverage_floor_applied=False,
    )
    recorder.save()
    return workflow


def test_the_budget_denominators_reach_the_report_model(
    workflow_with_escalation_provenance: Path,
):
    """Every number that chose the row count must be reconstructable from the report."""
    report = training_report.build_report(str(workflow_with_escalation_provenance))
    budgets = {budget.context_name: budget for budget in report.escalation_budgets}

    assert set(budgets) == {"TodoItem", "*"}
    todo = budgets["TodoItem"]
    assert todo.included is True
    assert todo.selected_rows == 87
    assert todo.own_rows == 40
    assert todo.selected_budget == 87
    assert todo.coverage_floor == 87
    assert todo.coverage_floor_applied is True
    assert todo.raw_candidate_rows == 412
    assert todo.deduplicated_candidate_rows == 203
    assert todo.always_include_rows == 7


def test_a_context_with_no_ancestors_is_reported_as_having_no_escalation_class(
    workflow_with_escalation_provenance: Path,
):
    """Absent is a legitimate answer, and "why is there no wildcard class here" needs one."""
    report = training_report.build_report(str(workflow_with_escalation_provenance))
    root = next(b for b in report.escalation_budgets if b.context_name == "*")

    assert root.included is False
    assert root.selected_rows == 0
    assert "no non-local ancestor utterances" in (root.reason or "")


def test_the_human_report_prints_the_budget_and_why_it_bound(
    workflow_with_escalation_provenance: Path,
):
    """The finding is specifically that `selected_budget` / `coverage_floor` never printed.

    A developer reading the text report after the output has scrolled away must be able to
    answer the AR6 question without opening the JSON.
    """
    report = training_report.build_report(str(workflow_with_escalation_provenance))
    rendered = training_report.format_report(report)

    assert "ESCALATION BUDGET (2)" in rendered
    assert "TodoItem" in rendered
    assert "87" in rendered
    assert "412/203" in rendered
    assert "coverage floor bound the budget" in rendered
    assert "no escalation class" in rendered
    assert WILDCARD_LABEL in rendered


def test_the_json_report_carries_the_budgets_for_a_ci_reader(
    workflow_with_escalation_provenance: Path,
):
    report = training_report.build_report(str(workflow_with_escalation_provenance))
    payload = report.to_dict()

    assert payload["summary"]["escalation_contexts"] == 2
    by_context = {
        entry["context_name"]: entry for entry in payload["escalation_budgets"]
    }
    assert by_context["TodoItem"]["selected_budget"] == 87
    assert by_context["TodoItem"]["coverage_floor"] == 87


def test_a_workflow_without_escalation_provenance_prints_no_budget_section(
    tmp_path: Path,
):
    """Older artifacts, and workflows with no ancestor contexts, must not grow an empty table."""
    workflow = tmp_path / "flat_workflow"
    workflow.mkdir()
    recorder = determinism.ProvenanceRecorder(str(workflow))
    _record_a_trained_context(recorder, "*", "add_two_numbers", 40)
    recorder.save()

    report = training_report.build_report(str(workflow))
    assert report.escalation_budgets == []
    assert "ESCALATION BUDGET" not in training_report.format_report(report)


def test_the_budget_section_survives_provenance_written_by_the_real_trainer(
    workflow_with_escalation_provenance: Path,
):
    """Guard the coupling: the report reads fields the recorder chose to persist.

    `record_context` serialises with `exclude_none`, so a field the trainer leaves unset is
    absent from the file rather than null. Reading it back through the shipped loader is
    what proves the two halves still agree.
    """
    path = training_report.provenance_path(str(workflow_with_escalation_provenance))
    persisted = json.loads(path.read_text(encoding="utf-8"))
    wildcard_record = persisted["context_training"]["TodoItem"][WILDCARD_LABEL]

    assert wildcard_record["selected_budget"] == 87
    assert "coverage_floor" in wildcard_record
    # The root context left `selected_budget` unset, so it must be missing rather than 0.
    root_record = persisted["context_training"]["*"][WILDCARD_LABEL]
    assert "selected_budget" not in root_record

    report = training_report.build_report(str(workflow_with_escalation_provenance))
    root = next(b for b in report.escalation_budgets if b.context_name == "*")
    assert root.selected_budget is None
