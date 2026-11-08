"""Integration tests for the internal consultation workflow (fix-kk46.5)."""
from __future__ import annotations

import json
import os

import pytest

import fastworkflow
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.review_engine import Change
from fastworkflow.review_proposals import SubTask, SubTaskProposal
from fastworkflow.train import selective_training

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

CONSULTATION = fastworkflow.get_internal_workflow_path("consultation")


def _workflow():
    return fastworkflow.Workflow.create(
        CONSULTATION,
        workflow_id_str="consultation-test",
        workflow_context={"NLU_Pipeline_Stage": fastworkflow.NLUPipelineStage.INTENT_DETECTION},
    )


def test_consultation_command_directory_has_no_core_commands():
    directory = CommandDirectory.load(CONSULTATION)
    assert "IntentDetection/go_up" not in directory.map_command_2_metadata
    assert "ErrorCorrection/abort" not in directory.map_command_2_metadata
    names = {key.split("/")[-1] for key in directory.map_command_2_metadata}
    assert names >= {"confirm", "decline", "drop", "add", "look_up", "edit"}


def test_app_training_does_not_include_consultation_contexts():
    library = os.path.join(
        os.path.dirname(fastworkflow.__file__), "examples", "library"
    )
    contexts = selective_training.contexts_for_training(library)
    assert "ProposalReview" not in contexts


def test_consultation_training_targets_proposal_review_only():
    contexts = selective_training.contexts_for_training(CONSULTATION)
    assert contexts == {"ProposalReview"}


@pytest.mark.parametrize(
    ("command_name", "utterance", "field", "expected"),
    [
        ("ProposalReview/confirm", "sounds good to me", "decision", "confirm"),
        ("ProposalReview/decline", "never mind", "decision", "decline"),
    ],
)
def test_confirm_and_decline_return_typed_outputs(command_name, utterance, field, expected):
    wf = _workflow()
    out = CommandExecutor.perform_action(
        wf,
        fastworkflow.Action(command_name=command_name, command=utterance),
    )
    assert out.success
    body = out.command_response.output
    assert getattr(body, field) == expected


def test_drop_returns_items():
    from fastworkflow.utils import python_utils

    drop_mod = python_utils.get_module(
        os.path.join(CONSULTATION, "_commands", "ProposalReview", "drop.py"),
        CONSULTATION,
    )
    wf = _workflow()
    params = drop_mod.Signature.Input(items=["2", "renewal step"])
    response = drop_mod.ResponseGenerator()(wf, "ProposalReview/drop", params)
    assert response.output.items == ["2", "renewal step"]


def test_edit_uses_scripted_reader_without_calling_a_model():
    proposal = SubTaskProposal(items=[SubTask(goal="reserve a book")])
    edits = [Change(index=1, field="goal", value="renew the card")]

    def reader(text: str, prop) -> list:
        assert text == "make it about renewing"
        assert prop is proposal
        return edits

    wf = _workflow()
    wf.context["consultation_proposal"] = proposal
    wf.context["consultation_edit_reader"] = reader
    out = CommandExecutor.perform_action(
        wf,
        fastworkflow.Action(
            command_name="ProposalReview/edit",
            command="make it about renewing",
        ),
    )
    assert out.success
    payload = out.command_response.output
    assert payload.reply == "make it about renewing"
    assert payload.edits == edits


def test_edit_without_context_keys_returns_no_edits():
    wf = _workflow()
    out = CommandExecutor.perform_action(
        wf,
        fastworkflow.Action(
            command_name="ProposalReview/edit",
            command="change the author to Lee",
        ),
    )
    assert out.success
    assert out.command_response.output.reply == "change the author to Lee"
    assert out.command_response.output.edits == []


def test_decline_seeds_include_never_mind_and_cancel():
    from fastworkflow.utils import python_utils

    confirm_mod = python_utils.get_module(
        os.path.join(CONSULTATION, "_commands", "ProposalReview", "confirm.py"),
        CONSULTATION,
    )
    decline_mod = python_utils.get_module(
        os.path.join(CONSULTATION, "_commands", "ProposalReview", "decline.py"),
        CONSULTATION,
    )
    decline_seeds = set(decline_mod.Signature.plain_utterances)
    confirm_seeds = set(confirm_mod.Signature.plain_utterances)
    assert "never mind" in decline_seeds
    assert "cancel" in decline_seeds
    assert "never mind" not in confirm_seeds
    assert "cancel" not in confirm_seeds


def _load_held_out_corpus() -> list[dict]:
    corpus_path = os.path.join(CONSULTATION, "heldout_reply_corpus.jsonl")
    rows = []
    with open(corpus_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _plain_utterance_seeds() -> dict[str, set[str]]:
    from fastworkflow.utils import python_utils

    seeds: dict[str, set[str]] = {}
    for command in ("confirm", "decline", "drop", "add", "look_up", "edit"):
        module = python_utils.get_module(
            os.path.join(CONSULTATION, "_commands", "ProposalReview", f"{command}.py"),
            CONSULTATION,
        )
        seeds[command] = {text.strip().lower() for text in module.Signature.plain_utterances}
    return seeds


def test_held_out_corpus_labels_are_valid_commands():
    rows = _load_held_out_corpus()
    assert len(rows) >= 36
    by_label: dict[str, list[str]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row["text"])
    assert set(by_label) == {
        "confirm",
        "decline",
        "drop",
        "add",
        "look_up",
        "edit",
    }
    assert all(len(texts) >= 6 for texts in by_label.values())
    seeds = _plain_utterance_seeds()
    for row in rows:
        assert row["text"].strip().lower() not in seeds[row["label"]]


def _proposal_review_model_dir() -> str:
    model_dir = os.path.join(CONSULTATION, "___command_info", "ProposalReview")
    threshold = os.path.join(model_dir, "threshold.json")
    if not os.path.isfile(threshold):
        pytest.skip("consultation ProposalReview head is not trained")
    return model_dir


def _predict_command_short_name(router, text: str) -> str:
    labels, details = router.predict_with_details(text)
    if not labels:
        return ""
    if details.get("confident") or len(labels) == 1:
        label = labels[0]
    else:
        label = labels[0]
    return label.split("/")[-1] if "/" in label else label


def test_trained_head_scores_held_out_corpus():
    from fastworkflow.model_pipeline_training import CommandRouter
    from fastworkflow.train.artifact_versioning import require_construction_record

    require_construction_record(CONSULTATION)
    router = CommandRouter(_proposal_review_model_dir())
    rows = _load_held_out_corpus()

    per_class: dict[str, list[bool]] = {name: [] for name in (
        "confirm", "decline", "drop", "add", "look_up", "edit"
    )}
    for row in rows:
        predicted = _predict_command_short_name(router, row["text"])
        per_class[row["label"]].append(predicted == row["label"])

    rates = {label: (sum(hits) / len(hits) if hits else 0.0) for label, hits in per_class.items()}
    overall = sum(row["label"] == _predict_command_short_name(router, row["text"]) for row in rows) / len(rows)

    # Shipped head: majority of held-out replies should route to the right command.
    assert overall >= 0.70, f"held-out overall={overall:.3f}, per-class={rates}"
    assert all(rate >= 0.50 for rate in rates.values()), f"per-class={rates}"
