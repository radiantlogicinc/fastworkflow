"""Integration tests for the fuzzy pre-match tie guard in intent detection.

Covers fix-0xn. Before the guard, `CommandNamePrediction.predict` took
`best_matched_commands[0]` from a list that `find_best_matches` documents as
*all* candidates tying for the minimum distance. Commands sharing a prefix tie
at distance 0.0, because scoring compares only the leading `len(input)`
characters, and `command_name_dict` is built by iterating a set — so the pick
varied between processes and bypassed both the classifier and the ambiguity
prompt.

Real components throughout: a real `RoutingDefinition` for a real test workflow
and the real `predict` path. Only the classifier result is injected, the same
seam `test_wildcard_escalation.py` uses, so no trained model artifacts are
needed.
"""

import logging
import os
import shutil
from io import StringIO

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingDefinition
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    CommandNamePrediction,
)
from fastworkflow.utils.fuzzy_match import find_best_matches

# tests/todo_list_workflow exposes two commands sharing the 'add_child_todo'
# prefix, which is the ordinary shape of a generated command set.
TIED_PREFIX = "add_child_todo"
TIED_COMMANDS = {"add_child_todoitem", "add_child_todolist"}

FUZZY_THRESHOLD = 0.3  # the literal at the call site under test


@pytest.fixture(scope="module")
def todolist_command_names() -> dict[str, str]:
    """Real short-name -> fully-qualified map for the TodoList context."""
    workflow_path = os.path.join(os.path.dirname(__file__), "todo_list_workflow")
    routing_definition = RoutingDefinition.build(workflow_path)
    valid = set(routing_definition.get_command_names("TodoList"))
    assert valid, "todo_list_workflow exposes no commands in 'TodoList'"
    return {fqn.split("/")[-1]: fqn for fqn in valid}


@pytest.fixture
def todolist_workflows(tmp_path):
    """A real app workflow plus the real CME workflow bound to it."""
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "todo_list_workflow"),
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info",
            "___workflow_contexts",
            "___convo_info",
            "__pycache__",
        ),
    )
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=str(workflow_path),
        workflow_id_str=f"fuzzy-tie-{tmp_path.name}",
    )
    cme_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        ),
        parent_workflow_id=app_workflow.id,
        workflow_context={"app_workflow": app_workflow},
    )
    return app_workflow, cme_workflow


def _router_returning(predictions, modelpipeline=...):
    class PredictingRouter:
        def __init__(self, model_artifact_path):
            # `store_utterance_cache` only computes an embedding when the
            # pipeline is not None, so passing None keeps the clarification
            # stages reachable without loading a transformer.
            self.modelpipeline = self if modelpipeline is ... else modelpipeline

        def predict(self, command):
            return predictions

    return PredictingRouter


# ---------------------------------------------------------------------------
# The precondition: the tie is real, and it is reachable with real names
# ---------------------------------------------------------------------------


def test_prefix_sharing_commands_tie_at_distance_zero(todolist_command_names):
    matches, distance = find_best_matches(
        TIED_PREFIX, todolist_command_names.keys(), threshold=FUZZY_THRESHOLD)
    assert set(matches) == TIED_COMMANDS
    assert distance == 0.0


def test_tie_is_also_reachable_from_a_spaced_utterance(todolist_command_names):
    """A user typing words, not a command name, reaches the same tie: predict
    replaces spaces with underscores before matching."""
    utterance = "add child todo"
    matches, distance = find_best_matches(
        utterance.replace(" ", "_"), todolist_command_names.keys(),
        threshold=FUZZY_THRESHOLD)
    assert set(matches) == TIED_COMMANDS
    assert distance == 0.0


# ---------------------------------------------------------------------------
# fix-0xn — INTENT_DETECTION defers to the classifier instead of guessing
# ---------------------------------------------------------------------------


def test_tie_defers_to_the_classifier_rather_than_picking_one(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """The whole point: a tied pre-match must not commit to one of the tied
    commands. Here the classifier is confident about a third command, and that
    is what must win."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        intent_detection, "CommandRouter",
        _router_returning(["TodoList/get_properties"]))

    result = CommandNamePrediction(cme_workflow).predict(
        "TodoList", TIED_PREFIX, fastworkflow.NLUPipelineStage.INTENT_DETECTION)

    assert result.command_name == "TodoList/get_properties"
    assert result.error_msg is None


def test_tie_reaches_the_ambiguity_prompt_when_the_classifier_is_unsure(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """Deferring restores the prompt that exists to ask which one was meant."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        intent_detection, "CommandRouter",
        _router_returning(
            ["TodoList/add_child_todoitem", "TodoList/add_child_todolist"]))

    result = CommandNamePrediction(cme_workflow).predict(
        "TodoList", TIED_PREFIX, fastworkflow.NLUPipelineStage.INTENT_DETECTION)

    assert result.command_name is None
    assert result.error_msg is not None
    assert "add_child_todoitem" in result.error_msg
    assert "add_child_todolist" in result.error_msg


def test_tie_is_logged(todolist_workflows, monkeypatch, setup_test_environment):
    """The pre-guard behaviour was invisible; a discarded tie must not be."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        intent_detection, "CommandRouter",
        _router_returning(["TodoList/get_properties"]))

    log_output = StringIO()
    log_handler = logging.StreamHandler(log_output)
    intent_detection.logger.addHandler(log_handler)
    try:
        CommandNamePrediction(cme_workflow).predict(
            "TodoList", TIED_PREFIX,
            fastworkflow.NLUPipelineStage.INTENT_DETECTION)
    finally:
        intent_detection.logger.removeHandler(log_handler)

    warning = log_output.getvalue()
    assert "Fuzzy pre-match tied across" in warning
    assert "add_child_todoitem" in warning
    assert "Deferring to the classifier" in warning


# ---------------------------------------------------------------------------
# Positive controls: the guard must not disarm the pre-match
# ---------------------------------------------------------------------------


def test_unique_fuzzy_match_still_routes_without_the_classifier(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """An unambiguous near-miss must still short-circuit the classifier. The
    stub router would return a different command if it were consulted."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        intent_detection, "CommandRouter",
        _router_returning(["TodoList/get_properties"]))

    result = CommandNamePrediction(cme_workflow).predict(
        "TodoList", "add_child_todoitm",
        fastworkflow.NLUPipelineStage.INTENT_DETECTION)

    assert result.command_name == "TodoList/add_child_todoitem"
    assert result.error_msg is None


def test_exact_command_name_still_routes(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """One of the tied names, given in full, is an exact dict hit and never
    reaches the fuzzy branch at all."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        intent_detection, "CommandRouter",
        _router_returning(["TodoList/get_properties"]))

    result = CommandNamePrediction(cme_workflow).predict(
        "TodoList", "add_child_todolist",
        fastworkflow.NLUPipelineStage.INTENT_DETECTION)

    assert result.command_name == "TodoList/add_child_todolist"


def test_clarification_stage_tie_still_resolves_to_one_of_the_tied_commands(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """fix-0xn's open half, pinned so the remaining decision gets made on
    purpose rather than inherited.

    The clarification stages have no classifier to defer to — fuzzy matching is
    the only mechanism there — so the guard is scoped to INTENT_DETECTION and a
    tie in these stages still resolves to whichever candidate came first. When
    that half is decided, this test is the one to change."""
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        intent_detection, "CommandRouter",
        _router_returning(["TodoList/get_properties"], modelpipeline=None))

    # These stages write the resolved name to the clarification cache, which
    # reads the in-flight utterance from the CME context.
    workflow_context = cme_workflow.context
    workflow_context["command"] = TIED_PREFIX
    cme_workflow.context = workflow_context

    result = CommandNamePrediction(cme_workflow).predict(
        "TodoList", TIED_PREFIX,
        fastworkflow.NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION)

    assert result.command_name in {f"TodoList/{name}" for name in TIED_COMMANDS}
