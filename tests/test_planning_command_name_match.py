"""A planning command typed by its name is matched by name, before the classifier.

The defect: `CommandNamePrediction._predict_impl` builds its exact-name candidate set
from the routed commands only (`cme` names plus `get_command_names(context)`). A
planning command (a skill file that runs a plan) is absent from that set, so typing it
by name never exact-matches. Its words fall through to fuzzy and then to the context's
classifier, which was trained without planning labels and answers with some routed
command instead. The agent's `inspect_entity <request>` reached `open_reconciliation`
that way.

The fix puts the planning commands runnable in the context into the same exact-name
set, and into the inventory the foreign-context guard reads, so a planning command is
matched, guarded and walked up exactly as a routed command is.

Every test here is integration: a real library copy with a real `RoutingDefinition`,
the real `CommandNamePrediction`, and the real CME `wildcard` and `CommandExecutor`
dispatch. The only double is the classifier, which raises if it is ever consulted:
the claim under test is that a planning name never reaches it.
"""

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT,
    CommandNamePrediction,
)
from fastworkflow.command_executor import CommandExecutor, _resolve_planning_command_name
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import trained_command_labels
from fastworkflow.workflow_agent import planner_commands_text
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_intent_detection_foreign_context import (
    _refusing_router as refusing_router,
)
from tests.test_nested_planning_commands import BOOK_LOANS
from tests.test_planning_commands import recipe_env  # noqa: F401 - recipe_env is a fixture
from tests.test_step_helpers import LibraryRunner, _library_workflow
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixture
    initialized_fastworkflow,
)

INTENT_DETECTION = fastworkflow.NLUPipelineStage.INTENT_DETECTION
MISUNDERSTANDING = fastworkflow.NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION

#: A planning command owned by the root context, the shape of the live `inspect_entity`:
#: a request-level argument. Its one phase is a guard (`do_not_run`), so the run calls no
#: nested command: a root command run as a nested step fails in agent mode in this copy
#: (`library_hours` answers with a pydantic Output the executor refuses), which is not this
#: change and is not investigated here.
INSPECT_ENTITY_SOURCE = '''"""inspect_entity: the entity the request names, as one planned step."""
from pydantic import BaseModel, Field

from fastworkflow.planning import do_not_run


class Signature:
    """Inspect the entity the request names."""

    class Input(BaseModel):
        query: str = Field(description="the words that name the entity",
                           json_schema_extra={"copied_from_command_text": True})


class PlanGenerator:
    def check(self, input):
        return [do_not_run("library_hours")]

    def __result__(self, input):
        return f"Inspected {input.query}."
'''

COUNT_BOOKS = "Library/count_books"
ROOT_PLANNING = "inspect_entity"
FREE_TEXT = ("inspect_entity Open the Jesse Rodgers-ADM login and tell me three things: "
             "what permissions it carries, who owns it, and how risky it is.")


@pytest.fixture
def planning_name_env(tmp_path, initialized_fastworkflow, monkeypatch):
    """A library copy with Library/count_books, Book/loan_report and a root planning command.

    The startup command leaves the workflow in Library, the context the live agent was in.
    The classifier is replaced by a double that raises if consulted (see `refusing_router`).
    """
    folder = _library_workflow(
        tmp_path,
        extra_commands=("_commands/Library/count_books.py", "_commands/Book/loan_report.py"),
    )
    Path(folder, "_commands", f"{ROOT_PLANNING}.py").write_text(
        INSPECT_ENTITY_SOURCE, encoding="utf-8")
    RoutingRegistry.clear_registry()
    RoutingRegistry.get_definition(folder, load_cached=False)
    wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"pn_{uuid.uuid4().hex[:8]}")
    ctx = WorkflowExecutionContext(run_as_agent=True)
    ctx.bind_app_workflow(wf)
    ctx.push_active_workflow(wf)
    LibraryRunner(wf)(ctx, "startup")
    yield SimpleNamespace(ctx=ctx, wf=wf, folder=folder)
    ctx.pop_active_workflow()
    RoutingRegistry.clear_registry()


@pytest.fixture
def classifier_consulted(planning_name_env, monkeypatch):
    """Route every classifier consultation to a double that records it and raises."""
    consulted: list[str] = []
    monkeypatch.setattr(intent_detection, "CommandRouter", refusing_router(consulted))
    return consulted


@pytest.fixture
def predictor(planning_name_env):
    return CommandNamePrediction(planning_name_env.ctx._cme_workflow)


def _predict(predictor, context: str, utterance: str, stage=INTENT_DETECTION):
    # The clarification stage reads the pending utterance the wildcard stored on the CME
    # context; store it as the wildcard would, so the stage sees a real pending turn.
    predictor.cme_workflow.context["command"] = utterance
    nlu_trace: dict = {}
    output = predictor._predict_impl(context, utterance, stage, nlu_trace)
    return output, nlu_trace


# ---------------------------------------------------------------------------
# The exact-name layer: a planning command is matched by name in its own context
# ---------------------------------------------------------------------------

def test_a_planning_command_is_matched_by_name_in_its_own_context(predictor, classifier_consulted):
    output, trace = _predict(predictor, "Library", "count_books query=Jane Austen")

    assert output.command_name == COUNT_BOOKS
    assert output.is_cme_command is False
    assert trace["matcher_layer"] == "exact_prefix"
    assert classifier_consulted == []


def test_a_root_planning_command_is_matched_by_name_in_the_root_context(
    predictor, classifier_consulted
):
    output, trace = _predict(predictor, "*", f"{ROOT_PLANNING} query=Jesse Rodgers-ADM")

    assert output.command_name == ROOT_PLANNING
    assert trace["matcher_layer"] == "exact_prefix"
    assert classifier_consulted == []


def test_free_text_after_a_root_planning_name_still_matches_the_name(
    predictor, classifier_consulted
):
    """The live shape: the name, then the request in free words, no key=value."""
    output, trace = _predict(predictor, "*", FREE_TEXT)

    assert output.command_name == ROOT_PLANNING
    assert trace["matcher_layer"] == "exact_prefix"
    assert classifier_consulted == []


def test_the_misunderstanding_stage_matches_a_planning_name_too(predictor, classifier_consulted):
    output, trace = _predict(predictor, "*", FREE_TEXT, stage=MISUNDERSTANDING)

    assert output.command_name == ROOT_PLANNING
    assert trace["matcher_layer"] == "exact_prefix"
    assert classifier_consulted == []


# ---------------------------------------------------------------------------
# The foreign-context guard: a planning command owned elsewhere is named, not guessed
# ---------------------------------------------------------------------------

def test_a_root_planning_name_typed_in_another_context_names_its_owner(
    predictor, classifier_consulted
):
    output, trace = _predict(predictor, "Library", FREE_TEXT)

    assert output.command_name is None
    assert trace["matcher_layer"] == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
    assert trace["known_name_owner_contexts"] == ["*"]
    assert classifier_consulted == []


def test_the_parent_walk_from_library_reaches_the_root_planning_command(
    predictor, classifier_consulted
):
    """`Library -> *`, the loop `_commands/wildcard.py` runs while the name is None."""
    visited = []
    resolved = None
    for context in ("Library", "*"):
        output, trace = _predict(predictor, context, FREE_TEXT)
        visited.append((context, output.command_name, trace["matcher_layer"]))
        if output.command_name is not None:
            resolved = output.command_name
            break

    assert resolved == ROOT_PLANNING
    assert visited == [
        ("Library", None, MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT),
        ("*", ROOT_PLANNING, "exact_prefix"),
    ]
    assert classifier_consulted == []


def test_a_sibling_planning_command_is_named_after_its_owner(predictor, classifier_consulted):
    output, trace = _predict(predictor, "Library", "loan_report Dune")

    assert output.command_name is None
    assert trace["matcher_layer"] == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
    assert "Book" in trace["known_name_owner_contexts"]
    assert "Library" not in trace["known_name_owner_contexts"]
    assert classifier_consulted == []


def test_the_inventory_lists_planning_commands_with_their_owners(predictor):
    inventory = predictor.command_inventory()

    assert inventory[ROOT_PLANNING] == ("*",)
    assert inventory["count_books"] == ("Library",)
    assert inventory["loan_report"] == ("Book",)


# ---------------------------------------------------------------------------
# Not a classifier label, and routed commands behave as before
# ---------------------------------------------------------------------------

def test_planning_commands_are_classifier_labels_of_their_heads(planning_name_env):
    """Phase 10 step 1 (production): a root planning command labels the global head, and a scoped one
    labels the heads on its lookup path. The baseline arm keeps the pre-step-1 rule (no labels)."""
    from fastworkflow.model_pipeline_training import (
        PLANNING_HEADS_NOT_HERE,
        experiment_planning_heads,
        experiment_reserved_class_arm,
    )

    crd = RoutingRegistry.get_definition(planning_name_env.folder, load_cached=False)

    assert ROOT_PLANNING in trained_command_labels("*", crd)
    assert "count_books" in {label.split("/")[-1] for label in trained_command_labels("Library", crd)}
    with experiment_reserved_class_arm("sparse_1"), experiment_planning_heads(PLANNING_HEADS_NOT_HERE):
        assert ROOT_PLANNING not in trained_command_labels("*", crd)
        assert "count_books" not in {label.split("/")[-1]
                                     for label in trained_command_labels("Library", crd)}


def test_a_routed_command_still_resolves_by_name_in_its_context(predictor, classifier_consulted):
    output, trace = _predict(predictor, "Library", "find_book query=Dune")

    assert output.command_name == "Library/find_book"
    assert trace["matcher_layer"] == "exact_prefix"
    assert classifier_consulted == []


def test_a_routed_command_typed_in_the_root_is_still_foreign(predictor, classifier_consulted):
    output, trace = _predict(predictor, "*", "find_book query=Dune")

    assert output.command_name is None
    assert trace["known_name_owner_contexts"] == ["Library"]
    assert classifier_consulted == []


# ---------------------------------------------------------------------------
# End to end: the real wildcard, the real extraction hop, run_planning_command
# ---------------------------------------------------------------------------

def test_a_planning_name_typed_to_the_executor_runs_as_a_planning_command(
    planning_name_env, classifier_consulted
):
    """The agent's call, `execute_workflow_query` -> `invoke_command`, with no stand-in.

    The classifier double raises if consulted; the extraction hop is the real
    `ParameterExtraction` (key=value form, so no model is asked), and the executor reaches
    `run_planning_command` with the extracted arguments.
    """
    ctx = planning_name_env.ctx
    # Agent mode, as `_initialize_agent_functionality` sets it on a live session (the
    # key=value extraction only runs in agent mode).
    planning_name_env.wf.context["run_as_agent"] = True
    ctx._cme_workflow.context["run_as_agent"] = True
    # The root ('*') context, where the live `inspect_entity` is owned. The library copy's
    # root object is Library, so the parent walk never reaches '*' from there; the walk is
    # covered at the matcher level above.
    planning_name_env.wf.record = None
    out = CommandExecutor.invoke_command(ctx, f"{ROOT_PLANNING} query=Jesse Rodgers-ADM")

    assert classifier_consulted == []
    assert out.success is True
    assert out.command_name == ROOT_PLANNING
    assert out.command_parameters.query == "Jesse Rodgers-ADM"
    assert out.command_response.response == "Inspected Jesse Rodgers-ADM."
    assert ctx._frame_stack()[1:] == []


def test_a_library_planning_command_is_named_on_the_listing_and_resolves_by_its_name(request):
    # A _commands/ planning command: its listing line is the name a plan step calls it by.
    workflow = request.getfixturevalue("recipe_env").ctx.get_active_workflow()
    assert "- book_loans\n" in planner_commands_text(workflow)
    assert _resolve_planning_command_name(workflow.folderpath, "book_loans") == BOOK_LOANS
