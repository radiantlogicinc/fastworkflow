"""A known command name is never answered by a foreign context.

The defect: the exact-command-name matcher in
`CommandNamePrediction._predict_impl` is scoped to the CURRENT context's command
set, so a command the workflow owns *somewhere else* -- typed verbatim, with its
parameters -- is invisible to the exact and fuzzy layers and is adjudicated by
this context's classifier instead. Measured on a large multi-context workflow,
that produced a hundred misroutes across thirty-one runs, most of them silent:
`fetch_result_page <handle>O9</handle> <contains>Alan Cooper</contains>` was
answered `Directory/find_permission` -> "No permissions found.", which reads as
a true negative.

The guard: after the context-scoped exact-prefix miss and BEFORE fuzzy, cache or
classifier, the first token is tested against the workflow's FULL inventory. A
real command name that this context does not own returns `command_name=None` at
once, which is already the signal that drives the parent-chain walk in
`_commands/wildcard.py`.

Two levels, both offline and deterministic:

* against a captured multi-context inventory
  (`tests/fixtures/multi_context_routing_inventory.json`, a copy of a real
  workflow's `___command_info/routing_definition.json`), table-driven from the
  observed misroutes. The real `predict` path runs; only the command inventory
  is supplied, and the classifier double raises if it is ever consulted;
* against the real `tests/todo_list_workflow` with its real `RoutingDefinition`,
  so the guard is also exercised end to end with no inventory injected at all.

No trained artifact is read or written by any of this, and no model is loaded.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction._commands import (
    wildcard as wildcard_module,
)
from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
    MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT,
    CommandNamePrediction,
    foreign_context_hint,
)
from tests.todo_list_workflow.application.todo_manager import TodoListManager

INVENTORY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "multi_context_routing_inventory.json"
)

INTENT_DETECTION = fastworkflow.NLUPipelineStage.INTENT_DETECTION


# ---------------------------------------------------------------------------
# The misroute table (investigation section 4.2), as rows
# ---------------------------------------------------------------------------

#: (issuing context, utterance as the agent wrote it, owning contexts).
#: Every row is an observed live misroute except where marked: the utterance is
#: reproduced with its parameters because the exact-prefix matcher splits on
#: space and '(' only -- never on '<' -- which is why the XML tail is part of
#: what the classifier was given.
MISROUTE_TABLE = [
    # The original report: four occurrences across four runs, each "No permissions found."
    ("Permission", "fetch_result_page <handle>O9</handle> <contains>Alan Cooper</contains>",
     ["*"]),
    ("DirectoryExplorer", "fetch_result_page <handle>O6</handle> <contains>Alan Cooper</contains>",
     ["*"]),
    # 7x / 2 runs, loud only because fetch_result_page has a required parameter.
    ("DirectoryExplorer", "show_holders <filter>Alan Cooper</filter>", ["Permission"]),
    ("Account", "show_holders <filter>Brandon Miller</filter>", ["Permission"]),
    ("Identity", "show_holders", ["Permission"]),
    # 12x / 4 runs, SILENT: answered with the directory-wide 64-permission catalogue.
    ("Identity", "list_permissions", ["Account"]),
    # 4x / 2 runs, SILENT.
    ("ControlsMonitor", "list_accounts", ["Application", "Identity", "Repository"]),
    ("DirectoryExplorer", "list_accounts", ["Application", "Identity", "Repository"]),
    # 4x / 3 runs, SILENT.
    ("DirectoryExplorer", "list_findings", ["ControlsMonitor"]),
    ("Account", "list_findings", ["ControlsMonitor"]),
    # 28x / 13 runs, SILENT -- that bucket came from the embedding cache, but the
    # name is foreign to the issuing context either way and the guard returns
    # before the cache is consulted.
    ("*", "open_identity_by_uid <identity_uid>81b86cf622ed7f1f3be7b964852e0f42</identity_uid>",
     ["DirectoryExplorer"]),
    # 1x, SILENT: answered by who_am_i.
    ("*", "who_has_access_to <resource>Active Directory</resource>",
     ["Directory", "DirectoryExplorer"]),
    # 1x, SILENT: answered by ControlCatalog/list_controls.
    ("Account", "list_entitlements", ["Identity"]),
    # 1x, SILENT: answered by open_reconciliation.
    ("ControlsMonitor", "open_account_by_uid <account_uid>3f2a</account_uid>",
     ["DirectoryExplorer"]),
]

#: First tokens that name no command of this workflow. `list_holders` and
#: `show_portrait` are names the agent invented; `search_memory` is the DSPy
#: agent-side tool, which the workflow has never heard of and which was answered
#: "No permissions found." three times. All three must still reach fuzzy and the
#: classifier -- the guard is about names the workflow owns, not about names it
#: does not.
#: The third element is the label the classifier double returns: the real router
#: returns a label the context owns, and `resolve_fully_qualified_command_name`
#: looks it up in that context's dict, so an invented label would fail for a
#: reason that has nothing to do with this guard.
NOT_A_COMMAND_ANYWHERE = [
    ("DirectoryExplorer", "list_holders <filter>Alan Cooper</filter>", "find_permission"),
    ("Account", "search_memory <question>who holds AD admin rights</question>",
     "show_metadata"),
    ("Permission", "show_portrait", "show_metadata"),
    ("DirectoryExplorer", "which identities hold the compliance officer permission",
     "find_permission"),
]


# ---------------------------------------------------------------------------
# Doubles: the real predict path, an injected inventory, no model
# ---------------------------------------------------------------------------

def _refusing_router(recorder):
    """A `CommandRouter` that records any consultation and refuses to answer.

    The guard's whole claim is that nothing below it runs. A double that returns
    a plausible label could not tell "the guard fired" from "the classifier
    happened to agree", so this one raises instead.
    """
    class RefusingRouter:
        def __init__(self, model_artifact_path):
            self.model_artifact_path = model_artifact_path
            # `cache_match` computes an embedding only when this is not None, so
            # None keeps the embedding cache off without loading a transformer.
            self.modelpipeline = None

        def predict(self, command):
            recorder.append(command)
            raise AssertionError(
                f"the classifier was consulted for {command!r}; the guard must "
                "return before any of fuzzy / cache / classifier runs"
            )

        def predict_with_details(self, command):
            return self.predict(command)

    return RefusingRouter


def _labelling_router(label):
    class LabellingRouter:
        def __init__(self, model_artifact_path):
            self.modelpipeline = None

        def predict(self, command):
            return self.predict_with_details(command)[0]

        def predict_with_details(self, command):
            return [label], {}

    return LabellingRouter


class _InventoryRoutingDefinition:
    """The parts of `RoutingDefinition` the prediction path reads.

    `contexts` is what the new inventory helper reads; `get_command_names` is
    what the per-context candidate set is built from. Supplying one object for
    both is the point: a test that injected a different inventory to the guard
    than to the matcher could pass while the two disagreed in production.
    """

    def __init__(self, contexts: dict[str, list[str]]):
        self.contexts = contexts

    def get_command_names(self, context: str) -> list[str]:
        if context not in self.contexts:
            raise ValueError(f"Context '{context}' not found in the workflow.")
        return [c for c in self.contexts[context] if c != "wildcard"]


@pytest.fixture(scope="module")
def ido_contexts() -> dict[str, list[str]]:
    return json.loads(INVENTORY_FIXTURE.read_text())["contexts"]


@pytest.fixture
def todolist_workflows(tmp_path):
    """A real app workflow plus the real CME workflow bound to it."""
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "todo_list_workflow"),
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
        ),
    )
    app_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=str(workflow_path),
        workflow_id_str=f"foreign-context-{tmp_path.name}",
    )
    cme_workflow = fastworkflow.Workflow.create(
        workflow_folderpath=fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        ),
        parent_workflow_id=app_workflow.id,
        workflow_context={"app_workflow": app_workflow},
    )
    return app_workflow, cme_workflow


@pytest.fixture
def ido_predictor(todolist_workflows, ido_contexts, monkeypatch):
    """A predictor whose app workflow reports the real IDO command inventory.

    The CME workflow keeps its real routing definition; only the app workflow's
    is replaced, which is exactly the seam the guard reads.
    """
    _, cme_workflow = todolist_workflows
    predictor = CommandNamePrediction(cme_workflow)
    app_folderpath = predictor.app_workflow_folderpath
    real_get_definition = fastworkflow.RoutingRegistry.get_definition
    ido_definition = _InventoryRoutingDefinition(ido_contexts)

    def get_definition(workflow_folderpath, *args, **kwargs):
        if str(workflow_folderpath) == str(app_folderpath):
            return ido_definition
        return real_get_definition(workflow_folderpath, *args, **kwargs)

    monkeypatch.setattr(
        fastworkflow.RoutingRegistry, "get_definition", staticmethod(get_definition)
    )
    return predictor


# ---------------------------------------------------------------------------
# The fixture is the real inventory
# ---------------------------------------------------------------------------

def test_fixture_carries_the_topology_the_defect_needs(ido_contexts):
    """`*` is a peer context that nothing inherits -- the structural exposure."""
    assert "fetch_result_page" in ido_contexts["*"]
    assert not any(
        "fetch_result_page" in commands
        for context, commands in ido_contexts.items() if context != "*"
    )
    assert "Permission/show_holders" in ido_contexts["Permission"]
    assert "Permission/show_holders" not in ido_contexts["DirectoryExplorer"]


# ---------------------------------------------------------------------------
# The table: None from the foreign context, resolved from the owner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "context,utterance,owners", MISROUTE_TABLE,
    ids=[f"{c}:{u.split()[0]}" for c, u, _ in MISROUTE_TABLE],
)
def test_known_name_is_not_answered_by_a_foreign_context(
    ido_predictor, monkeypatch, setup_test_environment, context, utterance, owners
):
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))

    nlu_trace: dict = {}
    result = ido_predictor._predict_impl(
        context, utterance, INTENT_DETECTION, nlu_trace)

    assert result.command_name is None
    assert result.error_msg is None
    assert nlu_trace["matcher_layer"] == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
    assert nlu_trace["known_name_foreign_context"] is True
    assert sorted(nlu_trace["known_name_owner_contexts"]) == sorted(owners)
    assert not consulted


@pytest.mark.parametrize(
    "context,utterance,owners", MISROUTE_TABLE,
    ids=[f"{c}:{u.split()[0]}" for c, u, _ in MISROUTE_TABLE],
)
def test_the_owning_context_still_resolves_the_same_name(
    ido_predictor, monkeypatch, setup_test_environment, context, utterance, owners
):
    """The other half of the walk: where the name lives, it resolves exactly."""
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))
    token = utterance.split(" ", 1)[0]

    for owner in owners:
        nlu_trace: dict = {}
        result = ido_predictor._predict_impl(
            owner, utterance, INTENT_DETECTION, nlu_trace)
        assert result.command_name.split("/")[-1] == token, (owner, token)
        assert nlu_trace["matcher_layer"] == "exact_prefix"
    assert not consulted


def test_the_walk_lands_the_bead_case_on_the_root_context(
    ido_predictor, monkeypatch, setup_test_environment
):
    """`Permission -> DirectoryExplorer -> *`, the chain the misrouted call walked.

    This is the loop `_commands/wildcard.py` runs: keep asking the parent
    while the name is None. Without the foreign-context guard it stopped at the
    first context whose classifier produced a label -- `Directory/find_permission`,
    at 0.284.
    """
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router([]))
    utterance = "fetch_result_page <handle>O9</handle> <contains>Alan Cooper</contains>"

    resolved = None
    visited = []
    for context in ("Permission", "DirectoryExplorer", "*"):
        nlu_trace: dict = {}
        output = ido_predictor._predict_impl(
            context, utterance, INTENT_DETECTION, nlu_trace)
        visited.append((context, output.command_name, nlu_trace["matcher_layer"]))
        if output.command_name is not None:
            resolved = output.command_name
            break

    assert resolved == "fetch_result_page"
    assert visited == [
        ("Permission", None, MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT),
        ("DirectoryExplorer", None, MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT),
        ("*", "fetch_result_page", "exact_prefix"),
    ]


# ---------------------------------------------------------------------------
# The separator: any whitespace ends the command name, not a space alone
# ---------------------------------------------------------------------------

#: The separator between a command name and its tail. An agent writes
#: `list_permissions\n<scope>all</scope>` as readily as it writes the same thing
#: with a space, and on a space-only split the whole line becomes the tentative
#: name: it matches nothing, the refusal guard never fires, and a context that
#: cannot run the command adjudicates it anyway -- the silent misroute this
#: guard exists to stop. The last two cases split either way and are here as
#: the controls that say so.
SEPARATORS = [
    ("newline", "\n"),
    ("tab", "\t"),
    ("carriage return", "\r"),
    ("newline after space", " \n"),
    ("two spaces", "  "),
]


@pytest.mark.parametrize("name,separator", SEPARATORS, ids=[n for n, _ in SEPARATORS])
def test_any_whitespace_ends_the_command_name_for_the_guard(
    ido_predictor, monkeypatch, setup_test_environment, name, separator
):
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))

    # `list_permissions` from Identity is row 6 of the misroute table: owned by
    # Account, 12 silent occurrences across 4 runs.
    nlu_trace: dict = {}
    result = ido_predictor._predict_impl(
        "Identity", f"list_permissions{separator}<scope>all</scope>",
        INTENT_DETECTION, nlu_trace)

    assert result.command_name is None
    assert nlu_trace["matcher_layer"] == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
    assert nlu_trace["known_name_foreign_context"] is True
    assert nlu_trace["known_name_owner_contexts"] == ["Account"]
    assert not consulted


@pytest.mark.parametrize("name,separator", SEPARATORS, ids=[n for n, _ in SEPARATORS])
def test_the_owner_still_resolves_the_name_across_separators(
    ido_predictor, monkeypatch, setup_test_environment, name, separator
):
    """The other half: the same parse must not break the owning context."""
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))

    nlu_trace: dict = {}
    result = ido_predictor._predict_impl(
        "Account", f"list_permissions{separator}<scope>all</scope>",
        INTENT_DETECTION, nlu_trace)

    assert result.command_name.split("/")[-1] == "list_permissions"
    assert nlu_trace["matcher_layer"] == "exact_prefix"
    assert not consulted


# ---------------------------------------------------------------------------
# Negative control: a token that is not a command anywhere is untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "context,utterance,label", NOT_A_COMMAND_ANYWHERE,
    ids=[f"{c}:{u.split()[0]}" for c, u, _ in NOT_A_COMMAND_ANYWHERE],
)
def test_a_token_that_is_not_a_command_still_reaches_the_lower_layers(
    ido_predictor, monkeypatch, setup_test_environment, context, utterance, label
):
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _labelling_router(label))

    nlu_trace: dict = {}
    result = ido_predictor._predict_impl(
        context, utterance, INTENT_DETECTION, nlu_trace)

    # Whatever the lower layers decide, they decided it -- the guard did not.
    assert nlu_trace["matcher_layer"] in {"fuzzy_prematch", "classifier"}
    assert "known_name_foreign_context" not in nlu_trace
    assert result.command_name is not None


def test_free_text_still_reaches_the_classifier_and_its_label_wins(
    ido_predictor, monkeypatch, setup_test_environment
):
    monkeypatch.setattr(
        intent_detection, "CommandRouter",
        _labelling_router("find_permission"))

    nlu_trace: dict = {}
    result = ido_predictor._predict_impl(
        "DirectoryExplorer", "who can read the payroll repository",
        INTENT_DETECTION, nlu_trace)

    assert nlu_trace["matcher_layer"] == "classifier"
    assert result.command_name == "Directory/find_permission"


def test_the_guard_is_scoped_to_intent_detection(
    ido_predictor, monkeypatch, setup_test_environment
):
    """Clarification stages match a constrained suggestion set, not this one.

    There, "the current context does not own this name" is the ordinary case
    rather than a misroute, so the guard must not fire and end the stage.
    """
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _labelling_router("show_metadata"))

    nlu_trace: dict = {}
    ido_predictor._predict_impl(
        "Permission", "fetch_result_page <handle>O9</handle>",
        fastworkflow.NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
        nlu_trace)

    assert nlu_trace["matcher_layer"] != MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
    assert "known_name_foreign_context" not in nlu_trace


# ---------------------------------------------------------------------------
# The same guard against a real RoutingDefinition, nothing injected
# ---------------------------------------------------------------------------

def test_real_workflow_foreign_name_returns_none(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """`create_todo_list` belongs to TodoListManager; TodoItem may not answer it."""
    _, cme_workflow = todolist_workflows
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))

    result = CommandNamePrediction(cme_workflow).predict(
        "TodoItem", "create_todo_list my list", INTENT_DETECTION)

    assert result.command_name is None
    assert result.error_msg is None
    assert not consulted


def test_real_workflow_owning_context_still_resolves_by_exact_prefix(
    todolist_workflows, monkeypatch, setup_test_environment
):
    _, cme_workflow = todolist_workflows
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))

    result = CommandNamePrediction(cme_workflow).predict(
        "TodoListManager", "create_todo_list my list", INTENT_DETECTION)

    assert result.command_name == "TodoListManager/create_todo_list"
    assert not consulted


def test_real_workflow_inherited_name_is_not_foreign(
    todolist_workflows, monkeypatch, setup_test_environment
):
    """Inheritance is ownership: TodoList has TodoItem as a base.

    The guard reads the same resolved command lists the candidate set is built
    from, so an inherited name is owned, not foreign.
    """
    _, cme_workflow = todolist_workflows
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router([]))

    predictor = CommandNamePrediction(cme_workflow)
    result = predictor.predict("TodoList", "assign_to bob", INTENT_DETECTION)

    assert result.command_name is not None
    assert result.command_name.split("/")[-1] == "assign_to"
    assert "assign_to" in predictor.command_inventory()


def test_inventory_excludes_reserved_labels(
    todolist_workflows, setup_test_environment
):
    """`wildcard` names no command; treating it as one would swallow the
    escalation utterances the classifier is trained on."""
    _, cme_workflow = todolist_workflows
    inventory = CommandNamePrediction(cme_workflow).command_inventory()
    assert "wildcard" not in inventory
    assert "parameter_value" not in inventory
    assert "create_todo_list" in inventory


# ---------------------------------------------------------------------------
# Mixed-case command names: a context never refuses its own command
# ---------------------------------------------------------------------------

MIXED_CASE_CONTEXTS = {
    "*": [],
    "Child": ["Child/ListPermissions", "Child/show_metadata"],
    "Sibling": ["Sibling/ShowHolders"],
}


@pytest.fixture
def mixed_case_predictor(todolist_workflows, monkeypatch):
    """A predictor whose app workflow reports mixed-case command names.

    `command_name_dict` keeps each command's spelling while the inventory is
    keyed lowercase; that pair is exactly what disagreed.
    """
    _, cme_workflow = todolist_workflows
    predictor = CommandNamePrediction(cme_workflow)
    app_folderpath = predictor.app_workflow_folderpath
    real_get_definition = fastworkflow.RoutingRegistry.get_definition
    definition = _InventoryRoutingDefinition(MIXED_CASE_CONTEXTS)

    def get_definition(workflow_folderpath, *args, **kwargs):
        if str(workflow_folderpath) == str(app_folderpath):
            return definition
        return real_get_definition(workflow_folderpath, *args, **kwargs)

    monkeypatch.setattr(
        fastworkflow.RoutingRegistry, "get_definition", staticmethod(get_definition)
    )
    return predictor


@pytest.mark.parametrize("utterance", [
    "ListPermissions",
    "listpermissions",
    "LISTPERMISSIONS <scope>all</scope>",
])
def test_a_mixed_case_command_resolves_in_its_own_context(
    mixed_case_predictor, monkeypatch, setup_test_environment, utterance
):
    """The token is lowercased and the context's own key is not. Before the
    fix the exact match missed and the guard named the current context as the
    foreign owner, refusing a command that belongs here."""
    consulted: list[str] = []
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router(consulted))
    nlu_trace: dict = {}
    result = mixed_case_predictor._predict_impl(
        "Child", utterance, INTENT_DETECTION, nlu_trace)

    assert result.command_name == "Child/ListPermissions"
    assert result.routing_hint is None
    assert nlu_trace["matcher_layer"] == "exact_prefix"
    assert "known_name_foreign_context" not in nlu_trace
    assert not consulted


def test_a_mixed_case_command_owned_elsewhere_is_still_foreign(
    mixed_case_predictor, monkeypatch, setup_test_environment
):
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router([]))
    nlu_trace: dict = {}
    result = mixed_case_predictor._predict_impl(
        "Child", "ShowHolders", INTENT_DETECTION, nlu_trace)

    assert result.command_name is None
    assert result.known_name_owner_contexts == ["Sibling"]
    assert "Sibling context" in result.routing_hint


def test_the_current_context_is_never_its_own_foreign_owner(ido_predictor):
    """Defence beyond the case fold: whatever the candidate set says, the
    context being asked is dropped from the owners it reports."""
    assert ido_predictor.foreign_owner_contexts(
        "show_holders", {}, "Permission") == []
    assert ido_predictor.foreign_owner_contexts(
        "show_holders", {}, "Identity") == ["Permission"]


# ---------------------------------------------------------------------------
# The hint at the end of the walk (owner-approved scope addition, 2026-09-15)
# ---------------------------------------------------------------------------

def _declare_enter_command(workflow_path, context_name: str, declaration: str) -> None:
    """Write an `enter_command` declaration into a COPY of the test workflow.

    The copy is what the fixture built in tmp_path, so the real
    `get_context_class` loads a real module with a real class attribute and
    nothing here stubs the lookup being tested. The shared fixture on disk is
    untouched.
    """
    path = workflow_path / "_commands" / context_name / f"_{context_name}.py"
    source = path.read_text()
    marker = "class Context:\n"
    assert marker in source, path
    path.write_text(source.replace(
        marker, f"class Context:\n    enter_command = {declaration!r}\n", 1))


class TestTheHintTextItself:
    """A pure composer: what it says, and what it refuses to invent."""

    def test_an_entity_context_hint_names_the_context_and_how_to_enter_it(self):
        hint = foreign_context_hint(
            "list_permissions", ["Account"],
            {"Account": ["open_account_by_uid <account_uid>"]})
        assert "'list_permissions'" in hint
        assert "Account context" in hint
        assert "open_account_by_uid <account_uid>" in hint

    def test_a_stateless_workspace_hint_names_the_command_that_enters_it(self):
        """No uid to supply: the workspace is entered by one bare command."""
        hint = foreign_context_hint(
            "list_findings", ["ControlsMonitor"],
            {"ControlsMonitor": ["open_controls_monitor"]})
        assert "ControlsMonitor context" in hint
        assert "'open_controls_monitor'" in hint
        assert "<" not in hint.split("Enter it with:")[1]

    def test_several_owners_are_all_named(self):
        hint = foreign_context_hint(
            "who_has_access_to", ["Directory", "DirectoryExplorer"], {})
        assert "Directory, DirectoryExplorer contexts" in hint

    def test_several_owners_each_get_their_own_entering_command(self):
        """A merged list cannot say which command opens which context."""
        hint = foreign_context_hint(
            "get_properties", ["TodoItem", "TodoList", "Board"],
            {"TodoItem": ["open_item <id>"], "TodoList": ["get_todo_list <id>"],
             "Board": []})
        assert "TodoItem, TodoList, Board contexts" in hint
        assert "Enter TodoItem with: 'open_item <id>'." in hint
        assert "Enter TodoList with: 'get_todo_list <id>'." in hint
        assert "Enter Board with" not in hint
        assert "Enter it with" not in hint

    def test_a_shared_entering_command_is_named_beside_each_context(self):
        hint = foreign_context_hint(
            "show", ["A", "B"], {"A": ["open_ab"], "B": ["open_ab"]})
        assert "Enter A with: 'open_ab'." in hint
        assert "Enter B with: 'open_ab'." in hint

    def test_without_a_declaration_it_names_the_context_and_nothing_else(self):
        """Nothing in the routing definition records which command enters a
        context, and guessing one from a command's NAME would bake one
        workflow's spelling conventions into the framework. Naming the owner
        alone is the honest floor."""
        hint = foreign_context_hint("list_permissions", ["Account"], {})
        assert "Account context" in hint
        assert "Enter it with" not in hint
        assert "what_can_i_do" in hint

    def test_without_a_declaration_it_never_says_then_after_nothing(self):
        """'Then run it there' needs an entry instruction before it; with no
        declaration the hint says where to go instead."""
        hint = foreign_context_hint("list_permissions", ["Account"], {})
        assert "Then run" not in hint
        assert "Move into that context first, then run 'list_permissions' there." in hint

    def test_several_undeclared_owners_are_one_of_those_contexts(self):
        hint = foreign_context_hint("who_has_access_to", ["Directory", "DirectoryExplorer"], {})
        assert "Then run" not in hint
        assert "Move into one of those contexts first" in hint

    def test_a_declared_entry_still_reads_enter_then_run(self):
        hint = foreign_context_hint(
            "list_permissions", ["Account"],
            {"Account": ["open_account_by_uid <account_uid>"]})
        assert ("Enter it with: 'open_account_by_uid <account_uid>'. "
                "Then run 'list_permissions' there.") in hint


class TestTheHintIsComposedAndRecorded:
    def test_the_guard_records_the_hint_on_its_own_span(
        self, ido_predictor, monkeypatch, setup_test_environment
    ):
        monkeypatch.setattr(
            intent_detection, "CommandRouter", _refusing_router([]))
        nlu_trace: dict = {}
        output = ido_predictor._predict_impl(
            "Identity", "list_permissions", INTENT_DETECTION, nlu_trace)

        assert output.command_name is None
        assert output.known_name_owner_contexts == ["Account"]
        assert nlu_trace["known_name_foreign_context_hint"] == output.routing_hint
        assert "Account context" in output.routing_hint

    def test_the_ido_workflow_declares_no_entering_command_so_the_hint_names_the_context(
        self, ido_predictor, monkeypatch, setup_test_environment
    ):
        """Recorded rather than asserted as a wish: `ido_workflow` declares no
        `enter_command` on any context, so its hints name the owning context
        and stop there. Adding the declaration is a workflow change, and this
        test is what would notice it."""
        monkeypatch.setattr(
            intent_detection, "CommandRouter", _refusing_router([]))
        nlu_trace: dict = {}
        ido_predictor._predict_impl(
            "Identity", "list_permissions", INTENT_DETECTION, nlu_trace)
        assert "Enter it with" not in nlu_trace["known_name_foreign_context_hint"]

    def test_the_guard_declines_and_records_no_flag(
        self, ido_predictor, monkeypatch, setup_test_environment
    ):
        """There is no auto-navigation setting for the routing event to carry,
        so the guard's own record says only what the guard did: it declined, at
        the known-name foreign-context layer, and the walk runs."""
        monkeypatch.setattr(
            intent_detection, "CommandRouter", _refusing_router([]))
        nlu_trace: dict = {}
        output = ido_predictor._predict_impl(
            "Identity", "list_permissions", INTENT_DETECTION, nlu_trace)

        assert output.command_name is None
        assert "auto_navigation_enabled" not in nlu_trace
        assert nlu_trace["matcher_layer"] == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT

    def test_a_resolved_prediction_carries_no_hint(
        self, ido_predictor, monkeypatch, setup_test_environment
    ):
        """A root command is reached by the walk and needs no hint at all."""
        monkeypatch.setattr(
            intent_detection, "CommandRouter", _refusing_router([]))
        nlu_trace: dict = {}
        output = ido_predictor._predict_impl(
            "*", "fetch_result_page <handle>O9</handle>", INTENT_DETECTION, nlu_trace)

        assert output.command_name == "fetch_result_page"
        assert output.routing_hint is None
        assert output.known_name_owner_contexts is None
        assert "known_name_foreign_context_hint" not in nlu_trace
        assert "auto_navigation_enabled" not in nlu_trace


class TestTheDeclaredEnteringCommandIsRead:
    """The declaration is read off the context's own callback class."""

    @pytest.fixture
    def declared_workflows(self, tmp_path):
        workflow_path = tmp_path / "todo_list_workflow"
        shutil.copytree(
            os.path.join(os.path.dirname(__file__), "todo_list_workflow"),
            workflow_path,
            ignore=shutil.ignore_patterns(
                "___command_info", "___workflow_contexts", "___convo_info",
                "__pycache__",
            ),
        )
        _declare_enter_command(workflow_path, "TodoList",
                               "get_todo_list <todo_list_id>")
        app_workflow = fastworkflow.Workflow.create(
            workflow_folderpath=str(workflow_path),
            workflow_id_str=f"enter-command-{tmp_path.name}",
        )
        cme_workflow = fastworkflow.Workflow.create(
            workflow_folderpath=fastworkflow.get_internal_workflow_path(
                "command_metadata_extraction"
            ),
            parent_workflow_id=app_workflow.id,
            workflow_context={"app_workflow": app_workflow},
        )
        return app_workflow, cme_workflow

    def test_the_hint_names_the_owner_and_its_declared_entering_command(
        self, declared_workflows, monkeypatch, setup_test_environment
    ):
        _, cme_workflow = declared_workflows
        monkeypatch.setattr(
            intent_detection, "CommandRouter", _refusing_router([]))
        predictor = CommandNamePrediction(cme_workflow)

        assert predictor.enter_commands_for("TodoList") == [
            "get_todo_list <todo_list_id>"]
        output = predictor.predict(
            "TodoItem", "mark_completed", INTENT_DETECTION)

        assert output.command_name is None
        assert "TodoList context" in output.routing_hint
        assert "get_todo_list <todo_list_id>" in output.routing_hint

    def test_an_undeclared_context_resolves_to_no_entering_command(
        self, declared_workflows, setup_test_environment
    ):
        _, cme_workflow = declared_workflows
        predictor = CommandNamePrediction(cme_workflow)
        assert predictor.enter_commands_for("TodoItem") == []
        assert predictor.enter_commands_for("no-such-context") == []


# ---------------------------------------------------------------------------
# The hint reaches the caller: wildcard's end-of-walk path
# ---------------------------------------------------------------------------

class TestTheHintReachesTheFailureMessage:
    """The walk ends at the root with no owner on the chain; say where it lives.

    The CME wildcard command is driven directly with doubles for the two
    collaborators it calls out to - the predictor and the executor - because
    what is under test is the lines between them: keep the first hint, carry it
    through the walk, and make it the message the caller actually reads.
    `TestTheHintPathOnARealWorkflow` below drives the same path with nothing
    replaced but the classifier.
    """

    class _Workflow:
        """The CME workflow: a context dict and an app workflow."""

        def __init__(self, app_workflow):
            self._context = {"app_workflow": app_workflow}

        @property
        def context(self):
            return self._context

        @context.setter
        def context(self, value):
            self._context = value

        def end_command_processing(self):
            self._context.pop("command", None)
            self._context["NLU_Pipeline_Stage"] = (
                fastworkflow.NLUPipelineStage.INTENT_DETECTION)

    class _AppWorkflow:
        """Context objects are their own names; the chain is a list."""

        def __init__(self, chain):
            self.chain = list(chain)
            self.current_command_context = chain[0]
            self.current_command_context_name = chain[0]
            self.command_context_for_response_generation = chain[0]
            self.context = {}

        @property
        def is_command_context_for_response_generation_root(self):
            return (self.command_context_for_response_generation
                    == self.chain[-1])

        def get_parent(self, context_object):
            return self.chain[self.chain.index(context_object) + 1]

    @staticmethod
    def _predictor_declining(hint, contexts_seen):
        class DecliningPredictor:
            def __init__(self, cme_workflow):
                pass

            def predict(self, context_name, command, stage):
                contexts_seen.append(context_name)
                return CommandNamePrediction.Output(
                    command_name=None,
                    known_name_owner_contexts=["Account"],
                    # Only the first context composes one in production; the
                    # later ones repeat it. Returning it once proves the walk
                    # does not lose it.
                    routing_hint=hint if len(contexts_seen) == 1 else None,
                )
        return DecliningPredictor

    def _run(self, monkeypatch, hint):
        from fastworkflow._workflows.command_metadata_extraction._commands import (
            wildcard as wildcard_command,
        )

        contexts_seen: list[str] = []
        actions_run: list[str] = []
        monkeypatch.setattr(wildcard_command, "CommandNamePrediction",
                            self._predictor_declining(hint, contexts_seen))
        monkeypatch.setattr(
            fastworkflow.Workflow, "get_command_context_name",
            staticmethod(lambda context_object: context_object))

        def perform_action(workflow, action):
            actions_run.append(action.command_name)
            return fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response="I couldn't determine which available command "
                             "matches your request.",
                    success=False))

        monkeypatch.setattr(
            wildcard_command.CommandExecutor, "perform_action",
            staticmethod(perform_action))

        app_workflow = self._AppWorkflow(["Identity", "DirectoryExplorer", "*"])
        cme_workflow = self._Workflow(app_workflow)
        output = wildcard_command.ResponseGenerator()(
            cme_workflow, "list_permissions")
        return output, contexts_seen, actions_run, cme_workflow

    def test_the_message_names_the_owner_and_how_to_get_there(
        self, monkeypatch, setup_test_environment
    ):
        hint = foreign_context_hint(
            "list_permissions", ["Account"],
            {"Account": ["open_account_by_uid <account_uid>"]})
        output, contexts_seen, actions_run, cme_workflow = self._run(
            monkeypatch, hint)

        response = output.command_response.response
        assert response == hint
        assert "Account context" in response
        assert "open_account_by_uid <account_uid>" in response
        assert output.success is False
        assert output.command_handled is True
        # The whole chain was asked before the message was composed.
        assert contexts_seen == ["Identity", "DirectoryExplorer", "*"]
        # Not a misunderstanding: no you_misunderstood, and no clarification
        # stage waiting to swallow the entering command the hint names.
        assert actions_run == []
        assert cme_workflow.context["NLU_Pipeline_Stage"] == (
            fastworkflow.NLUPipelineStage.INTENT_DETECTION)
        assert "command" not in cme_workflow.context

    def test_a_walk_with_no_hint_is_left_exactly_as_it_was(
        self, monkeypatch, setup_test_environment
    ):
        """Free text that no context could route still gets the plain message:
        the hint is for a name the workflow owns, and nothing else."""
        output, _, actions_run, cme_workflow = self._run(monkeypatch, None)
        assert output.command_response.response == (
            "I couldn't determine which available command matches your request.")
        assert actions_run == ["ErrorCorrection/you_misunderstood"]
        assert cme_workflow.context["NLU_Pipeline_Stage"] == (
            fastworkflow.NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION)


# ---------------------------------------------------------------------------
# The reply to you_misunderstood: nothing matched is not a KeyError
# ---------------------------------------------------------------------------

MISUNDERSTANDING = fastworkflow.NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION


@pytest.mark.parametrize("utterance", [
    "gibberish that matches nothing",
    "list_holders <filter>Alan Cooper</filter>",
])
def test_an_unmatched_clarification_reply_falls_back_to_what_can_i_do(
    ido_predictor, monkeypatch, setup_test_environment, utterance
):
    """Both clarification stages substitute 'what can i do?' when no matcher
    claims the reply. Only the ambiguity stage used to register that key, so in
    the misunderstanding stage the substitution raised KeyError for any reply
    that was not one of this context's own commands -- including a command the
    workflow owns elsewhere."""
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router([]))
    nlu_trace: dict = {}
    result = ido_predictor._predict_impl(
        "Identity", utterance, MISUNDERSTANDING, nlu_trace)

    assert result.command_name == "IntentDetection/what_can_i_do"
    assert result.is_cme_command is True
    assert nlu_trace["matcher_layer"] == "clarification_default"


@pytest.mark.parametrize("utterance,owners", [
    ("open_account_by_uid <account_uid>3f2a</account_uid>", ["DirectoryExplorer"]),
    ("list_permissions", ["Account"]),
])
def test_a_foreign_name_in_the_misunderstanding_reply_is_declined_with_its_owner(
    ido_predictor, monkeypatch, setup_test_environment, utterance, owners
):
    """The misunderstanding stage matches this context's full command set, so a
    real command owned elsewhere is the same misroute there as in intent
    detection. It used to fall to the 'what can i do?' default, which lists
    this context's commands and never names the owner."""
    monkeypatch.setattr(
        intent_detection, "CommandRouter", _refusing_router([]))
    nlu_trace: dict = {}
    result = ido_predictor._predict_impl(
        "Identity", utterance, MISUNDERSTANDING, nlu_trace)

    assert result.command_name is None
    assert result.known_name_owner_contexts == owners
    assert nlu_trace["matcher_layer"] == MATCHER_LAYER_KNOWN_NAME_FOREIGN_CONTEXT
    for owner in owners:
        assert f"{owner} context" in result.routing_hint


# ---------------------------------------------------------------------------
# The end of the walk on a real workflow, then the message after it
# ---------------------------------------------------------------------------

class TestTheHintPathOnARealWorkflow:
    """The real CME wildcard command, executor and `you_misunderstood`, over a
    copy of `tests/todo_list_workflow` whose TodoList declares its entering
    command. Only the classifier is replaced; no LLM is called and no trained
    model is read.

    From TodoListManager the chain is `TodoListManager -> *`, and
    `get_properties` is owned by TodoItem and TodoList, both off it.
    """

    @pytest.fixture
    def workflows(self, tmp_path):
        workflow_path = tmp_path / "todo_list_workflow"
        shutil.copytree(
            os.path.join(os.path.dirname(__file__), "todo_list_workflow"),
            workflow_path,
            ignore=shutil.ignore_patterns(
                "___command_info", "___workflow_contexts", "___convo_info",
                "__pycache__",
            ),
        )
        _declare_enter_command(workflow_path, "TodoList",
                               "get_todo_list <todo_list_id>")
        app_workflow = fastworkflow.Workflow.create(
            workflow_folderpath=str(workflow_path),
            workflow_id_str=f"hint-path-{tmp_path.name}",
        )
        app_workflow.current_command_context = TodoListManager()
        cme_workflow = fastworkflow.Workflow.create(
            workflow_folderpath=fastworkflow.get_internal_workflow_path(
                "command_metadata_extraction"
            ),
            parent_workflow_id=app_workflow.id,
            workflow_context={"app_workflow": app_workflow},
        )
        return app_workflow, cme_workflow

    def test_a_foreign_name_gets_the_hint_and_the_entering_command_then_routes(
        self, workflows, monkeypatch, setup_test_environment
    ):
        app_workflow, cme_workflow = workflows
        monkeypatch.setattr(
            intent_detection, "CommandRouter", _refusing_router([]))

        output = wildcard_module.ResponseGenerator()(cme_workflow, "get_properties")

        response = output.command_response.response
        assert output.success is False
        assert output.command_handled is True
        assert "'get_properties' is a command of the TodoItem, TodoList contexts" in response
        assert "'get_todo_list <todo_list_id>'" in response
        assert "I couldn't determine" not in response

        # Doing what the hint says: the next message is the entering command,
        # and it is routed by ordinary intent detection in this context.
        stage = cme_workflow.context["NLU_Pipeline_Stage"]
        assert stage == fastworkflow.NLUPipelineStage.INTENT_DETECTION
        assert "command" not in cme_workflow.context
        following = CommandNamePrediction(cme_workflow).predict(
            app_workflow.current_command_context_name, "get_todo_list 1", stage)
        assert following.command_name == "TodoListManager/get_todo_list"

    def test_free_text_still_ends_at_you_misunderstood_and_a_stray_reply_is_answered(
        self, workflows, monkeypatch, setup_test_environment
    ):
        """No known name, so no hint: every context's classifier escalates, the
        walk ends at the root, and the real `you_misunderstood` runs. A reply
        that matches nothing in the clarification stage lists the options
        instead of raising."""
        app_workflow, cme_workflow = workflows
        monkeypatch.setattr(
            intent_detection, "CommandRouter", _labelling_router("wildcard"))

        output = wildcard_module.ResponseGenerator()(cme_workflow, "tell me a joke")

        assert output.command_response.response.startswith(
            "I couldn't determine which available command matches your request.")
        assert output.command_handled is True
        stage = cme_workflow.context["NLU_Pipeline_Stage"]
        assert stage == MISUNDERSTANDING

        following = CommandNamePrediction(cme_workflow).predict(
            app_workflow.current_command_context_name, "none of those", stage)
        assert following.command_name == "IntentDetection/what_can_i_do"

    def test_a_foreign_name_as_the_misunderstanding_reply_gets_the_hint(
        self, workflows, monkeypatch, setup_test_environment
    ):
        """Free text puts the workflow in the misunderstanding stage; the reply
        is a real command owned off the chain. The walk runs in that stage,
        finds no owner, and the reply ends with the hint and the stage closed,
        instead of a list of this context's commands."""
        app_workflow, cme_workflow = workflows
        monkeypatch.setattr(
            intent_detection, "CommandRouter", _labelling_router("wildcard"))
        wildcard_module.ResponseGenerator()(cme_workflow, "tell me a joke")
        assert cme_workflow.context["NLU_Pipeline_Stage"] == MISUNDERSTANDING

        monkeypatch.setattr(
            intent_detection, "CommandRouter", _refusing_router([]))
        output = wildcard_module.ResponseGenerator()(cme_workflow, "get_properties")

        response = output.command_response.response
        assert output.success is False
        assert output.command_handled is True
        assert "'get_properties' is a command of the TodoItem, TodoList contexts" in response
        assert "Enter TodoList with: 'get_todo_list <todo_list_id>'." in response
        assert cme_workflow.context["NLU_Pipeline_Stage"] == INTENT_DETECTION
        assert "command" not in cme_workflow.context


