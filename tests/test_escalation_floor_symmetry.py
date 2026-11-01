"""The escalation coverage floor must not depend on the order contexts are visited.

bd fix-4ej. `train()` shares one utterance cache across every context in a run, and two
paths write it with different command sets:

* `cache_ancestor_utterances` walks `context_model.commands(ancestor)`, which excludes
  the core commands;
* `cache_context_command_utterances` walks `crd.contexts[ctx] | core_command_names`,
  which includes them.

The retired coverage floor (`class_balance.coverage_floor_of`, now deleted) counted cache
entries, so whichever path reached a
context first used to decide how many "sources" its descendants believed they had to
cover. Visit order is `sorted(context_set_for_training)`, so the deciding fact was a
pair of context NAMES: renaming one context moved a different context's escalation
budget, and when the floor bound the budget it moved that context's training rows.
Nothing in the diff would explain it.

The fix counts core commands on neither path (`skip_commands` in
`group_ancestor_utterances`). These tests state that as three properties:

1. renaming a context leaves every other context's escalation record identical,
   including the selected rows, on a hierarchy deep enough that the floor binds;
2. the same holds on a real bundled workflow, `messaging_app_4`;
3. permuting the visit order alone changes nothing -- the property the rename tests
   depend on, asserted directly so a failure says which of the two broke.

A fourth test pins the DIRECTION of the fix, because "count core commands everywhere"
would also be symmetric and would give small contexts more escalation data. It asserts
the reason core commands are not sources: every one of their rows is already a local
row, so the coverage pass can never give them one, and a floor that counted them would
promise coverage the selection does not deliver.

Phase 9 step 5: the ancestor escalation class and its coverage floor are retired. The shipped
class is the sparse "not here" class, which draws seeds of every foreign command and reads no
cache, so the properties below now hold of the sparse class: a rename or a visit order cannot
move another head's rows, and core commands are never escalation rows. The floor-specific
premises of the original tests (the floor binds the budget; the floor counts five ancestor
commands) describe the retired construction and are replaced by assertions that the floor is
zero and the budget is absent.

No training happens here. `select_escalation_rows` is the shipped decision and it is
called directly, so these run in about a second while still exercising the code
`train()` runs. What is supplied rather than generated is the per-command corpus: real
generation needs an LLM, so each generated command's own `Signature.generate_utterances`
returns its declared seeds (the pattern in `tests/test_training_determinism.py`), and
the core commands -- whose generators live in the internal workflow -- are pre-placed in
the `command_utterance_cache` that `_get_cached_command_utterances` exists to read. Both
sides of every comparison get byte-identical corpora, so the corpus cannot be what
makes them differ.
"""

import contextlib
import io
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import (
    cache_context_command_utterances,
    select_escalation_rows,
    train,
    trained_command_labels,
)
from fastworkflow.nlu_labels import (
    PARAMETER_VALUE_LABEL,
    WILDCARD_LABEL,
    label_of,
)
from fastworkflow.train import generate_synthetic
from fastworkflow.train.determinism import (
    ProvenanceRecorder,
    UtteranceProvenance,
    get_provenance_recorder,
    set_provenance_recorder,
)
from fastworkflow.train.selective_training import contexts_for_training

MESSAGING_APP_PATH = os.path.join("fastworkflow", "examples", "messaging_app_4")

@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    """Real init; `Workflow.create` needs the context store configured."""
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


def _copy_messaging_app(root: Path, name: str) -> str:
    """A private copy, so nothing here can write into `fastworkflow/examples/`."""
    path = root / name
    shutil.copytree(
        MESSAGING_APP_PATH,
        path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__"),
    )
    return str(path)


def _seed_corpora(crd) -> dict[str, list[str]]:
    """Every command's declared seed utterances, as the warm `command_utterance_cache`.

    `_get_cached_command_utterances` reads this cache before generating, so pre-placing
    entries here is the supported way to run the trainer's assembly without an LLM. A
    command with no `Signature.Input` has no utterance metadata and gets an empty list,
    which is what the real generator returns for it too.

    `wildcard` is left out, exactly as `train()` leaves it out: it is never a training
    label, so neither cache-fill path asks for it, and `train()` reads its rows through
    `_get_utterances` instead. Putting the seven bare-value literals it declares in here
    would silently make them this run's always-include rows, which is not what a real
    run does -- its generator returns only the humanised command name.
    """
    cmd_dir = crd.command_directory
    corpora: dict[str, list[str]] = {}
    for command_name in sorted(
        set(cmd_dir.get_commands()) | set(cmd_dir.core_command_names)
    ):
        if label_of(command_name) == WILDCARD_LABEL:
            continue
        cmd_dir.ensure_command_hydrated(command_name)
        metadata = cmd_dir.get_utterance_metadata(command_name)
        corpora[command_name] = list(metadata.plain_utterances) if metadata else []
    return corpora


def _escalation_records(
    workflow_path: str,
    tag: str,
    visit_order: list[str] | None = None,
) -> dict[str, dict]:
    """Replay `train()`'s per-context row assembly and return each escalation decision.

    The loop is `train()`'s: for every context in visit order, fill the shared cache
    with the labels that context trains, then ask `select_escalation_rows` for the
    reserved class. Both of those are the functions `train()` calls; only model fitting
    and evaluation are left out.

    `visit_order` defaults to `train()`'s own `sorted(contexts_for_training(...))`. A
    caller passing something else is asking "what would this run have produced if the
    names had sorted differently", which is what a rename does.
    """
    RoutingRegistry.clear_registry()
    crd = RoutingRegistry.get_definition(workflow_path)
    command_cache = _seed_corpora(crd)
    context_cache: dict[str, dict[str, list[str]]] = {}
    records: dict[str, dict] = {}

    workflow = fastworkflow.Workflow.create(
        workflow_path, workflow_id_str=f"escalation-symmetry-{tag}")
    try:
        order = visit_order or sorted(contexts_for_training(workflow_path))
        for context_name in order:
            rows_by_command = cache_context_command_utterances(
                context_name, crd, workflow, context_cache, command_cache)
            context_utterances: set[str] = set()
            own_row_count = 0
            for rows in rows_by_command.values():
                own_row_count += len(rows)
                context_utterances |= set(rows)
            if not context_utterances:
                continue
            selection = select_escalation_rows(
                context_name,
                crd,
                workflow,
                context_utterances=context_utterances,
            )
            records[context_name] = {
                "own_rows": own_row_count,
                "included": selection.included,
                "coverage_floor": selection.coverage_floor,
                "budget": selection.budget,
                "raw_candidates": selection.raw_candidate_count,
                "deduplicated_candidates": selection.deduplicated_candidate_count,
                "always_include": len(selection.always_include_rows),
                "selected_rows": selection.rows,
            }
    finally:
        workflow.close()
        RoutingRegistry.clear_registry()
    return records


@contextlib.contextmanager
def _quiet():
    """The assembly prints one line per (context, command), which on the chain fixture is
    over two hundred lines of noise around whatever the assertion has to say."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


# ---------------------------------------------------------------------
# 0. The real train() glue loop must stay cheaply reachable
# ---------------------------------------------------------------------

def test_train_assembles_every_context_before_model_fitting(tmp_path, monkeypatch):
    """Drive real ``train()`` through every copied workflow context without fitting.

    The copy protects the bundled example's untracked trained artifacts. Synthetic
    generation is replaced only at its LLM boundary with deterministic command-specific
    rows; routing, command hydration, inheritance, assembly, balancing, and provenance
    recording all remain real.
    """
    workflow_path = _copy_messaging_app(tmp_path, "train_glue_copy")

    def deterministic_generation(seed_utterances, command_name, **_kwargs):
        rows = [
            f"{command_name} deterministic utterance one",
            f"{command_name} deterministic utterance two",
        ]
        return rows, UtteranceProvenance(
            command_name=command_name,
            seed=42,
            seed_utterance_count=len(seed_utterances),
            generated_count=len(rows),
            final_count=len(rows),
        )

    monkeypatch.setattr(
        generate_synthetic,
        "generate_diverse_utterances_with_provenance",
        deterministic_generation,
    )

    RoutingRegistry.clear_registry()
    expected_contexts = set(contexts_for_training(workflow_path))
    recorder = ProvenanceRecorder(workflow_path)
    previous_recorder = get_provenance_recorder()
    set_provenance_recorder(recorder)
    workflow = fastworkflow.Workflow.create(
        workflow_path, workflow_id_str="train-glue-stop-before-fit")
    try:
        with _quiet() as output:
            train(workflow, stop_before_fit=True)
    finally:
        workflow.close()
        set_provenance_recorder(previous_recorder)
        RoutingRegistry.clear_registry()

    context_records = recorder.context_records
    wildcard_contexts = {
        context_name
        for context_name, command_name in context_records
        if command_name == WILDCARD_LABEL
    }
    parameter_value_contexts = {
        context_name
        for context_name, command_name in context_records
        if command_name == PARAMETER_VALUE_LABEL
    }

    assert len(expected_contexts) > 1
    assert wildcard_contexts == expected_contexts
    assert parameter_value_contexts == expected_contexts
    assert output.getvalue().count("stopping before model fitting") == len(
        expected_contexts)
    assert "Loading google/bert" not in output.getvalue()
    assert not list(Path(workflow_path).rglob("tinymodel.pth"))
    assert not list(Path(workflow_path).rglob("largemodel.pth"))


# ---------------------------------------------------------------------
# 1. A rename must not move another context's rows
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2. Visit order alone must not move anything
# ---------------------------------------------------------------------

def test_escalation_records_are_independent_of_the_order_contexts_are_visited(tmp_path):
    """Same workflow, two visit orders -- the determinism property the rename tests relied on.

    Phase 9 step 5 retired the ancestor cache and the hierarchy file this test used to build.
    The shipped sparse class reads seeds, not the cache, so the property now has to hold of
    the 'not here' rows themselves: visiting the contexts in reverse order must not change any
    context's escalation record. Runs on the copied messaging_app_4, which has no hierarchy file.
    """
    workflow_path = _copy_messaging_app(tmp_path, "visit_orders")
    forward_order = sorted(contexts_for_training(workflow_path))
    backward_order = list(reversed(forward_order))
    assert len(forward_order) > 1, "need more than one context to permute"

    with _quiet():
        forwards = _escalation_records(workflow_path, "fwd", forward_order)
        backwards = _escalation_records(workflow_path, "bwd", backward_order)

    assert forwards.keys() == backwards.keys()
    for context_name in forwards:
        assert forwards[context_name] == backwards[context_name], (
            f"{context_name}'s escalation record depends on visit order: "
            f"{forwards[context_name]} vs {backwards[context_name]}"
        )


# ---------------------------------------------------------------------
# 3. Which way the asymmetry was resolved, and why
# ---------------------------------------------------------------------

def test_core_commands_are_not_escalation_sources(tmp_path):
    """Core commands are never escalation rows, on the sparse class.

    A core command is a label in EVERY context (`command_routing.build` unions
    `core_command_names` into each one), so all of its rows are local rows in every head. The
    sparse class samples foreign commands only, and `exclude` removes every local row, so no
    core command's seed can be an escalation row. The retired ancestor floor counted the
    application commands of a parent (five for `User`'s `ChatRoom`); that count is gone with it.
    """
    workflow_path = _copy_messaging_app(tmp_path, "messaging_direction")

    with _quiet():
        records = _escalation_records(workflow_path, "direction")

        RoutingRegistry.clear_registry()
        crd = RoutingRegistry.get_definition(workflow_path)
        command_cache = _seed_corpora(crd)

    core_commands = set(crd.command_directory.core_command_names)
    ancestor_commands = set(crd.context_model.commands("ChatRoom"))
    assert not ancestor_commands & core_commands, (
        "context_model.commands() is supposed to exclude core commands; if it stopped "
        "doing so the two cache-fill paths would agree for a different reason"
    )

    user = records["User"]
    assert user["coverage_floor"] == 0 and user["budget"] is None
    core_rows = {row for name in core_commands for row in command_cache.get(name, [])}
    assert not (set(user["selected_rows"]) & core_rows), (
        "a core command's seed reached User's escalation class"
    )

    # The reason, not just the number: every core-command row is also a row `User`
    # trains locally, so a floor that counted core commands would be counting sources
    # the coverage pass provably cannot serve.
    user_labels = trained_command_labels("User", crd)
    user_rows = {
        row
        for label in user_labels
        if label_of(label) != WILDCARD_LABEL
        for row in command_cache[label]
    }
    covered_core_commands = [
        command_name
        for command_name in sorted(core_commands)
        if label_of(command_name) != WILDCARD_LABEL
        and command_cache[command_name]
        and not set(command_cache[command_name]) <= user_rows
    ]
    assert not covered_core_commands, (
        f"these core commands have ancestor rows that are NOT local to User, so the "
        f"argument for excluding them from the floor does not hold: "
        f"{covered_core_commands}"
    )
    assert core_commands & user_labels == core_commands


