"""The escalation coverage floor must not depend on the order contexts are visited.

bd fix-4ej. `train()` shares one utterance cache across every context in a run, and two
paths write it with different command sets:

* `cache_ancestor_utterances` walks `context_model.commands(ancestor)`, which excludes
  the core commands;
* `cache_context_command_utterances` walks `crd.contexts[ctx] | core_command_names`,
  which includes them.

`class_balance.coverage_floor_of` counts cache entries, so whichever path reached a
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
import json
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import (
    _get_utterances,
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
from fastworkflow.train import class_balance, generate_synthetic
from fastworkflow.train.determinism import (
    ProvenanceRecorder,
    UtteranceProvenance,
    get_provenance_recorder,
    set_provenance_recorder,
)
from fastworkflow.train.selective_training import contexts_for_training

MESSAGING_APP_PATH = os.path.join("fastworkflow", "examples", "messaging_app_4")

# Ten commands per ancestor context, two rows each. Ten is not decorative: the floor
# only changes what trains when it exceeds the context's own row count, and `Leaf`'s own
# rows are dominated by the core commands' real seed lists (26 rows). Three ancestor
# contexts of ten commands each put the floor at 30 -- above 26, so it binds -- and each
# ancestor context contributes four phantom core sources when it is visited as a
# trainee, so the defect is a 4-row swing per ancestor context.
COMMANDS_PER_ANCESTOR = 10

# Topics are attached to a context's POSITION in the chain, never to its name, so the
# rename under test cannot change a single utterance. That is what "differing only by a
# context rename" has to mean: renaming a context does not rewrite its author's seeds.
CHAIN_TOPICS = ("inventory", "shipping", "billing")
LEAF_TOPICS = {"Leaf": "onboarding", "Mint": "archival"}

_COMMAND_MODULE = '''\
from pydantic import BaseModel


class Signature:
    class Input(BaseModel):
        target: str

    plain_utterances = [
        "{first}",
        "{second}",
    ]

    @staticmethod
    def generate_utterances(workflow, command_name):
        """Deterministic, and independent of `command_name`: a context rename must not
        be able to change this command's corpus, or a rename test could not tell a
        moved budget from a moved utterance."""
        return list(Signature.plain_utterances)
'''


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    """Real init; `Workflow.create` needs the context store configured."""
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


def _write_command(folder: Path, topic: str, index: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    name = f"task_{index:02d}"
    phrase = f"{topic} {name.replace('_', ' ')}"
    (folder / f"{name}.py").write_text(
        _COMMAND_MODULE.format(
            first=f"please run the {phrase} routine",
            second=f"kindly start {phrase} for me now",
        ),
        encoding="utf-8",
    )


def _build_chain_workflow(root: Path, name: str, chain: tuple[str, ...]) -> str:
    """A workflow shaped `chain[0] <- chain[1] <- chain[2] <- {Leaf, Mint}`.

    `chain` names the three ancestor contexts, outermost first. Everything else --
    command names, seed utterances, hierarchy shape, row counts -- is a function of
    position, so two workflows built from two different `chain` tuples differ in exactly
    the context names and nothing else.
    """
    path = root / name
    commands = path / "_commands"
    for context, topic in zip(chain, CHAIN_TOPICS):
        for index in range(COMMANDS_PER_ANCESTOR):
            _write_command(commands / context, topic, index)
    for leaf, topic in LEAF_TOPICS.items():
        _write_command(commands / leaf, topic, 0)

    hierarchy = {
        chain[1]: {"parent": [chain[0]]},
        chain[2]: {"parent": [chain[1]]},
    }
    for leaf in LEAF_TOPICS:
        hierarchy[leaf] = {"parent": [chain[2]]}
    (path / "context_hierarchy_model.json").write_text(
        json.dumps(hierarchy, indent=2), encoding="utf-8")
    return str(path)


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


def _rename_context(workflow_path: str, old: str, new: str) -> None:
    """Rename a context the way an author would: the folder, its context class module,
    and the references in the two model files. Nothing else in the workflow moves."""
    commands = Path(workflow_path) / "_commands"
    (commands / old).rename(commands / new)
    context_class = commands / new / f"_{old}.py"
    if context_class.is_file():
        context_class.rename(commands / new / f"_{new}.py")
    for relative in (
        "context_hierarchy_model.json",
        os.path.join("_commands", "context_inheritance_model.json"),
    ):
        model_path = Path(workflow_path) / relative
        if model_path.is_file():
            model_path.write_text(
                model_path.read_text(encoding="utf-8").replace(f'"{old}"', f'"{new}"'),
                encoding="utf-8",
            )


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
        # `train()`'s own line. The wildcard command's generator is deterministic and
        # LLM-free, so this one really runs rather than being supplied.
        wildcard_utterances = set(_get_utterances(
            workflow, workflow_path, crd.command_directory, WILDCARD_LABEL))
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
                context_cache,
                command_cache,
                context_utterances=context_utterances,
                own_row_count=own_row_count,
                wildcard_utterances=wildcard_utterances,
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

def test_renaming_an_ancestor_context_does_not_change_a_descendant_escalation_rows(
    tmp_path,
):
    """The defect, at the only place it can be seen: the rows that actually train.

    Two workflows differing in one context name. `Billing` sorts before `Leaf` and
    `Mint`, so it is trained first and its cache entry holds its core commands when they
    read it; `Zulu` sorts after both, so they read it while it holds only
    `context_model.commands()`. Every other input is identical -- the seed utterances are
    keyed to a context's position in the chain, not its name.

    `Leaf` and `Mint` are asserted together because one rename moves BOTH: every
    descendant of the renamed context reads the same cache entry. Renaming a context is
    not expected to be free for the context itself, whose command labels change; it is
    expected to be free for everything else, and it was not.
    """
    baseline = _build_chain_workflow(
        tmp_path, "baseline", ("Alpha", "Bravo", "Billing"))
    renamed = _build_chain_workflow(
        tmp_path, "renamed", ("Alpha", "Bravo", "Zulu"))

    with _quiet():
        before = _escalation_records(baseline, "baseline")
        after = _escalation_records(renamed, "renamed")

    # The premise: the rename really did move the ancestor across the two descendants in
    # visit order. Without this the test could pass by testing nothing.
    assert sorted(before) == ["*", "Alpha", "Billing", "Bravo", "Leaf", "Mint"]
    assert sorted(after) == ["*", "Alpha", "Bravo", "Leaf", "Mint", "Zulu"]

    for context_name in ("Leaf", "Mint"):
        assert before[context_name] == after[context_name], (
            f"renaming a third context changed {context_name}'s escalation class: "
            f"{before[context_name]} vs {after[context_name]}"
        )

    # And the floor really is the binding constraint here, so the equality above is a
    # statement about training rows and not only about a reported number.
    leaf = before["Leaf"]
    assert leaf["coverage_floor"] > leaf["own_rows"], (
        "the fixture no longer binds the budget on the coverage floor, so this test no "
        "longer covers the case where the asymmetry changes what trains"
    )
    assert leaf["budget"] == leaf["coverage_floor"]
    assert len(leaf["selected_rows"]) > leaf["coverage_floor"]


def test_renaming_a_parent_context_does_not_change_its_children_escalation_budgets(
    tmp_path,
):
    """The same property on a real bundled workflow rather than a built fixture.

    `messaging_app_4` is the only bundled example with more than one trainable context
    (`User` and `PremiumUser` both under `ChatRoom`). Its corpora are small enough that
    the cost ratio, not the floor, bounds the budget -- so the rows come out equal even
    with the defect present, and what moves is `coverage_floor` and `raw_candidate_rows`:
    the two numbers `fix-k0i.34` put in the ESCALATION BUDGET table for a person to
    read. A report that says a different thing about two identical runs is a broken
    instrument, so they are asserted too.
    """
    baseline = _copy_messaging_app(tmp_path, "messaging_baseline")
    renamed = _copy_messaging_app(tmp_path, "messaging_renamed")
    _rename_context(renamed, "ChatRoom", "ZChatRoom")

    with _quiet():
        before = _escalation_records(baseline, "msg-baseline")
        after = _escalation_records(renamed, "msg-renamed")

    assert sorted(before) == ["*", "ChatRoom", "PremiumUser", "User"]
    assert sorted(after) == ["*", "PremiumUser", "User", "ZChatRoom"]

    for context_name in ("User", "PremiumUser"):
        assert before[context_name] == after[context_name], (
            f"renaming ChatRoom changed {context_name}'s escalation record: "
            f"{before[context_name]} vs {after[context_name]}"
        )


# ---------------------------------------------------------------------
# 2. Visit order alone must not move anything
# ---------------------------------------------------------------------

def test_escalation_records_are_independent_of_the_order_contexts_are_visited(tmp_path):
    """Same workflow, two visit orders -- the mechanism the rename tests rely on.

    A rename is only interesting because it permutes `sorted(context_set_for_training)`.
    Asserting the permutation directly separates the two things a rename test conflates:
    if this passes and the rename test fails, the rename changed an input rather than
    the visit order.

    The ancestor-last order is the one the defect needed: it makes every descendant read
    its ancestors' cache entries before those contexts are trained, which is when they
    hold no core commands.
    """
    workflow_path = _build_chain_workflow(
        tmp_path, "orders", ("Alpha", "Bravo", "Billing"))

    ancestors_first = ["*", "Alpha", "Bravo", "Billing", "Leaf", "Mint"]
    ancestors_last = ["*", "Leaf", "Mint", "Billing", "Bravo", "Alpha"]

    with _quiet():
        forwards = _escalation_records(workflow_path, "fwd", ancestors_first)
        backwards = _escalation_records(workflow_path, "bwd", ancestors_last)

    for context_name in ("Leaf", "Mint", "Billing", "Bravo"):
        assert forwards[context_name] == backwards[context_name], (
            f"{context_name}'s escalation class depends on visit order: "
            f"{forwards[context_name]} vs {backwards[context_name]}"
        )


# ---------------------------------------------------------------------
# 3. Which way the asymmetry was resolved, and why
# ---------------------------------------------------------------------

def test_core_commands_are_not_escalation_sources(tmp_path):
    """Pins the direction: the floor counts ancestor commands, minus the core ones.

    "Count core commands on both paths" would also be symmetric, so symmetry alone does
    not choose. This asserts the reason it is the wrong choice. A core command is a
    label in EVERY context (`command_routing.build` unions `core_command_names` into
    each one), so all of its rows are local rows in the descendant, `exclude` removes
    every one of them, and the coverage pass cannot give it the row the floor claims it
    keeps. Counting it would raise the budget by four per ancestor context to cover
    sources that cannot be covered.

    Asserted on `messaging_app_4`, where `User`'s single ancestor `ChatRoom` has five
    application commands and four core commands with utterances: the floor is 5.
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
    assert user["coverage_floor"] == len(ancestor_commands) == 5

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


def test_grouping_drops_core_commands_whichever_path_wrote_the_cache():
    """The fix at its narrowest: one cache, two shapes, one answer.

    `cache_ancestor_utterances` writes the left-hand shape and
    `cache_context_command_utterances` the right-hand one. Both must produce the same
    escalation population and therefore the same floor. This is the assertion that
    fails first if `skip_commands` stops being passed, and it needs no workflow at all.
    """
    core = ["IntentDetection/go_up", "wildcard"]
    ancestor_path_shape = {
        "ChatRoom": {
            "ChatRoom/add_user": ["add a user", "invite someone"],
            "ChatRoom/list_users": ["who is here"],
        }
    }
    trainee_path_shape = {
        "ChatRoom": {
            **ancestor_path_shape["ChatRoom"],
            "IntentDetection/go_up": ["go up", "back out"],
            "wildcard": ["something else"],
        }
    }

    grouped = [
        class_balance.group_ancestor_utterances(
            ["ChatRoom"],
            shape,
            skip_labels=(WILDCARD_LABEL,),
            skip_commands=core,
        )
        for shape in (ancestor_path_shape, trainee_path_shape)
    ]
    assert grouped[0] == grouped[1] == ancestor_path_shape
    assert (
        class_balance.coverage_floor_of(grouped[0])
        == class_balance.coverage_floor_of(grouped[1])
        == 2
    )
    assert class_balance.reserved_candidate_counts(
        grouped[0]) == class_balance.reserved_candidate_counts(grouped[1]) == (3, 3)
