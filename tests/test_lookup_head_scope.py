"""Phase 9 step 3: the scoped head layout, its global persona holdout, and their provenance.

Integration tests on a copy of ``fastworkflow/examples/library`` (no mocks, no model fitting),
plus pure tests on hand-built rows for the split. Nothing here trains an intent head: the
signature, the planner and the publish check read files, and the pilot is run with --dry-run.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.model_pipeline_training import (
    FLATTENED,
    HEAD_SCOPES,
    RESERVED_CLASS_ARMS,
    SCOPED,
    _is_action_only,
    _is_planning_name,
    active_experiment_head_scope,
    experiment_head_scope,
    experiment_reserved_class_arm,
    lookup_holdout_split,
    lookup_split_command_names,
    not_here_seed_rows,
    planning_head_labels,
    own_command_names,
    resolve_head_scope,
    scoped_head_split,
    trained_command_labels,
)
from fastworkflow.nlu_labels import PARAMETER_VALUE_LABEL, WILDCARD_LABEL
from fastworkflow.train import artifact_versioning, determinism, not_here_pilot
from fastworkflow.train import selective_training as st
from fastworkflow.train import utterance_cache
from fastworkflow.train.heldout_evaluation import SEED_PERSONA_ID, LabeledUtterance

REPO = Path(__file__).resolve().parent.parent
LIBRARY = REPO / "fastworkflow" / "examples" / "library"
SEED = 42
HEAD = "Book"  # inherits Record, so it is the head where the two layouts differ


@pytest.fixture(scope="module", autouse=True)
def initialized_fastworkflow():
    fastworkflow.init({})
    yield
    RoutingRegistry.clear_registry()


def _copy(tmp_path: Path, name: str) -> str:
    destination = tmp_path / name
    shutil.copytree(LIBRARY, destination, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()
    return str(destination)


def _labels(workflow: str, scope: str, ctx: str = HEAD) -> set[str]:
    crd = RoutingRegistry.get_definition(workflow, load_cached=False)
    with experiment_reserved_class_arm("sparse_1"), experiment_head_scope(scope):
        return trained_command_labels(ctx, crd)


# ---------------------------------------------------------------- the switch

def test_the_default_layout_is_flattened_and_nothing_is_active():
    assert active_experiment_head_scope() is None
    assert resolve_head_scope() == FLATTENED
    assert HEAD_SCOPES == ("flattened", "scoped")


def test_an_unknown_layout_is_refused():
    with pytest.raises(ValueError, match="unknown head scope"):
        resolve_head_scope("bogus")
    with pytest.raises(ValueError, match="unknown head scope"):
        with experiment_head_scope("bogus"):
            pass


def test_scoped_without_an_arm_is_refused_and_nothing_is_left_set():
    with pytest.raises(ValueError, match="reserved-class arm"):
        with experiment_head_scope(SCOPED):
            pass
    assert active_experiment_head_scope() is None
    with pytest.raises(ValueError, match="reserved-class arm"):
        resolve_head_scope(SCOPED)


def test_scoped_under_a_retired_arm_is_refused_before_any_layout_is_set():
    """Phase 9 step 5 (was: scoped under the control arm is refused because it needs the sparse
    class). The control arm no longer exists, so it is refused by name, before the layout."""
    with pytest.raises(ValueError, match="retired"):
        with experiment_reserved_class_arm("control"):
            pass  # pragma: no cover - refused on entry
    assert active_experiment_head_scope() is None


def test_scoped_under_a_sparse_arm_is_set_and_restored_even_after_an_error():
    with experiment_reserved_class_arm("sparse_1"):
        with experiment_head_scope(SCOPED):
            assert active_experiment_head_scope() == SCOPED
            assert resolve_head_scope() == SCOPED
        assert active_experiment_head_scope() is None
        with pytest.raises(RuntimeError):
            with experiment_head_scope(SCOPED):
                raise RuntimeError("boom")
        assert active_experiment_head_scope() is None


def test_flattened_is_a_layout_too_and_needs_the_sparse_arm():
    # Both layouts draw the one global persona holdout alongside the sparse class.
    with pytest.raises(ValueError, match="reserved-class arm"):
        with experiment_head_scope(FLATTENED):
            pass
    with pytest.raises(ValueError, match="retired"):
        with experiment_reserved_class_arm("control"):
            pass  # pragma: no cover - refused on entry
    with experiment_reserved_class_arm("sparse_8"):
        with experiment_head_scope(FLATTENED):
            assert active_experiment_head_scope() == FLATTENED
        assert active_experiment_head_scope() is None


# ---------------------------------------------------------------- label sets

def test_flattened_labels_are_today_the_inherited_union_plus_core(tmp_path):
    workflow = _copy(tmp_path, "labels_flat")
    crd = RoutingRegistry.get_definition(workflow, load_cached=False)
    core = set(crd.command_directory.core_command_names)
    # Phase 10 step 1 (production): the head's planning labels are added on top of today's labels.
    expected = {
        name for name in set(crd.contexts[HEAD]) | core
        if not _is_planning_name(crd.command_directory, name)
    } | planning_head_labels(HEAD, crd)
    assert _labels(workflow, FLATTENED) == expected
    assert planning_head_labels(HEAD, crd) == {"Book/loan_report", "Book/shelf_report"}
    assert any(name.startswith("Record/") for name in expected)  # inherited, as today


def test_scoped_labels_are_the_own_list_plus_core_and_no_inherited_command(tmp_path):
    workflow = _copy(tmp_path, "labels_scoped")
    crd = RoutingRegistry.get_definition(workflow, load_cached=False)
    core = set(crd.command_directory.core_command_names)
    own = own_command_names(crd, HEAD)
    labels = _labels(workflow, SCOPED)

    assert own and all(name.startswith("Book/") for name in own)
    assert not any(name.startswith("Record/") for name in labels)
    assert labels == {
        name for name in own | core
        if not _is_planning_name(crd.command_directory, name)
        and not _is_action_only(crd.command_directory, name)
    } | planning_head_labels(HEAD, crd)
    assert core <= labels  # core commands stay in every head, including wildcard


def test_the_global_head_never_learns_the_startup_command_under_either_layout(tmp_path):
    workflow = _copy(tmp_path, "labels_star")
    crd = RoutingRegistry.get_definition(workflow, load_cached=False)
    assert "startup" in own_command_names(crd, "*")  # it is a global command in the model
    assert "startup" not in _labels(workflow, SCOPED, "*")
    # Under the flattened layout startup is still a label name, but it has no training rows:
    # it is action-only, so no utterances are generated or seeded for it.
    assert _is_action_only(crd.command_directory, "startup")
    picks, _ = not_here_seed_rows(workflow, crd, _labels(workflow, FLATTENED, "*"), 1)
    assert not picks.get("startup")


def test_the_global_split_names_skip_planning_and_startup(tmp_path):
    workflow = _copy(tmp_path, "split_names")
    crd = RoutingRegistry.get_definition(workflow, load_cached=False)
    contexts = sorted(st.contexts_for_training(workflow))
    names = lookup_split_command_names(crd, contexts)
    owned = set().union(*(own_command_names(crd, ctx) for ctx in contexts))
    # Planning commands are in the planning files, not in any context's own "/" list, so the
    # split never sees them; the filter is the second line of defence.
    assert "Book/loan_report" not in owned
    assert "startup" not in names
    assert not any(_is_planning_name(crd.command_directory, name) for name in names)
    assert set(names) <= owned


# ---------------------------------------------------------------- not-here rows

def test_scoped_not_here_rows_cover_the_base_commands_flattened_had_as_labels(tmp_path):
    workflow = _copy(tmp_path, "not_here_scoped")
    crd = RoutingRegistry.get_definition(workflow, load_cached=False)
    core = set(crd.command_directory.core_command_names)
    execution = set(core)
    for names in crd.contexts.values():
        execution |= set(names)

    scoped_labels = _labels(workflow, SCOPED)
    picks, _available = not_here_seed_rows(workflow, crd, scoped_labels, 1)
    inherited = {name for name in execution if name.startswith("Record/")}
    assert inherited and inherited <= set(picks)  # base commands are foreign under scoped
    # every execution command the head does not label is covered, or is a deliberate exclusion
    uncovered = {
        name for name in execution - scoped_labels
        if name != "wildcard" and not _is_planning_name(crd.command_directory, name)
        and not _is_action_only(crd.command_directory, name)
    }
    assert uncovered <= set(picks)

    flat_labels = _labels(workflow, FLATTENED)
    flat_picks, _ = not_here_seed_rows(workflow, crd, flat_labels, 1)
    assert not any(name.startswith("Record/") for name in flat_picks)


# ---------------------------------------------------------------- the global split (pure)

def _rows():
    """Own rows of three commands across six personas, plus hand-written seeds, as LabeledUtterance."""
    personas = [f"persona-{index}" for index in range(6)]
    rows: list[LabeledUtterance] = []
    for label in ("Book/loan", "Member/join", "Record/add_note"):
        for index, persona in enumerate(personas):
            for phrase in ("first", "second"):
                rows.append(LabeledUtterance(
                    utterance=f"{label} {phrase} {index}", label=label, persona=persona))
    rows.append(LabeledUtterance(utterance="loan seed", label="Book/loan", persona=SEED_PERSONA_ID))
    return rows


def test_the_global_split_is_one_fixed_split_whichever_head_asks():
    rows = _rows()
    first = lookup_holdout_split(rows, seed=7, persona_batches=None)
    second = lookup_holdout_split(list(rows), seed=7, persona_batches=None)
    assert [r.utterance for r in first.heldout] == [r.utterance for r in second.heldout]
    assert first.heldout  # the test has teeth: something is held out

    book_labels = {"Book/loan", "core"}
    member_labels = {"Member/join", "core"}
    book_rows = [r for r in rows if r.label in book_labels]
    member_rows = [r for r in rows if r.label in member_labels]

    # asked in either order, each head sees the same global holdout
    member_first = scoped_head_split(first, member_rows, member_labels)
    book_view = scoped_head_split(first, book_rows, book_labels)
    member_view = scoped_head_split(first, member_rows, member_labels)
    assert [r.utterance for r in member_first.heldout] == [r.utterance for r in member_view.heldout]
    # each head's held-out set is exactly the global held-out rows of its own labels
    assert {r.utterance for r in book_view.heldout} == {
        r.utterance for r in first.heldout if r.label in book_labels}
    assert {r.utterance for r in member_view.heldout} == {
        r.utterance for r in first.heldout if r.label in member_labels}


def test_a_globally_held_out_utterance_never_reaches_any_heads_training_rows():
    rows = _rows()
    global_split = lookup_holdout_split(rows, seed=7, persona_batches=None)
    held = {r.utterance for r in global_split.heldout}
    # a wildcard row and a parameter_value row carrying a held-out text must be dropped too
    head_rows = [
        *[r for r in rows if r.label == "Book/loan"],
        LabeledUtterance(utterance=next(iter(held)), label=WILDCARD_LABEL, persona=SEED_PERSONA_ID),
        LabeledUtterance(utterance=next(iter(held)), label=PARAMETER_VALUE_LABEL, persona=SEED_PERSONA_ID),
    ]
    view = scoped_head_split(global_split, head_rows, {"Book/loan", "core"})
    assert not ({r.utterance for r in view.train} & held)
    assert all(r.label in {"Book/loan", "core"} for r in view.heldout)
    assert all(r.utterance in held for r in view.heldout)
    # seeds are never held out, so the hand-written seed stays in training
    assert any(r.utterance == "loan seed" for r in view.train)


# ---------------------------------------------------------------- provenance and signature

def test_provenance_carries_the_layout_next_to_the_arm(tmp_path):
    armed = tmp_path / "armed"
    armed.mkdir()
    recorder = determinism.ProvenanceRecorder(str(armed))
    recorder.run_metadata = {"reserved_class_arm": "sparse_1", "train_seed": 1, "head_scope": SCOPED}
    recorder.save()
    assert determinism.ProvenanceRecorder.load_run_metadata(str(armed))["head_scope"] == SCOPED


def test_the_signature_records_the_layout_and_the_own_label_space(tmp_path):
    workflow = _copy(tmp_path, "signature_scoped")
    contexts = sorted(st.contexts_for_training(workflow))
    scoped_sig, _ = st.compute_training_signature(
        workflow, contexts, seed=SEED, reserved_class_arm="sparse_1", head_scope=SCOPED)
    book = scoped_sig.contexts[HEAD]
    assert book.head_scope == SCOPED
    assert not any(name.startswith("Record/") for name in book.label_space)

    fields = st.context_contribution_manifest_fields(scoped_sig, st._full_retrain(contexts, "baseline"))
    record = fields[artifact_versioning.CONTEXT_CONTRIBUTIONS_KEY][HEAD]
    assert record["head_scope"] == SCOPED
    assert record["reserved_class_construction"] == "not_here"

    flat_sig, _ = st.compute_training_signature(
        workflow, contexts, seed=SEED, reserved_class_arm="sparse_1")
    assert flat_sig.contexts[HEAD].head_scope is None
    assert any(name.startswith("Record/") for name in flat_sig.contexts[HEAD].label_space)
    assert "head_scope" not in st.context_contribution_manifest_fields(
        flat_sig, st._full_retrain(contexts, "baseline"))[artifact_versioning.CONTEXT_CONTRIBUTIONS_KEY][HEAD]


def test_a_layout_change_is_a_reason_to_retrain_and_none_equals_flattened():
    base = st.ContextSignature(context_name=HEAD, reserved_class_arm="sparse_1")
    scoped = st.ContextSignature(context_name=HEAD, reserved_class_arm="sparse_1", head_scope=SCOPED)
    flat = st.ContextSignature(context_name=HEAD, reserved_class_arm="sparse_1", head_scope=FLATTENED)
    assert any("head layout changed" in d for d in st._diff_context_signature(base, scoped))
    assert st._diff_context_signature(base, flat) == []


# ---------------------------------------------------------------- publish check and planner

def _seed_utterance_cache(workflow: str) -> None:
    root = os.path.join(workflow, "___command_info", utterance_cache.CACHE_DIRNAME)
    os.makedirs(root, exist_ok=True)
    context_commands, _u = st.build_context_maps(workflow, st.contexts_for_training(workflow))
    for commands in context_commands.values():
        for command in commands:
            with open(os.path.join(root, f"{utterance_cache.slugify(command)}.baseline.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"entries": {}}, handle)


def _publish_baseline(workflow: str, arm: str, head_scope=None) -> str:
    """Publish a complete-looking version under the layout. No model is trained."""
    RoutingRegistry.clear_registry()
    _seed_utterance_cache(workflow)
    contexts = sorted(st.contexts_for_training(workflow))
    version = artifact_versioning.new_version_id()
    for context_name in contexts:
        folder = artifact_versioning.version_dir(workflow, version) / artifact_versioning.context_folder_name(
            context_name)
        folder.mkdir(parents=True, exist_ok=True)
        for artifact in st.REQUIRED_CONTEXT_ARTIFACTS:
            (folder / artifact).touch()
    signature, _unresolved = st.compute_training_signature(
        workflow, contexts, seed=SEED, reserved_class_arm=arm, head_scope=head_scope)
    fields = st.context_contribution_manifest_fields(signature, st._full_retrain(contexts, "baseline"))
    artifact_versioning.write_manifest(workflow, version, seed=SEED, **fields)
    st.save_training_signature(workflow, version, signature)
    artifact_versioning.publish_version(workflow, version)
    return version


def _plan(workflow: str, version: str, arm: str, head_scope):
    RoutingRegistry.clear_registry()
    plan, _signature = st.compute_training_plan(
        workflow, sorted(st.contexts_for_training(workflow)),
        changed_only=True, seed=SEED, carry_forward_from=version,
        reserved_class_arm=arm, head_scope=head_scope)
    return plan


def _edit_seed(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("plain_utterances = [",
                                 'plain_utterances = [\n        "seed added by the scoped test",', 1),
                    encoding="utf-8")


def test_the_publish_check_accepts_a_scoped_version_and_refuses_a_mixed_one(tmp_path):
    workflow = _copy(tmp_path, "publish_scoped")
    version = _publish_baseline(workflow, arm="sparse_1", head_scope=SCOPED)
    assert artifact_versioning.verify_version_consistency(workflow, version).problems == []

    manifest_path = artifact_versioning.version_dir(workflow, version) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    block = manifest[artifact_versioning.CONTEXT_CONTRIBUTIONS_KEY]
    block[sorted(block)[0]]["head_scope"] = FLATTENED
    manifest_path.write_text(json.dumps(manifest))
    problems = artifact_versioning.verify_version_consistency(workflow, version).problems
    assert any("mixes head layouts" in problem for problem in problems)


def test_a_scoped_seed_edit_retrains_every_head(tmp_path):
    workflow = _copy(tmp_path, "scoped_edit")
    version = _publish_baseline(workflow, arm="sparse_1", head_scope=SCOPED)
    _edit_seed(Path(workflow) / "_commands" / "Record" / "add_note.py")

    plan = _plan(workflow, version, "sparse_1", SCOPED)
    assert set(plan.contexts_to_train) == set(st.contexts_for_training(workflow))
    assert any("persona holdout" in reason for reasons in plan.reasons.values() for reason in reasons)


def test_a_flattened_seed_edit_also_retrains_every_head_under_the_global_holdout(tmp_path):
    workflow = _copy(tmp_path, "flat_edit")
    version = _publish_baseline(workflow, arm="sparse_1", head_scope=FLATTENED)
    _edit_seed(Path(workflow) / "_commands" / "Record" / "add_note.py")
    plan = _plan(workflow, version, "sparse_1", FLATTENED)
    assert set(plan.contexts_to_train) == set(st.contexts_for_training(workflow))


def test_switching_the_layout_marks_every_head_dirty(tmp_path):
    workflow = _copy(tmp_path, "layout_switch")
    flat_version = _publish_baseline(workflow, arm="sparse_1", head_scope=None)
    plan = _plan(workflow, flat_version, "sparse_1", SCOPED)
    assert set(plan.contexts_to_train) == set(st.contexts_for_training(workflow))


# ---------------------------------------------------------------- the pilot driver

def test_the_driver_layout_choices_match_the_trainer():
    assert tuple(not_here_pilot.HEAD_SCOPE_CHOICES) == HEAD_SCOPES
    assert all(RESERVED_CLASS_ARMS[arm][0] == "not_here" for arm in not_here_pilot.SCOPED_ARMS)


@pytest.mark.parametrize("layout", ["flattened", "scoped"])
def test_driver_refuses_any_layout_under_a_non_sparse_arm(tmp_path, layout):
    with pytest.raises(SystemExit) as refused:
        not_here_pilot.main(["--arm", "control", "--head-scope", layout,
                             "--scratch", str(tmp_path / "s"), "--env-file", "x.env", "--dry-run"])
    assert refused.value.code == 2


def test_driver_dry_run_prints_the_head_scope_and_touches_nothing(tmp_path):
    scratch = tmp_path / "scratch"
    source = tmp_path / "source"
    source.mkdir()
    cache = tmp_path / "cache"
    (cache / "___command_info" / "utterance_cache").mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, "-m", "fastworkflow.train.not_here_pilot", "--arm", "sparse_1",
         "--head-scope", "scoped", "--scratch", str(scratch), "--env-file", "fastworkflow.env.nemotron",
         "--source", str(source), "--cache-from", str(cache), "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "head scope scoped" in result.stdout
    assert not scratch.exists()


# ---------------------------------------------------------------- train() glue, either layout

def _glue_run(tmp_path, monkeypatch, scope: str, name: str):
    """Drive the real ``train()`` under *scope* and return (export document, per-head views).

    Synthetic generation is replaced only at its LLM boundary (the seam
    test_escalation_floor_symmetry.py uses), with rows that carry persona attribution so the
    persona split has something to hold out. Routing, the per-head assembly, the global split,
    the held-out removal and the export are the shipped code. ``stop_before_fit`` stops
    before any model is fitted.
    """
    from fastworkflow.model_pipeline_training import LOOKUP_HOLDOUT_FILENAME, train
    from fastworkflow.train import generate_synthetic
    from fastworkflow.train.determinism import (
        ProvenanceRecorder,
        UtteranceProvenance,
        set_provenance_recorder,
    )
    import fastworkflow.model_pipeline_training as mpt

    workflow_path = str(tmp_path / name)
    shutil.copytree(REPO / "fastworkflow" / "examples" / "messaging_app_4", workflow_path,
                    ignore=shutil.ignore_patterns("___*", "__pycache__"))
    RoutingRegistry.clear_registry()

    def deterministic_generation(seed_utterances, command_name, **_kwargs):
        rows = [f"{command_name} phrase {index}" for index in range(12)]
        personas = {row: f"persona-{index % 4}" for index, row in enumerate(rows)}
        return rows, UtteranceProvenance(
            command_name=command_name, seed=SEED,
            persona_ids=sorted(set(personas.values())), utterance_personas=personas,
            # one persona per batch, so a one-persona holdout fits
            generator_config={"personas_per_batch": 1},
            seed_utterance_count=len(seed_utterances),
            generated_count=len(rows), final_count=len(rows))

    monkeypatch.setattr(generate_synthetic, "generate_diverse_utterances_with_provenance",
                        deterministic_generation)

    recorder = ProvenanceRecorder(workflow_path)
    set_provenance_recorder(recorder)
    mpt._HOLDOUT_EXPORT = {}
    workflow = fastworkflow.Workflow.create(workflow_path, workflow_id_str=f"lookup-glue-{name}")
    try:
        with experiment_reserved_class_arm("sparse_1"), experiment_head_scope(scope):
            with contextlib.redirect_stdout(io.StringIO()):
                train(workflow, stop_before_fit=True)
        heads = dict(mpt._HOLDOUT_EXPORT)
    finally:
        mpt._HOLDOUT_EXPORT = None
        set_provenance_recorder(None)
        workflow.close()
        RoutingRegistry.clear_registry()

    export = Path(workflow_path, "___command_info", LOOKUP_HOLDOUT_FILENAME)
    assert export.is_file(), f"a {scope} run must write one global holdout"
    document = json.loads(export.read_text(encoding="utf-8"))
    return document, heads


def test_the_global_split_is_identical_under_flattened_and_scoped(tmp_path, monkeypatch):
    flat_doc, _ = _glue_run(tmp_path, monkeypatch, FLATTENED, "glue_flat")
    scoped_doc, _ = _glue_run(tmp_path, monkeypatch, SCOPED, "glue_scoped")
    assert flat_doc["head_scope"] == FLATTENED and scoped_doc["head_scope"] == SCOPED
    assert flat_doc["arm"] == scoped_doc["arm"] == "sparse_1"
    flat_held = {case["utterance"] for case in flat_doc["heldout"]}
    scoped_held = {case["utterance"] for case in scoped_doc["heldout"]}
    assert flat_held, "the test has teeth: a persona holdout was drawn"
    assert flat_held == scoped_held
    assert flat_doc["heldout"] == scoped_doc["heldout"]


def test_under_flattened_no_head_trains_on_a_globally_held_out_text(tmp_path, monkeypatch):
    document, heads = _glue_run(tmp_path, monkeypatch, FLATTENED, "glue_flat_leak")
    held_texts = {case["utterance"] for case in document["heldout"]}
    assert held_texts and heads
    for ctx_name, head in heads.items():
        head_labels = set(head["own_labels"])
        assert not (set(head["training_utterances"]) & held_texts), ctx_name
        assert {r["utterance"] for r in head["own_heldout"]} == {
            c["utterance"] for c in document["heldout"] if c["label"] in head_labels}, ctx_name


def test_under_scoped_no_head_trains_on_a_globally_held_out_text(tmp_path, monkeypatch):
    document, heads = _glue_run(tmp_path, monkeypatch, SCOPED, "glue_scoped_leak")
    held_texts = {case["utterance"] for case in document["heldout"]}
    assert held_texts and heads
    for ctx_name, head in heads.items():
        assert not (set(head["training_utterances"]) & held_texts), ctx_name
